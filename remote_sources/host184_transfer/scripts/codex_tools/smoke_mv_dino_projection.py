#!/usr/bin/env python3
"""Standalone projection smoke for offline DINO cache.

It validates the protocol-critical part before full training: MV pkl img_paths
map to cache files, 2D feature maps have expected shape, and points project into
feature-grid coordinates with high valid ratio under the same pose convention
used by v5 (`inv`, with optional auto comparison).
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib.util
_proj_path = REPO_ROOT / "oneformer3d" / "projection_utils.py"
_spec = importlib.util.spec_from_file_location("codex_projection_utils", _proj_path)
_projection_utils = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_projection_utils)
project_points_to_uv = _projection_utils.project_points_to_uv
sample_img_feat = _projection_utils.sample_img_feat

SCANNT_INTR = (577.870605, 577.870605, 319.5, 239.5)


def load_data_list(path: Path) -> List[dict]:
    with path.open('rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and 'data_list' in obj:
        return obj['data_list']
    if isinstance(obj, list):
        return obj
    raise TypeError(type(obj))


def read_bin(path: Path, dtype, cols: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=dtype)
    if arr.size % cols != 0:
        raise RuntimeError(f'{path} size {arr.size} not divisible by {cols}')
    return arr.reshape(-1, cols)


def load_points(data_root: Path, item: dict, frame_idx: int) -> torch.Tensor:
    pts_paths = item.get('pts_paths') or item.get('point_cloud', {}).get('pts_paths')
    if pts_paths is None:
        raise KeyError('pts_paths')
    rel = pts_paths[frame_idx]
    path = data_root / rel
    # ESAM/AutoSeg ScanNet pkl stores float32 points with xyzrgb-like features.
    raw = np.fromfile(path, dtype=np.float32)
    for cols in (6, 9, 12, 3):
        if raw.size % cols == 0:
            arr = raw.reshape(-1, cols)
            if arr.shape[1] >= 3:
                return torch.from_numpy(arr[:, :3].copy()).float()
    raise RuntimeError(f'cannot infer point columns for {path}, raw_size={raw.size}')


def parse_cache(cache_path: Path) -> torch.Tensor:
    payload = torch.load(cache_path, map_location='cpu')
    feat = payload['feat'] if isinstance(payload, dict) and 'feat' in payload else payload
    if not torch.is_tensor(feat):
        raise TypeError(f'cache {cache_path} does not contain tensor feat')
    if tuple(feat.shape[-2:]) != (30, 40) or feat.shape[0] != 1024:
        raise RuntimeError(f'unexpected feature shape {tuple(feat.shape)} for {cache_path}')
    return feat.float().unsqueeze(0)


def project_one(points: torch.Tensor, pose: Any, feat: torch.Tensor, mode: str) -> Tuple[float, int, int, str]:
    device = feat.device
    xyz_world = points.to(device)
    pose_t = torch.as_tensor(pose, device=device, dtype=xyz_world.dtype)
    candidates = []
    if mode in ('inv', 'auto'):
        candidates.append(('inv', torch.linalg.inv(pose_t)))
    if mode in ('direct', 'auto'):
        candidates.append(('direct', pose_t))
    if mode in ('identity', 'auto'):
        candidates.append(('identity', None))
    best = None
    H_feat, W_feat = feat.shape[-2:]
    # Match BuildCamInfoFromPoses + ResizeForDINO: ScanNet intrinsics are
    # first scaled from original 640x480 image to target 560x420, then mapped
    # to DINO patch grid.
    H0, W0 = 480.0, 640.0
    H_img, W_img = 420.0, 560.0
    fx0, fy0, cx0, cy0 = SCANNT_INTR
    fx = fx0 * (W_img / W0)
    fy = fy0 * (H_img / H0)
    cx = cx0 * (W_img / W0)
    cy = cy0 * (H_img / H0)
    scale_w, scale_h = W_feat / W_img, H_feat / H_img
    intr_feat = (fx * scale_w, fy * scale_h, (cx + 0.5) * scale_w - 0.5, (cy + 0.5) * scale_h - 0.5)
    ones = torch.ones((xyz_world.shape[0], 1), device=device, dtype=xyz_world.dtype)
    xyz_h = torch.cat([xyz_world, ones], dim=1)
    for name, T in candidates:
        xyz_cam = xyz_world if T is None else (xyz_h @ T.t())[:, :3]
        uv, valid = project_points_to_uv(
            xyz_cam,
            feat_hw=(H_feat, W_feat),
            max_depth=20.0,
            standard_intrinsics=intr_feat,
            already_scaled=True,
        )
        cnt = int(valid.sum().item())
        total = int(valid.numel())
        rate = cnt / max(total, 1)
        if best is None or rate > best[0]:
            best = (rate, cnt, total, name, uv, valid)
    assert best is not None
    # Also sample once to catch grid_sample / coordinate errors.
    _ = sample_img_feat(feat, best[4], best[5], align_corners=False)
    return float(best[0]), int(best[1]), int(best[2]), str(best[3])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default='data/scannet200-mv_fast/scannet200_mv_oneformer3d_infos_train.pkl')
    ap.add_argument('--data-root', default='data/scannet200-mv_fast')
    ap.add_argument('--cache-root', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--num-scenes', type=int, default=8)
    ap.add_argument('--frames-per-scene', type=int, default=4)
    ap.add_argument('--pose-mode', choices=['inv', 'direct', 'identity', 'auto'], default='auto')
    ap.add_argument('--summary', required=True)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    cache_root = Path(args.cache_root)
    manifest_rels = []
    with Path(args.manifest).open() as f:
        for line in f:
            if line.strip():
                manifest_rels.append(json.loads(line)['rel_img_path'])
    manifest_set = set(manifest_rels)
    dl = load_data_list(Path(args.pkl))
    rows = []
    total_valid = 0
    total_points = 0
    pose_counts: Dict[str, int] = {}
    checked = 0

    # Prefer exact scene/frame references stored in the manifest. This lets a
    # random subset cache smoke validate only frames that were actually cached.
    target_pairs = []
    manifest_records = []
    with Path(args.manifest).open() as f:
        for line in f:
            if line.strip():
                manifest_records.append(json.loads(line))
    for rec in manifest_records:
        cache_path = cache_root / rec['cache_rel_path']
        if not cache_path.exists():
            continue
        refs = rec.get('refs') or []
        chosen_ref = None
        for ref in refs:
            if ref.get('split') == 'train':
                chosen_ref = ref
                break
        # This smoke loads the train pkl; skip val-only cached frames here.
        if chosen_ref is not None:
            target_pairs.append((int(chosen_ref['scene_idx']), int(chosen_ref['frame_idx']), rec['rel_img_path']))
    if not target_pairs:
        # Fallback: deterministic first scenes, useful only after a full cache exists.
        for scene_idx, item in enumerate(dl[:args.num_scenes]):
            img_paths = item['img_paths']
            if not img_paths:
                continue
            idxs = np.linspace(0, len(img_paths) - 1, num=min(args.frames_per_scene, len(img_paths)), dtype=int).tolist()
            for frame_idx in idxs:
                target_pairs.append((scene_idx, int(frame_idx), img_paths[int(frame_idx)]))
    max_checks = args.num_scenes * args.frames_per_scene
    target_pairs = target_pairs[:max_checks]

    for scene_idx, frame_idx, rel in target_pairs:
        item = dl[scene_idx]
        poses = item['poses']
        if rel not in manifest_set:
            raise RuntimeError(f'{rel} absent from manifest')
        cache_path = cache_root / Path(rel).with_suffix('.pt')
        if not cache_path.exists():
            rows.append({'scene_idx': scene_idx, 'frame_idx': frame_idx, 'rel_img_path': rel, 'status': 'missing_cache'})
            continue
        feat = parse_cache(cache_path)
        points = load_points(data_root, item, frame_idx)
        rate, cnt, total, chosen = project_one(points, poses[frame_idx], feat, args.pose_mode)
        total_valid += cnt
        total_points += total
        pose_counts[chosen] = pose_counts.get(chosen, 0) + 1
        checked += 1
        rows.append({
            'scene_idx': scene_idx,
            'frame_idx': frame_idx,
            'rel_img_path': rel,
            'cache_path': str(cache_path),
            'valid_rate': rate,
            'valid_count': cnt,
            'total_points': total,
            'pose_mode': chosen,
            'feature_shape': list(feat.shape),
            'status': 'ok',
        })
    rates = [r['valid_rate'] for r in rows if r.get('status') == 'ok']
    summary = {
        'pkl': args.pkl,
        'data_root': args.data_root,
        'cache_root': str(cache_root),
        'manifest': args.manifest,
        'num_checked': checked,
        'num_missing_cache': sum(1 for r in rows if r.get('status') == 'missing_cache'),
        'mean_valid_rate': (sum(rates) / len(rates)) if rates else None,
        'min_valid_rate': min(rates) if rates else None,
        'max_valid_rate': max(rates) if rates else None,
        'global_valid_rate': (total_valid / total_points) if total_points else None,
        'pose_mode_counts': pose_counts,
        'rows': rows,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if checked == 0:
        raise SystemExit('no cached frames checked')
    if summary['min_valid_rate'] is not None and summary['min_valid_rate'] < 0.90:
        raise SystemExit(f"projection valid rate below 0.90: min={summary['min_valid_rate']}")


if __name__ == '__main__':
    main()
