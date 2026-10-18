#!/usr/bin/env python3
"""Granularity and DINO-edge diagnostics for GCI first-principles validation.

This script is intentionally standalone: it reads the preprocessed MV pkl +
point/instance/superpoint files directly and does not instantiate the training
model. It answers two questions:
  1) Are lifted VFM fragments over-fragmenting GT objects?
  2) Can projected DINOv2 fragment prototypes separate same-object from
     different-object fragment pairs?
"""
import argparse
import csv
import json
import math
import os
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MIN_DEPTH = 0.3
BASE_INTR = (577.870605, 577.870605, 319.5, 239.5)


def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'data_list' in data:
        data = data['data_list']
    return data


def scene_id_from_item(item):
    m = re.search(r'scene\d{4}_\d{2}', str(item))
    return m.group(0) if m else 'unknown'


def cache_path_from_img(cache_root, img_path, data_root='data/scannet200-mv_fast/'):
    p = str(img_path)
    data_root = data_root.rstrip('/') + '/'
    if p.startswith(data_root):
        rel = p[len(data_root):]
    else:
        marker = 'scannet200-mv_fast/'
        rel = p.split(marker, 1)[1] if marker in p else p
    rel = os.path.splitext(rel)[0] + '.pt'
    return os.path.join(cache_root, rel)


def load_bin(path, dtype, shape_last=None):
    arr = np.fromfile(path, dtype=dtype)
    if shape_last is not None:
        arr = arr.reshape(-1, shape_last)
    return arr


def project_points(points_xyz, pose, feat_hw, img_hw=(420, 560), max_depth=20.0):
    """Auto-select inv/direct/identity pose mode by valid projection ratio."""
    xyz = torch.as_tensor(points_xyz, dtype=torch.float32)
    pose_t = torch.as_tensor(pose, dtype=torch.float32)
    C_H, C_W = feat_hw
    img_h, img_w = img_hw
    fx, fy, cx, cy = BASE_INTR
    # BuildCamInfoFromPoses scales base 480x640 to 420x560, then model scales to feature map.
    intr_img = (fx * img_w / 640.0, fy * img_h / 480.0, cx * img_w / 640.0, cy * img_h / 480.0)
    scale_w = C_W / float(img_w)
    scale_h = C_H / float(img_h)
    intr_feat = (
        intr_img[0] * scale_w,
        intr_img[1] * scale_h,
        (intr_img[2] + 0.5) * scale_w - 0.5,
        (intr_img[3] + 0.5) * scale_h - 0.5,
    )

    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype)
    xyz_h = torch.cat([xyz, ones], dim=1)
    modes = []
    try:
        modes.append(('inv', torch.linalg.inv(pose_t)))
    except Exception:
        pass
    modes.append(('direct', pose_t))
    modes.append(('identity', torch.eye(4, dtype=torch.float32)))
    best = None
    for mode, mat in modes:
        cam = (mat @ xyz_h.t()).t()[:, :3]
        x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
        valid_z = (z > MIN_DEPTH) & (z < max_depth)
        safe_z = torch.where(valid_z, z, torch.ones_like(z))
        u = intr_feat[0] * (x / safe_z) + intr_feat[2]
        v = intr_feat[1] * (y / safe_z) + intr_feat[3]
        valid = valid_z & (u >= 0) & (u < C_W) & (v >= 0) & (v < C_H)
        ratio = float(valid.float().mean().item()) if valid.numel() else 0.0
        if best is None or ratio > best[2]:
            uv = torch.stack([u, v], dim=1)
            best = (uv, valid, ratio, mode)
    return best


def sample_feat(feat_chw, uv, valid):
    if feat_chw.dtype == torch.float16 and feat_chw.device.type == 'cpu':
        feat_chw = feat_chw.float()
    C, H, W = feat_chw.shape
    idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    out = torch.zeros((uv.shape[0], C), dtype=torch.float32)
    if idx.numel() == 0:
        return out
    uvv = uv[idx]
    # align_corners=False, same as v5 path.
    x_norm = 2.0 * (uvv[:, 0] + 0.5) / float(W) - 1.0
    y_norm = 2.0 * (uvv[:, 1] + 0.5) / float(H) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(feat_chw.float().unsqueeze(0), grid, mode='bilinear', align_corners=False).squeeze(3).squeeze(0).T
    out[idx] = sampled.float()
    return out


def compute_sp_centroids(points_xyz, sp_mask, sp_ids):
    id_to_idx = {int(s): i for i, s in enumerate(sp_ids.tolist())}
    cent = np.zeros((len(sp_ids), 3), dtype=np.float32)
    counts = np.zeros(len(sp_ids), dtype=np.int64)
    inv = np.asarray([id_to_idx[int(x)] for x in sp_mask], dtype=np.int64)
    np.add.at(cent, inv, points_xyz.astype(np.float32))
    np.add.at(counts, inv, 1)
    cent = cent / np.maximum(counts[:, None], 1)
    return cent, counts


def pool_sp_dino(sampled, valid, sp_mask, min_valid_points=5):
    sp_ids, inv = np.unique(sp_mask, return_inverse=True)
    n = len(sp_ids)
    valid_np = valid.cpu().numpy().astype(bool)
    feats = sampled.cpu().numpy().astype(np.float32)
    C = feats.shape[1]
    sums = np.zeros((n, C), dtype=np.float32)
    counts = np.zeros(n, dtype=np.int64)
    inv_valid = inv[valid_np]
    if inv_valid.size:
        np.add.at(sums, inv_valid, feats[valid_np])
        np.add.at(counts, inv_valid, 1)
    ok = counts >= min_valid_points
    proto = np.zeros_like(sums)
    proto[ok] = sums[ok] / np.maximum(counts[ok, None], 1)
    norms = np.linalg.norm(proto, axis=1, keepdims=True) + 1e-6
    proto = proto / norms
    return sp_ids, proto, counts, ok


def frame_granularity(sp_mask, inst_mask, min_overlap_points=20, min_gt_frac=0.01, purity_thr=0.8):
    valid = inst_mask >= 0
    sp_ids = np.unique(sp_mask)
    gt_ids = np.unique(inst_mask[valid]) if valid.any() else np.array([], dtype=np.int64)
    gt_sizes = {int(g): int((inst_mask == g).sum()) for g in gt_ids}

    sp_records = []
    gt_to_sp = defaultdict(list)
    for sp in sp_ids:
        idx = sp_mask == sp
        total = int(idx.sum())
        gt_vals = inst_mask[idx]
        gt_vals = gt_vals[gt_vals >= 0]
        if gt_vals.size == 0:
            sp_records.append((int(sp), total, -1, 0.0, 0, 0))
            continue
        cnt = Counter(gt_vals.tolist())
        dom_gt, dom_count = cnt.most_common(1)[0]
        purity = dom_count / max(int(gt_vals.size), 1)
        n_gt_nontriv = sum(1 for _, c in cnt.items() if c >= min_overlap_points)
        sp_records.append((int(sp), total, int(dom_gt), float(purity), int(n_gt_nontriv), int(gt_vals.size)))
        for g, c in cnt.items():
            if c >= min_overlap_points and c / max(gt_sizes.get(int(g), 1), 1) >= min_gt_frac:
                gt_to_sp[int(g)].append((int(sp), int(c)))

    frags_per_gt = [len(v) for g, v in gt_to_sp.items() if gt_sizes.get(g, 0) >= min_overlap_points]
    boundary_cross = [r for r in sp_records if r[4] >= 2]
    low_purity = [r for r in sp_records if r[2] >= 0 and r[3] < purity_thr]
    valid_sp = [r for r in sp_records if r[2] >= 0]
    purity_vals = [r[3] for r in valid_sp]
    return dict(
        n_sp=int(len(sp_ids)),
        n_gt=int(len(gt_ids)),
        mean_fragments_per_gt=float(np.mean(frags_per_gt)) if frags_per_gt else 0.0,
        median_fragments_per_gt=float(np.median(frags_per_gt)) if frags_per_gt else 0.0,
        max_fragments_per_gt=int(max(frags_per_gt)) if frags_per_gt else 0,
        mean_fragment_purity=float(np.mean(purity_vals)) if purity_vals else 0.0,
        boundary_cross_rate=float(len(boundary_cross) / max(len(valid_sp), 1)),
        low_purity_rate=float(len(low_purity) / max(len(valid_sp), 1)),
        n_valid_sp=int(len(valid_sp)),
        sp_records=sp_records,
        gt_to_sp=gt_to_sp,
    )


def sample_edge_scores(sp_records, sp_ids, sp_proto, sp_ok, sp_centroids=None, max_pos=200, max_neg=400, purity_thr=0.8, rng=None):
    rng = rng or random.Random(0)
    id_to_idx = {int(s): i for i, s in enumerate(sp_ids.tolist())}
    groups = defaultdict(list)
    for sp, total, dom_gt, purity, n_gt, valid_points in sp_records:
        if dom_gt >= 0 and purity >= purity_thr and sp in id_to_idx and sp_ok[id_to_idx[sp]]:
            groups[int(dom_gt)].append(int(sp))
    pos = []
    for g, sps in groups.items():
        if len(sps) < 2:
            continue
        for _ in range(min(max_pos, len(sps) * 2)):
            a, b = rng.sample(sps, 2)
            pos.append((a, b, g, g))
    if len(pos) > max_pos:
        pos = rng.sample(pos, max_pos)
    all_sps = [(sp, g) for g, sps in groups.items() for sp in sps]
    neg = []
    if len(all_sps) >= 2:
        tries = 0
        while len(neg) < max_neg and tries < max_neg * 20:
            (a, ga), (b, gb) = rng.sample(all_sps, 2)
            if ga != gb:
                neg.append((a, b, ga, gb))
            tries += 1
    def score(row, label):
        a, b, ga, gb = row
        ia, ib = id_to_idx[a], id_to_idx[b]
        sim = float(np.dot(sp_proto[ia], sp_proto[ib]))
        dist = ''
        if sp_centroids is not None:
            dist = float(np.linalg.norm(sp_centroids[ia] - sp_centroids[ib]))
        return sim, dict(label=label, sp_a=int(a), sp_b=int(b), gt_a=int(ga), gt_b=int(gb), dino_sim=sim, centroid_dist=dist)
    pos_items = [score(p, 1) for p in pos]
    neg_items = [score(p, 0) for p in neg]
    pos_scores = [x[0] for x in pos_items]
    neg_scores = [x[0] for x in neg_items]
    pair_rows = [x[1] for x in pos_items] + [x[1] for x in neg_items]
    return pos_scores, neg_scores, pair_rows


def threshold_report(pair_rows, taus=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95), dist_gates=(None, 0.25, 0.5, 1.0)):
    reports = []
    for gate in dist_gates:
        rows = pair_rows
        if gate is not None:
            rows = [r for r in pair_rows if r.get('centroid_dist', '') != '' and float(r['centroid_dist']) <= gate]
        for tau in taus:
            accepted = [r for r in rows if float(r['dino_sim']) >= tau]
            tp = sum(1 for r in accepted if int(r['label']) == 1)
            fp = sum(1 for r in accepted if int(r['label']) == 0)
            pos_total = sum(1 for r in rows if int(r['label']) == 1)
            neg_total = sum(1 for r in rows if int(r['label']) == 0)
            reports.append(dict(
                dist_gate=gate if gate is not None else 'none', tau=tau,
                accepted=len(accepted), tp=tp, fp=fp,
                precision=float(tp / max(tp + fp, 1)),
                recall=float(tp / max(pos_total, 1)),
                fpr=float(fp / max(neg_total, 1))))
    return reports


def auc_from_scores(pos, neg, max_pairs=200000, seed=0):
    if not pos or not neg:
        return None
    rng = random.Random(seed)
    total = len(pos) * len(neg)
    if total <= max_pairs:
        vals = [(p > n) + 0.5 * (p == n) for p in pos for n in neg]
        return float(np.mean(vals))
    wins = 0.0
    for _ in range(max_pairs):
        p = rng.choice(pos); n = rng.choice(neg)
        wins += (p > n) + 0.5 * (p == n)
    return float(wins / max_pairs)


def choose_frame_indices(n, max_frames):
    if max_frames <= 0 or n <= max_frames:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, num=max_frames).round().astype(int).tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann-file', required=True)
    ap.add_argument('--data-root', default='data/scannet200-mv_fast')
    ap.add_argument('--dino-cache-root', default='')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--max-scenes', type=int, default=10)
    ap.add_argument('--max-frames-per-scene', type=int, default=8)
    ap.add_argument('--seed', type=int, default=2050318485)
    ap.add_argument('--min-overlap-points', type=int, default=20)
    ap.add_argument('--purity-thr', type=float, default=0.8)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_pkl(args.ann_file)
    if args.max_scenes > 0:
        data = data[:args.max_scenes]

    frame_rows = []
    edge_pos, edge_neg = [], []
    edge_pair_rows = []
    missing_dino = 0
    processed_frames = 0

    for scene_idx, item in enumerate(data):
        scene = scene_id_from_item(item)
        frame_indices = choose_frame_indices(len(item['pts_paths']), args.max_frames_per_scene)
        for frame_i in frame_indices:
            pts_path = os.path.join(args.data_root, item['pts_paths'][frame_i])
            sp_path = os.path.join(args.data_root, item['super_pts_paths'][frame_i])
            ins_path = os.path.join(args.data_root, item['pts_instance_mask_paths'][frame_i])
            img_path = item.get('img_paths', [None])[frame_i]
            pts = load_bin(pts_path, np.float32, 6)
            sp = load_bin(sp_path, np.int64)
            ins = load_bin(ins_path, np.int64)
            if not (len(pts) == len(sp) == len(ins)):
                raise RuntimeError(f'shape mismatch scene={scene} frame={frame_i}: pts={len(pts)} sp={len(sp)} ins={len(ins)}')
            g = frame_granularity(sp, ins, min_overlap_points=args.min_overlap_points, purity_thr=args.purity_thr)
            dino_valid_rate = None
            pos_scores = []
            neg_scores = []
            if args.dino_cache_root:
                cp = cache_path_from_img(args.dino_cache_root, img_path)
                if os.path.exists(cp):
                    payload = torch.load(cp, map_location='cpu')
                    feat = payload.get('feat') if isinstance(payload, dict) else payload
                    feat = feat.contiguous()
                    uv, valid, ratio, mode = project_points(pts[:, :3], item['poses'][frame_i], tuple(feat.shape[-2:]))
                    sampled = sample_feat(feat, uv, valid)
                    sp_ids, proto, counts, ok = pool_sp_dino(sampled, valid, sp)
                    sp_centroids, _ = compute_sp_centroids(pts[:, :3], sp, sp_ids)
                    dino_valid_rate = ratio
                    pos_scores, neg_scores, pair_rows = sample_edge_scores(g['sp_records'], sp_ids, proto, ok, sp_centroids=sp_centroids, rng=rng, purity_thr=args.purity_thr)
                    for row in pair_rows:
                        row.update(scene=scene, frame_i=int(frame_i))
                    edge_pair_rows.extend(pair_rows)
                    edge_pos.extend(pos_scores)
                    edge_neg.extend(neg_scores)
                else:
                    missing_dino += 1
            frame_rows.append(dict(
                scene=scene,
                scene_idx=scene_idx,
                frame_i=frame_i,
                n_points=len(pts),
                n_sp=g['n_sp'],
                n_gt=g['n_gt'],
                mean_fragments_per_gt=g['mean_fragments_per_gt'],
                median_fragments_per_gt=g['median_fragments_per_gt'],
                max_fragments_per_gt=g['max_fragments_per_gt'],
                mean_fragment_purity=g['mean_fragment_purity'],
                boundary_cross_rate=g['boundary_cross_rate'],
                low_purity_rate=g['low_purity_rate'],
                dino_valid_rate=dino_valid_rate if dino_valid_rate is not None else '',
                n_pos_edges=len(pos_scores),
                n_neg_edges=len(neg_scores),
                pos_dino_mean=float(np.mean(pos_scores)) if pos_scores else '',
                neg_dino_mean=float(np.mean(neg_scores)) if neg_scores else '',
            ))
            processed_frames += 1
            if processed_frames % 20 == 0:
                print(f'[progress] frames={processed_frames} scenes={scene_idx+1} edge_pos={len(edge_pos)} edge_neg={len(edge_neg)}')

    csv_path = out_dir / 'frame_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
        writer.writeheader(); writer.writerows(frame_rows)

    def mean_key(k):
        vals = [r[k] for r in frame_rows if r[k] != '']
        return float(np.mean(vals)) if vals else None
    def median_key(k):
        vals = [r[k] for r in frame_rows if r[k] != '']
        return float(np.median(vals)) if vals else None
    auc = auc_from_scores(edge_pos, edge_neg, seed=args.seed)
    summary = dict(
        ann_file=args.ann_file,
        data_root=args.data_root,
        dino_cache_root=args.dino_cache_root,
        scenes=len(data),
        frames=processed_frames,
        missing_dino_frames=missing_dino,
        mean_n_sp=mean_key('n_sp'),
        mean_n_gt=mean_key('n_gt'),
        mean_fragments_per_gt=mean_key('mean_fragments_per_gt'),
        median_frame_fragments_per_gt=median_key('median_fragments_per_gt'),
        mean_max_fragments_per_gt=mean_key('max_fragments_per_gt'),
        mean_fragment_purity=mean_key('mean_fragment_purity'),
        mean_boundary_cross_rate=mean_key('boundary_cross_rate'),
        mean_low_purity_rate=mean_key('low_purity_rate'),
        mean_dino_valid_rate=mean_key('dino_valid_rate'),
        edge_pos_count=len(edge_pos),
        edge_neg_count=len(edge_neg),
        edge_pos_dino_mean=float(np.mean(edge_pos)) if edge_pos else None,
        edge_pos_dino_std=float(np.std(edge_pos)) if edge_pos else None,
        edge_neg_dino_mean=float(np.mean(edge_neg)) if edge_neg else None,
        edge_neg_dino_std=float(np.std(edge_neg)) if edge_neg else None,
        edge_dino_auc=auc,
    )
    threshold_reports = threshold_report(edge_pair_rows)
    summary['threshold_reports'] = threshold_reports
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(out_dir / 'edge_scores.npz', pos=np.asarray(edge_pos, dtype=np.float32), neg=np.asarray(edge_neg, dtype=np.float32))
    if edge_pair_rows:
        with open(out_dir / 'edge_pairs.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(edge_pair_rows[0].keys()))
            writer.writeheader(); writer.writerows(edge_pair_rows)
    if threshold_reports:
        with open(out_dir / 'threshold_report.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(threshold_reports[0].keys()))
            writer.writeheader(); writer.writerows(threshold_reports)
    print(json.dumps(summary, indent=2))
    print(f'[saved] {out_dir}')


if __name__ == '__main__':
    main()
