#!/usr/bin/env python3
"""Build a DINO cache manifest from ScanNet MV pkl img_paths only.

This script intentionally does not scan raw 2D directories. It only records
frames referenced by the MV info pkl files, so the offline cache is aligned
with the actual ESAM/AutoSeg3D MV protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_data_list(pkl_path: Path) -> List[dict]:
    with pkl_path.open('rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and 'data_list' in obj:
        return obj['data_list']
    if isinstance(obj, list):
        return obj
    raise TypeError(f'Unsupported pkl format: {pkl_path}, type={type(obj)}')


def cache_rel_for_img(rel_img: str, suffix: str = '.pt') -> str:
    p = Path(rel_img)
    return str(p.with_suffix(suffix))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='data/scannet200-mv_fast')
    ap.add_argument('--train-pkl', default='data/scannet200-mv_fast/scannet200_mv_oneformer3d_infos_train.pkl')
    ap.add_argument('--val-pkl', default='data/scannet200-mv_fast/scannet200_mv_oneformer3d_infos_val.pkl')
    ap.add_argument('--cache-root', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--subset-size', type=int, default=128)
    ap.add_argument('--seed', type=int, default=2050318485)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    cache_root = Path(args.cache_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: Dict[str, dict] = {}
    split_counts = {}
    pkl_info = {}
    for split, pkl in [('train', Path(args.train_pkl)), ('val', Path(args.val_pkl))]:
        dl = load_data_list(pkl)
        pkl_info[split] = {
            'path': str(pkl),
            'sha256': sha256_file(pkl),
            'num_scenes': len(dl),
        }
        n_refs = 0
        for scene_idx, item in enumerate(dl):
            for frame_idx, rel in enumerate(item.get('img_paths', []) or []):
                n_refs += 1
                rel = str(rel)
                abs_img = data_root / rel
                rec = records.setdefault(rel, {
                    'rel_img_path': rel,
                    'abs_img_path': str(abs_img),
                    'cache_rel_path': cache_rel_for_img(rel),
                    'cache_path': str(cache_root / cache_rel_for_img(rel)),
                    'splits': [],
                    'refs': [],
                    'exists': abs_img.exists(),
                })
                if split not in rec['splits']:
                    rec['splits'].append(split)
                if len(rec['refs']) < 8:
                    rec['refs'].append({'split': split, 'scene_idx': scene_idx, 'frame_idx': frame_idx})
        split_counts[split] = {'num_scenes': len(dl), 'num_frame_refs': n_refs}

    all_recs = [records[k] for k in sorted(records)]
    missing = [r for r in all_recs if not r['exists']]
    manifest = out_dir / 'scannet200_mv_fast_dino_used_frames_manifest.jsonl'
    with manifest.open('w') as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    import random
    rng = random.Random(args.seed)
    subset = list(all_recs)
    rng.shuffle(subset)
    subset = subset[: min(args.subset_size, len(subset))]
    subset_path = out_dir / f'scannet200_mv_fast_dino_used_frames_subset{len(subset)}_seed{args.seed}.jsonl'
    with subset_path.open('w') as f:
        for r in subset:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    summary = {
        'data_root': str(data_root),
        'cache_root': str(cache_root),
        'manifest': str(manifest),
        'subset_manifest': str(subset_path),
        'seed': args.seed,
        'pkl_info': pkl_info,
        'split_counts': split_counts,
        'unique_frames': len(all_recs),
        'missing_images': len(missing),
        'missing_examples': missing[:10],
    }
    summary_path = out_dir / 'scannet200_mv_fast_dino_used_frames_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if missing:
        raise SystemExit(f'{len(missing)} images referenced by pkl are missing')


if __name__ == '__main__':
    main()
