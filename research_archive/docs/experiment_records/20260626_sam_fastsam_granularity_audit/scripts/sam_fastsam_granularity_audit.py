#!/usr/bin/env python3
"""SAM/FastSAM proposal-granularity diagnostics.

Standalone, read-only against processed data. It does not instantiate models.
"""
import csv
import json
import math
import os
import pickle
import re
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and 'data_list' in obj:
        return obj['data_list']
    return obj


def scene_id_from_item(item):
    s = str(item)
    m = re.search(r'scene\d{4}_\d{2}', s)
    return m.group(0) if m else 'unknown'


def frame_id_from_path(p):
    base = os.path.basename(str(p))
    stem = os.path.splitext(base)[0]
    try:
        return int(stem)
    except Exception:
        return None


def choose_frame_indices(n, max_frames):
    if max_frames <= 0 or n <= max_frames:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, num=max_frames).round().astype(int).tolist()))


def load_bin(path, dtype, shape_last=None):
    arr = np.fromfile(path, dtype=dtype)
    if shape_last is not None:
        arr = arr.reshape(-1, shape_last)
    return arr


def frame_granularity(sp_mask, inst_mask, min_overlap_points=20, min_gt_frac=0.01, purity_thr=0.8):
    valid = inst_mask >= 0
    sp_ids = np.unique(sp_mask)
    gt_ids = np.unique(inst_mask[valid]) if valid.any() else np.array([], dtype=np.int64)
    gt_sizes = {int(g): int((inst_mask == g).sum()) for g in gt_ids}

    gt_to_sp = defaultdict(list)
    valid_sp = 0
    purity_vals = []
    boundary_cross = 0
    low_purity = 0
    sp_point_counts = []

    for sp in sp_ids:
        idx = sp_mask == sp
        total = int(idx.sum())
        sp_point_counts.append(total)
        vals = inst_mask[idx]
        vals = vals[vals >= 0]
        if vals.size == 0:
            continue
        valid_sp += 1
        cnt = Counter(vals.tolist())
        dom_gt, dom_count = cnt.most_common(1)[0]
        purity = dom_count / max(int(vals.size), 1)
        purity_vals.append(float(purity))
        n_nontriv = sum(1 for _, c in cnt.items() if c >= min_overlap_points)
        if n_nontriv >= 2:
            boundary_cross += 1
        if purity < purity_thr:
            low_purity += 1
        for g, c in cnt.items():
            if c >= min_overlap_points and c / max(gt_sizes.get(int(g), 1), 1) >= min_gt_frac:
                gt_to_sp[int(g)].append((int(sp), int(c)))

    frags_per_gt = [len(v) for g, v in gt_to_sp.items() if gt_sizes.get(g, 0) >= min_overlap_points]
    return dict(
        n_sp=int(len(sp_ids)),
        n_gt=int(len(gt_ids)),
        mean_fragments_per_gt=float(np.mean(frags_per_gt)) if frags_per_gt else 0.0,
        median_fragments_per_gt=float(np.median(frags_per_gt)) if frags_per_gt else 0.0,
        max_fragments_per_gt=int(max(frags_per_gt)) if frags_per_gt else 0,
        mean_fragment_purity=float(np.mean(purity_vals)) if purity_vals else 0.0,
        boundary_cross_rate=float(boundary_cross / max(valid_sp, 1)),
        low_purity_rate=float(low_purity / max(valid_sp, 1)),
        median_sp_points=float(np.median(sp_point_counts)) if sp_point_counts else 0.0,
        mean_sp_points=float(np.mean(sp_point_counts)) if sp_point_counts else 0.0,
    )


def mean(vals):
    vals = [v for v in vals if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def median(vals):
    vals = [v for v in vals if v is not None]
    return float(stats.median(vals)) if vals else None


def summarize_rows(rows, keys):
    out = {'n': len(rows)}
    for k in keys:
        vals = [r.get(k) for r in rows if isinstance(r.get(k), (int, float))]
        out[f'{k}_mean'] = mean(vals)
        out[f'{k}_median'] = median(vals)
        out[f'{k}_min'] = float(min(vals)) if vals else None
        out[f'{k}_max'] = float(max(vals)) if vals else None
    return out


def parse_seg2d_jsonl(paths):
    rows = []
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    keys = ['num_masks_raw', 'num_masks_used', 'pixel_coverage', 'mean_mask_area', 'median_mask_area', 'num_boxes']
    by_split = {}
    for split in sorted(set(r.get('split', 'unknown') for r in rows)):
        by_split[split] = summarize_rows([r for r in rows if r.get('split', 'unknown') == split], keys)
    return rows, {'overall': summarize_rows(rows, keys), 'by_split': by_split}


def parse_sam_auto_csv(path):
    if not os.path.exists(path):
        return [], {}
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rr = dict(r)
            for k in ['sam_auto_mask_count', 'sam_auto_area_mean', 'sam_auto_area_median']:
                rr[k] = float(rr[k])
            m = re.search(r'(scene\d{4}_\d{2}).*/(\d+)\.jpg', rr.get('image', ''))
            if m:
                rr['scene'] = m.group(1); rr['frame_id'] = int(m.group(2))
            rows.append(rr)
    return rows, summarize_rows(rows, ['sam_auto_mask_count', 'sam_auto_area_mean', 'sam_auto_area_median'])


def compare_sam_sample_to_seg2d(sam_rows, seg_rows):
    index = {}
    for r in seg_rows:
        scene = r.get('scan_name')
        frame = r.get('frame_id')
        if scene is not None and frame is not None:
            index[(scene, int(frame))] = r
    out = []
    for sr in sam_rows:
        key = (sr.get('scene'), sr.get('frame_id'))
        fr = index.get(key)
        row = dict(scene=key[0], frame_id=key[1], sam_auto_mask_count=sr.get('sam_auto_mask_count'), sam_auto_area_median=sr.get('sam_auto_area_median'))
        if fr:
            row.update(fastsam_masks=fr.get('num_masks_used'), fastsam_area_median=fr.get('median_mask_area'), fastsam_coverage=fr.get('pixel_coverage'))
            if fr.get('num_masks_used'):
                row['sam_to_fastsam_mask_ratio'] = float(sr.get('sam_auto_mask_count') / max(fr.get('num_masks_used'), 1))
        out.append(row)
    return out


def compute_3d_for_run(name, data_root, ann_file, out_csv, max_scenes=50, max_frames_per_scene=4, scene_filter=None):
    data = load_pkl(ann_file)
    if scene_filter:
        wanted = set(scene_filter)
        data = [it for it in data if scene_id_from_item(it) in wanted]
    if max_scenes > 0:
        data = data[:max_scenes]
    rows = []
    for scene_idx, item in enumerate(data):
        scene = scene_id_from_item(item)
        for frame_i in choose_frame_indices(len(item['pts_paths']), max_frames_per_scene):
            img_path = item.get('img_paths', [''])[frame_i]
            frame_id = frame_id_from_path(img_path)
            sp_path = os.path.join(data_root, item['super_pts_paths'][frame_i])
            ins_path = os.path.join(data_root, item['pts_instance_mask_paths'][frame_i])
            if not os.path.exists(sp_path) or not os.path.exists(ins_path):
                continue
            sp = load_bin(sp_path, np.int64)
            ins = load_bin(ins_path, np.int64)
            if len(sp) != len(ins):
                raise RuntimeError(f'shape mismatch {name} {scene} {frame_i}: sp={len(sp)} ins={len(ins)}')
            g = frame_granularity(sp, ins)
            g.update(run=name, scene=scene, scene_idx=scene_idx, frame_i=int(frame_i), frame_id=frame_id)
            rows.append(g)
    if rows:
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    summary = summarize_rows(rows, ['n_sp','n_gt','mean_fragments_per_gt','median_fragments_per_gt','max_fragments_per_gt','mean_fragment_purity','boundary_cross_rate','low_purity_rate','median_sp_points','mean_sp_points'])
    summary.update(run=name, data_root=data_root, ann_file=ann_file, scenes=len(data), frames=len(rows))
    return rows, summary


def main():
    out_dir = Path(os.environ.get('AUDIT_OUT', '/tmp/sam_fastsam_granularity_audit'))
    out_dir.mkdir(parents=True, exist_ok=True)

    fast_root = '/datadisk1/xxy/data/data-ESAM/scannet200-mv_fast/runs_mv/fastsam_everything_stride20_img640'
    yolo_fast_root = '/datadisk1/xxy/data/data-ESAM/scannet200-mv_fast/runs_mv/yolo_fastsam_prompt_conf0p25_expand1p10_stride20_img640'
    dedup_root = '/datadisk1/xxy/data/data-ESAM/scannet200-mv_fast/runs_mv/fastsam_dedup_everything_stride40_img640_20260123'
    m2f_root = '/datadisk1/xxy/data/data-ESAM/scannet200-mv_fast/runs_mv/mask2former_panoptic_val_full_rerun_20260122'

    seg_fast_rows, seg_fast_summary = parse_seg2d_jsonl([fast_root + '/seg2d_stats.rank0.jsonl', fast_root + '/seg2d_stats.rank1.jsonl'])
    seg_yolo_rows, seg_yolo_summary = parse_seg2d_jsonl([yolo_fast_root + '/seg2d_stats.rank0.jsonl', yolo_fast_root + '/seg2d_stats.rank1.jsonl'])
    seg_dedup_rows, seg_dedup_summary = parse_seg2d_jsonl([dedup_root + '/seg2d_stats.rank0.jsonl'])

    sam_csv = '/datadisk1/xxy/3D_Reconstruction_v5_184sync_20260603/test/_sam_overseg_outputs_v2/sam_auto_mask_counts.csv'
    yolo_sam_csv = '/datadisk1/xxy/3D_Reconstruction_v5_184sync_20260603/test/_sam_overseg_outputs_v2/yolo_sam_metrics.csv'
    sam_rows, sam_summary = parse_sam_auto_csv(sam_csv)
    sam_vs_fast = compare_sam_sample_to_seg2d(sam_rows, seg_fast_rows + seg_dedup_rows)
    if sam_vs_fast:
        with open(out_dir / 'sam_auto_vs_fastsam_sample.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(sam_vs_fast[0].keys()))
            w.writeheader(); w.writerows(sam_vs_fast)

    # Same scene filter from fast subset50 where possible.
    subset_ann = fast_root + '/scannet200_mv_oneformer3d_infos_val_subset50.pkl'
    subset_scenes = [scene_id_from_item(it) for it in load_pkl(subset_ann)]

    run_specs = [
        ('fastsam_everything_stride20_subset50', fast_root, subset_ann),
        ('yolo_fastsam_prompt_stride20_same50', yolo_fast_root, yolo_fast_root + '/scannet200_mv_oneformer3d_infos_val.pkl'),
        ('fastsam_dedup_stride40_same50', dedup_root, dedup_root + '/scannet200_mv_oneformer3d_infos_val.pkl'),
        ('mask2former_val_rerun_same50', m2f_root, m2f_root + '/scannet200_mv_oneformer3d_infos_val.pkl'),
    ]
    three_d_summaries = []
    for name, root, ann in run_specs:
        if not os.path.exists(ann):
            three_d_summaries.append({'run': name, 'missing_ann': ann})
            continue
        rows, summary = compute_3d_for_run(name, root, ann, out_dir / f'{name}_frame_metrics.csv', max_scenes=50, max_frames_per_scene=4, scene_filter=subset_scenes if 'subset50' not in name else None)
        three_d_summaries.append(summary)

    result = {
        'seg2d': {
            'fastsam_everything_stride20': seg_fast_summary,
            'yolo_fastsam_prompt_stride20': seg_yolo_summary,
            'fastsam_dedup_stride40': seg_dedup_summary,
            'sam_auto_sample': sam_summary,
        },
        'sam_vs_fastsam_sample': summarize_rows(sam_vs_fast, ['sam_auto_mask_count','fastsam_masks','sam_to_fastsam_mask_ratio','sam_auto_area_median','fastsam_area_median']),
        'three_d': three_d_summaries,
        'paths': {
            'fast_root': fast_root,
            'yolo_fast_root': yolo_fast_root,
            'dedup_root': dedup_root,
            'm2f_root': m2f_root,
            'sam_csv': sam_csv,
            'yolo_sam_csv': yolo_sam_csv,
        }
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(result, f, indent=2)

    # Markdown summary for quick copy.
    with open(out_dir / 'summary.md', 'w') as f:
        f.write('# SAM/FastSAM granularity audit summary\n\n')
        f.write('## 2D proposal summaries\n\n')
        for name, summ in result['seg2d'].items():
            f.write(f'### {name}\n\n```json\n{json.dumps(summ, indent=2)}\n```\n\n')
        f.write('## 3D lifting/query granularity summaries\n\n')
        for summ in three_d_summaries:
            f.write(f'### {summ.get("run")}\n\n```json\n{json.dumps(summ, indent=2)}\n```\n\n')
        f.write('## SAM auto vs matched FastSAM sampled frames\n\n')
        f.write('```json\n' + json.dumps(result['sam_vs_fastsam_sample'], indent=2) + '\n```\n')

    print(json.dumps(result, indent=2))
    print(f'[saved] {out_dir}')


if __name__ == '__main__':
    main()
