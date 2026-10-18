#!/usr/bin/env python3
"""Audit FastSAM fragment quality against visible GT instance masks.

This is a read-only data audit. It loads the same preprocessed annotation pkl
and per-frame binary masks used by the existing codex calibration scripts, then
compares each FastSAM superpoint fragment with visible GT instances.
"""
import argparse
import csv
import json
import pickle
import re
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_BINS = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0000001)


def load_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "data_list" in data:
        data = data["data_list"]
    return data


def load_bin(path, dtype, shape_last=None):
    arr = np.fromfile(path, dtype=dtype)
    if shape_last is not None:
        arr = arr.reshape(-1, shape_last)
    return arr


def scene_id_from_item(item):
    m = re.search(r"scene\d{4}_\d{2}", str(item))
    return m.group(0) if m else "unknown"


def choose_frame_indices(n_frames, max_frames):
    if max_frames <= 0 or n_frames <= max_frames:
        return list(range(n_frames))
    return sorted(set(np.linspace(0, n_frames - 1, num=max_frames).round().astype(int).tolist()))


def resolve_path(path, base_dir):
    path = Path(path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_config(path):
    path = resolve_path(path, Path.cwd())
    try:
        from mmengine.config import Config

        return Config.fromfile(str(path)).to_dict(), path
    except Exception as exc:
        namespace = {"__file__": str(path)}
        try:
            code = compile(path.read_text(), str(path), "exec")
            exec(code, namespace)
        except Exception as fallback_exc:
            raise RuntimeError(
                f"failed to load config with mmengine ({exc}) or direct exec ({fallback_exc})"
            ) from fallback_exc
        return {k: v for k, v in namespace.items() if not k.startswith("__")}, path


def dataset_from_config(cfg):
    for loader_key in ("val_dataloader", "test_dataloader", "train_dataloader"):
        loader = cfg.get(loader_key)
        if not isinstance(loader, Mapping):
            continue
        dataset = loader.get("dataset", loader)
        while isinstance(dataset, Mapping) and "dataset" in dataset and "ann_file" not in dataset:
            dataset = dataset["dataset"]
        if isinstance(dataset, Mapping) and dataset.get("ann_file"):
            return loader_key, dataset
    raise KeyError("could not find dataset.ann_file in val/test/train dataloader config")


def resolve_dataset_paths(config_path):
    cfg, config_path = load_config(config_path)
    loader_key, dataset = dataset_from_config(cfg)
    data_root = dataset.get("data_root") or cfg.get("data_root") or ""
    if not data_root:
        raise KeyError("could not resolve data_root from config dataset or top-level config")
    data_root = resolve_path(data_root, REPO_ROOT)
    ann_file = Path(dataset["ann_file"])
    if not ann_file.is_absolute():
        ann_file = data_root / ann_file
    return loader_key, data_root, ann_file.resolve()


def audit_frame(scene, scene_idx, frame_i, sp_mask, inst_mask):
    if len(sp_mask) != len(inst_mask):
        raise RuntimeError(
            f"shape mismatch scene={scene} frame={frame_i}: sp={len(sp_mask)} ins={len(inst_mask)}"
        )
    visible = inst_mask >= 0
    gt_ids, gt_counts = np.unique(inst_mask[visible], return_counts=True) if visible.any() else ([], [])
    gt_sizes = {int(g): int(c) for g, c in zip(gt_ids, gt_counts)}

    rows = []
    for fragment_id in np.unique(sp_mask):
        idx = sp_mask == fragment_id
        size = int(idx.sum())
        visible_gt = inst_mask[idx]
        visible_gt = visible_gt[visible_gt >= 0]
        visible_size = int(visible_gt.size)
        valid = visible_size > 0
        dominant_gt = -1
        dominant_overlap = 0
        n_overlapping_gt = 0
        purity = 0.0
        coverage = 0.0
        if valid:
            counts = Counter(int(x) for x in visible_gt.tolist())
            dominant_gt, dominant_overlap = counts.most_common(1)[0]
            n_overlapping_gt = len(counts)
            purity = float(dominant_overlap / max(size, 1))
            coverage = float(dominant_overlap / max(gt_sizes.get(int(dominant_gt), 1), 1))
        rows.append(
            dict(
                scene=scene,
                scene_idx=int(scene_idx),
                frame_i=int(frame_i),
                fragment_id=int(fragment_id),
                size=size,
                visible_size=visible_size,
                dominant_gt=int(dominant_gt),
                dominant_overlap=int(dominant_overlap),
                dominant_gt_visible_size=int(gt_sizes.get(int(dominant_gt), 0)),
                n_overlapping_gt=int(n_overlapping_gt),
                purity=purity,
                coverage=coverage,
                valid=bool(valid),
            )
        )
    return rows, dict(n_visible_gt=int(len(gt_sizes)), visible_points=int(visible.sum()))


def quantiles(values):
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    qs = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)
    return {f"p{q}": float(np.percentile(arr, q)) for q in qs}


def bin_counts(values, bins=DEFAULT_BINS):
    counts = {}
    arr = np.asarray(values, dtype=np.float64)
    for lo, hi in zip(bins[:-1], bins[1:]):
        label = f"[{lo:.2f},{min(hi, 1.0):.2f}{']' if hi > 1.0 else ')'}"
        if arr.size == 0:
            counts[label] = 0
        elif hi > 1.0:
            counts[label] = int(((arr >= lo) & (arr <= 1.0)).sum())
        else:
            counts[label] = int(((arr >= lo) & (arr < hi)).sum())
    return counts


def rankdata(values):
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def correlations(rows):
    valid_rows = [r for r in rows if r["valid"]]
    pairs = {
        "size_vs_purity": ("size", "purity"),
        "size_vs_coverage": ("size", "coverage"),
        "purity_vs_coverage": ("purity", "coverage"),
        "visible_size_vs_purity": ("visible_size", "purity"),
        "visible_size_vs_coverage": ("visible_size", "coverage"),
    }
    out = {}
    for name, (a_key, b_key) in pairs.items():
        a = [r[a_key] for r in valid_rows]
        b = [r[b_key] for r in valid_rows]
        out[name] = {
            "pearson": pearson(a, b),
            "spearman": pearson(rankdata(a), rankdata(b)) if len(a) >= 2 else None,
            "n": len(a),
        }
    return out


def summarize(rows, frame_rows, args, loader_key, data_root, ann_file):
    valid_rows = [r for r in rows if r["valid"]]
    invalid_rows = [r for r in rows if not r["valid"]]
    purity = [r["purity"] for r in valid_rows]
    coverage = [r["coverage"] for r in valid_rows]
    sizes = [r["size"] for r in rows]
    visible_sizes = [r["visible_size"] for r in valid_rows]
    return dict(
        config=args.config,
        loader_key=loader_key,
        ann_file=str(ann_file),
        data_root=str(data_root),
        max_scenes=args.max_scenes,
        max_frames=args.max_frames,
        scenes_processed=len({r["scene"] for r in frame_rows}),
        frames_processed=len(frame_rows),
        fragments_total=len(rows),
        fragments_valid=len(valid_rows),
        fragments_invalid=len(invalid_rows),
        valid_fragment_rate=float(len(valid_rows) / max(len(rows), 1)),
        size_quantiles=quantiles(sizes),
        visible_size_quantiles=quantiles(visible_sizes),
        purity_quantiles=quantiles(purity),
        coverage_quantiles=quantiles(coverage),
        purity_bins=bin_counts(purity),
        coverage_bins=bin_counts(coverage),
        correlations=correlations(rows),
        frames=dict(
            n_visible_gt_quantiles=quantiles([r["n_visible_gt"] for r in frame_rows]),
            visible_points_quantiles=quantiles([r["visible_points"] for r in frame_rows]),
            fragments_per_frame_quantiles=quantiles([r["n_fragments"] for r in frame_rows]),
            valid_fragments_per_frame_quantiles=quantiles([r["n_valid_fragments"] for r in frame_rows]),
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="mmengine config with dataset ann_file/data_root")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-scenes", type=int, default=0, help="0 audits all scenes")
    ap.add_argument("--max-frames", type=int, default=0, help="0 audits all frames per scene")
    args = ap.parse_args()

    loader_key, data_root, ann_file = resolve_dataset_paths(args.config)
    data = load_pkl(ann_file)
    if args.max_scenes > 0:
        data = data[: args.max_scenes]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fragment_rows = []
    frame_rows = []
    for scene_idx, item in enumerate(data):
        scene = scene_id_from_item(item)
        frame_indices = choose_frame_indices(len(item["pts_paths"]), args.max_frames)
        for frame_i in frame_indices:
            sp_path = data_root / item["super_pts_paths"][frame_i]
            ins_path = data_root / item["pts_instance_mask_paths"][frame_i]
            sp = load_bin(sp_path, np.int64)
            ins = load_bin(ins_path, np.int64)
            rows, frame_info = audit_frame(scene, scene_idx, frame_i, sp, ins)
            fragment_rows.extend(rows)
            frame_rows.append(
                dict(
                    scene=scene,
                    scene_idx=int(scene_idx),
                    frame_i=int(frame_i),
                    n_fragments=len(rows),
                    n_valid_fragments=sum(1 for r in rows if r["valid"]),
                    **frame_info,
                )
            )
        if (scene_idx + 1) % 10 == 0:
            print(f"[progress] scenes={scene_idx + 1} frames={len(frame_rows)} fragments={len(fragment_rows)}")

    if not fragment_rows:
        raise RuntimeError("no fragments audited; check ann_file and max limits")

    with open(out_dir / "fragments.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fragment_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fragment_rows)

    summary = summarize(fragment_rows, frame_rows, args, loader_key, data_root, ann_file)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
