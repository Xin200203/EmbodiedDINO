#!/usr/bin/env python3
"""Evaluate raw/fixed/learned GCI seed granularity without full model inference."""
import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from gci_granularity_diagnostics import (  # noqa: E402
    cache_path_from_img,
    choose_frame_indices,
    compute_sp_centroids,
    frame_granularity,
    load_bin,
    load_pkl,
    pool_sp_dino,
    project_points,
    sample_feat,
    scene_id_from_item,
)

spec = importlib.util.spec_from_file_location("gci_core", REPO_ROOT / "oneformer3d" / "gci.py")
gci = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gci
spec.loader.exec_module(gci)


def load_calibrator(path, device="cpu"):
    payload = torch.load(path, map_location=device)
    model = gci.GCIEdgeMLP(in_dim=len(payload["feature_names"]))
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return payload, model


def summarize(rows):
    out = {}
    if not rows:
        return out
    keys = rows[0].keys()
    for k in keys:
        if k in ("scene", "frame_i", "mode", "param"):
            continue
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float, np.integer, np.floating))]
        if vals:
            out[k] = float(np.mean(vals))
    return out


def fixed_scores(proto, pairs):
    proto = gci.l2_normalize(proto.astype(np.float32))
    return np.sum(proto[pairs[:, 0]] * proto[pairs[:, 1]], axis=1).astype(np.float32) if len(pairs) else np.zeros(0, dtype=np.float32)


def learned_scores(model, payload, feats):
    if len(feats) == 0:
        return np.zeros(0, dtype=np.float32)
    mean = payload["feature_mean"].astype(np.float32)
    std = payload["feature_std"].astype(np.float32)
    x = torch.from_numpy((feats - mean) / std).float()
    with torch.no_grad():
        p = torch.sigmoid(model(x)).cpu().numpy().astype(np.float32)
    return p


def add_row(rows, scene, frame_i, mode, param, valid_rate, raw, seed, labels, accepted, stats):
    rows.append(dict(
        scene=scene, frame_i=int(frame_i), mode=mode, param=param,
        dino_valid_rate=float(valid_rate),
        raw_n_sp=raw["n_sp"], seed_n_sp=seed["n_sp"], sp_reduction=raw["n_sp"] - seed["n_sp"],
        raw_fragments_per_gt=raw["mean_fragments_per_gt"], seed_fragments_per_gt=seed["mean_fragments_per_gt"],
        raw_max_fragments_per_gt=raw["max_fragments_per_gt"], seed_max_fragments_per_gt=seed["max_fragments_per_gt"],
        raw_purity=raw["mean_fragment_purity"], seed_purity=seed["mean_fragment_purity"],
        raw_boundary_cross=raw["boundary_cross_rate"], seed_boundary_cross=seed["boundary_cross_rate"],
        raw_low_purity=raw["low_purity_rate"], seed_low_purity=seed["low_purity_rate"],
        accepted_edges=len(accepted),
        n_clusters=int(len(np.unique(labels))),
        candidate_edges=stats.n_candidate_edges,
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-file", required=True)
    ap.add_argument("--data-root", default="data/scannet200-mv_fast")
    ap.add_argument("--dino-cache-root", required=True)
    ap.add_argument("--calibrator", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-scenes", type=int, default=50)
    ap.add_argument("--max-frames-per-scene", type=int, default=4)
    ap.add_argument("--max-pair-dist", type=float, default=1.0)
    ap.add_argument("--fixed-taus", default="0.90,0.95")
    ap.add_argument("--learned-thresholds", default="auto")
    ap.add_argument("--min-valid-points", type=int, default=5)
    ap.add_argument("--min-valid-rate", type=float, default=0.90)
    ap.add_argument("--purity-thr", type=float, default=0.8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fixed_taus = [float(x) for x in args.fixed_taus.split(",") if x]
    payload = model = None
    learned_thresholds = []
    if args.calibrator:
        payload, model = load_calibrator(args.calibrator)
        for x in args.learned_thresholds.split(","):
            x = x.strip()
            if not x:
                continue
            learned_thresholds.append(float(payload.get("threshold", payload.get("summary", {}).get("chosen_threshold", {}).get("threshold", 0.5))) if x == "auto" else float(x))

    data = load_pkl(args.ann_file)
    if args.max_scenes > 0:
        data = data[:args.max_scenes]

    rows = []
    frames = 0
    missing = 0
    for scene_idx, item in enumerate(data):
        scene = scene_id_from_item(item)
        for frame_i in choose_frame_indices(len(item["pts_paths"]), args.max_frames_per_scene):
            pts = load_bin(os.path.join(args.data_root, item["pts_paths"][frame_i]), np.float32, 6)
            sp = load_bin(os.path.join(args.data_root, item["super_pts_paths"][frame_i]), np.int64)
            ins = load_bin(os.path.join(args.data_root, item["pts_instance_mask_paths"][frame_i]), np.int64)
            raw = frame_granularity(sp, ins, purity_thr=args.purity_thr)
            cp = cache_path_from_img(args.dino_cache_root, item["img_paths"][frame_i])
            if not os.path.exists(cp):
                missing += 1
                continue
            payload_feat = torch.load(cp, map_location="cpu")
            feat = payload_feat.get("feat") if isinstance(payload_feat, dict) else payload_feat
            uv, valid, ratio, pose_mode = project_points(pts[:, :3], item["poses"][frame_i], tuple(feat.shape[-2:]))
            if ratio < args.min_valid_rate:
                print(f"[WARN][projection] scene={scene} frame={frame_i} valid_rate={ratio:.4f} < {args.min_valid_rate:.2f}")
            sampled = sample_feat(feat, uv, valid)
            sp_ids, proto, valid_counts, ok = pool_sp_dino(sampled, valid, sp, min_valid_points=args.min_valid_points)
            centroids, point_counts = compute_sp_centroids(pts[:, :3], sp, sp_ids)
            pairs = gci.make_candidate_pairs(ok, centroids=centroids, max_dist=args.max_pair_dist)
            feats = gci.edge_features(proto, centroids, point_counts, valid_counts, pairs)
            for tau in fixed_taus:
                scores = fixed_scores(proto, pairs)
                labels, accepted, stats = gci.complete_link_clusters(len(sp_ids), pairs, scores, threshold=tau)
                seed_mask = gci.make_seed_mask(sp, sp_ids, labels)
                seed = frame_granularity(seed_mask, ins, purity_thr=args.purity_thr)
                add_row(rows, scene, frame_i, "fixed_dino", f"tau={tau:.3f},dist={args.max_pair_dist}", ratio, raw, seed, labels, accepted, stats)
            if model is not None:
                probs = learned_scores(model, payload, feats)
                for thr in learned_thresholds:
                    labels, accepted, stats = gci.complete_link_clusters(len(sp_ids), pairs, probs, threshold=thr)
                    seed_mask = gci.make_seed_mask(sp, sp_ids, labels)
                    seed = frame_granularity(seed_mask, ins, purity_thr=args.purity_thr)
                    add_row(rows, scene, frame_i, "learned", f"thr={thr:.3f},dist={args.max_pair_dist}", ratio, raw, seed, labels, accepted, stats)
            frames += 1
            if frames % 20 == 0:
                print(f"[progress] frames={frames} rows={len(rows)} missing={missing}")

    if not rows:
        raise RuntimeError("no rows produced")
    with open(out_dir / "seed_frame_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = dict(args=vars(args), frames=frames, scenes=len(data), missing_dino_frames=missing, configs={})
    for mode in sorted(set(r["mode"] for r in rows)):
        for param in sorted(set(r["param"] for r in rows if r["mode"] == mode)):
            sub = [r for r in rows if r["mode"] == mode and r["param"] == param]
            summary["configs"][f"{mode}:{param}"] = summarize(sub)
    with open(out_dir / "seed_summary.json", "w") as f: json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
