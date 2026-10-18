#!/usr/bin/env python3
"""Train a lightweight learned GCI edge calibrator from GT-labeled fragments.

This is an offline mechanism test.  It does not change the model and should not
be used to claim test-set performance by itself.
"""
import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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


def auc_score(y, s):
    y = np.asarray(y).astype(np.int64)
    s = np.asarray(s).astype(np.float64)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(s), dtype=np.float64) + 1.0
    # tie handling: average ranks for equal scores
    vals, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for k, c in enumerate(counts):
            if c > 1:
                idx = np.nonzero(inv == k)[0]
                ranks[idx] = ranks[idx].mean()
    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def metrics_at_threshold(y, p, thr):
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p).astype(np.float64)
    pred = p >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return dict(threshold=float(thr), tp=tp, fp=fp, fn=fn, tn=tn, precision=float(precision), recall=float(recall), f1=float(f1))


def best_threshold(y, p, min_precision=0.95):
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 91), np.asarray(p, dtype=np.float64)]))
    rows = [metrics_at_threshold(y, p, t) for t in candidates]
    feasible = [r for r in rows if r["precision"] >= min_precision and (r["tp"] + r["fp"]) > 0]
    if feasible:
        return max(feasible, key=lambda r: (r["recall"], r["f1"], r["threshold"])), rows
    return max(rows, key=lambda r: (r["f1"], r["precision"])), rows


def sample_frame_pairs(sp_records, sp_ids, proto, ok, centroids, point_counts, valid_counts, purity_thr, max_dist, max_pos, max_neg, rng):
    id_to_idx = {int(s): i for i, s in enumerate(sp_ids.tolist())}
    dom = {}
    for sp, total, dom_gt, purity, n_gt, valid_points in sp_records:
        if dom_gt >= 0 and purity >= purity_thr and int(sp) in id_to_idx:
            idx = id_to_idx[int(sp)]
            if bool(ok[idx]):
                dom[idx] = int(dom_gt)
    idxs = sorted(dom)
    pos, neg = [], []
    for ai, i in enumerate(idxs):
        for j in idxs[ai + 1:]:
            if max_dist > 0 and float(np.linalg.norm(centroids[i] - centroids[j])) > max_dist:
                continue
            if dom[i] == dom[j]:
                pos.append((i, j))
            else:
                neg.append((i, j))
    if len(pos) > max_pos:
        pos = rng.sample(pos, max_pos)
    if len(neg) > max_neg:
        neg = rng.sample(neg, max_neg)
    pairs = np.asarray(pos + neg, dtype=np.int64).reshape(-1, 2)
    labels = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.float32)
    if len(labels) == 0:
        return None, None
    feats = gci.edge_features(proto, centroids, point_counts, valid_counts, pairs)
    return feats, labels


def collect_dataset(args):
    rng = random.Random(args.seed)
    data = load_pkl(args.ann_file)
    if args.max_scenes > 0:
        data = data[:args.max_scenes]
    Xs, ys, scenes = [], [], []
    frame_rows = []
    missing = 0
    frames = 0
    for scene_idx, item in enumerate(data):
        scene = scene_id_from_item(item)
        for frame_i in choose_frame_indices(len(item["pts_paths"]), args.max_frames_per_scene):
            pts = load_bin(os.path.join(args.data_root, item["pts_paths"][frame_i]), np.float32, 6)
            sp = load_bin(os.path.join(args.data_root, item["super_pts_paths"][frame_i]), np.int64)
            ins = load_bin(os.path.join(args.data_root, item["pts_instance_mask_paths"][frame_i]), np.int64)
            if not (len(pts) == len(sp) == len(ins)):
                raise RuntimeError(f"shape mismatch scene={scene} frame={frame_i}: pts={len(pts)} sp={len(sp)} ins={len(ins)}")
            cp = cache_path_from_img(args.dino_cache_root, item["img_paths"][frame_i])
            if not os.path.exists(cp):
                missing += 1
                continue
            payload = torch.load(cp, map_location="cpu")
            feat = payload.get("feat") if isinstance(payload, dict) else payload
            uv, valid, ratio, mode = project_points(pts[:, :3], item["poses"][frame_i], tuple(feat.shape[-2:]))
            if ratio < args.min_valid_rate:
                print(f"[WARN][projection] scene={scene} frame={frame_i} valid_rate={ratio:.4f} < {args.min_valid_rate:.2f}")
            sampled = sample_feat(feat, uv, valid)
            sp_ids, proto, valid_counts, ok = pool_sp_dino(sampled, valid, sp, min_valid_points=args.min_valid_points)
            centroids, point_counts = compute_sp_centroids(pts[:, :3], sp, sp_ids)
            gran = frame_granularity(sp, ins, purity_thr=args.purity_thr)
            out = sample_frame_pairs(
                gran["sp_records"], sp_ids, proto, ok, centroids, point_counts, valid_counts,
                purity_thr=args.purity_thr, max_dist=args.max_pair_dist,
                max_pos=args.max_pos_per_frame, max_neg=args.max_neg_per_frame, rng=rng)
            n_pos = n_neg = 0
            if out[0] is not None:
                X, y = out
                Xs.append(X); ys.append(y); scenes.extend([scene] * len(y))
                n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
            frame_rows.append(dict(scene=scene, frame_i=int(frame_i), valid_rate=float(ratio), n_sp=int(len(sp_ids)), n_pairs=n_pos+n_neg, n_pos=n_pos, n_neg=n_neg))
            frames += 1
            if frames % 20 == 0:
                print(f"[progress] frames={frames} pairs={sum(len(y) for y in ys)} missing={missing}")
    if not Xs:
        raise RuntimeError("no training pairs collected")
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0), np.asarray(scenes), frame_rows, dict(frames=frames, missing_dino_frames=missing, scenes=len(data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-file", required=True)
    ap.add_argument("--data-root", default="data/scannet200-mv_fast")
    ap.add_argument("--dino-cache-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-scenes", type=int, default=50)
    ap.add_argument("--max-frames-per-scene", type=int, default=4)
    ap.add_argument("--max-pos-per-frame", type=int, default=256)
    ap.add_argument("--max-neg-per-frame", type=int, default=512)
    ap.add_argument("--max-pair-dist", type=float, default=1.0)
    ap.add_argument("--min-valid-points", type=int, default=5)
    ap.add_argument("--min-valid-rate", type=float, default=0.90)
    ap.add_argument("--purity-thr", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--val-scene-frac", type=float, default=0.25)
    ap.add_argument("--min-precision", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=2050318485)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed & 0xFFFFFFFF); torch.manual_seed(args.seed & 0xFFFFFFFF)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    X, y, scenes, frame_rows, collection = collect_dataset(args)
    unique = sorted(set(scenes.tolist()))
    rng = random.Random(args.seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * args.val_scene_frac))) if len(unique) > 1 else 0
    val_set = set(unique[:n_val])
    val_mask = np.asarray([s in val_set for s in scenes], dtype=bool)
    if val_mask.sum() == 0 or (~val_mask).sum() == 0:
        idx = np.arange(len(y)); np.random.shuffle(idx); val_mask = np.zeros(len(y), dtype=bool); val_mask[idx[: max(1, len(idx)//4)]] = True
    train_mask = ~val_mask
    mean = X[train_mask].mean(axis=0).astype(np.float32)
    std = X[train_mask].std(axis=0).astype(np.float32); std[std < 1e-6] = 1.0
    Xt = torch.from_numpy((X[train_mask] - mean) / std).float()
    yt = torch.from_numpy(y[train_mask]).float()
    Xv = torch.from_numpy((X[val_mask] - mean) / std).float()
    yv = torch.from_numpy(y[val_mask]).float()
    model = gci.GCIEdgeMLP(in_dim=X.shape[1], hidden_dim=args.hidden_dim)
    pos = float((yt == 1).sum().item()); neg = float((yt == 0).sum().item())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_state = None; best_auc = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); opt.zero_grad(set_to_none=True)
        logits = model(Xt)
        loss = F.binary_cross_entropy_with_logits(logits, yt, pos_weight=pos_weight)
        loss.backward(); opt.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(Xv)).cpu().numpy()
            auc = auc_score(yv.cpu().numpy(), pv)
            row = dict(epoch=epoch, loss=float(loss.item()), val_auc=auc)
            history.append(row)
            if auc is not None and auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[train] epoch={epoch} loss={loss.item():.4f} val_auc={auc}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(model(Xv)).cpu().numpy()
        pt = torch.sigmoid(model(Xt)).cpu().numpy()
    chosen, threshold_rows = best_threshold(yv.cpu().numpy(), pv, min_precision=args.min_precision)
    summary = dict(
        args=vars(args),
        collection=collection,
        feature_names=list(gci.FEATURE_NAMES),
        n_pairs=int(len(y)),
        n_train=int(train_mask.sum()),
        n_val=int(val_mask.sum()),
        train_pos=int(y[train_mask].sum()), train_neg=int((1-y[train_mask]).sum()),
        val_pos=int(y[val_mask].sum()), val_neg=int((1-y[val_mask]).sum()),
        train_auc=auc_score(y[train_mask], pt),
        val_auc=auc_score(y[val_mask], pv),
        chosen_threshold=chosen,
        history=history,
        val_scenes=sorted(val_set),
    )
    ckpt = dict(
        model_state=model.state_dict(),
        feature_mean=mean,
        feature_std=std,
        feature_names=list(gci.FEATURE_NAMES),
        threshold=float(chosen["threshold"]),
        summary=summary,
    )
    torch.save(ckpt, out_dir / "gci_edge_calibrator.pt")
    with open(out_dir / "summary.json", "w") as f: json.dump(summary, f, indent=2)
    with open(out_dir / "frame_sampling.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys())); w.writeheader(); w.writerows(frame_rows)
    with open(out_dir / "threshold_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(threshold_rows[0].keys())); w.writeheader(); w.writerows(threshold_rows)
    print(json.dumps(summary, indent=2))
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
