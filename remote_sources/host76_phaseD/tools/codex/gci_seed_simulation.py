#!/usr/bin/env python3
"""Simulate latent object seed initialization from raw fragments.

This script uses projected DINO prototypes + centroid gates to cluster raw
superpoints into high-precision seeds, then evaluates whether seed-level
partitions reduce GT fragmentation without excessive boundary crossing.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from gci_granularity_diagnostics import (
    load_pkl, scene_id_from_item, choose_frame_indices, load_bin,
    cache_path_from_img, project_points, sample_feat, pool_sp_dino,
    compute_sp_centroids, frame_granularity,
)


def complete_link_clusters(proto, centroids, ok, tau, dist_gate):
    n = proto.shape[0]
    sim = proto @ proto.T
    dist = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    parent = list(range(n))
    comps = {i: [i] for i in range(n)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []
    for i in range(n):
        if not ok[i]:
            continue
        for j in range(i + 1, n):
            if ok[j] and sim[i, j] >= tau and dist[i, j] <= dist_gate:
                pairs.append((float(sim[i, j]), i, j))
    pairs.sort(reverse=True)
    accepted_edges = []
    for score, i, j in pairs:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        ci, cj = comps[ri], comps[rj]
        # Complete-link guard: every cross pair must satisfy both gates.
        good = True
        for a in ci:
            for b in cj:
                if sim[a, b] < tau or dist[a, b] > dist_gate:
                    good = False
                    break
            if not good:
                break
        if not good:
            continue
        if len(ci) < len(cj):
            ri, rj = rj, ri
            ci, cj = cj, ci
        parent[rj] = ri
        comps[ri] = ci + cj
        del comps[rj]
        accepted_edges.append((i, j, score))
    # Normalize cluster ids.
    root_to_new = {}
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        r = find(i)
        if r not in root_to_new:
            root_to_new[r] = len(root_to_new)
        labels[i] = root_to_new[r]
    return labels, accepted_edges


def make_seed_mask(sp_mask, sp_ids, cluster_labels):
    mapping = {int(sp): int(cluster_labels[i]) for i, sp in enumerate(sp_ids.tolist())}
    seed = np.empty_like(sp_mask, dtype=np.int64)
    for sp, lab in mapping.items():
        seed[sp_mask == sp] = lab
    # Re-contiguous just in case.
    uniq = np.unique(seed)
    remap = {int(v): i for i, v in enumerate(uniq.tolist())}
    for v, i in remap.items():
        seed[seed == v] = i
    return seed


def summarize_rows(rows):
    out = {}
    keys = rows[0].keys()
    for k in keys:
        if k in ('scene', 'frame_i', 'tau', 'dist_gate'):
            continue
        vals = [r[k] for r in rows if isinstance(r[k], (int, float, np.floating))]
        if vals:
            out[k] = float(np.mean(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann-file', required=True)
    ap.add_argument('--data-root', default='data/scannet200-mv_fast')
    ap.add_argument('--dino-cache-root', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--max-scenes', type=int, default=50)
    ap.add_argument('--max-frames-per-scene', type=int, default=4)
    ap.add_argument('--taus', default='0.9,0.95')
    ap.add_argument('--dist-gates', default='0.25,0.5')
    args = ap.parse_args()

    taus = [float(x) for x in args.taus.split(',') if x]
    gates = [float(x) for x in args.dist_gates.split(',') if x]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data = load_pkl(args.ann_file)
    if args.max_scenes > 0:
        data = data[:args.max_scenes]

    rows = []
    missing = 0
    frames = 0
    for scene_idx, item in enumerate(data):
        scene = scene_id_from_item(item)
        for frame_i in choose_frame_indices(len(item['pts_paths']), args.max_frames_per_scene):
            pts = load_bin(os.path.join(args.data_root, item['pts_paths'][frame_i]), np.float32, 6)
            sp = load_bin(os.path.join(args.data_root, item['super_pts_paths'][frame_i]), np.int64)
            ins = load_bin(os.path.join(args.data_root, item['pts_instance_mask_paths'][frame_i]), np.int64)
            raw = frame_granularity(sp, ins)
            cp = cache_path_from_img(args.dino_cache_root, item['img_paths'][frame_i])
            if not os.path.exists(cp):
                missing += 1
                continue
            payload = torch.load(cp, map_location='cpu')
            feat = payload.get('feat') if isinstance(payload, dict) else payload
            uv, valid, ratio, mode = project_points(pts[:, :3], item['poses'][frame_i], tuple(feat.shape[-2:]))
            sampled = sample_feat(feat, uv, valid)
            sp_ids, proto, counts, ok = pool_sp_dino(sampled, valid, sp)
            centroids, sp_counts = compute_sp_centroids(pts[:, :3], sp, sp_ids)
            for tau in taus:
                for gate in gates:
                    labels, accepted_edges = complete_link_clusters(proto, centroids, ok, tau=tau, dist_gate=gate)
                    seed = make_seed_mask(sp, sp_ids, labels)
                    gci = frame_granularity(seed, ins)
                    rows.append(dict(
                        scene=scene, frame_i=int(frame_i), tau=tau, dist_gate=gate,
                        dino_valid_rate=float(ratio),
                        raw_n_sp=raw['n_sp'], seed_n_sp=gci['n_sp'], sp_reduction=raw['n_sp'] - gci['n_sp'],
                        raw_fragments_per_gt=raw['mean_fragments_per_gt'], seed_fragments_per_gt=gci['mean_fragments_per_gt'],
                        raw_max_fragments_per_gt=raw['max_fragments_per_gt'], seed_max_fragments_per_gt=gci['max_fragments_per_gt'],
                        raw_purity=raw['mean_fragment_purity'], seed_purity=gci['mean_fragment_purity'],
                        raw_boundary_cross=raw['boundary_cross_rate'], seed_boundary_cross=gci['boundary_cross_rate'],
                        raw_low_purity=raw['low_purity_rate'], seed_low_purity=gci['low_purity_rate'],
                        accepted_edges=len(accepted_edges)))
            frames += 1
            if frames % 20 == 0:
                print(f'[progress] frames={frames} scenes={scene_idx+1}')

    with open(out_dir / 'seed_frame_metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    summary = dict(ann_file=args.ann_file, frames=frames, scenes=len(data), missing_dino_frames=missing, configs={})
    for tau in taus:
        for gate in gates:
            sub = [r for r in rows if r['tau'] == tau and r['dist_gate'] == gate]
            summary['configs'][f'tau{tau}_dist{gate}'] = summarize_rows(sub)
    with open(out_dir / 'seed_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f'[saved] {out_dir}')


if __name__ == '__main__':
    main()
