"""Granularity-Calibrated Instance Initialization utilities.

This module is intentionally lightweight and side-effect free.  It contains the
seed/edge logic shared by offline diagnostics and later model integration.
The model path should import these functions only behind explicit config flags.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


FEATURE_NAMES = (
    "dino_sim",
    "centroid_dist",
    "exp_neg_centroid_dist",
    "abs_log_count_ratio",
    "min_log_points",
    "max_log_points",
    "min_valid_rate",
    "max_valid_rate",
)


@dataclass
class GCIClusterStats:
    n_nodes: int
    n_valid_nodes: int
    n_candidate_edges: int
    n_accepted_edges: int
    n_clusters: int
    n_merged_clusters: int

    def to_dict(self) -> Dict[str, int]:
        return dict(
            n_nodes=int(self.n_nodes),
            n_valid_nodes=int(self.n_valid_nodes),
            n_candidate_edges=int(self.n_candidate_edges),
            n_accepted_edges=int(self.n_accepted_edges),
            n_clusters=int(self.n_clusters),
            n_merged_clusters=int(self.n_merged_clusters),
        )


class GCIEdgeMLP(nn.Module):
    """Tiny edge calibrator for same-object fragment affinity."""

    def __init__(self, in_dim: int = len(FEATURE_NAMES), hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(hidden_dim // 2, 8), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def l2_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def edge_features(
    proto: np.ndarray,
    centroids: np.ndarray,
    point_counts: Optional[np.ndarray],
    valid_counts: Optional[np.ndarray],
    pairs: np.ndarray,
) -> np.ndarray:
    """Build pair features for candidate superpoint/seed edges.

    Args:
        proto: [N, C] normalized DINO prototypes.
        centroids: [N, 3] fragment centroids.
        point_counts: [N] total point counts per fragment.
        valid_counts: [N] DINO-valid point counts per fragment.
        pairs: [M, 2] int pair indices.
    """
    if pairs.size == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    i = pairs[:, 0].astype(np.int64)
    j = pairs[:, 1].astype(np.int64)
    proto = l2_normalize(proto.astype(np.float32))
    dino_sim = np.sum(proto[i] * proto[j], axis=1)
    dist = np.linalg.norm(centroids[i].astype(np.float32) - centroids[j].astype(np.float32), axis=1)
    exp_neg_dist = np.exp(-dist)
    if point_counts is None:
        point_counts = np.ones(proto.shape[0], dtype=np.float32)
    if valid_counts is None:
        valid_counts = np.ones(proto.shape[0], dtype=np.float32)
    point_counts = np.maximum(point_counts.astype(np.float32), 1.0)
    valid_counts = np.maximum(valid_counts.astype(np.float32), 0.0)
    li = np.log1p(point_counts[i])
    lj = np.log1p(point_counts[j])
    abs_log_count_ratio = np.abs(li - lj)
    min_log_points = np.minimum(li, lj)
    max_log_points = np.maximum(li, lj)
    valid_rate = np.clip(valid_counts / point_counts, 0.0, 1.0)
    min_valid_rate = np.minimum(valid_rate[i], valid_rate[j])
    max_valid_rate = np.maximum(valid_rate[i], valid_rate[j])
    feats = np.stack([
        dino_sim,
        dist,
        exp_neg_dist,
        abs_log_count_ratio,
        min_log_points,
        max_log_points,
        min_valid_rate,
        max_valid_rate,
    ], axis=1)
    return feats.astype(np.float32)


def make_candidate_pairs(valid_mask: np.ndarray, centroids: Optional[np.ndarray] = None, max_dist: Optional[float] = None) -> np.ndarray:
    valid_idx = np.nonzero(valid_mask.astype(bool))[0]
    pairs: List[Tuple[int, int]] = []
    for a_pos, i in enumerate(valid_idx):
        for j in valid_idx[a_pos + 1:]:
            if centroids is not None and max_dist is not None:
                if float(np.linalg.norm(centroids[int(i)] - centroids[int(j)])) > float(max_dist):
                    continue
            pairs.append((int(i), int(j)))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def make_candidate_pairs_topk(
    valid_mask: np.ndarray,
    centroids: np.ndarray,
    max_dist: Optional[float] = None,
    topk: int = 16,
    max_pairs: int = 0,
) -> np.ndarray:
    """Build local candidate pairs by nearest valid neighbors per node.

    Uses KDTree when available so model-time GCI does not materialize an NxN
    distance matrix on frames with many superpoints.
    """
    valid_idx = np.nonzero(valid_mask.astype(bool))[0].astype(np.int64)
    if valid_idx.size <= 1 or topk <= 0:
        return make_candidate_pairs(valid_mask, centroids=centroids, max_dist=max_dist)
    cent = centroids.astype(np.float32)
    k = min(int(topk) + 1, int(valid_idx.size))  # +1 for self neighbor.
    pairs = set()
    used_kdtree = False
    try:
        from scipy.spatial import cKDTree
        sub = cent[valid_idx]
        tree = cKDTree(sub)
        upper = float(max_dist) if max_dist is not None else np.inf
        dists, neigh = tree.query(sub, k=k, distance_upper_bound=upper, workers=1)
        if np.ndim(neigh) == 1:
            neigh = neigh[:, None]
            dists = dists[:, None]
        for row in range(neigh.shape[0]):
            for col_pos in range(neigh.shape[1]):
                col = int(neigh[row, col_pos])
                if col < 0 or col >= valid_idx.size or col == row:
                    continue
                if not np.isfinite(float(dists[row, col_pos])):
                    continue
                a = int(valid_idx[row]); b = int(valid_idx[col])
                if a > b:
                    a, b = b, a
                pairs.add((a, b))
        used_kdtree = True
    except Exception:
        used_kdtree = False

    if not used_kdtree:
        sub = cent[valid_idx]
        # Fallback for environments without scipy.spatial. This is intentionally
        # kept only as a fallback; normal ESAM env has scipy.
        dist = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
        np.fill_diagonal(dist, np.inf)
        k_fallback = min(int(topk), max(int(valid_idx.size) - 1, 1))
        for row in range(valid_idx.size):
            row_dist = dist[row]
            if max_dist is not None:
                valid_cols = np.nonzero(row_dist <= float(max_dist))[0]
                if valid_cols.size == 0:
                    continue
                if valid_cols.size > k_fallback:
                    nearest_local = valid_cols[np.argpartition(row_dist[valid_cols], k_fallback - 1)[:k_fallback]]
                else:
                    nearest_local = valid_cols
            else:
                nearest_local = np.argpartition(row_dist, k_fallback - 1)[:k_fallback]
            for col in nearest_local.tolist():
                a = int(valid_idx[row]); b = int(valid_idx[col])
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                pairs.add((a, b))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    pair_list = list(pairs)
    if max_pairs and len(pair_list) > int(max_pairs):
        pair_list.sort(key=lambda ab: float(np.linalg.norm(cent[ab[0]] - cent[ab[1]])))
        pair_list = pair_list[:int(max_pairs)]
    return np.asarray(pair_list, dtype=np.int64).reshape(-1, 2)


def complete_link_clusters(
    n: int,
    candidate_pairs: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, List[Tuple[int, int, float]], GCIClusterStats]:
    """Greedy complete-link clustering from scored candidate edges."""
    parent = list(range(n))
    comps: Dict[int, List[int]] = {i: [i] for i in range(n)}
    score_map: Dict[Tuple[int, int], float] = {}
    for (a, b), s in zip(candidate_pairs.tolist(), scores.tolist()):
        if a > b:
            a, b = b, a
        score_map[(int(a), int(b))] = float(s)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    order = [(float(s), int(a), int(b)) for (a, b), s in zip(candidate_pairs.tolist(), scores.tolist()) if float(s) >= threshold]
    order.sort(reverse=True)
    accepted: List[Tuple[int, int, float]] = []
    for score, i, j in order:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        ci, cj = comps[ri], comps[rj]
        good = True
        for a in ci:
            for b in cj:
                x, y = (a, b) if a < b else (b, a)
                if score_map.get((x, y), -1e9) < threshold:
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
        accepted.append((i, j, score))

    root_to_new: Dict[int, int] = {}
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        r = find(i)
        if r not in root_to_new:
            root_to_new[r] = len(root_to_new)
        labels[i] = root_to_new[r]
    stats = GCIClusterStats(
        n_nodes=int(n),
        n_valid_nodes=int(n),
        n_candidate_edges=int(len(candidate_pairs)),
        n_accepted_edges=int(len(accepted)),
        n_clusters=int(len(root_to_new)),
        n_merged_clusters=int(sum(1 for c in comps.values() if len(c) > 1)),
    )
    return labels, accepted, stats


def connected_component_clusters(
    n: int,
    candidate_pairs: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, List[Tuple[int, int, float]], GCIClusterStats]:
    """Fast union-find clustering over thresholded learned edges."""
    parent = list(range(n))
    size = [1 for _ in range(n)]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    accepted: List[Tuple[int, int, float]] = []
    order = [(float(s), int(a), int(b)) for (a, b), s in zip(candidate_pairs.tolist(), scores.tolist()) if float(s) >= threshold]
    order.sort(reverse=True)
    for score, a, b in order:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if size[ra] < size[rb]:
            ra, rb = rb, ra
            a, b = b, a
        parent[rb] = ra
        size[ra] += size[rb]
        accepted.append((a, b, score))

    root_to_new: Dict[int, int] = {}
    labels = np.zeros(n, dtype=np.int64)
    comp_sizes: Dict[int, int] = {}
    for i in range(n):
        r = find(i)
        if r not in root_to_new:
            root_to_new[r] = len(root_to_new)
            comp_sizes[r] = 0
        labels[i] = root_to_new[r]
        comp_sizes[r] += 1
    stats = GCIClusterStats(
        n_nodes=int(n),
        n_valid_nodes=int(n),
        n_candidate_edges=int(len(candidate_pairs)),
        n_accepted_edges=int(len(accepted)),
        n_clusters=int(len(root_to_new)),
        n_merged_clusters=int(sum(1 for v in comp_sizes.values() if v > 1)),
    )
    return labels, accepted, stats


def make_seed_mask(sp_mask: np.ndarray, sp_ids: Sequence[int], cluster_labels: np.ndarray) -> np.ndarray:
    mapping = {int(sp): int(cluster_labels[i]) for i, sp in enumerate(list(sp_ids))}
    seed = np.empty_like(sp_mask, dtype=np.int64)
    for sp, lab in mapping.items():
        seed[sp_mask == sp] = lab
    uniq = np.unique(seed)
    remap = {int(v): i for i, v in enumerate(uniq.tolist())}
    for v, i in remap.items():
        seed[seed == v] = i
    return seed
