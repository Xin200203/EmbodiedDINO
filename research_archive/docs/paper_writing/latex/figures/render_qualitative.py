#!/usr/bin/env python3
"""Render GT-matched qualitative panels from host/EmbodiedDINO NPZ dumps."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = np.asarray([plt.cm.tab20(i % 20)[:3] for i in range(256)])


def compact_ground_truth(gt):
    ids = [int(x) for x in np.unique(gt) if x >= 0]
    lookup = {value: index for index, value in enumerate(ids)}
    compact = np.asarray([lookup.get(int(x), -1) for x in gt], dtype=np.int32)
    return compact, ids


def match_predictions(gt_compact, pred):
    gt_ids = [int(x) for x in np.unique(gt_compact) if x >= 0]
    pred_ids = [int(x) for x in np.unique(pred) if x >= 0]
    candidates = []
    for pred_id in pred_ids:
        pred_mask = pred == pred_id
        for gt_id in gt_ids:
            gt_mask = gt_compact == gt_id
            intersection = np.count_nonzero(pred_mask & gt_mask)
            if intersection == 0:
                continue
            union = np.count_nonzero(pred_mask | gt_mask)
            candidates.append((intersection / max(union, 1), pred_id, gt_id))

    mapping = {}
    used_gt = set()
    for iou, pred_id, gt_id in sorted(candidates, reverse=True):
        if iou < 0.1 or pred_id in mapping or gt_id in used_gt:
            continue
        mapping[pred_id] = gt_id
        used_gt.add(gt_id)

    matched = np.full_like(pred, -1)
    for pred_id, gt_id in mapping.items():
        matched[pred == pred_id] = gt_id
    return matched


def colors(instances):
    output = np.full((instances.shape[0], 3), 0.86, dtype=np.float32)
    valid = instances >= 0
    output[valid] = PALETTE[instances[valid] % len(PALETTE)]
    return output


def render(ax, xyz, point_colors, indices, elev=58, azim=-65):
    points = xyz[indices]
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=point_colors[indices], s=1.8, linewidths=0, rasterized=True)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    spans = np.maximum(np.ptp(points, axis=0), 1e-3)
    ax.set_box_aspect((spans[0], spans[1], spans[2] * 0.9))
    lower, upper = points.min(0), points.max(0)
    pad = 0.015 * (upper - lower)
    ax.set_xlim(lower[0] + pad[0], upper[0] - pad[0])
    ax.set_ylim(lower[1] + pad[1], upper[1] - pad[1])
    ax.set_zlim(lower[2], upper[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", nargs="+", type=int, default=[7, 4, 3])
    parser.add_argument(
        "--row", action="append", nargs=3, metavar=("DUMP_DIR", "INDEX", "LABEL"),
        help="Add a row from a dump directory, scene index, and display label.")
    parser.add_argument("--max-points", type=int, default=90000)
    args = parser.parse_args()

    if args.row:
        rows = [(Path(directory), int(index), label)
                for directory, index, label in args.row]
    else:
        if not args.dump_dir:
            parser.error("--dump-dir is required when --row is not used")
        scene_names = {
            7: "scene0412_01 (gain)",
            4: "scene0488_00 (gain)",
            3: "scene0304_00 (failure)",
        }
        rows = [(Path(args.dump_dir), scene, scene_names.get(scene, f"scene {scene:03d}"))
                for scene in args.scenes]
    column_names = ["Ground truth", "Matched host", "EmbodiedDINO"]
    figure = plt.figure(figsize=(15.6, 12.0), dpi=180, facecolor="white")

    for row, (row_dump_dir, scene, row_label) in enumerate(rows):
        host = np.load(row_dump_dir / f"scene{scene:03d}_ab0.npz")
        ours = np.load(row_dump_dir / f"scene{scene:03d}_ab1.npz")
        gt, _ = compact_ground_truth(host["gt_inst"])
        host_pred = match_predictions(gt, host["pred_inst"])
        ours_pred = match_predictions(gt, ours["pred_inst"])

        rng = np.random.RandomState(scene)
        count = min(args.max_points, host["xyz"].shape[0])
        indices = rng.choice(host["xyz"].shape[0], count, replace=False)
        for column, instances in enumerate((gt, host_pred, ours_pred)):
            axis = figure.add_subplot(len(rows), 3, row * 3 + column + 1, projection="3d")
            render(axis, host["xyz"], colors(instances), indices)
            if row == 0:
                axis.set_title(column_names[column], fontsize=15, pad=2)
            if column == 0:
                axis.text2D(
                    -0.03, 0.93, row_label,
                    transform=axis.transAxes,
                    fontsize=13, fontweight="bold", ha="left", va="top")

    figure.subplots_adjust(left=0.005, right=0.995, top=0.96, bottom=0.005,
                           wspace=-0.12, hspace=-0.14)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


if __name__ == "__main__":
    main()
