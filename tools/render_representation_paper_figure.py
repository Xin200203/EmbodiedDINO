#!/usr/bin/env python3

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render the reviewer-facing representation diagnostic.")
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def select(rows, metrics, **filters):
    selected = []
    for metric, label in metrics:
        matches = [
            row for row in rows
            if row["baseline"] == "host" and
            row["comparison"] == "embodieddino" and
            row["metric"] == metric and
            all(row.get(key) == str(value) for key, value in filters.items())]
        if len(matches) != 1:
            raise ValueError(
                f"expected one row for {metric} with {filters}, "
                f"found {len(matches)}")
        selected.append((label, matches[0]))
    return selected


def plot_panel(axis, rows, title, scene_count):
    labels = [label for label, _ in rows]
    means = np.asarray([float(row["mean_delta"]) for _, row in rows])
    lows = np.asarray([float(row["ci95_low"]) for _, row in rows])
    highs = np.asarray([float(row["ci95_high"]) for _, row in rows])
    positions = np.arange(len(rows))
    axis.errorbar(
        means,
        positions,
        xerr=np.vstack([means - lows, highs - means]),
        fmt="o",
        color="#2563A6",
        ecolor="#2563A6",
        markersize=5,
        capsize=3,
        linewidth=1.4)
    axis.axvline(0.0, color="#222222", linewidth=0.8)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_title(
        f"{title}\n{scene_count} paired scenes", fontsize=10, pad=7)
    axis.grid(axis="x", color="#D1D5DB", linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)


def main():
    args = parse_args()
    representation_rows = read_csv(os.path.join(
        args.analysis_dir, "representation_paired_scene_deltas.csv"))
    proposal_rows = read_csv(os.path.join(
        args.analysis_dir, "proposal_paired_scene_deltas.csv"))

    representation = select(
        representation_rows,
        [("retrieval_r1", "Cross-frame R@1"),
         ("retrieval_map", "Cross-frame mAP"),
         ("cosine_margin", "Cosine margin")])
    proposals = select(
        proposal_rows,
        [("mean_best_iou", "Mean best IoU"),
         ("recall_iou_0.25", "Recall at IoU 0.25"),
         ("recall_iou_0.50", "Recall at IoU 0.50"),
         ("recall_iou_0.75", "Recall at IoU 0.75")],
        topk=5)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75))
    plot_panel(
        axes[0], representation,
        "(a) Cross-frame feature consistency",
        representation[0][1]["scenes"])
    plot_panel(
        axes[1], proposals,
        "(b) Proposals under a top-5 budget",
        proposals[0][1]["scenes"])
    fig.supxlabel(
        "Paired scene difference (EmbodiedDINO - matched host)",
        fontsize=9, y=0.01)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.4)

    output_dir = os.path.dirname(os.path.abspath(args.output_prefix))
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output_prefix + ".pdf", bbox_inches="tight")
    fig.savefig(args.output_prefix + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
