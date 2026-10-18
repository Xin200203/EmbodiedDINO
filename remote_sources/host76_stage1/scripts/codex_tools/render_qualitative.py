#!/usr/bin/env python
"""Render GT | host | ours instance panels from dumped npz files (matplotlib Agg)."""
import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = np.array([plt.cm.tab20(i % 20)[:3] for i in range(600)])
np.random.seed(0)
PALETTE = PALETTE[np.random.permutation(len(PALETTE))]


def colorize(inst, floor_mask=None):
    c = np.full((inst.shape[0], 3), 0.82)
    v = inst >= 0
    c[v] = PALETTE[inst[v] % len(PALETTE)]
    return c


def render(ax, xyz, colors, elev=58, azim=-65, n=120000):
    idx = np.random.choice(xyz.shape[0], min(n, xyz.shape[0]), replace=False)
    p, c = xyz[idx], colors[idx]
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=c, s=2.6, linewidths=0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect((np.ptp(p[:, 0]), np.ptp(p[:, 1]), np.ptp(p[:, 2]) * 0.9))
    m0, m1 = p.min(0), p.max(0)
    pad = 0.02 * (m1 - m0)
    ax.set_xlim(m0[0] + pad[0], m1[0] - pad[0])
    ax.set_ylim(m0[1] + pad[1], m1[1] - pad[1])
    ax.set_zlim(m0[2], m1[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--tag-a", default="ab0")
    ap.add_argument("--tag-b", default="ab1")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    scenes = sorted({os.path.basename(f).split("_")[0]
                     for f in glob.glob(os.path.join(args.dump_dir, "scene*_*.npz"))})
    for sc in scenes:
        try:
            a = np.load(os.path.join(args.dump_dir, f"{sc}_{args.tag_a}.npz"))
            b = np.load(os.path.join(args.dump_dir, f"{sc}_{args.tag_b}.npz"))
        except FileNotFoundError:
            continue
        # GT instance ids: map raw ids to compact
        gt = a["gt_inst"].copy()
        u = np.unique(gt[gt >= 0]) if (gt >= 0).any() else np.array([], dtype=np.int64)
        remap = {x: i for i, x in enumerate(u.tolist())}
        gtc = np.array([remap.get(x, -1) for x in gt.tolist()], dtype=np.int32)
        fig = plt.figure(figsize=(16.5, 6.2), dpi=150)
        for j, (inst, title) in enumerate(
                [(gtc, "Ground truth"), (a["pred_inst"], "Host (no feature)"),
                 (b["pred_inst"], "EmbodiedDINO")]):
            ax = fig.add_subplot(1, 3, j + 1, projection="3d")
            render(ax, a["xyz"], colorize(inst))
            ax.set_title(title, fontsize=13)
        fig.subplots_adjust(left=0.0, right=1.0, top=0.94, bottom=0.0, wspace=-0.12)
        out = os.path.join(args.out_dir, f"{sc}_panel.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print("[render]", out, flush=True)


if __name__ == "__main__":
    main()
