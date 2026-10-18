#!/usr/bin/env python
"""Dump per-scene qualitative data: rec_xyz + rgb, GT instance ids, and
predicted instance ids for one model checkpoint. One .npz per scene."""
import argparse, os, sys
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy("file_system")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint
from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--score-thr", type=float, default=0.3)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    register_all_modules()
    import oneformer3d  # noqa
    cfg = Config.fromfile(args.config)
    loader = Runner.build_dataloader(cfg.test_dataloader)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.ckpt, map_location="cpu")
    model = model.cuda().eval()
    for si, batch in enumerate(loader):
        if si >= args.scenes:
            break
        raw = batch["data_samples"][0]
        ann = raw.eval_ann_info
        with torch.no_grad():
            out = model.test_step(batch)[0]
        pred = out.pred_pts_seg
        masks = np.asarray(pred["pts_instance_mask"][0])          # [K, N]
        scores = np.asarray(pred["instance_scores"])
        order = np.argsort(-scores)
        inst = np.full(masks.shape[1], -1, dtype=np.int32)
        rank = 0
        for k in order:
            if scores[k] < args.score_thr:
                continue
            sel = masks[k].astype(bool) & (inst == -1)
            if sel.sum() < 20:
                continue
            inst[sel] = rank
            rank += 1
        rec = np.asarray(ann["rec_xyz"])                          # [N, 3(+rgb)]
        np.savez_compressed(
            os.path.join(args.out_dir, f"scene{si:03d}_{args.tag}.npz"),
            xyz=rec[:, :3].astype(np.float32),
            rgb=(rec[:, 3:6].astype(np.float32) if rec.shape[1] >= 6 else np.zeros((rec.shape[0], 3), np.float32)),
            pred_inst=inst,
            gt_inst=np.asarray(ann["pts_instance_mask"]).astype(np.int32),
            gt_sem=np.asarray(ann["pts_semantic_mask"]).astype(np.int32),
        )
        print(f"[dump] scene {si} pts={rec.shape[0]} inst={rank}", flush=True)


if __name__ == "__main__":
    main()
