#!/usr/bin/env python
"""End-to-end 3D-pipeline latency: per-frame wall time of model.predict
(backbone + fusion + decode + online merge), excluding data loading and eval.
Same dataloader order for every config -> paired per-scene comparison."""
import argparse, json, os, sys, time
import numpy as np
import torch

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
    ap.add_argument("--scenes", type=int, default=22)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    register_all_modules()
    import oneformer3d  # noqa
    cfg = Config.fromfile(args.config)
    loader = Runner.build_dataloader(cfg.test_dataloader)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.ckpt, map_location="cpu")
    model = model.cuda().eval()
    per_frame_ms, scene_stats = [], []
    for si, batch in enumerate(loader):
        if si >= args.scenes:
            break
        data = model.data_preprocessor(batch, False)
        n_frames = int(data["inputs"]["points"][0].shape[0])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model._run_forward(data, mode="predict")
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        if si >= args.warmup:
            per_frame_ms.append(dt / n_frames)
            scene_stats.append({"scene": si, "frames": n_frames,
                                "ms_per_frame": dt / n_frames})
        print(f"[bench] scene {si} frames={n_frames} {dt/n_frames:.1f} ms/frame", flush=True)
    arr = np.array(per_frame_ms)
    res = {"config": os.path.basename(args.config),
           "n_scenes": len(arr),
           "mean_ms_per_frame": float(arr.mean()),
           "median_ms_per_frame": float(np.median(arr)),
           "p90_ms_per_frame": float(np.percentile(arr, 90)),
           "scenes": scene_stats}
    print("[bench] RESULT:", json.dumps({k: v for k, v in res.items() if k != "scenes"}), flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
