#!/usr/bin/env python
"""Open-vocabulary evaluation: v5ab1 class-agnostic masks + multi-view lifted
CLIP features matched to class-name text embeddings.

Per scene: model.test_step yields merged masks on the reconstructed cloud
(eval_ann_info['rec_xyz'], map_to_rec_pcd=True). Scene points are projected
into every frame's cached CLIP map and averaged over valid views; instances
pool member-point features, go through post_layernorm + visual_projection
into the joint space, and take the argmax class over text embeddings. The
cls198 evaluator (v5cls1 config) scores the relabeled predictions.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint
from mmdet3d.registry import MODELS, METRICS, DATASETS
from mmdet3d.utils import register_all_modules
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "proj_utils", os.path.join(REPO, "oneformer3d", "projection_utils.py"))
proj_utils = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(proj_utils)

TEMPLATES = ["a {} in a scene", "there is a {} in the scene",
             "a photo of a {}", "{}"]


def build_text_embeddings(hf_dir, class_names, device):
    from transformers import CLIPModel
    import clip as openai_clip
    model = CLIPModel.from_pretrained(hf_dir).to(device).eval()
    embs = []
    with torch.no_grad():
        for name in class_names:
            prompts = [t.format(name.replace("_", " ")) for t in TEMPLATES]
            toks = openai_clip.tokenize(prompts).to(device)
            feats = model.get_text_features(input_ids=toks,
                                            attention_mask=(toks != 0).long())
            feats = torch.nn.functional.normalize(feats.float(), dim=-1).mean(0)
            embs.append(torch.nn.functional.normalize(feats, dim=-1))
    text = torch.stack(embs)          # [C, 768]
    return text, model


def clip_cache_path(cache_root, img_path):
    rel = str(img_path).split("2D/")[-1]
    return os.path.join(cache_root, "2D", os.path.splitext(rel)[0] + ".pt")


def flatten1(x):
    return x[0] if isinstance(x, (list, tuple)) and len(x) == 1 else x


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def project_all_frames(rec_xyz, cam_infos, device, hw=(480, 640)):
    """Project rec points into every frame at full image resolution.
    Returns list of (uv [N,2] cpu fp16, valid [N] cpu bool)."""
    xyz_w = rec_xyz.to(device).float()
    out = []
    for cam in cam_infos:
        pose = cam["pose"] if isinstance(cam, dict) else cam
        pose_t = torch.as_tensor(np.asarray(pose), dtype=torch.float32, device=device).reshape(4, 4)
        w2c = torch.linalg.inv(pose_t)
        xyz_c = (w2c[:3, :3] @ xyz_w.T).T + w2c[:3, 3]
        uv, valid = proj_utils.project_points_to_uv(xyz_c, hw, max_depth=20.0)
        out.append((uv.half().cpu(), valid.cpu()))
    return out


def crop_instance_views(masks_t_cpu, projs, img_paths, data_root, n_views=3, pad=0.15,
                        masks_gpu=None, device="cuda"):
    """For each instance pick top-N visible frames and return crop boxes.
    Returns list per instance: [(frame_idx, (x0,y0,x1,y1)), ...]."""
    K = masks_t_cpu.shape[0]
    if masks_gpu is not None:
        valid_stack = torch.stack([v for _, v in projs]).to(device=device, dtype=torch.float16)
        vis = (masks_gpu.half() @ valid_stack.T).int().cpu()
    else:
        vis = torch.zeros((K, len(projs)), dtype=torch.int32)
        for f, (uv, valid) in enumerate(projs):
            vis[:, f] = (masks_t_cpu & valid.unsqueeze(0)).sum(1)
    plans = []
    for k in range(K):
        topv = torch.topk(vis[k], min(n_views, vis.shape[1]))
        entries = []
        for cnt, f in zip(topv.values.tolist(), topv.indices.tolist()):
            if cnt < 8:
                continue
            uv, valid = projs[f]
            sel = masks_t_cpu[k] & valid
            pts = uv[sel].float()
            x0, y0 = pts[:, 0].quantile(0.02).item(), pts[:, 1].quantile(0.02).item()
            x1, y1 = pts[:, 0].quantile(0.98).item(), pts[:, 1].quantile(0.98).item()
            w, h = x1 - x0, y1 - y0
            if w < 8 or h < 8:
                continue
            x0 = max(0.0, x0 - pad * w); y0 = max(0.0, y0 - pad * h)
            x1 = min(640.0, x1 + pad * w); y1 = min(480.0, y1 + pad * h)
            entries.append((f, (x0, y0, x1, y1)))
        plans.append(entries)
    return plans


def encode_crops(plans, img_paths, data_root, clip_model, device):
    """Batch-encode all planned crops; return per-instance mean joint-space embedding (or None)."""
    from PIL import Image
    frames = {}
    for entries in plans:
        for f, _ in entries:
            frames.setdefault(f, None)
    for f in frames:
        ip = str(flatten1(img_paths[f]))
        path = ip if os.path.isabs(ip) and os.path.exists(ip) else os.path.join(data_root, ip.split("2D/")[-1])
        if not os.path.exists(path):
            path = os.path.join(data_root, "2D", ip.split("2D/")[-1])
        frames[f] = Image.open(path).convert("RGB")
    crops, owners = [], []
    for k, entries in enumerate(plans):
        for f, (x0, y0, x1, y1) in entries:
            im = frames[f].crop((x0, y0, x1, y1)).resize((224, 224), Image.BICUBIC)
            t = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0
            crops.append((t - CLIP_MEAN) / CLIP_STD)
            owners.append(k)
    if not crops:
        return {}
    batch = torch.stack(crops).to(device)
    feats = []
    with torch.no_grad():
        for i in range(0, batch.shape[0], 256):
            f = clip_model.get_image_features(pixel_values=batch[i:i + 256])
            feats.append(torch.nn.functional.normalize(f.float(), dim=-1))
    feats = torch.cat(feats)
    owners = torch.tensor(owners)
    out = {}
    for k in owners.unique().tolist():
        e = feats[owners == k].mean(0, keepdim=True)
        out[k] = torch.nn.functional.normalize(e, dim=-1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cls-config", required=True)
    ap.add_argument("--clip-cache", required=True)
    ap.add_argument("--clip-hf-dir", required=True)
    ap.add_argument("--data-root", default="data/scannet200-mv_fast")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-scenes", type=int, default=0)
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()
    device = "cuda"
    register_all_modules()
    import oneformer3d  # noqa: registers project modules

    cfg = Config.fromfile(args.config)
    loader = Runner.build_dataloader(cfg.test_dataloader)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.ckpt, map_location="cpu")
    model = model.to(device).eval()

    cls_cfg = Config.fromfile(args.cls_config)
    metric = METRICS.build(cls_cfg.val_evaluator)
    cls_ds = DATASETS.build(cls_cfg.val_dataloader.dataset)
    metric.dataset_meta = cls_ds.metainfo
    thing_inds = cls_cfg.val_evaluator.thing_class_inds
    classes = metric.dataset_meta["classes"]
    thing_names = [classes[i] for i in thing_inds]
    print(f"[ov] {len(thing_names)} thing classes; first5={thing_names[:5]}", flush=True)

    text, clip_model = build_text_embeddings(args.clip_hf_dir, thing_names, device)
    print(f"[ov] text bank {tuple(text.shape)}", flush=True)

    results, t0 = [], time.time()
    no_valid_instances = 0
    for si, batch in enumerate(loader):
        if args.max_scenes and si >= args.max_scenes:
            break
        with torch.no_grad():
            outputs = model.test_step(batch)
        ds = outputs[0]
        pred = ds.pred_pts_seg
        masks = pred["pts_instance_mask"][0]
        masks_t = torch.as_tensor(np.asarray(masks), device=device)
        if masks_t.dtype != torch.bool:
            masks_t = masks_t > 0.5
        raw_ds = batch["data_samples"][0]
        eval_ann = raw_ds.eval_ann_info
        rec_xyz = torch.as_tensor(np.asarray(eval_ann["rec_xyz"]))[:, :3]
        inp = batch["inputs"]
        img_paths = flatten1(inp["img_paths"])
        cam_infos = flatten1(inp["cam_info"])
        if args.inspect and si == 0:
            print("[inspect] masks", tuple(masks_t.shape), "rec_xyz", tuple(rec_xyz.shape),
                  "frames", len(img_paths), flush=True)
        assert masks_t.shape[1] == rec_xyz.shape[0], \
            f"mask {tuple(masks_t.shape)} vs rec {tuple(rec_xyz.shape)}"
        masks_cpu = masks_t.cpu()
        projs = project_all_frames(rec_xyz, cam_infos, device)
        plans = crop_instance_views(masks_cpu, projs, img_paths, args.data_root,
                                    masks_gpu=masks_t, device=device)
        embs = encode_crops(plans, img_paths, args.data_root, clip_model, device)
        K = masks_t.shape[0]
        labels = torch.zeros(K, dtype=torch.long)
        scores_arr = np.asarray(pred["instance_scores"]).copy()
        sims_dbg = []
        for k in range(K):
            if k not in embs:
                no_valid_instances += 1
                scores_arr[k] = 0.0  # unlabeled: remove from ranking
                continue
            s = (embs[k] @ text.T).squeeze(0)
            labels[k] = int(s.argmax()); sims_dbg.append(float(s.max()))
        if args.inspect:
            import collections
            hist = collections.Counter(labels.tolist()).most_common(6)
            print(f"[diag] scene {si} label-hist(top6): "
                  + ", ".join(f"{thing_names[c]}x{n}" for c, n in hist), flush=True)
            k0 = int(np.argmax(np.asarray(pred["instance_scores"])))
            if k0 in embs:
                s0 = (embs[k0] @ text.T).squeeze(0); top = s0.topk(3)
                print(f"[diag] top-score inst {k0} (npts={int(masks_cpu[k0].sum())}, views={len(plans[k0])}): "
                      + ", ".join(f"{thing_names[i]}:{v:.3f}" for v, i in zip(top.values.tolist(), top.indices.tolist())), flush=True)
            if sims_dbg:
                print(f"[diag] sim max mean={np.mean(sims_dbg):.3f} p90={np.percentile(sims_dbg, 90):.3f}", flush=True)
        single = dict(
            pts_instance_mask=[masks_t.cpu().numpy()],
            instance_labels=labels.numpy(),
            instance_scores=scores_arr,
        )
        results.append((eval_ann, single))
        del masks_t
        torch.cuda.empty_cache()
        if si % 25 == 0:
            print(f"[ov] scene {si} K={K} crops_ok={len(embs)}/{K} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"[ov] scenes={len(results)} instances_wo_valid_proj={no_valid_instances}", flush=True)
    print("[diag] first-scene labels max:", int(results[0][1]["instance_labels"].max()),
          "-> branch:", "CAT-AGNOSTIC" if results[0][1]["instance_labels"].max() == 0 else "CLASS-SENSITIVE", flush=True)
    metrics = metric.compute_metrics(results)
    print("[ov] METRICS:", json.dumps(metrics, default=float), flush=True)
    with open(args.out, "w") as f:
        json.dump({"metrics": {k: float(v) for k, v in metrics.items()},
                   "n_scenes": len(results)}, f, indent=2)


if __name__ == "__main__":
    main()
