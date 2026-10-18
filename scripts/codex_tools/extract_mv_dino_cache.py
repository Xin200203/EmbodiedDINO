#!/usr/bin/env python3
"""Extract offline DINOv2 feature maps for a JSONL manifest.

The cache unit is an RGB frame feature map after deterministic resize to
420x560. Output tensor shape is Cx30x40 for ViT-L/14. Projection to 3D remains
online during training/smoke, so cache is independent of point sampling.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from typing import List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from tools.extract_dinov2_features import DINOv2FeatureExtractor

DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 14


def load_jsonl(path: Path) -> List[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def preprocess(path: Path, target_hw: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    img = Image.open(path).convert('RGB')
    src_wh = img.size
    h, w = target_hw
    img = img.resize((w, h), Image.BILINEAR)
    x = T.ToTensor()(img)
    x = T.Normalize(DINO_MEAN, DINO_STD)(x)
    return x, (src_wh[1], src_wh[0])


@torch.inference_mode()
def forward_batch(model, imgs: torch.Tensor) -> torch.Tensor:
    weight_dtype = model.patch_embed.proj.weight.dtype
    imgs = imgs.to(device=next(model.parameters()).device, dtype=weight_dtype)
    feats = model.forward_features(imgs)
    if isinstance(feats, dict):
        tokens = feats.get('x_norm_patchtokens', None)
        h = feats.get('h', None)
        w = feats.get('w', None)
        if isinstance(h, torch.Tensor):
            h = int(h.item())
        if isinstance(w, torch.Tensor):
            w = int(w.item())
    else:
        tokens, h, w = feats, None, None
    if tokens is None:
        raise RuntimeError('DINO forward_features did not return patch tokens')
    b, l, c = tokens.shape
    if h is None or w is None:
        h = imgs.shape[-2] // PATCH_SIZE
        w = imgs.shape[-1] // PATCH_SIZE
    if h * w > l:
        raise RuntimeError(f'Invalid DINO grid h*w={h*w} > tokens={l}')
    return tokens[:, :h*w, :].transpose(1, 2).reshape(b, c, h, w).contiguous()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--cache-root', required=True)
    ap.add_argument('--checkpoint', default='/datadisk1/xxy/data/models/dinov2_vitl14_reg4_pretrain.pth')
    ap.add_argument('--arch', default='dinov2_vitl14_reg')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--target-h', type=int, default=420)
    ap.add_argument('--target-w', type=int, default=560)
    ap.add_argument('--dtype', choices=['fp16', 'fp32'], default='fp16')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--summary', required=True)
    args = ap.parse_args()

    if args.device.startswith('cuda:'):
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.device.split(':')[-1])
        device = 'cuda:0'
    else:
        device = args.device
    manifest = Path(args.manifest)
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    recs = load_jsonl(manifest)
    recs = [r for i, r in enumerate(recs) if i % args.num_shards == args.shard_index]
    if args.limit > 0:
        recs = recs[:args.limit]

    dtype = torch.float16 if args.dtype == 'fp16' else torch.float32
    extractor = DINOv2FeatureExtractor(arch=args.arch, device=device, dtype=dtype, checkpoint=args.checkpoint)
    model = extractor.model.eval()

    done = skipped = failed = 0
    shapes = {}
    t0 = time.time()
    batch_imgs, batch_recs, batch_src = [], [], []

    def flush():
        nonlocal done, batch_imgs, batch_recs, batch_src
        if not batch_imgs:
            return
        imgs = torch.stack(batch_imgs, dim=0)
        feats = forward_batch(model, imgs)
        feats = feats.half().cpu() if dtype == torch.float16 else feats.float().cpu()
        for feat, rec, src_hw in zip(feats, batch_recs, batch_src):
            # IMPORTANT: feat is a slice/view into the batch tensor. Saving a view with
            # torch.save serializes the whole underlying batch storage (~8x larger for
            # batch_size=8). Clone to compact standalone storage before caching.
            feat = feat.contiguous().clone()
            out = cache_root / rec['cache_rel_path']
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'feat': feat,
                'rel_img_path': rec['rel_img_path'],
                'src_hw': src_hw,
                'target_hw': [args.target_h, args.target_w],
                'arch': args.arch,
                'checkpoint': args.checkpoint,
                'dtype': args.dtype,
            }
            tmp_out = out.with_suffix(out.suffix + '.tmp')
            torch.save(payload, tmp_out)
            tmp_out.replace(out)
            shapes[str(tuple(feat.shape))] = shapes.get(str(tuple(feat.shape)), 0) + 1
            done += 1
        batch_imgs, batch_recs, batch_src = [], [], []

    for rec in tqdm(recs, desc=f'shard {args.shard_index}/{args.num_shards}'):
        out = cache_root / rec['cache_rel_path']
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            x, src_hw = preprocess(Path(rec['abs_img_path']), (args.target_h, args.target_w))
            batch_imgs.append(x)
            batch_recs.append(rec)
            batch_src.append(src_hw)
            if len(batch_imgs) >= args.batch_size:
                flush()
        except Exception as e:
            failed += 1
            print(json.dumps({'failed': rec, 'error': repr(e)}, ensure_ascii=False), flush=True)
    flush()
    summary = {
        'manifest': str(manifest),
        'cache_root': str(cache_root),
        'checkpoint': args.checkpoint,
        'arch': args.arch,
        'device': args.device,
        'batch_size': args.batch_size,
        'target_hw': [args.target_h, args.target_w],
        'dtype': args.dtype,
        'num_input_records_after_shard_limit': len(recs),
        'num_done': done,
        'num_skipped': skipped,
        'num_failed': failed,
        'shape_counts': shapes,
        'elapsed_sec': round(time.time() - t0, 3),
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(f'{failed} extraction failures')


if __name__ == '__main__':
    main()
