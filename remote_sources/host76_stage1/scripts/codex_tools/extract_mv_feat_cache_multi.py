#!/usr/bin/env python3
"""Extract offline 2D feature maps for a JSONL manifest — multi-backend (D8).

Backends:
  dinov2 : original path (DINOv2FeatureExtractor), patch 14
  dinov3 : local torch.hub repo + .pth weights, patch 16
  clip   : transformers CLIPVisionModel + interpolated pos-emb, patch 14

Cache payload format identical to extract_mv_dino_cache.py (feat CxHxW fp16).
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

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def load_jsonl(path: Path) -> List[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_model(backend: str, checkpoint: str, device: str, dtype, dinov3_repo: str, clip_dir: str, target_hw):
    """Returns (model, forward_fn(imgs)->BxCxhxw, patch_size, mean, std)."""
    if backend == 'dinov2':
        from tools.extract_dinov2_features import DINOv2FeatureExtractor
        ex = DINOv2FeatureExtractor(arch='dinov2_vitl14_reg', device=device, dtype=dtype, checkpoint=checkpoint)
        model = ex.model.eval()
        return model, make_dino_forward(model, 14), 14, IMAGENET_MEAN, IMAGENET_STD

    if backend == 'dinov3':
        model = torch.hub.load(dinov3_repo, 'dinov3_vitl16', source='local', pretrained=False)
        sd = torch.load(checkpoint, map_location='cpu')
        for key in ('model', 'state_dict', 'teacher'):
            if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'[dinov3] load_state_dict missing={len(missing)} unexpected={len(unexpected)}')
        if len(missing) > 5:
            raise RuntimeError(f'dinov3 weights mismatch, missing e.g. {missing[:5]}')
        model = model.to(device=device, dtype=dtype).eval()
        return model, make_dino_forward(model, 16), 16, IMAGENET_MEAN, IMAGENET_STD

    if backend == 'clip':
        from transformers import CLIPVisionModel
        model = CLIPVisionModel.from_pretrained(clip_dir)
        vm = model.vision_model
        emb = vm.embeddings
        ph = target_hw[0] // 14
        pw = target_hw[1] // 14
        old = emb.position_embedding.weight.data  # (1+G*G, C), G=16 for 224/14
        cls_pe, patch_pe = old[:1], old[1:]
        g = int(patch_pe.shape[0] ** 0.5)
        patch_pe = patch_pe.reshape(1, g, g, -1).permute(0, 3, 1, 2)
        patch_pe = torch.nn.functional.interpolate(patch_pe, size=(ph, pw), mode='bicubic', align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(ph * pw, -1)
        new_pe = torch.nn.Embedding(1 + ph * pw, old.shape[1])
        new_pe.weight.data = torch.cat([cls_pe, patch_pe], dim=0)
        emb.position_embedding = new_pe
        emb.register_buffer('position_ids', torch.arange(1 + ph * pw).unsqueeze(0), persistent=False)
        emb.num_positions = 1 + ph * pw
        emb.num_patches = ph * pw
        model = model.to(device=device, dtype=dtype).eval()

        @torch.inference_mode()
        def fwd(imgs):
            imgs = imgs.to(device=device, dtype=dtype)
            out = model(pixel_values=imgs).last_hidden_state  # (B, 1+N, C)
            tokens = out[:, 1:, :]
            b, l, c = tokens.shape
            assert l == ph * pw, f'clip tokens {l} != {ph*pw}'
            return tokens.transpose(1, 2).reshape(b, c, ph, pw).contiguous()

        return model, fwd, 14, CLIP_MEAN, CLIP_STD

    raise ValueError(backend)


def make_dino_forward(model, patch_size):
    @torch.inference_mode()
    def fwd(imgs):
        weight_dtype = next(model.parameters()).dtype
        imgs = imgs.to(device=next(model.parameters()).device, dtype=weight_dtype)
        feats = model.forward_features(imgs)
        if isinstance(feats, dict):
            tokens = feats.get('x_norm_patchtokens', None)
        else:
            tokens = feats
        if tokens is None:
            raise RuntimeError('forward_features did not return patch tokens')
        b, l, c = tokens.shape
        h = imgs.shape[-2] // patch_size
        w = imgs.shape[-1] // patch_size
        if h * w > l:
            raise RuntimeError(f'grid {h}x{w} > tokens {l}')
        return tokens[:, :h * w, :].transpose(1, 2).reshape(b, c, h, w).contiguous()
    return fwd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', required=True, choices=['dinov2', 'dinov3', 'clip'])
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--cache-root', required=True)
    ap.add_argument('--checkpoint', default='')
    ap.add_argument('--dinov3-repo', default='/datadisk3/xxy/feat_weights/dinov3_repo')
    ap.add_argument('--clip-dir', default='/datadisk3/xxy/feat_weights/clip_vitl14_hf')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--target-h', type=int, default=420)
    ap.add_argument('--target-w', type=int, default=560)
    ap.add_argument('--dtype', choices=['fp16', 'fp32'], default='fp16')
    ap.add_argument('--compute-dtype', choices=['fp16', 'fp32', 'bf16'], default='')
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
    cmap = {'fp16': torch.float16, 'fp32': torch.float32, 'bf16': torch.bfloat16}
    compute_dtype = cmap[args.compute_dtype] if args.compute_dtype else dtype
    assert args.target_h % {'dinov2': 14, 'dinov3': 16, 'clip': 14}[args.backend] == 0
    assert args.target_w % {'dinov2': 14, 'dinov3': 16, 'clip': 14}[args.backend] == 0
    model, forward_fn, patch, mean, std = build_model(
        args.backend, args.checkpoint, device, compute_dtype, args.dinov3_repo, args.clip_dir,
        (args.target_h, args.target_w))
    arch_tag = {'dinov2': 'dinov2_vitl14_reg', 'dinov3': 'dinov3_vitl16', 'clip': 'clip_vitl14'}[args.backend]

    def preprocess(path: Path):
        img = Image.open(path).convert('RGB')
        src_wh = img.size
        img = img.resize((args.target_w, args.target_h), Image.BILINEAR)
        x = T.ToTensor()(img)
        x = T.Normalize(mean, std)(x)
        return x, (src_wh[1], src_wh[0])

    done = skipped = failed = 0
    shapes = {}
    t0 = time.time()
    batch_imgs, batch_recs, batch_src = [], [], []

    def flush():
        nonlocal done, batch_imgs, batch_recs, batch_src
        if not batch_imgs:
            return
        imgs = torch.stack(batch_imgs, dim=0)
        feats = forward_fn(imgs)
        if not torch.isfinite(feats).all():
            raise RuntimeError(f'non-finite features in batch (backend={args.backend}, compute={args.compute_dtype or args.dtype})')
        feats = feats.half().cpu() if dtype == torch.float16 else feats.float().cpu()
        for feat, rec, src_hw in zip(feats, batch_recs, batch_src):
            feat = feat.contiguous().clone()  # avoid batch-storage serialization
            out = cache_root / rec['cache_rel_path']
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'feat': feat,
                'rel_img_path': rec['rel_img_path'],
                'src_hw': src_hw,
                'target_hw': [args.target_h, args.target_w],
                'arch': arch_tag,
                'checkpoint': args.checkpoint,
                'dtype': args.dtype,
            }
            tmp_out = out.with_suffix(out.suffix + '.tmp')
            torch.save(payload, tmp_out)
            tmp_out.replace(out)
            shapes[str(tuple(feat.shape))] = shapes.get(str(tuple(feat.shape)), 0) + 1
            done += 1
        batch_imgs, batch_recs, batch_src = [], [], []

    for rec in tqdm(recs, desc=f'{args.backend} shard {args.shard_index}/{args.num_shards}', mininterval=30):
        out = cache_root / rec['cache_rel_path']
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            x, src_hw = preprocess(Path(rec['abs_img_path']))
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
        'backend': args.backend, 'arch': arch_tag,
        'manifest': str(manifest), 'cache_root': str(cache_root),
        'checkpoint': args.checkpoint, 'device': args.device,
        'batch_size': args.batch_size, 'target_hw': [args.target_h, args.target_w],
        'dtype': args.dtype,
        'num_input_records_after_shard_limit': len(recs),
        'num_done': done, 'num_skipped': skipped, 'num_failed': failed,
        'shape_counts': shapes, 'elapsed_sec': round(time.time() - t0, 3),
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(f'{failed} extraction failures')


if __name__ == '__main__':
    main()
