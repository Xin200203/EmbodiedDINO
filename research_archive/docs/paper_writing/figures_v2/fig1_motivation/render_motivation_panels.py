#!/usr/bin/env python3
"""
Render the three 2D panels used by Fig. 1 (motivation).

WHY 2D IS THE RIGHT SPACE HERE
------------------------------
Fig. 1 makes a claim about the *interface*: mask lifting keeps region membership
and drops the descriptors, and it drops them *before* anything reaches 3D.  Both
objects therefore live in the image plane at the moment of the loss.  A 2D
rendering is not a cheap substitute for a 3D one -- it is the semantically
correct space for this particular claim, and it needs no prediction dumps.

OUTPUTS (into ./assets/)
------------------------
  panel_rgb.png        the source frame
  panel_mask.png       FastSAM masks, each filled with a FLAT distinct colour
  panel_dino.png       DINOv2 patch features, PCA -> RGB, mask outlines overlaid
  panel_dino_pure.png  the same PCA map without the outline overlay

FAITHFULNESS NOTES (2026-07-27, after checking the server code)
---------------------------------------------------------------
* Masks come from the project's own FastSAM path
  (data/scannet200-mv/load_scannet_mv_data_fast.py): thirdparty `fastsam`
  package, run on the original frame file, retina_masks=True, imgsz=640,
  conf=0.1, iou=0.9 (retry iou=0.7), then flattened big-to-small into ONE
  partition map -- that partition, not the overlapping raw masks, is what the
  model lifts.  ultralytics 8.0.120 on the server has no FastSAM class.
* DINO arch is dinov2_vitl14_reg (reg4): project configs say
  arch='dinov2_vitl14_reg' and the training-time feature cache is named
  dinov2_vitl14_reg4_420x560_fp16.

DESIGN DECISIONS AND WHY
------------------------
* Masks get *distinct flat* colours, not a single binary colour.  The paper's own
  wording is that mask lifting "conveys region support effectively, but ...
  discards the dense descriptors".  A pure binary render would strawman the
  baseline; distinct flat regions say "useful structure, zero appearance".
* PCA channels are normalised by PERCENTILE CLIPPING, not min-max.  A handful of
  outlier patches otherwise washes the whole map out to grey.
* DINOv2 runs at the same resolution the paper uses (420x560), so the panel shows
  what the method actually consumes.  Patch size 14 gives 30x40 tokens, which is
  blocky; we bilinearly upsample for display and say so in the caption.

RUN ON THE SERVER (the local checkout has no torch / no data):
    conda activate ESAM
    python3 render_motivation_panels.py \
        --scene scene0011_00 --frame 40 \
        --scannet-root /path/to/scannet-mv
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# DINOv2 expects ImageNet statistics
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# flat, mutually distinguishable, print-safe; deliberately NOT a rainbow
FLAT = np.array([
    [222, 72, 60], [236, 150, 44], [226, 206, 62], [110, 190, 90],
    [58, 168, 168], [62, 110, 196], [136, 86, 190], [214, 96, 168],
    [150, 120, 90], [110, 110, 118], [240, 180, 170], [170, 210, 240],
], dtype=np.uint8)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True, help='e.g. scene0011_00')
    ap.add_argument('--frame', type=int, required=True)
    ap.add_argument('--scannet-root', required=True)
    ap.add_argument('--out', default='assets')
    ap.add_argument('--height', type=int, default=420)
    ap.add_argument('--width', type=int, default=560)
    ap.add_argument('--dino', default='dinov2_vitl14_reg',
                    help='must match the paper: ViT-L/14 with registers (reg4). '
                         'Project configs use arch="dinov2_vitl14_reg" and the '
                         'training cache is dinov2_vitl14_reg4_420x560_fp16.')
    ap.add_argument('--clip-pct', type=float, default=2.0,
                    help='percentile clipped from each end before normalising')
    ap.add_argument('--fastsam-weights', default=None,
                    help='if omitted, the mask panel is skipped and must be '
                         'produced by the project FastSAM path instead')
    return ap.parse_args()


def load_frame(root, scene, idx, h, w):
    """Locate one RGB frame. Tries the common ScanNet-MV layouts.

    Returns (resized PIL image, path of the ORIGINAL full-res frame).  The
    original path matters because the project's FastSAM step runs on the file
    itself (load_scannet_mv_data_fast.py), not on a resized array.
    """
    cands = [
        os.path.join(root, scene, 'color', f'{idx}.jpg'),
        os.path.join(root, scene, 'color', f'{idx:05d}.jpg'),
        os.path.join(root, scene, 'rgb', f'{idx}.png'),
        os.path.join(root, '2D', scene, 'color', f'{idx}.jpg'),
    ]
    for c in cands:
        if os.path.isfile(c):
            print(f'[frame] {c}')
            return Image.open(c).convert('RGB').resize((w, h), Image.BILINEAR), c
    raise FileNotFoundError(
        'no RGB frame found; tried:\n  ' + '\n  '.join(cands) +
        '\nPass the layout that exists on this machine.')


@torch.no_grad()
def dino_pca(img, model_name, h, w, clip_pct):
    """DINOv2 patch tokens -> 3 principal components -> RGB image."""
    try:
        model = torch.hub.load('facebookresearch/dinov2', model_name)
    except Exception as e:                      # offline: use the hub cache dir
        cache = os.path.expanduser('~/.cache/torch/hub/facebookresearch_dinov2_main')
        print(f'[dino] hub load failed ({e}); retrying from local cache {cache}')
        model = torch.hub.load(cache, model_name, source='local')
    model.eval().cuda()

    x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
    x = ((x - MEAN) / STD).unsqueeze(0).cuda()

    feats = model.forward_features(x)['x_norm_patchtokens'][0]      # (N, C)
    p = model.patch_size
    gh, gw = h // p, w // p
    assert feats.shape[0] == gh * gw, (feats.shape, gh, gw)

    f = feats.float()
    f = f - f.mean(0, keepdim=True)
    # torch.pca_lowrank is enough for 3 components and avoids a sklearn dep
    _, _, v = torch.pca_lowrank(f, q=3)
    comp = (f @ v[:, :3]).reshape(gh, gw, 3).cpu().numpy()

    # percentile clipping per channel -- min-max would let outliers wash it out
    out = np.empty_like(comp)
    for c in range(3):
        lo, hi = np.percentile(comp[..., c], [clip_pct, 100 - clip_pct])
        out[..., c] = np.clip((comp[..., c] - lo) / max(hi - lo, 1e-8), 0, 1)

    # upsample for display; the underlying map really is gh x gw and the caption
    # must say so rather than implying per-pixel descriptors
    t = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)
    print(f'[dino] {model_name}  token grid {gh}x{gw}  -> upsampled {h}x{w}')
    return (t[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def fastsam_group_ids(img_path, weights):
    """FastSAM masks flattened into one partition map, exactly as the project
    lifts them (data/scannet200-mv/load_scannet_mv_data_fast.py:192-206):

      * thirdparty `fastsam` package, NOT `ultralytics` (8.0.120 has no FastSAM)
      * run on the ORIGINAL image file, retina_masks=True, imgsz=640,
        conf=0.1, iou=0.9, with the project's iou=0.7 retry
      * masks sorted by area DESCENDING and painted in that order, so smaller
        masks overwrite larger ones -> a single flat partition (group ids)

    Returns int32 (H, W) with -1 for unassigned pixels.
    """
    from fastsam import FastSAM
    model = FastSAM(weights)
    res = model(img_path, device='cuda', retina_masks=True,
                imgsz=640, conf=0.1, iou=0.9)
    try:
        m = res[0].masks.data.cpu().numpy().astype(bool)
    except Exception:
        res = model(img_path, device='cuda', retina_masks=True,
                    imgsz=640, conf=0.1, iou=0.7)
        m = res[0].masks.data.cpu().numpy().astype(bool)
    order = np.argsort([-mm.sum() for mm in m])            # big first
    gid = np.full(m.shape[1:], -1, np.int32)
    for rank, k in enumerate(order):
        gid[m[k]] = rank
    print(f'[mask] {m.shape[0]} raw masks -> {len(np.unique(gid)) - (1 if (gid == -1).any() else 0)} '
          f'surviving regions after flattening')
    return gid


def resize_group_ids(gid, h, w):
    """Nearest-neighbour resize of an int partition map (keeps regions flat)."""
    return np.asarray(Image.fromarray(gid, mode='I').resize((w, h), Image.NEAREST),
                      dtype=np.int32)


def flat_masks(gid):
    """Partition map -> flat colours; greedy assignment so touching regions
    never share a colour (12 print-safe colours cycle otherwise)."""
    h, w = gid.shape
    # adjacency across 4-neighbour boundaries
    pairs = set()
    for a, b in ((gid[:, :-1], gid[:, 1:]), (gid[:-1, :], gid[1:, :])):
        d = a != b
        pairs |= set(map(tuple, np.unique(
            np.stack([a[d], b[d]], 1), axis=0).tolist()))
    adj = {}
    for x, y in pairs:
        if x < 0 or y < 0:
            continue
        adj.setdefault(x, set()).add(y)
        adj.setdefault(y, set()).add(x)
    ids = [i for i in np.unique(gid) if i >= 0]            # ascending = big first
    colour, nxt = {}, 0
    for i in ids:
        used = {colour[j] for j in adj.get(i, ()) if j in colour}
        pick = next((c for off in range(len(FLAT))
                     if (c := (nxt + off) % len(FLAT)) not in used), nxt % len(FLAT))
        colour[i] = pick
        nxt = (pick + 1) % len(FLAT)
    canvas = np.full((h, w, 3), 255, np.uint8)
    for i in ids:
        canvas[gid == i] = FLAT[colour[i]]
    return canvas


def outline(gid):
    """1px boundaries of the flattened partition (what actually gets lifted)."""
    edge = np.zeros(gid.shape, bool)
    d = gid[:, :-1] != gid[:, 1:]
    edge[:, :-1] |= d; edge[:, 1:] |= d
    d = gid[:-1, :] != gid[1:, :]
    edge[:-1, :] |= d; edge[1:, :] |= d
    edge &= gid >= 0                       # don't outline pure background
    return edge


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    img, orig_path = load_frame(a.scannet_root, a.scene, a.frame, a.height, a.width)
    img.save(os.path.join(a.out, 'panel_rgb.png'))

    pca = dino_pca(img, a.dino, a.height, a.width, a.clip_pct)
    # the PCA map with no overlay, in case the figure assembly wants it bare
    Image.fromarray(pca).save(os.path.join(a.out, 'panel_dino_pure.png'))

    gid = None
    if a.fastsam_weights:
        gid = fastsam_group_ids(orig_path, a.fastsam_weights)
        gid = resize_group_ids(gid, a.height, a.width)
        Image.fromarray(flat_masks(gid)).save(
            os.path.join(a.out, 'panel_mask.png'))
    else:
        print('[mask] SKIPPED (no --fastsam-weights). Produce panel_mask.png '
              'from the project FastSAM path so the masks match the ones the '
              'model actually lifts.')

    if gid is not None:
        e = outline(gid)
        pca = pca.copy()
        pca[e] = (0.35 * pca[e] + 0.65 * np.array([255, 255, 255])).astype(np.uint8)
    Image.fromarray(pca).save(os.path.join(a.out, 'panel_dino.png'))

    print(f'\n[done] wrote panels to {a.out}/')
    print(f'[provenance] scene={a.scene} frame={a.frame} model={a.dino} '
          f'res={a.height}x{a.width} clip={a.clip_pct}%  '
          '<-- record this in the figure README, the frame choice must be reproducible')


if __name__ == '__main__':
    main()
