#!/usr/bin/env python3
"""
Render a scene's instance decomposition as a coloured point cloud, in the
"dollhouse" style of the reference panel (assets/example.png): one flat colour
per instance, white background, elevated oblique view.

WHAT IS BEING SHOWN -- read this before writing a caption
---------------------------------------------------------
Colours come from the dataset's GROUND-TRUTH instance annotation
(data/scannet200/instance_mask/<scene>.bin), NOT from any model prediction.
This panel therefore illustrates what a scene-level instance decomposition IS.
It must NOT be captioned as our output, or as anyone's output.  If a predicted
decomposition is wanted instead, that needs prediction dumps and a different
script.

DATA LAYOUT (verified on the server, 2026-07-29)
------------------------------------------------
  points/<scene>.bin         float32, N x 6  -> xyz (metres) + rgb (1..255)
  instance_mask/<scene>.bin  int64,   N      -> instance id per point
For scene0030_00: N = 293,811, 91 instance ids (0..90), and the coordinates are
already axis aligned (bbox min sits at the origin), so no axis_align_matrix has
to be applied here.

TWO TRAPS ALREADY PAID FOR (see render_scene_3d.py)
---------------------------------------------------
1. Open3DScene.setup_camera derives near/far clip planes from the CURRENT scene
   bounding box.  Called before add_geometry it clips everything away and the
   render comes out blank.  Camera setup therefore happens AFTER add_geometry.
2. A near-uniform output means the geometry was not visible, not that the scene
   is plain.  The renderer checks pixel std and warns.

RUN (server; needs the EGL-capable open3d in the ESAM env)
    /home/nebula/miniconda3/envs/ESAM/bin/python render_scene_instances.py \
        --scene scene0030_00 --out assets
"""

import argparse
import colorsys
import os

import numpy as np
import open3d as o3d
from PIL import Image

ROOT = os.path.expanduser('~/xxy/3D_Reconstruction/data/scannet200')


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='scene0030_00')
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--out', default='assets')
    ap.add_argument('--name', default='scene_instances.png')
    ap.add_argument('--width', type=int, default=1600)
    ap.add_argument('--height', type=int, default=1000)
    ap.add_argument('--down', type=int, default=2, help='supersample factor')
    ap.add_argument('--point-size', type=float, default=6.0)
    ap.add_argument('--elev', type=float, default=32.0, help='degrees above horizon')
    ap.add_argument('--azim', type=float, default=225.0, help='degrees, 0=+x')
    ap.add_argument('--dist', type=float, default=0.95,
                    help='eye distance as a multiple of the bbox diagonal')
    ap.add_argument('--fov', type=float, default=40.0)
    ap.add_argument('--sat', type=float, default=0.62, help='palette saturation')
    ap.add_argument('--val', type=float, default=0.88, help='palette value')
    ap.add_argument('--no-crop', action='store_true',
                    help='keep the full frame instead of trimming background margins')
    ap.add_argument('--margin', type=int, default=8, help='pixels kept around the crop')
    ap.add_argument('--drop', default='',
                    help="comma list of structure to remove: walls,ceiling,floor. "
                         "Detected GEOMETRICALLY (per-instance PCA), because this "
                         "scene's semantic_mask indices do not line up with the "
                         "class-name list in the training configs -- verified by "
                         "cross-checking z-extents, so labels are not trusted here.")
    ap.add_argument('--top-k', type=int, default=0,
                    help='after structure removal, keep only the K largest instances. '
                         'This is the lever that makes the panel readable: the full '
                         'GT decomposition of this scene has 91 instances.')
    ap.add_argument('--frames-upto', type=int, default=0,
                    help='keep only instances that appear in the per-frame 2D instance '
                         'maps up to this frame index (partial scene state). Assumes '
                         'the 2D maps use scene-level instance ids, which is consistent '
                         'with their value range but has not been proven.')
    ap.add_argument('--floor-grey-level', type=float, default=0.78,
                    help='brightness of the neutral floor. Must stay below the '
                         'background-keying threshold (~0.88 after tone mapping), '
                         'otherwise the floor is keyed out as background and '
                         'disappears -- which is exactly what 0.87 did.')
    ap.add_argument('--grey-floor', action='store_true',
                    help='keep the floor as a neutral ground plane instead of giving it '
                         'a hue and a slot in --top-k')
    ap.add_argument('--neutral-largest', type=int, default=0,
                    help='paint the N largest instances light grey (floor/wall) '
                         'instead of giving them a hue')
    return ap.parse_args()


def load(root, scene):
    p = np.fromfile(os.path.join(root, 'points', f'{scene}.bin'),
                    dtype=np.float32).reshape(-1, 6)
    m = np.fromfile(os.path.join(root, 'instance_mask', f'{scene}.bin'),
                    dtype=np.int64)
    sem_path = os.path.join(root, 'semantic_mask', f'{scene}.bin')
    sem = (np.fromfile(sem_path, dtype=np.int64)
           if os.path.exists(sem_path) else np.zeros(len(m), np.int64))
    assert len(p) == len(m) == len(sem), (p.shape, m.shape, sem.shape)
    print(f'[load] {scene}: {len(p)} points, {len(np.unique(m))} instances, '
          f'bbox {np.round(p[:, :3].min(0), 2)}..{np.round(p[:, :3].max(0), 2)}')
    return p[:, :3].astype(np.float64), m, sem


def palette(ids, sat, val, neutral_largest, counts):
    """One flat pastel colour per instance.

    Hues walk the circle by the golden angle so that consecutive ids -- which in
    ScanNet tend to be spatially adjacent -- land far apart, without needing a
    3D adjacency graph.
    """
    order = np.argsort(-counts)
    neutral = set(ids[order[:neutral_largest]].tolist()) if neutral_largest else set()
    lut = {}
    g = 0.6180339887498949
    h = 0.11
    for i in ids.tolist():
        if i in neutral:
            lut[i] = (0.86, 0.86, 0.88)
            continue
        h = (h + g) % 1.0
        # nudge saturation per id so neighbouring hues are also tonally distinct
        s = sat * (0.85 + 0.3 * ((i * 7) % 5) / 4.0)
        lut[i] = colorsys.hsv_to_rgb(h, min(s, 0.95), val)
    return lut


def classify(xyz, inst, ids):
    """Label each INSTANCE as wall / ceiling / floor / object from its own shape.

    Two independent wall tests, because one is not enough:
      * planar with a near-horizontal normal and a substantial vertical extent --
        catches an ordinary wall face;
      * spanning most of the room height, whatever its planarity -- catches an
        instance that unions several faces, which defeats the PCA test (id=0 of
        scene0030_00 is exactly this and was missed until this clause was added).
    Judging per instance rather than per semantic class matters: a class aggregates
    instances at different heights, which inflates its span and would misfile e.g.
    boxes as structure.
    """
    zlo, zhi = xyz[:, 2].min(), xyz[:, 2].max()
    H = max(zhi - zlo, 1e-6)
    kind = {}
    for i in ids.tolist():
        q = xyz[inst == i]
        if len(q) < 200:
            kind[i] = 'object'; continue
        c = q - q.mean(0)
        w, v = np.linalg.eigh(c.T @ c / len(q))
        n = v[:, 0]                          # least-variance direction = normal
        planar = w[0] / max(w.sum(), 1e-12) < 0.03
        ez = (q[:, 2].max() - q[:, 2].min()) / H
        zc = (q[:, 2].mean() - zlo) / H
        if planar and abs(n[2]) > 0.85 and zc < 0.22:
            kind[i] = 'floor'
        elif planar and abs(n[2]) > 0.85 and zc > 0.78:
            kind[i] = 'ceiling'
        elif (planar and abs(n[2]) < 0.35 and ez > 0.45) or ez > 0.70:
            kind[i] = 'wall'
        else:
            kind[i] = 'object'
    return kind


def camera(xyz, elev, azim, dist, fov, renderer):
    lo, hi = xyz.min(0), xyz.max(0)
    center = (lo + hi) / 2.0
    diag = float(np.linalg.norm(hi - lo))
    e, a = np.radians(elev), np.radians(azim)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    eye = center + d * (dist * diag)
    renderer.setup_camera(fov, center.astype(np.float32), eye.astype(np.float32),
                          np.array([0., 0., 1.], np.float32))
    print(f'[camera] eye={np.round(eye, 2)} lookat={np.round(center, 2)} '
          f'elev={elev} azim={azim} fov={fov}')


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    xyz, inst, sem = load(a.root, a.scene)
    ids_all, counts_all = np.unique(inst, return_counts=True)
    lut = palette(ids_all, a.sat, a.val, a.neutral_largest, counts_all)

    ids, counts = np.unique(inst, return_counts=True)

    _alias = {'walls': 'wall', 'ceilings': 'ceiling', 'floors': 'floor'}
    drop = {_alias.get(x.strip(), x.strip()) for x in a.drop.split(',') if x.strip()}
    if drop:
        kind = classify(xyz, inst, ids)
        tally = {}
        for i in ids.tolist():
            tally[kind[i]] = tally.get(kind[i], 0) + int((inst == i).sum())
        print('[classify] ' + '  '.join(f'{k}={v}pts' for k, v in sorted(tally.items())))
        bad = np.isin(inst, [i for i in ids.tolist() if kind[i] in drop])
        print(f'[drop] {sorted(drop)}: {int(bad.sum())} points '
              f'({100.0 * bad.mean():.1f}%)')
        xyz, inst = xyz[~bad], inst[~bad]
        ids, counts = np.unique(inst, return_counts=True)
        print(f'[keep] {len(xyz)} points, {len(ids)} instances')

    floor_ids = set()
    if a.grey_floor:
        k2 = classify(xyz, inst, ids)
        floor_ids = {i for i in ids.tolist() if k2[i] == 'floor'}
        print(f'[grey-floor] {len(floor_ids)} instance(s) held out as ground plane')

    if a.frames_upto:
        import glob
        d = os.path.join(os.path.dirname(a.root.rstrip('/')), 'scannet200-mv', '2D',
                         a.scene, 'instance')
        seen = set()
        used = 0
        for f in sorted(glob.glob(os.path.join(d, '*')),
                        key=lambda q: int(os.path.basename(q).split('.')[0])):
            if int(os.path.basename(f).split('.')[0]) > a.frames_upto:
                break
            seen |= set(np.unique(np.array(Image.open(f))).tolist())
            used += 1
        keep = np.array(sorted((seen | floor_ids) & set(ids.tolist())))
        print(f'[frames<={a.frames_upto}] {used} maps, {len(seen)} ids seen, '
              f'{len(keep)} kept of {len(ids)}')
        m = np.isin(inst, keep)
        xyz, inst = xyz[m], inst[m]
        ids, counts = np.unique(inst, return_counts=True)

    if a.top_k:
        rank = [i for i in ids[np.argsort(-counts)].tolist() if i not in floor_ids]
        keep = np.array(sorted(set(rank[:a.top_k]) | floor_ids))
        print(f'[top-k] keeping {len(keep)} instances '
              f'({a.top_k} largest + {len(floor_ids)} floor) of {len(ids)}')
        m = np.isin(inst, keep)
        xyz, inst = xyz[m], inst[m]
        ids, counts = np.unique(inst, return_counts=True)
        print(f'[final] {len(xyz)} points, {len(ids)} instances')


    rgb = np.empty((len(xyz), 3), np.float64)
    for i in ids.tolist():
        g = a.floor_grey_level
        rgb[inst == i] = (g, g, g * 1.02) if i in floor_ids else lut[i]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(rgb)

    r = o3d.visualization.rendering.OffscreenRenderer(a.width * a.down,
                                                      a.height * a.down)
    r.scene.set_background(np.array([1., 1., 1., 1.], np.float32))
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'                 # flat colours, no shading
    mat.point_size = a.point_size * a.down

    r.scene.add_geometry('inst', pc, mat)       # geometry FIRST -- see docstring
    camera(xyz, a.elev, a.azim, a.dist, a.fov, r)

    img = np.asarray(r.render_to_image())
    if img[..., :3].std() < 1.0:
        print(f'[WARN] output looks uniform (std={img[..., :3].std():.3f}); the '
              'geometry was probably not visible -- check the camera, not the data')
    # Open3D tone-maps the offscreen buffer, so a (1,1,1) background lands near
    # (236,235,235) rather than pure white.  Force it: a pixel is background only
    # if it is BOTH very bright and nearly neutral, which no hued instance is.
    arr = img[..., :3].astype(np.int16)
    mx, mn = arr.max(2), arr.min(2)
    bg = (mn >= 225) & ((mx - mn) <= 3)
    arr[bg] = 255
    img = arr.astype(np.uint8)
    print(f'[bg] {100.0 * bg.mean():.1f}% of pixels keyed to white')

    if not a.no_crop:
        ys, xs = np.where(~bg)
        m = a.margin * a.down
        y0, y1 = max(0, ys.min() - m), min(img.shape[0], ys.max() + m + 1)
        x0, x1 = max(0, xs.min() - m), min(img.shape[1], xs.max() + m + 1)
        img = img[y0:y1, x0:x1]
        print(f'[crop] -> {img.shape[1]}x{img.shape[0]}')

    im = Image.fromarray(img)
    if a.down > 1:
        im = im.resize((im.width // a.down, im.height // a.down), Image.LANCZOS)
    path = os.path.join(a.out, a.name)
    im.save(path)
    print(f'[write] {path} {im.size}  (ground-truth instances, not a prediction)')


if __name__ == '__main__':
    main()
