#!/usr/bin/env python3
"""Render GT / matched-host / EmbodiedDINO instance panels for one scene.

WHAT MAKES THIS DIFFERENT FROM fig1_motivation/render_scene_instances.py
------------------------------------------------------------------------
That script renders GROUND TRUTH as a point cloud.  Ground truth cannot show an
advantage (there is nothing to compare), and point splats leave pinholes.  Here:
  * panels come from real PREDICTION dumps (dump_pred_instances.py), so the
    figure shows what each model actually output;
  * geometry is the ScanNet MESH, so surfaces are continuous instead of sparse.

THE MAPPING IS THE IDENTITY -- verified, do not add a KD-tree
------------------------------------------------------------
data/scannet200/points/<scene>.bin holds exactly the vertices of
scans/<scene>/<scene>_vh_clean_2.ply, in the same order:
    N = 293811 both, max |mesh_raw - pts| = 0.00000   (scene0030_00)
so a per-point label array indexes mesh vertices directly.  The points are in
the RAW mesh frame, NOT axis aligned -- against the axis-aligned mesh the same
comparison gives 8.4 m.  (render_scene_instances.py's docstring claims the
points "are already axis aligned" because the bbox min sits near the origin.
That inference is wrong; its conclusion -- don't apply the matrix -- is right.)

The axis-align matrix IS applied here, but for the CAMERA only: it is a
rotation about z plus a translation, so it leaves labels untouched while making
every room share one orientation.  That is what lets a single (elev, azim) mean
the same relative viewpoint in all scenes, which a multi-scene figure needs.

TWO TRAPS INHERITED FROM THE EARLIER RENDERER
---------------------------------------------
1. setup_camera derives clip planes from the current scene bbox, so it must be
   called AFTER add_geometry or everything is clipped away and the frame is blank.
2. Open3D tone-maps the offscreen buffer: a (1,1,1) background lands near
   (236,235,235).  Background is keyed by "bright AND neutral"; any neutral grey
   brighter than ~0.88 is keyed out as background and vanishes.  The floor grey
   therefore stays at 0.78.

COLOURING -- the choice that decides whether the figure shows anything
----------------------------------------------------------------------
--scheme own : each panel is coloured by its OWN instance ids.  Fragmentation
    reads directly as several colours on one object.  Colours do not correspond
    across panels.
--scheme gt  : each predicted instance takes the colour of the GT instance it
    overlaps most, so the same object is the same hue in every panel.  On its
    own this HIDES fragmentation -- two fragments of one object match the same
    GT and come out identically coloured -- so instance seams are drawn as well.
    The seams are what make this scheme work; they are not decoration.
"""

import argparse
import colorsys
import json
import os
import os.path as osp

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

DATA = os.path.expanduser('~/xxy/3D_Reconstruction/data')


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--dump-host', required=True, help='dir holding npz/<scene>.npz')
    ap.add_argument('--dump-ours', required=True)
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--out', default='renders')
    ap.add_argument('--scheme', default='own', choices=('own', 'gt'))
    ap.add_argument('--drop', default='structure,unlabeled',
                    help="roles to remove from every panel. 'unlabeled' is the "
                         'region ScanNet never annotated; it is excluded from '
                         'evaluation, so keeping it would invite the reader to '
                         'judge a region no method is scored on.')
    ap.add_argument('--drop-classes', default='',
                    help='extra class names to treat as structure, on top of '
                         'STRUCTURE_CLASSES')
    ap.add_argument('--grey-floor', action='store_true', default=True)
    ap.add_argument('--floor-grey-level', type=float, default=0.78)
    ap.add_argument('--width', type=int, default=1100)
    ap.add_argument('--height', type=int, default=850)
    ap.add_argument('--down', type=int, default=2)
    ap.add_argument('--elev', type=float, default=38.0)
    ap.add_argument('--azim', type=float, default=225.0)
    ap.add_argument('--azims', default='', help='comma list; renders one set per azimuth')
    ap.add_argument('--dist', type=float, default=0.92)
    ap.add_argument('--fov', type=float, default=40.0)
    ap.add_argument('--sat', type=float, default=0.62)
    ap.add_argument('--val', type=float, default=0.88)
    ap.add_argument('--min-inst-pts', type=int, default=100,
                    help='predicted instances smaller than this are treated as '
                         'unassigned; they are invisible at figure scale but '
                         'would each consume a hue')
    ap.add_argument('--highlight', type=int, default=3,
                    help='ring the N objects the host fragments most and '
                         'we fragment less; 0 disables. Boxes are drawn on '
                         'ALL panels at the same place.')
    ap.add_argument('--margin', type=int, default=10)
    return ap.parse_args()


# --------------------------------------------------------------------- loading
def load_scene(data, scene):
    mesh = o3d.io.read_triangle_mesh(
        osp.join(data, 'scannet200', 'scans', scene, f'{scene}_vh_clean_2.ply'))
    pts = np.fromfile(osp.join(data, 'scannet200', 'points', f'{scene}.bin'),
                      dtype=np.float32).reshape(-1, 6)
    v = np.asarray(mesh.vertices)
    assert len(v) == len(pts), (
        f'{scene}: mesh has {len(v)} vertices but points/.bin has {len(pts)}. '
        'The identity mapping this script relies on does not hold here.')
    d = np.abs(v - pts[:, :3]).max()
    assert d < 1e-3, (f'{scene}: mesh and points differ by {d:.4f} m; they are '
                      'not the same cloud in the same frame.')
    A = np.load(osp.join(data, 'scannet200-mv', 'axis_align_matrix',
                         f'{scene}.npy'))
    return mesh, A


def load_dump(d, scene):
    z = np.load(osp.join(d, 'npz', f'{scene}.npz'))
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ structure
# The dump's gt_sem is a 0-based index into the dataset's 200-class tuple, and
# 200 means unlabeled.  Verified on scene0568_00: index 0 covers 22.1% of
# vertices and index 1 covers 17.1% (wall, floor), and gt_inst < 0 coincides
# point-for-point with the unlabeled region.
#
# WHY THIS IS A NAME LIST AND NOT A LIST OF INDICES
# ScanNet200 puts building structure in classes that are nominally objects:
# 'door', 'window', 'curtain', 'shower wall', 'bathroom stall'.  A first pass
# dropped only wall/floor/ceiling and two of these slabs then filled the entire
# frame of scene0458_00 -- the panel showed a door, not a room of objects.
# Indices are resolved from the dataset's own METAINFO at runtime rather than
# counted by hand off the source, because miscounting one position silently
# deletes the wrong class.
STRUCTURE_CLASSES = (
    'wall', 'ceiling', 'door', 'doorframe', 'window', 'windowsill', 'curtain',
    'blinds', 'shower curtain', 'shower curtain rod', 'shower door',
    'shower wall', 'shower floor', 'closet door', 'closet wall', 'closet',
    'bathroom stall', 'bathroom stall door', 'mirror', 'divider', 'column',
    'pillar', 'structure', 'ledge', 'stairs', 'stair rail', 'rail',
    'ceiling light', 'vent', 'light switch', 'power outlet',
)
FLOOR_CLASS = 'floor'
SEM_UNLABELED = 200


def class_tuple():
    """The dataset's own 200-class tuple, found by scanning its METAINFOs."""
    import oneformer3d.scannet_dataset as sd
    best = None
    for name in dir(sd):
        mi = getattr(getattr(sd, name), 'METAINFO', None)
        if isinstance(mi, dict):
            c = mi.get('classes')
            if isinstance(c, (list, tuple)) and len(c) >= 200 \
                    and tuple(c[:2]) == ('wall', 'floor'):
                best = tuple(c)
    if best is None:
        raise RuntimeError(
            'no METAINFO in oneformer3d.scannet_dataset has a >=200-entry class '
            'tuple beginning (wall, floor); refusing to guess semantic indices')
    return best


def roles(xyz, gt_inst, gt_sem, drop_names):
    """Per-vertex role: structure / floor / unlabeled / object."""
    names = class_tuple()
    idx = {n: i for i, n in enumerate(names)}
    unknown = [n for n in drop_names if n not in idx]
    if unknown:
        print(f'[roles] not classes in this dataset, ignored: {unknown}')
    struct = np.array([idx[n] for n in drop_names if n in idx], np.int64)

    role = np.full(len(xyz), 'object', dtype='<U10')
    role[np.isin(gt_sem, struct)] = 'structure'
    role[gt_sem == idx[FLOOR_CLASS]] = 'floor'
    role[(gt_sem == SEM_UNLABELED) | (gt_inst < 0)] = 'unlabeled'

    # A ceiling that carries an object class rather than 'ceiling': planar,
    # near-horizontal, high in the room.  Most ScanNet ceilings fall in the
    # unlabeled region and are already caught above.
    zlo, zhi = xyz[:, 2].min(), xyz[:, 2].max()
    H = max(zhi - zlo, 1e-6)
    for i in np.unique(gt_inst[(gt_inst >= 0) & (role == 'object')]):
        m = gt_inst == i
        q = xyz[m]
        if len(q) < 200:
            continue
        c = q - q.mean(0)
        w, vec = np.linalg.eigh(c.T @ c / len(q))
        if (w[0] / max(w.sum(), 1e-12) < 0.03 and abs(vec[2, 0]) > 0.85
                and (q[:, 2].mean() - zlo) / H > 0.78):
            role[m] = 'structure'
    per_class = {}
    for s in np.unique(gt_sem[role == 'structure']):
        if 0 <= s < len(names):
            per_class[names[s]] = int((gt_sem == s).sum())
    return role, per_class


# -------------------------------------------------------------------- palette
def palette(ids, sat, val, seed_hue=0.11):
    """One flat pastel colour per id, hues walking the circle by the golden angle.

    Consecutive ids in ScanNet tend to be spatially adjacent, so a golden-angle
    walk puts neighbours far apart in hue without needing an adjacency graph.
    """
    g = 0.6180339887498949
    h = seed_hue
    lut = {}
    for i in sorted(int(x) for x in ids):
        h = (h + g) % 1.0
        s = sat * (0.85 + 0.3 * ((i * 7) % 5) / 4.0)
        lut[i] = colorsys.hsv_to_rgb(h, min(s, 0.95), val)
    return lut


def clean_pred(label, min_pts):
    """Drop predicted instances too small to be seen, so they cost no hue."""
    out = label.copy()
    ids, cnt = np.unique(out[out >= 0], return_counts=True)
    for i in ids[cnt < min_pts]:
        out[out == i] = -1
    return out


def match_to_gt(pred, gt):
    """For each predicted instance, the GT instance it overlaps most.

    Plain per-instance intersection counting.  A confusion matrix over
    (n_pred x n_gt) would be cleaner to read but allocates n_pred*n_gt and both
    can exceed a hundred, so this stays a loop over predictions.
    """
    m = {}
    for p in np.unique(pred[pred >= 0]):
        sel = gt[pred == p]
        sel = sel[sel >= 0]
        if len(sel) == 0:
            m[int(p)] = None
            continue
        u, c = np.unique(sel, return_counts=True)
        m[int(p)] = int(u[np.argmax(c)])
    return m


def seam_vertices(faces, label):
    """Vertices of faces that straddle an instance boundary.

    One vertex wide, which at figure scale reads as a line.  This is what keeps
    fragmentation visible under --scheme gt, where two fragments of one object
    would otherwise share a colour.
    """
    l0, l1, l2 = label[faces[:, 0]], label[faces[:, 1]], label[faces[:, 2]]
    mixed = (l0 != l1) | (l1 != l2) | (l0 != l2)
    return np.unique(faces[mixed].reshape(-1))


# ------------------------------------------------------------------ rendering
def camera(xyz, elev, azim, dist, fov, r):
    lo, hi = xyz.min(0), xyz.max(0)
    center = (lo + hi) / 2.0
    diag = float(np.linalg.norm(hi - lo))
    e, a = np.radians(elev), np.radians(azim)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    eye = center + d * (dist * diag)
    # World +z as "up" becomes parallel to the view direction as elev -> 90 and
    # the camera degenerates (blank or wildly rolled frame).  Near straight down,
    # use a horizontal up instead, so azim keeps controlling the in-plane
    # rotation of a top view rather than doing nothing.
    up = (np.array([0., 0., 1.]) if elev < 88.0
          else np.array([-np.cos(a), -np.sin(a), 0.]))
    r.setup_camera(fov, center.astype(np.float32), eye.astype(np.float32),
                   up.astype(np.float32))


def frag_count(pred, gt, ids, frac=0.1):
    """How many predicted instances each GT object is carved into."""
    out = {}
    for g in ids:
        m = gt == g
        tot = m.sum()
        p, c = np.unique(pred[m], return_counts=True)
        out[int(g)] = sum(1 for pi, ci in zip(p, c)
                          if pi >= 0 and ci >= frac * tot)
    return out


def highlight_boxes(base, keep, gt, targets, a, cam_xyz):
    """Image-space boxes for the highlighted GT objects.

    Found by rendering a mask pass -- the object black, everything else white --
    and taking the extent of the non-white pixels.  This is deliberately not a
    camera projection: getting Open3D's FoV convention and aspect handling wrong
    would place a box slightly off its object, which looks like a claim about the
    wrong thing and is hard to notice.  A mask pass cannot be off by definition.
    """
    boxes = {}
    for g in targets:
        col = np.ones((int(keep.sum()), 3))
        col[gt[keep] == g] = 0.0
        img, _ = render(base, col, a, cam_xyz)
        ink = img.min(2) < 128
        if not ink.any():
            continue                        # occluded from this azimuth
        ys, xs = np.where(ink)
        boxes[int(g)] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return boxes


def draw_dashed(d, box, colour, w, dash):
    x0, y0, x1, y1 = box
    for x in range(x0, x1, dash * 2):
        d.rectangle([x, y0, min(x + dash, x1), y0 + w], fill=colour)
        d.rectangle([x, y1 - w, min(x + dash, x1), y1], fill=colour)
    for y in range(y0, y1, dash * 2):
        d.rectangle([x0, y, x0 + w, min(y + dash, y1)], fill=colour)
        d.rectangle([x1 - w, y, x1, min(y + dash, y1)], fill=colour)


def render(mesh, colors, a, cam_xyz):
    m = o3d.geometry.TriangleMesh(mesh)
    m.vertex_colors = o3d.utility.Vector3dVector(colors)
    m.compute_vertex_normals()
    r = o3d.visualization.rendering.OffscreenRenderer(a.width * a.down,
                                                      a.height * a.down)
    r.scene.set_background(np.array([1., 1., 1., 1.], np.float32))
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'          # flat colours; shading would alter hue
    r.scene.add_geometry('m', m, mat)    # geometry FIRST -- see docstring trap 1
    camera(cam_xyz, a.elev, a.azim, a.dist, a.fov, r)
    img = np.asarray(r.render_to_image())[..., :3].astype(np.int16)
    mx, mn = img.max(2), img.min(2)
    bg = (mn >= 225) & ((mx - mn) <= 3)
    img[bg] = 255
    del r
    return img.astype(np.uint8), bg


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    mesh, A = load_scene(a.data, a.scene)
    host = load_dump(a.dump_host, a.scene)
    ours = load_dump(a.dump_ours, a.scene)

    v = np.asarray(mesh.vertices)
    gt = host['gt_inst'].astype(np.int64)
    assert np.array_equal(gt, ours['gt_inst'].astype(np.int64)), \
        'the two dumps disagree on ground truth -- different splits or scenes'
    assert len(gt) == len(v), f'{len(gt)} labels vs {len(v)} vertices'

    # -- structure removal, decided ONCE on GT and applied to every panel, so
    # -- no panel is judged on a different set of vertices than another.
    extra = [x.strip() for x in a.drop_classes.split(',') if x.strip()]
    role, per_class = roles(v, gt, host['gt_sem'],
                            tuple(STRUCTURE_CLASSES) + tuple(extra))
    drop = {x.strip() for x in a.drop.split(',') if x.strip()}
    keep = ~np.isin(role, list(drop)) if drop else np.ones(len(role), bool)
    floor_mask = role == 'floor'
    tally = {r: int((role == r).sum()) for r in np.unique(role)}
    print(f'[roles] {tally}')
    print(f'[structure] {dict(sorted(per_class.items(), key=lambda t: -t[1]))}')
    print(f'[drop] {sorted(drop)}: {int((~keep).sum())} verts '
          f'({100.0 * (~keep).mean():.1f}%)')
    ids = np.unique(gt[(gt >= 0) & keep & ~floor_mask])
    # Coverage on the things the figure is actually about.  If this is high the
    # panels will read as grey patches rather than as instances, and no amount
    # of palette work will fix it -- so it is printed, not assumed.
    obj = keep & ~floor_mask & (gt >= 0)
    for tag, pl in (('host', host['pred_label']), ('ours', ours['pred_label'])):
        print(f'[coverage] {tag}: {100.0 * (pl[obj] < 0).mean():.1f}% of shown '
              f'object vertices have no predicted instance')

    p_host = clean_pred(host['pred_label'].astype(np.int64), a.min_inst_pts)
    p_ours = clean_pred(ours['pred_label'].astype(np.int64), a.min_inst_pts)
    print(f'[instances] gt={len(ids)}  host={len(np.unique(p_host[p_host >= 0]))}'
          f'  ours={len(np.unique(p_ours[p_ours >= 0]))}')

    gt_lut = palette(ids, a.sat, a.val)
    GREY = (0.86, 0.86, 0.88)            # unassigned / unmatched
    FLOOR = (a.floor_grey_level,) * 2 + (a.floor_grey_level * 1.02,)
    SEAM = (0.25, 0.25, 0.27)

    def colorize(label, is_gt):
        c = np.tile(np.array(GREY), (len(v), 1))
        if is_gt:
            for i in ids.tolist():
                c[gt == i] = gt_lut[i]
        elif a.scheme == 'own':
            for i, col in palette(np.unique(label[label >= 0]),
                                  a.sat, a.val).items():
                c[label == i] = col
        else:
            for p, g in match_to_gt(label, gt).items():
                c[label == p] = gt_lut[g] if g in gt_lut else GREY
        # The floor is a neutral ground plane in every panel: it is structure,
        # it is excluded from instance evaluation anyway, and giving it a hue
        # would spend the reader's attention on the one region no method is
        # being judged on.
        if a.grey_floor:
            c[floor_mask] = FLOOR
        if a.scheme == 'gt':
            c[seam_vertices(np.asarray(mesh.triangles), gt if is_gt else label)] = SEAM
        return c

    panels = [('gt', colorize(gt, True)),
              ('host', colorize(p_host, False)),
              ('ours', colorize(p_ours, False))]

    # Cull once, so all three panels carry identical geometry and the camera and
    # crop are therefore identical too.
    base = o3d.geometry.TriangleMesh(mesh)
    base.vertices = o3d.utility.Vector3dVector(
        (np.c_[v, np.ones(len(v))] @ A.T)[:, :3])   # camera frame only
    if not keep.all():
        base.remove_vertices_by_mask(~keep)
    cam_xyz = np.asarray(base.vertices)

    # Which objects to ring: the ones the host carves up and we do not.  Chosen
    # by count, not by eye, so the choice is reportable -- and the same boxes go
    # on all three panels, so a box is an invitation to compare, not a claim that
    # something is wrong only in the panel it is drawn on.
    targets = []
    if a.highlight:
        gk = np.where(keep, gt, -1)
        fh = frag_count(np.where(keep, p_host, -1), gk, ids)
        fo = frag_count(np.where(keep, p_ours, -1), gk, ids)
        # Tie-break by SIZE, not by instance id.  A first version sorted on
        # (fragment difference, id); most objects tie at a difference of 1, so
        # the ring landed on whichever clutter happened to have the highest id
        # while the large obviously-shattered surface went unmarked.  What the
        # eye notices is area.
        size = {int(g): int(((gt == g) & keep).sum()) for g in ids}
        cand = [(fh[g] - fo[g], size[g], int(g)) for g in fh
                if fh[g] >= 2 and fh[g] > fo[g]]
        cand.sort(reverse=True)
        targets = [g for _, _, g in cand[:a.highlight]]
        print(f'[highlight] {len(targets)} of {len(cand)} candidates; '
              f'(id, host_frags, ours_frags, verts) = '
              f'{[(g, fh[g], fo[g], size[g]) for g in targets]}')

    azims = ([float(x) for x in a.azims.split(',') if x.strip()]
             if a.azims else [a.azim])
    for az in azims:
        a.azim = az
        imgs, bgs = {}, []
        for name, col in panels:
            img, bg = render(base, col[keep], a, cam_xyz)
            imgs[name] = img
            bgs.append(bg)
        boxes = (highlight_boxes(base, keep, gt, targets, a, cam_xyz)
                 if targets else {})
        fg = ~np.logical_and.reduce(bgs)      # union of ink over the 3 panels
        ys, xs = np.where(fg)
        m = a.margin * a.down
        y0, y1 = max(0, ys.min() - m), min(fg.shape[0], ys.max() + m + 1)
        x0, x1 = max(0, xs.min() - m), min(fg.shape[1], xs.max() + m + 1)
        tag = f'{a.scene}_az{int(az)}_{a.scheme}'
        pad = 4 * a.down
        for name, img in imgs.items():
            im = Image.fromarray(img[y0:y1, x0:x1])
            if boxes:
                d = ImageDraw.Draw(im)
                for bx in boxes.values():
                    draw_dashed(d, (max(0, bx[0] - x0 - pad),
                                    max(0, bx[1] - y0 - pad),
                                    min(im.width - 1, bx[2] - x0 + pad),
                                    min(im.height - 1, bx[3] - y0 + pad)),
                                (40, 40, 45), max(1, a.down), 6 * a.down)
            if a.down > 1:
                im = im.resize((im.width // a.down, im.height // a.down),
                               Image.LANCZOS)
            im.save(osp.join(a.out, f'{tag}_{name}.png'))
        print(f'[write] {tag}_*.png  {(x1 - x0) // a.down}x{(y1 - y0) // a.down}'
              f'  boxes={len(boxes)}')

    with open(osp.join(a.out, f'{a.scene}_stats.json'), 'w') as f:
        json.dump({'scene': a.scene, 'n_gt': int(len(ids)),
                   'n_host': int(len(np.unique(p_host[p_host >= 0]))),
                   'n_ours': int(len(np.unique(p_ours[p_ours >= 0]))),
                   'dropped_frac': float((~keep).mean()),
                   'classify': tally}, f, indent=1)


if __name__ == '__main__':
    main()
