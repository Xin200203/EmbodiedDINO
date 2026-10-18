#!/usr/bin/env python3
"""
Render the 3D-scene panels used by Fig. 2 (pipeline).

OUTPUTS (into --out)
--------------------
  scene_mesh.png           ScanNet reconstructed mesh (*_vh_clean_2.ply), vertex
                           colours, unlit  -> a continuous surface CANNOT look
                           sparse, which is why the mesh variant exists
  scene_mesh_lit.png       same mesh with a lit shader (pick one after seeing)
  scene_points.png         accumulated per-frame back-projected points (all
                           frames concatenated -- single-frame clouds show holes
                           from almost any angle), big splats, rendered 2x and
                           downsampled
  scene_frame_partial.png  ONE frame's back-projected points from the same
                           camera: the honest input-side quantity

DESIGN NOTES
------------
* The mv per-frame bins and the raw ScanNet mesh may live in different frames
  (there is an axis_align_matrix per scene).  We DETECT which frame the bins
  use by comparing bounding boxes instead of guessing.
* Camera: mean capture position of the trajectory, elevated and pulled back to
  fit the room -- "roughly along the capture direction", per the EmbodiedSAM
  reference look; NOT an oblique low view that exposes scan gaps.
* Colours are the scan's own RGB (no model predictions -- this must not become
  a results claim).

Run on the server (needs EGL-capable open3d; 0.17 on the server works):
    /home/nebula/miniconda3/envs/ESAM/bin/python render_scene_3d.py \
        --scene scene0030_00 --frame 400 --out assets_fig2
"""

import argparse
import glob
import os

import numpy as np
import open3d as o3d
from PIL import Image


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='scene0030_00')
    ap.add_argument('--frame', type=int, default=400,
                    help='frame used for the single-frame partial cloud; must '
                         'match the 2D panels')
    ap.add_argument('--scans-root',
                    default='/home/nebula/xxy/dataset/scannet_v2/scans')
    ap.add_argument('--mv-root',
                    default='/home/nebula/xxy/3D_Reconstruction/data/scannet200-mv')
    ap.add_argument('--out', default='assets_fig2')
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--height', type=int, default=1440)
    ap.add_argument('--down', type=int, default=2, help='downsample factor')
    ap.add_argument('--point-size', type=float, default=6.0)
    ap.add_argument('--partial-point-size', type=float, default=9.0)
    ap.add_argument('--fov', type=float, default=62.0)
    ap.add_argument('--height-boost', type=float, default=0.85,
                    help='camera elevation; 0.85 gives ~52deg, 0.55 ~40deg')
    ap.add_argument('--back-boost', type=float, default=0.85)
    ap.add_argument('--suffix', default='',
                    help='appended to output stems, e.g. _v2 for a variant pass')
    ap.add_argument('--zoom-partial', action='store_true',
                    help='also render the single-frame cloud framed on its own '
                         'bbox (same view direction), as an input-side icon')
    return ap.parse_args()


def load_bins(pts_dir):
    out = {}
    for p in glob.glob(os.path.join(pts_dir, '*.bin')):
        idx = int(os.path.basename(p).split('.')[0])
        d = np.fromfile(p, dtype=np.float32).reshape(-1, 6)
        out[idx] = d
    if not out:
        raise FileNotFoundError(f'no .bin files under {pts_dir}')
    return out


def bbox_iou(a, b):
    lo = np.maximum(a[0], b[0]); hi = np.minimum(a[1], b[1])
    inter = np.prod(np.clip(hi - lo, 0, None))
    va = np.prod(a[1] - a[0]); vb = np.prod(b[1] - b[0])
    return inter / max(va + vb - inter, 1e-9)


def pick_mesh_transform(mesh_v, pts, aam):
    """Do the bins live in the raw or the axis-aligned frame? Compare bboxes."""
    pb = (pts.min(0), pts.max(0))
    raw = (mesh_v.min(0), mesh_v.max(0))
    va = (mesh_v @ aam[:3, :3].T) + aam[:3, 3]
    ali = (va.min(0), va.max(0))
    iou_raw, iou_ali = bbox_iou(pb, raw), bbox_iou(pb, ali)
    print(f'[frame-detect] bbox IoU points-vs-rawmesh={iou_raw:.3f} '
          f'points-vs-alignedmesh={iou_ali:.3f}')
    if iou_ali >= iou_raw:
        print('[frame-detect] bins are AXIS-ALIGNED; transforming mesh by AAM')
        return aam, True
    print('[frame-detect] bins are in the RAW scanner frame; mesh untouched')
    return np.eye(4), False


def capture_camera(pose_dir, transform):
    """Mean camera position + mean horizontal view direction of the capture."""
    centers, fwd = [], []
    for p in glob.glob(os.path.join(pose_dir, '*.txt')):
        if os.path.basename(p).startswith('intrinsic'):
            continue
        m = np.loadtxt(p)
        if m.shape != (4, 4) or not np.isfinite(m).all():
            continue
        m = transform @ m                       # C2W, mapped into render frame
        centers.append(m[:3, 3])
        fwd.append(m[:3, :3] @ np.array([0., 0., 1.]))
    centers = np.stack(centers); fwd = np.stack(fwd)
    f = fwd.mean(0); f[2] = 0.0
    n = np.linalg.norm(f)
    f = f / n if n > 1e-6 else np.array([1., 0., 0.])
    return centers.mean(0), f


def setup_view(renderer, pts, cam_pos, view_dir, fov, height_boost=0.85,
               back_boost=0.85):
    """MUST be called with geometry already in the scene: Open3DScene's
    setup_camera derives the near/far clip planes from the CURRENT scene
    bounding box, so calling it on an empty scene clips everything away
    (first run of this script produced four blank images exactly this way)."""
    lo, hi = pts.min(0), pts.max(0)
    center = (lo + hi) / 2
    diag = float(np.linalg.norm(hi - lo))
    eye = center - view_dir * (back_boost * diag)
    eye[2] = hi[2] + height_boost * diag * 0.55
    # keep the eye on the capture side of the room
    eye[:2] = 0.55 * eye[:2] + 0.45 * cam_pos[:2]
    renderer.setup_camera(fov, center.astype(np.float32),
                          eye.astype(np.float32), np.array([0., 0., 1.], np.float32))
    print(f'[camera] eye={np.round(eye, 2)} lookat={np.round(center, 2)} fov={fov}')
    return eye, center


def render(renderer, name, geom, mat, path, down, cam):
    renderer.scene.clear_geometry()
    renderer.scene.add_geometry(name, geom, mat)
    setup_view(renderer, *cam)                 # after add_geometry, see above
    img = np.asarray(renderer.render_to_image())
    if img[..., :3].std() < 1.0:
        print(f'[WARN] {path} looks uniform (std={img[..., :3].std():.3f}) '
              '-- geometry probably not visible')
    im = Image.fromarray(img)
    if down > 1:
        im = im.resize((im.width // down, im.height // down), Image.LANCZOS)
    im.save(path)
    print(f'[write] {path} {im.size}')


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)

    mesh_path = os.path.join(a.scans_root, a.scene, f'{a.scene}_vh_clean_2.ply')
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    print(f'[mesh] {mesh_path}: {len(mesh.vertices)} verts, '
          f'{len(mesh.triangles)} tris, colors={mesh.has_vertex_colors()}')

    bins = load_bins(os.path.join(a.mv_root, 'points', a.scene))
    acc = np.concatenate([bins[k] for k in sorted(bins)], 0)
    print(f'[points] {len(bins)} frames accumulated -> {acc.shape[0]} points')
    if a.frame not in bins:
        raise KeyError(f'frame {a.frame} has no bin; available e.g. '
                       f'{sorted(bins)[:5]}...')

    aam = np.load(os.path.join(a.mv_root, 'axis_align_matrix', f'{a.scene}.npy'))
    T, _ = pick_mesh_transform(np.asarray(mesh.vertices), acc[:, :3], aam)
    mesh.transform(T)

    cam_pos, view_dir = capture_camera(
        os.path.join(a.mv_root, '2D', a.scene, 'pose'), T)

    renderer = o3d.visualization.rendering.OffscreenRenderer(a.width, a.height)
    renderer.scene.set_background(np.array([1.0, 1.0, 1.0, 1.0], np.float32))
    # camera params applied inside render(), AFTER geometry is in the scene
    cam = (acc[:, :3], cam_pos, view_dir, a.fov, a.height_boost, a.back_boost)
    S = a.suffix

    # ---- mesh, unlit (vertex colours carry the scan's own shading) ----------
    unlit = o3d.visualization.rendering.MaterialRecord()
    unlit.shader = 'defaultUnlit'
    render(renderer, 'mesh_unlit', mesh, unlit,
           os.path.join(a.out, f'scene_mesh{S}.png'), a.down, cam)

    # ---- mesh, lit variant (choose after seeing both) -----------------------
    mesh.compute_vertex_normals()
    lit = o3d.visualization.rendering.MaterialRecord()
    lit.shader = 'defaultLit'
    render(renderer, 'mesh_lit', mesh, lit,
           os.path.join(a.out, f'scene_mesh_lit{S}.png'), a.down, cam)

    # ---- accumulated point cloud, big splats --------------------------------
    def cloud(arr):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(arr[:, :3].astype(np.float64))
        pc.colors = o3d.utility.Vector3dVector((arr[:, 3:6] / 255.0).astype(np.float64))
        return pc

    pmat = o3d.visualization.rendering.MaterialRecord()
    pmat.shader = 'defaultUnlit'
    pmat.point_size = a.point_size * (a.width / 960.0)   # scale with supersampling
    render(renderer, 'points_acc', cloud(acc), pmat,
           os.path.join(a.out, f'scene_points{S}.png'), a.down, cam)

    # ---- one frame's partial cloud, same camera ------------------------------
    pmat2 = o3d.visualization.rendering.MaterialRecord()
    pmat2.shader = 'defaultUnlit'
    pmat2.point_size = a.partial_point_size * (a.width / 960.0)
    render(renderer, 'points_one', cloud(bins[a.frame]), pmat2,
           os.path.join(a.out, f'scene_frame_partial{S}.png'), a.down, cam)

    # ---- optional: the same single frame framed on its own bbox --------------
    if a.zoom_partial:
        camz = (bins[a.frame][:, :3], cam_pos, view_dir, a.fov,
                a.height_boost, a.back_boost)
        render(renderer, 'points_one_zoom', cloud(bins[a.frame]), pmat2,
               os.path.join(a.out, f'scene_frame_partial_zoom{S}.png'),
               a.down, camz)

    print(f'[provenance] scene={a.scene} partial_frame={a.frame} '
          f'mesh={mesh_path} bins={len(bins)}x20000 render={a.width}x{a.height}'
          f'/down{a.down} point_size={pmat.point_size}/{pmat2.point_size} '
          f'height_boost={a.height_boost} back_boost={a.back_boost}')


if __name__ == '__main__':
    main()
