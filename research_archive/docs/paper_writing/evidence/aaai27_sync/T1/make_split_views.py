#!/usr/bin/env python3
"""Build scene-filtered dump views (complement262 / subset50) from the val312 dumps.

Dump files are named like r0_0000_scene0488_00_f0000.pt; filter by scene id via
symlinks so the analyzer can be rerun on a subset without re-evaluating.
"""
import os, pickle, re, sys

ROOT = "/datadisk3/xxy/work_dirs/object_world_20260727/representation_diagnostics_val312"
D = "/datadisk1/xxy/data/data-ESAM/scannet200-mv_fast/runs_mv/fastsam_everything_stride20_img640/"

def scene_ids(pkl):
    with open(D + pkl, "rb") as f:
        payload = pickle.load(f)
    dl = payload["data_list"] if isinstance(payload, dict) else payload
    return {it["point_cloud"]["lidar_idx"] for it in dl}

SPLITS = {
    "complement262": scene_ids("scannet200_mv_oneformer3d_infos_val_complement262_ocwe20260720.pkl"),
    "subset50": scene_ids("scannet200_mv_oneformer3d_infos_val_subset50.pkl"),
}
PAT = re.compile(r"(scene\d{4}_\d{2})")

for split, keep in SPLITS.items():
    for variant in ("host", "embodieddino"):
        src = os.path.join(ROOT, "dumps", variant)
        dst = os.path.join(ROOT, "dumps_" + split, variant)
        os.makedirs(dst, exist_ok=True)
        n = 0
        covered = set()
        for fn in sorted(os.listdir(src)):
            m = PAT.search(fn)
            if not m or m.group(1) not in keep:
                continue
            link = os.path.join(dst, fn)
            if not os.path.lexists(link):
                os.symlink(os.path.join(src, fn), link)
            covered.add(m.group(1))
            n += 1
        print(f"{split}/{variant}: {n} dumps, {len(covered)}/{len(keep)} scenes covered")
