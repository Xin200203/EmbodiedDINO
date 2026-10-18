#!/usr/bin/env python3
"""Wave 0: create v5ab7 (no aug-reverse projection) and v5ab8 (zero-init fuse) configs,
then static-build both models plus v5ab1 as a default-behavior regression check."""
import pathlib

root = pathlib.Path('/datadisk1/xxy/AutoSeg3D_v5_stage1_clean_20260604')
src = root / 'configs/codex_v5/AutoSeg3D_scannet200_stage1_v5ab1_offline_dino_injection_only.py'
base = src.read_text()

wd_old = "work_dir = '/datadisk3/xxy/work_dirs/autoseg3d_v5_stage1_clean_20260619/v5ab1_offline_dino_injection_only_stage1_seed2050318485'"
assert base.count(wd_old) == 1

# ---------- v5ab7 ----------
old_mu = """model.update(dict(
    dino_enable=True,
    dino_require=True,
    dino_debug=False,
    dino_pose_mode='auto',
    dino_valid_rate_warn_thr=0.90))"""
new_mu = """model.update(dict(
    dino_enable=True,
    dino_require=False,
    dino_debug=False,
    dino_pose_mode='auto',
    dino_valid_rate_warn_thr=0.90,
    dino_skip_aug_reverse=True))"""
assert base.count(old_mu) == 1, 'model.update anchor missing'
t7 = base.replace(old_mu, new_mu)
t7 = t7.replace(wd_old, "work_dir = '/datadisk3/xxy/work_dirs/autoseg3d_v5_stage1_clean_20260619/v5ab7_pv_no_aug_reverse_stage1_seed2050318485'")
p7 = root / 'configs/codex_v5/AutoSeg3D_scannet200_stage1_v5ab7_pv_no_aug_reverse.py'
p7.write_text(t7)
print('written:', p7.name)

# ---------- v5ab8 ----------
old_bc = """model['backbone']['config'].update(dict(
    dino_strict=True,
    dino_min_hit_ratio=0.90,
    dino_residual=True))"""
new_bc = """model['backbone']['config'].update(dict(
    dino_strict=True,
    dino_min_hit_ratio=0.90,
    dino_residual=True,
    dino_fuse_init='zero'))"""
assert base.count(old_bc) == 1, 'backbone config anchor missing'
t8 = base.replace(old_bc, new_bc)
t8 = t8.replace(wd_old, "work_dir = '/datadisk3/xxy/work_dirs/autoseg3d_v5_stage1_clean_20260619/v5ab8_pv_zeroinit_fuse_stage1_seed2050318485'")
p8 = root / 'configs/codex_v5/AutoSeg3D_scannet200_stage1_v5ab8_pv_zeroinit_fuse.py'
p8.write_text(t8)
print('written:', p8.name)

# ---------- static build checks ----------
import sys
sys.path.insert(0, str(root))
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet3d.registry import MODELS
import oneformer3d  # noqa: F401
init_default_scope('mmdet3d')


def fuse_kernel_absmax(model):
    vals = []
    for name in ('dino_fuse_8', 'dino_fuse_4', 'dino_fuse_2', 'dino_fuse_1'):
        m = getattr(model.backbone, name, None)
        if m is not None and hasattr(m[0], 'kernel') and m[0].kernel is not None:
            vals.append(float(m[0].kernel.data.abs().max()))
    return vals


# v5ab7: flag must be set; init must stay default (nonzero kernels)
cfg7 = Config.fromfile(str(p7))
m7 = MODELS.build(cfg7.model)
assert getattr(m7, 'dino_skip_aug_reverse', None) is True, 'ab7 flag not set'
v7 = fuse_kernel_absmax(m7)
assert v7 and all(v > 0 for v in v7), f'ab7 kernels unexpectedly zero: {v7}'
print('ab7 OK: dino_skip_aug_reverse=True, fuse kernel absmax:', ['%.2e' % v for v in v7])
del m7

# v5ab8: kernels must be exactly zero; flag must stay default False
cfg8 = Config.fromfile(str(p8))
m8 = MODELS.build(cfg8.model)
assert getattr(m8, 'dino_skip_aug_reverse', None) is False, 'ab8 skip flag should be False'
assert getattr(m8.backbone, '_dino_fuse_init', None) == 'zero', 'ab8 fuse init flag not set'
v8 = fuse_kernel_absmax(m8)
assert v8 and all(v == 0.0 for v in v8), f'ab8 kernels not zero: {v8}'
print('ab8 OK: _dino_fuse_init=zero, fuse kernel absmax:', v8)
del m8

# v5ab1 regression: defaults unchanged (nonzero small-normal init, no skip flag)
cfg1 = Config.fromfile(str(src))
m1 = MODELS.build(cfg1.model)
assert getattr(m1, 'dino_skip_aug_reverse', None) is False, 'ab1 default flag changed!'
assert getattr(m1.backbone, '_dino_fuse_init', None) == 'normal', 'ab1 default init changed!'
v1 = fuse_kernel_absmax(m1)
assert v1 and all(v > 0 for v in v1), f'ab1 kernels unexpectedly zero: {v1}'
print('ab1 regression OK: defaults preserved, fuse kernel absmax:', ['%.2e' % v for v in v1])
print('ALL STATIC CHECKS PASSED')
