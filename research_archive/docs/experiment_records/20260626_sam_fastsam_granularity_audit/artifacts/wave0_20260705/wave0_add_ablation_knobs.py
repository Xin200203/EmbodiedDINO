#!/usr/bin/env python3
"""Wave 0 (2026-07-05): add two config-gated ablation knobs.

1. mixformer3d.py: model-level `dino_skip_aug_reverse` (default False)
   -> v5ab7 "unverified projection" ablation (skip 3D-augmentation reversal).
2. mink_unet.py: backbone-config `dino_fuse_init` ('normal' default | 'zero')
   -> v5ab8 "zero-init deadlock" ablation.

Both defaults preserve current behavior exactly. Exact-string replace, assert on miss.
"""
import pathlib
import sys

root = pathlib.Path('/datadisk1/xxy/AutoSeg3D_v5_stage1_clean_20260604')

# ---------- Edit 1: mixformer3d.py ----------
p1 = root / 'oneformer3d/mixformer3d.py'
t = p1.read_text()

old_param = """                 dino_valid_rate_warn_thr=0.90,
                 use_query_merge_fusion=True"""
new_param = """                 dino_valid_rate_warn_thr=0.90,
                 dino_skip_aug_reverse=False,
                 use_query_merge_fusion=True"""
assert t.count(old_param) == 1, f'param anchor count={t.count(old_param)}'
t = t.replace(old_param, new_param)

old_assign = """        self.dino_valid_rate_warn_thr = float(dino_valid_rate_warn_thr)"""
new_assign = """        self.dino_valid_rate_warn_thr = float(dino_valid_rate_warn_thr)
        self.dino_skip_aug_reverse = bool(dino_skip_aug_reverse)"""
assert t.count(old_assign) == 1, f'assign anchor count={t.count(old_assign)}'
t = t.replace(old_assign, new_assign)

old_proj = """            try:
                img_meta = getattr(batch_data_samples[b_idx], 'img_metas', {})
                xyz_world_np = apply_3d_transformation(xyz_train.clone(), coord_type='DEPTH', img_meta=img_meta, reverse=True)
                xyz_world = torch.as_tensor(xyz_world_np, device=device, dtype=xyz_train.dtype)
            except Exception:
                xyz_world = xyz_train"""
new_proj = """            if getattr(self, 'dino_skip_aug_reverse', False):
                # ablation v5ab7: unverified projection, no 3D-augmentation reversal
                xyz_world = xyz_train
            else:
                try:
                    img_meta = getattr(batch_data_samples[b_idx], 'img_metas', {})
                    xyz_world_np = apply_3d_transformation(xyz_train.clone(), coord_type='DEPTH', img_meta=img_meta, reverse=True)
                    xyz_world = torch.as_tensor(xyz_world_np, device=device, dtype=xyz_train.dtype)
                except Exception:
                    xyz_world = xyz_train"""
assert t.count(old_proj) == 1, f'proj anchor count={t.count(old_proj)}'
t = t.replace(old_proj, new_proj)
p1.write_text(t)
print('mixformer3d.py edited')

# ---------- Edit 2: mink_unet.py ----------
p2 = root / 'oneformer3d/mink_unet.py'
t = p2.read_text()

old_cfg = """            self._dino_residual = bool(getattr(config, 'dino_residual', True))"""
new_cfg = """            self._dino_residual = bool(getattr(config, 'dino_residual', True))
            self._dino_fuse_init = str(getattr(config, 'dino_fuse_init', 'normal'))"""
assert t.count(old_cfg) == 1, f'cfg anchor count={t.count(old_cfg)}'
t = t.replace(old_cfg, new_cfg)

old_init = """                        if hasattr(conv1x1, 'kernel') and conv1x1.kernel is not None:
                            conv1x1.kernel.data.normal_(mean=0.0, std=1e-4)"""
new_init = """                        if hasattr(conv1x1, 'kernel') and conv1x1.kernel is not None:
                            if getattr(self, '_dino_fuse_init', 'normal') == 'zero':
                                # ablation v5ab8: zero-init reproduces the sparse-conv gradient deadlock
                                conv1x1.kernel.data.zero_()
                            else:
                                conv1x1.kernel.data.normal_(mean=0.0, std=1e-4)"""
assert t.count(old_init) == 1, f'init anchor count={t.count(old_init)}'
t = t.replace(old_init, new_init)
p2.write_text(t)
print('mink_unet.py edited')

# ---------- compile check ----------
import py_compile
py_compile.compile(str(p1), doraise=True)
py_compile.compile(str(p2), doraise=True)
print('py_compile OK for both files')
