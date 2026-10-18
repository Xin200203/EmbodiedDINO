"""Class-agnostic object-only adapter protocol for the host-184 run.

This file is staged into ``configs/codex_owf`` in the remote experiment
workspace, so the relative base path below is resolved there.
"""

_base_ = ['./AutoSeg3D_scannet200_stage1_v5ab1_owf_fullval.py']

# DINO descriptors remain available to the object feedback module, while the
# dense point-level fusion path is disabled. The host is frozen by the base
# adapter config, making object feedback the only learned intervention.
model = dict(
    object_feedback_freeze_base=True,
    backbone=dict(
        config=dict(
            dino_point_fusion_enable=False,
            object_feedback_enable=True)))
