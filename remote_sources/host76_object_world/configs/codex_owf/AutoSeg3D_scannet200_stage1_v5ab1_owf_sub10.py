import os

_base_ = [
    '../codex_lgqc/AutoSeg3D_scannet200_stage1_v5ab1_lgqc_adapter_sub10_smoke.py'
]


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


OWF_MODE = os.environ.get('OWF_MODE', 'real')
OWF_PLACEMENT = os.environ.get('OWF_PLACEMENT', 'pre_memory')
if OWF_PLACEMENT not in ('pre_memory', 'post_memory'):
    raise ValueError(
        f'OWF_PLACEMENT must be pre_memory or post_memory, got {OWF_PLACEMENT}')
OWF_INCLUDE_DINO = _env_bool('OWF_INCLUDE_DINO', True)
OWF_INCLUDE_S2 = _env_bool('OWF_INCLUDE_S2', True)
OWF_INCLUDE_XYZ = _env_bool('OWF_INCLUDE_XYZ', True)
OWF_ORACLE_PURITY = _env_bool('OWF_ORACLE_PURITY', False)
OWF_ORACLE_PURITY_THRESHOLD = float(
    os.environ.get('OWF_ORACLE_PURITY_THRESHOLD', '0.8'))
OWF_PERSISTENT_ORACLE = _env_bool('OWF_PERSISTENT_ORACLE', False)
OWF_MEMORY_MOMENTUM = float(os.environ.get('OWF_MEMORY_MOMENTUM', '0.8'))
OWF_OUTPUT_SCALE = float(os.environ.get('OWF_OUTPUT_SCALE', '1.0'))
OWF_INIT_GATE = float(os.environ.get('OWF_INIT_GATE', '0.001'))
OWF_MAX_EPOCHS = int(os.environ.get('OWF_MAX_EPOCHS', '2'))
OWF_LR = float(os.environ.get('OWF_LR', '0.0001'))
DINO_CACHE_ROOT = os.environ.get(
    'OWF_DINO_CACHE',
    '/datadisk3/xxy/dino_cache/autoseg3d_v5_stage1_clean_20260604/'
    'scannet200_mv_fast/dinov2_vitl14_reg4_420x560_fp16')

model = dict(
    gci_enable=False,
    gci_learnable_residual=False,
    gci_freeze_base=False,
    gci_debug=False,
    object_feedback_freeze_base=True,
    object_feedback_debug=True,
    backbone=dict(
        config=dict(
            object_feedback_enable=True,
            object_feedback_mode=OWF_MODE,
            object_feedback_placement=OWF_PLACEMENT,
            object_feedback_include_dino=OWF_INCLUDE_DINO,
            object_feedback_include_s2=OWF_INCLUDE_S2,
            object_feedback_include_xyz=OWF_INCLUDE_XYZ,
            object_feedback_oracle_purity=OWF_ORACLE_PURITY,
            object_feedback_oracle_purity_threshold=OWF_ORACLE_PURITY_THRESHOLD,
            object_feedback_persistent_oracle=OWF_PERSISTENT_ORACLE,
            object_feedback_memory_momentum=OWF_MEMORY_MOMENTUM,
            object_feedback_output_scale=OWF_OUTPUT_SCALE,
            object_feedback_hidden_dim=64,
            object_feedback_init_gate=OWF_INIT_GATE,
            object_feedback_min_hit=0.90)))

load_from = os.environ.get(
    'OWF_CHECKPOINT',
    '/datadisk3/xxy/work_dirs/autoseg3d_v5_stage1_clean_20260619/'
    'v5ab1_offline_dino_injection_only_stage1_seed2050318485/epoch_96.pth')
work_dir = os.environ.get(
    'OWF_WORK_DIR',
    f'/datadisk3/xxy/work_dirs/object_world_20260719/sub10_{OWF_MODE}')

default_hooks = dict(
    logger=dict(interval=1),
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=4,
        save_best=['all_ap', 'all_ap_50%'],
        rule='greater'))
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=OWF_MAX_EPOCHS,
    val_interval=1)
param_scheduler = dict(
    type='PolyLR',
    begin=0,
    end=OWF_MAX_EPOCHS,
    power=0.9)
optim_wrapper = dict(optimizer=dict(lr=OWF_LR))
