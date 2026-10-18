import os


_base_ = [
    '../codex_v5/AutoSeg3D_scannet200_stage1_v5cls1_dino_cls198.py'
]


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


OWF_JOINT_SEED = int(os.environ.get('OWF_JOINT_SEED', '2050318485'))
OWF_JOINT_MODE = os.environ.get('OWF_JOINT_MODE', 'real')
OWF_JOINT_OUTPUT_SCALE = float(
    os.environ.get('OWF_JOINT_OUTPUT_SCALE', '1.0'))
OWF_JOINT_MAX_EPOCHS = int(
    os.environ.get('OWF_JOINT_MAX_EPOCHS', '128'))
OWF_JOINT_VAL_INTERVAL = int(
    os.environ.get('OWF_JOINT_VAL_INTERVAL', '16'))
OWF_JOINT_BATCH_SIZE = int(
    os.environ.get('OWF_JOINT_BATCH_SIZE', '2'))
OWF_JOINT_ACCUMULATIVE_COUNTS = int(
    os.environ.get('OWF_JOINT_ACCUMULATIVE_COUNTS', '2'))
OWF_JOINT_DEBUG = _env_bool('OWF_JOINT_DEBUG', False)

if OWF_JOINT_MODE not in ('real', 'shuffled', 'stats-only'):
    raise ValueError(
        f'OWF_JOINT_MODE must be real/shuffled/stats-only, got '
        f'{OWF_JOINT_MODE}')
if OWF_JOINT_OUTPUT_SCALE < 0.0:
    raise ValueError('OWF_JOINT_OUTPUT_SCALE must be non-negative')
if OWF_JOINT_BATCH_SIZE < 1:
    raise ValueError('OWF_JOINT_BATCH_SIZE must be positive')
if OWF_JOINT_ACCUMULATIVE_COUNTS < 1:
    raise ValueError('OWF_JOINT_ACCUMULATIVE_COUNTS must be positive')

# Both matched arms instantiate the same modules. The disabled arm sets the
# output scale to zero, preserving model construction and shared initialization.
model = dict(
    object_feedback_freeze_base=False,
    object_feedback_debug=OWF_JOINT_DEBUG,
    backbone=dict(
        config=dict(
            object_feedback_enable=True,
            object_feedback_mode=OWF_JOINT_MODE,
            object_feedback_placement='pre_memory',
            object_feedback_include_dino=True,
            object_feedback_include_s2=False,
            object_feedback_include_xyz=False,
            object_feedback_oracle_purity=False,
            object_feedback_persistent_oracle=False,
            object_feedback_output_scale=OWF_JOINT_OUTPUT_SCALE,
            object_feedback_hidden_dim=64,
            object_feedback_init_gate=0.05,
            object_feedback_min_hit=0.90)))

randomness = dict(seed=OWF_JOINT_SEED, deterministic=False)
train_dataloader = dict(batch_size=OWF_JOINT_BATCH_SIZE)
optim_wrapper = dict(accumulative_counts=OWF_JOINT_ACCUMULATIVE_COUNTS)
work_dir = os.environ.get(
    'OWF_JOINT_WORK_DIR',
    f'/home/nebula/xxy/work_dirs/object_world_joint_cls198_20260722/'
    f'seed{OWF_JOINT_SEED}_scale{OWF_JOINT_OUTPUT_SCALE:g}')

# Keep the original 128-epoch LR schedule even for the 16-epoch pilot so the
# pilot checkpoint can resume without changing the scheduler definition.
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=OWF_JOINT_MAX_EPOCHS,
    val_interval=OWF_JOINT_VAL_INTERVAL)
