import os

_base_ = ['./AutoSeg3D_scannet200_stage1_v5ab1_owf_sub10.py']

OWF_MODE = os.environ.get('OWF_MODE', 'real')
OWF_MAX_EPOCHS = int(os.environ.get('OWF_MAX_EPOCHS', '8'))
OWF_VAL_INTERVAL = int(os.environ.get('OWF_VAL_INTERVAL', '4'))

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        ann_file='scannet200_mv_oneformer3d_infos_train_subset50_lgqc20260625.pkl'))
val_dataloader = dict(
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        ann_file='scannet200_mv_oneformer3d_infos_val_subset50.pkl'))
test_dataloader = val_dataloader

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=OWF_MAX_EPOCHS,
    val_interval=OWF_VAL_INTERVAL)
param_scheduler = dict(
    type='PolyLR',
    begin=0,
    end=OWF_MAX_EPOCHS,
    power=0.9)
work_dir = os.environ.get(
    'OWF_WORK_DIR',
    f'/datadisk3/xxy/work_dirs/object_world_20260719/subset50_{OWF_MODE}')
