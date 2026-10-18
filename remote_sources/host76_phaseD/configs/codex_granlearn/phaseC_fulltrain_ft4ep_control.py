_base_ = './phaseB_control_subset50_ft4ep.py'
# Phase C: full-train short fine-tune control from V5 epoch96.
train_dataloader = dict(
    num_workers=4,
    persistent_workers=True,
    dataset=dict(ann_file='scannet200_mv_oneformer3d_infos_train.pkl'))
val_dataloader = dict(
    num_workers=0,
    persistent_workers=False,
    dataset=dict(ann_file='scannet200_mv_oneformer3d_infos_val.pkl'))
test_dataloader = val_dataloader
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=4, val_interval=1)
param_scheduler = dict(type='PolyLR', begin=0, end=4, power=0.9)
optim_wrapper = dict(optimizer=dict(lr=5e-5))
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=4, save_best=['all_ap'], rule='greater'))
model = dict(seedcalib_enable=False, dino_debug=False, dino_valid_rate_warn_thr=0.90)
work_dir = '/datadisk4/xxy/work_dirs/granlearn_20260627/phaseC_fulltrain_ft4ep_control'
