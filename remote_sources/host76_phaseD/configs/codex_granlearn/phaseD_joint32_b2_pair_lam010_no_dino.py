_base_ = './phaseC_fulltrain_ft4ep_lam010_no_dino.py'
load_from = '/datadisk1/xxy/3D_Reconstruction_v5_184sync_20260603/work_dirs/source_ckpts/184_ESAM_sv_dino_best_all_ap50_epoch115.pth'
randomness = dict(seed=2050318485, deterministic=False)
train_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True, dataset=dict(ann_file='scannet200_mv_oneformer3d_infos_train.pkl'))
val_dataloader = dict(num_workers=0, persistent_workers=False, dataset=dict(ann_file='scannet200_mv_oneformer3d_infos_val.pkl'))
test_dataloader = val_dataloader
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=32, val_interval=8)
param_scheduler = dict(type='PolyLR', begin=0, end=32, power=0.9)
optim_wrapper = dict(optimizer=dict(lr=1e-4))
default_hooks = dict(logger=dict(type='LoggerHook', interval=50), checkpoint=dict(type='CheckpointHook', interval=8, max_keep_ckpts=4, save_best=['all_ap'], rule='greater'))
model = dict(granlearn_enable=True, granlearn_loss_weight=0.10, granlearn_max_pairs=256, granlearn_use_dino=False, granlearn_dino_weighted=False, granlearn_debug=False, seedcalib_enable=False, dino_debug=False, dino_valid_rate_warn_thr=0.90)
work_dir = '/datadisk4/xxy/work_dirs/granlearn_20260704/phaseD_joint32_b2_pair_lam010_no_dino'
