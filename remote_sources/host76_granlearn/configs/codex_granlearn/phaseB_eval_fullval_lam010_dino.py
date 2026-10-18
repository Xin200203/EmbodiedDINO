_base_ = './phaseB_pairloss_subset50_ft4ep_lam010_dino.py'
# Full ScanNet200-MV val audit for plain DINO pair-loss variant. No training.
val_dataloader = dict(num_workers=0, persistent_workers=False, dataset=dict(ann_file='scannet200_mv_oneformer3d_infos_val.pkl'))
test_dataloader = dict(num_workers=0, persistent_workers=False, dataset=dict(ann_file='scannet200_mv_oneformer3d_infos_val.pkl'))
model = dict(granlearn_debug=False, dino_debug=False, dino_valid_rate_warn_thr=0.90, seedcalib_enable=False)
work_dir = '/datadisk4/xxy/work_dirs/granlearn_20260627/phaseB_eval_fullval_lam010_dino_r1'
