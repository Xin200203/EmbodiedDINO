_base_ = './phaseB_control_subset50_ft4ep.py'
# Full-val no-finetune base checkpoint control: v5 DINO-injection stage1 epoch_96.
val_dataloader = dict(num_workers=0, persistent_workers=False, dataset=dict(ann_file='scannet200_mv_oneformer3d_infos_val.pkl'))
test_dataloader = dict(num_workers=0, persistent_workers=False, dataset=dict(ann_file='scannet200_mv_oneformer3d_infos_val.pkl'))
model = dict(dino_debug=False, dino_valid_rate_warn_thr=0.90, seedcalib_enable=False)
work_dir = '/datadisk4/xxy/work_dirs/granlearn_20260627/phaseB_eval_fullval_base_epoch96_r1'
