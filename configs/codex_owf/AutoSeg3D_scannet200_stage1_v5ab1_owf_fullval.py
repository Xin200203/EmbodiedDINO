_base_ = ['./AutoSeg3D_scannet200_stage1_v5ab1_owf_subset50.py']

val_dataloader = dict(
    num_workers=6,
    persistent_workers=True,
    dataset=dict(
        ann_file='scannet200_mv_oneformer3d_infos_val.pkl'))
test_dataloader = val_dataloader
