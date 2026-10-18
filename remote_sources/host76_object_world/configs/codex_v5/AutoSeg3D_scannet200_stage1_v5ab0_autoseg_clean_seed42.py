# Codex clean AutoSeg3D stage1 baseline under the same clean workspace/protocol.
# No v5 DINO injection, no dual-path query, no offline DINO pipeline.
_base_ = '../scannet200/AutoSeg3D_scannet200_stage1.py'

load_from = '/datadisk1/xxy/3D_Reconstruction_v5_184sync_20260603/work_dirs/source_ckpts/184_ESAM_sv_dino_best_all_ap50_epoch115.pth'
randomness = dict(seed=42, deterministic=False)
work_dir = '/datadisk3/xxy/work_dirs/autoseg3d_v5_stage1_clean_20260619/v5ab0_autoseg_clean_stage1_seed42'
