#!/usr/bin/env bash
# T1 (AAAI-27): representation diagnostics on the FULL 312-scene ScanNet200 validation split.
# Two arms only: host (v5ab0) and embodieddino (v5ab1), per task scope.
# Protocol identical to the archived subset50 run (artifacts/host76_representation_subset50):
# same seed, stride-4 frame sampling, <=3 GT-bearing dumps/scene; only ann_file and the
# scene/global-dump caps are scaled: 50->312 scenes, 150->936 max dumps per variant.
# Launched 2026-07-27 on host 76 (dell), GPUs 2,3.
set -euo pipefail
export OWF_REPO_ROOT=/datadisk3/xxy/AutoSeg3D_v5_object_world_20260719
export OWF_PYTHON=/datadisk1/xxy/miniconda3/envs/ESAM/bin/python
export OWF_DINO_CACHE=/datadisk3/xxy/dino_cache/autoseg3d_v5_stage1_clean_20260604/scannet200_mv_fast/dinov2_vitl14_reg4_420x560_fp16
export OWF_HOST_CHECKPOINT=/datadisk3/xxy/work_dirs/autoseg3d_v5_stage1_clean_20260619/v5ab0_autoseg_clean_stage1_seed2050318485/epoch_96.pth
export OWF_OURS_CHECKPOINT=/datadisk3/xxy/work_dirs/autoseg3d_v5_stage1_clean_20260619/v5ab1_offline_dino_injection_only_stage1_seed2050318485/epoch_96.pth
export OWF_OUTPUT_ROOT=/datadisk3/xxy/work_dirs/object_world_20260727/representation_diagnostics_val312
export OWF_ANN_FILE=scannet200_mv_oneformer3d_infos_val.pkl
export OWF_GPU_LIST=2,3
export OWF_SEED=2050318485
export OWF_REPR_MAX_SCENES=312
export OWF_REPR_MAX_FRAMES=1000
export OWF_REPR_FRAME_STRIDE=4
export OWF_REPR_MAX_DUMPS_PER_VARIANT=936
export OWF_REPR_MAX_DUMPS_PER_SCENE=3
export OWF_REPR_REQUIRE_VALID=1
echo "[launch] ann pkl sha256:"
sha256sum /datadisk1/xxy/data/data-ESAM/scannet200-mv_fast/runs_mv/fastsam_everything_stride20_img640/scannet200_mv_oneformer3d_infos_val.pkl
exec bash /datadisk3/xxy/work_dirs/object_world_20260727/representation_diagnostics_val312/run_representation_diagnostics_2arm.sh
