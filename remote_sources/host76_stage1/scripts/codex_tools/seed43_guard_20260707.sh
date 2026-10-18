#!/usr/bin/env bash
cd /datadisk1/xxy/AutoSeg3D_v5_stage1_clean_20260604
source /datadisk1/xxy/miniconda3/etc/profile.d/conda.sh
conda activate ESAM
export OMP_NUM_THREADS=4 PYTHONPATH=$PWD
CFG=configs/codex_v5/AutoSeg3D_scannet200_stage1_v5ab1_offline_dino_injection_only_seed43.py
WD=/datadisk3/xxy/work_dirs/autoseg3d_v5_stage1_clean_20260619/v5ab1_offline_dino_injection_only_stage1_seed43
log(){ echo "[$(date -u +%F\ %T) guard43] $*"; }
for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24; do
  if [ -f "$WD/epoch_128.pth" ]; then log "complete"; break; fi
  log "attempt $attempt (resume=auto)"
  CUDA_VISIBLE_DEVICES=3 python tools/train.py "$CFG" --resume > "logs/train_v5ab1_seed43_gpu3_a${attempt}.log" 2>&1
  rc=$?
  log "attempt $attempt rc=$rc"
  [ "$rc" -eq 0 ] && break
  sleep 90
done
log "guard43 finished"
