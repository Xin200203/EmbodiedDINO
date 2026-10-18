#!/usr/bin/env bash
source /home/nebula/miniconda3/etc/profile.d/conda.sh
conda activate ESAM
cd /home/nebula/xxy/AutoSeg3D_v5_transfer_20260706
export OMP_NUM_THREADS=4 PYTHONPATH=$PWD PYTHONUNBUFFERED=1
OUT=/home/nebula/xxy/work_dirs_transfer/validrate_20260727
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/codex_v5/validrate_v5ab1_eval_scannet200_20260727.py ckpts/epoch_96.pth \
  --work-dir "$OUT/scannet200_v5ab1_e96" > "$OUT/logs/scannet200.log" 2>&1
echo "VALIDRATE_scannet200_retry_DONE rc=$?" | tee -a "$OUT/logs/driver.log"
touch "$OUT/all.complete"
