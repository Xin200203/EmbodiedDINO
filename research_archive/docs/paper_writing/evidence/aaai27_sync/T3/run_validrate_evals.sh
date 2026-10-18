#!/usr/bin/env bash
source /home/nebula/miniconda3/etc/profile.d/conda.sh
conda activate ESAM
cd /home/nebula/xxy/AutoSeg3D_v5_transfer_20260706
export OMP_NUM_THREADS=4 PYTHONPATH=$PWD PYTHONUNBUFFERED=1
OUT=/home/nebula/xxy/work_dirs_transfer/validrate_20260727
CKPT=ckpts/epoch_96.pth
for dom in 3rscan scenenn scannet200; do
  cfg=configs/codex_v5/validrate_v5ab1_eval_${dom}_20260727.py
  CUDA_VISIBLE_DEVICES=0 python tools/test.py "$cfg" "$CKPT" \
    --work-dir "$OUT/${dom}_v5ab1_e96" > "$OUT/logs/${dom}.log" 2>&1
  echo "VALIDRATE_${dom}_DONE rc=$?" | tee -a "$OUT/logs/driver.log"
done
touch "$OUT/all.complete"
