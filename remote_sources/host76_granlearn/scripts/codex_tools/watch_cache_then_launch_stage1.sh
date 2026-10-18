#!/usr/bin/env bash
set -eo pipefail
cd /datadisk1/xxy/AutoSeg3D_v5_stage1_clean_20260604
source /datadisk1/xxy/miniconda3/etc/profile.d/conda.sh
conda activate ESAM
export OMP_NUM_THREADS=4
export PYTHONPATH=$PWD:${PYTHONPATH:-}
CACHE_ROOT=/datadisk3/xxy/dino_cache/autoseg3d_v5_stage1_clean_20260604/scannet200_mv_fast/dinov2_vitl14_reg4_420x560_fp16
TARGET=108662
STAMP=20260619
mkdir -p logs metrics
log(){ echo "[$(date -u '+%F %T UTC')] $*"; }
log "watcher started; waiting for DATA2 full DINO cache target=$TARGET"
while true; do
  count=$(find "$CACHE_ROOT" -type f -name '*.pt' | wc -l)
  summaries=0
  failed=0
  for gpu in 0 1 2 3 4 5 6; do
    sum="metrics/data2_full_dino_cache_summary_20260619_retry_compactfix_shard${gpu}_of7.json"
    if [ -f "$sum" ]; then
      summaries=$((summaries+1))
      if grep -q '"num_failed": 0' "$sum"; then :; else failed=1; fi
    fi
  done
  log "cache_count=$count summaries=$summaries failed_flag=$failed"
  if [ "$count" -ge "$TARGET" ] && [ "$summaries" -eq 7 ] && [ "$failed" -eq 0 ]; then
    break
  fi
  sleep 300
done
log "DATA2 complete; aggregating summaries and starting GPU loss smokes"
python - <<'PYAGG' > metrics/data2_full_dino_cache_aggregate_20260619.json
import json, pathlib
root=pathlib.Path('metrics')
sums=[]
for i in range(7):
 p=root/f'data2_full_dino_cache_summary_20260619_retry_compactfix_shard{i}_of7.json'
 sums.append(json.loads(p.read_text()))
out={'num_shards':7,'total_done':sum(x.get('num_done',0) for x in sums),'total_skipped':sum(x.get('num_skipped',0) for x in sums),'total_failed':sum(x.get('num_failed',0) for x in sums),'shards':sums}
print(json.dumps(out, indent=2, ensure_ascii=False))
PYAGG
run_job(){
  local tag="$1"; local gpu="$2"; local cfg="$3"
  local smoke_log="logs/smoke_${tag}_loss_gpu${gpu}_20260619.log"
  local smoke_json="metrics/smoke_${tag}_loss_gpu${gpu}_20260619.json"
  local train_log="logs/train_${tag}_stage1_gpu${gpu}_20260619.log"
  local pid_file="logs/train_${tag}_stage1_gpu${gpu}.pid"
  if [ -f "$pid_file" ] && ps -p "$(cat "$pid_file")" >/dev/null 2>&1; then
    log "$tag already running pid=$(cat "$pid_file")"
    return 0
  fi
  log "running $tag GPU loss smoke on GPU$gpu cfg=$cfg"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/codex_tools/smoke_v5ab4_one_batch.py \
    --config "$cfg" \
    --device cuda:0 \
    --max-tries 20 \
    --summary "$smoke_json" \
    > "$smoke_log" 2>&1
  log "$tag smoke passed; launching training on GPU$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python tools/train.py "$cfg" \
    > "$train_log" 2>&1 &
  echo $! > "$pid_file"
  sleep 5
  ps -p "$(cat "$pid_file")" -o pid,etime,stat,args || true
}
# Main v5 and core v5-only ablations. These all keep AutoSeg3D stage1 flags/seed/data.
run_job v5ab4 0 configs/codex_v5/AutoSeg3D_scannet200_stage1_v5ab4_offline_dino_dual.py
run_job v5ab1 1 configs/codex_v5/AutoSeg3D_scannet200_stage1_v5ab1_offline_dino_injection_only.py
run_job v5ab5 2 configs/codex_v5/AutoSeg3D_scannet200_stage1_v5ab5_offline_dino_dual_no_detach.py
run_job v5ab6 3 configs/codex_v5/AutoSeg3D_scannet200_stage1_v5ab6_offline_dino_dual_no_mergefusion.py
log "watcher finished launch phase"
