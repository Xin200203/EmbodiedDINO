#!/usr/bin/env bash
set -eo pipefail
cd /datadisk1/xxy/AutoSeg3D_v5_stage1_clean_20260604
source /datadisk1/xxy/miniconda3/etc/profile.d/conda.sh
conda activate ESAM
export OMP_NUM_THREADS=4
export PYTHONPATH=$PWD:${PYTHONPATH:-}
tag=$1; gpu=$2; cfg=$3; waitpid=${4:-}
STAMP=20260705
log(){ echo "[$(date -u "+%F %T UTC")] $*"; }
if [ -n "$waitpid" ]; then
  log "waiting for pid $waitpid to exit before starting $tag"
  while ps -p "$waitpid" >/dev/null 2>&1; do sleep 60; done
  log "pid $waitpid exited"
fi
smoke_log="logs/smoke_${tag}_gpu${gpu}_${STAMP}.log"
smoke_json="metrics/smoke_${tag}_gpu${gpu}_${STAMP}.json"
train_log="logs/train_${tag}_gpu${gpu}_${STAMP}.log"
pid_file="logs/train_${tag}_gpu${gpu}.pid"
if [ -f "$pid_file" ] && ps -p "$(cat "$pid_file")" >/dev/null 2>&1; then
  log "$tag already running pid=$(cat "$pid_file")"; exit 0
fi
log "running $tag smoke on GPU$gpu cfg=$cfg"
CUDA_VISIBLE_DEVICES="$gpu" python scripts/codex_tools/smoke_v5ab4_one_batch.py --config "$cfg" --device cuda:0 --max-tries 20 --summary "$smoke_json" > "$smoke_log" 2>&1
log "$tag smoke passed; launching training on GPU$gpu"
CUDA_VISIBLE_DEVICES="$gpu" nohup python tools/train.py "$cfg" > "$train_log" 2>&1 &
echo $! > "$pid_file"
sleep 5
ps -p "$(cat "$pid_file")" -o pid,etime,stat,args | tail -1 || true
log "$tag launched"
