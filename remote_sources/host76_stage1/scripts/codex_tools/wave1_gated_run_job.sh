#!/usr/bin/env bash
set -eo pipefail
cd /datadisk1/xxy/AutoSeg3D_v5_stage1_clean_20260604
source /datadisk1/xxy/miniconda3/etc/profile.d/conda.sh
conda activate ESAM
export OMP_NUM_THREADS=4
export PYTHONPATH=$PWD:${PYTHONPATH:-}
tag=$1; gpu=$2; cfg=$3; waitpid=$4; gatefile=$5
log(){ echo "[$(date -u "+%F %T UTC")] $*"; }
log "gated runner $tag: waitpid=$waitpid gate=$gatefile"
while ps -p "$waitpid" >/dev/null 2>&1; do sleep 120; done
log "lane free (pid $waitpid exited)"
while [ ! -f "$gatefile" ]; do sleep 120; done
log "gate open ($gatefile)"
smoke_json="metrics/smoke_${tag}_gpu${gpu}.json"
CUDA_VISIBLE_DEVICES="$gpu" python scripts/codex_tools/smoke_v5ab4_one_batch.py --config "$cfg" --device cuda:0 --max-tries 20 --summary "$smoke_json" > "logs/smoke_${tag}_gpu${gpu}.log" 2>&1
log "$tag smoke passed; launching"
CUDA_VISIBLE_DEVICES="$gpu" nohup python tools/train.py "$cfg" > "logs/train_${tag}_gpu${gpu}.log" 2>&1 &
echo $! > "logs/train_${tag}_gpu${gpu}.pid"
sleep 5; ps -p "$(cat logs/train_${tag}_gpu${gpu}.pid)" -o pid,etime,args | tail -1
log "$tag launched"
