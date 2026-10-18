#!/usr/bin/env bash

set -uo pipefail

HOST="${OWF_CA_OBJECTONLY_HOST:-184}"
INTERVAL_SECONDS="${OWF_CA_OBJECTONLY_MONITOR_INTERVAL_SECONDS:-120}"
RETRY_SECONDS="${OWF_CA_OBJECTONLY_MONITOR_RETRY_SECONDS:-30}"
LOG_FILE="${OWF_CA_OBJECTONLY_MONITOR_LOG:-.omx/logs/owf_ca_objectonly_184.log}"
ROOT="${OWF_CA_OBJECTONLY_ROOT:-/home/nebula/xxy/work_dirs/object_world_ca_objectonly_20260725/seed2050318485_e4}"

mkdir -p "$(dirname "$LOG_FILE")"

while true; do
    timestamp="$(date '+%Y-%m-%d %H:%M:%S %z')"
    if output="$(ssh \
            -o BatchMode=yes \
            -o ConnectTimeout=10 \
            -o ConnectionAttempts=1 \
            -o ServerAliveInterval=5 \
            -o ServerAliveCountMax=1 \
            "$HOST" "
root='$ROOT'
line=\"\$(grep 'Epoch(train)' \"\$root/logs/train.log\" 2>/dev/null | tail -n 1)\"
if [ -z \"\$line\" ]; then line=initializing; fi
printf 'train %s\\n' \"\$line\"
error=\"\$(grep -E 'Traceback|CUDA out of memory|FileNotFoundError|loss: *nan|grad_norm: *nan' \"\$root/logs/train.log\" \"\$root/logs/driver.log\" 2>/dev/null | tail -n 1)\"
if [ -n \"\$error\" ]; then printf 'ALERT %s\\n' \"\$error\"; fi
if [ -s \"\$root/summary.tsv\" ]; then tail -n 4 \"\$root/summary.tsv\"; fi
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
" 2>&1)"; then
        {
            printf '%s status=ok\n' "$timestamp"
            printf '%s\n' "$output"
        } >>"$LOG_FILE"
        sleep "$INTERVAL_SECONDS"
    else
        {
            printf '%s status=unreachable\n' "$timestamp"
            printf '%s\n' "$output"
        } >>"$LOG_FILE"
        sleep "$RETRY_SECONDS"
    fi
done
