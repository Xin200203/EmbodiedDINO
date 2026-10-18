#!/usr/bin/env bash

set -uo pipefail

TARGET="${OWF_CLS198_MONITOR_TARGET:-184}"
INTERVAL_SECONDS="${OWF_CLS198_MONITOR_INTERVAL_SECONDS:-180}"
RETRY_SECONDS="${OWF_CLS198_MONITOR_RETRY_SECONDS:-60}"
LOG_FILE="${OWF_CLS198_MONITOR_LOG:-.omx/logs/owf_cls198_184.log}"

mkdir -p "$(dirname "$LOG_FILE")"

while true; do
    timestamp="$(date '+%Y-%m-%d %H:%M:%S %z')"
    if output="$(ssh \
            -o BatchMode=yes \
            -o ConnectTimeout=10 \
            -o ConnectionAttempts=1 \
            -o ServerAliveInterval=5 \
            -o ServerAliveCountMax=1 \
            "$TARGET" '
root=/home/nebula/xxy/work_dirs/object_world_joint_cls198_20260722/seed2050318485_enabled_e128_b2a2_retry2
log="$root/logs/train.log"
line="$(grep -E "Epoch\((train|val)\)" "$log" 2>/dev/null | tail -n 1)"
if [ -z "$line" ]; then
    line=initializing
fi
printf "class_sensitive %s\n" "$line"
error="$(grep -E "Traceback|CUDA out of memory|FileNotFoundError|[Ll]oss: *nan|grad_norm: *nan" \
    "$log" 2>/dev/null | tail -n 1)"
if [ -n "$error" ]; then
    printf "class_sensitive ALERT %s\n" "$error"
fi
checkpoint="$(cat "$root/last_checkpoint" 2>/dev/null)"
printf "class_sensitive state complete=%s last_checkpoint=%s\n" \
    "$([ -f "$root/train.complete" ] && printf yes || printf no)" \
    "${checkpoint:-missing}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader
' 2>&1)"; then
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
