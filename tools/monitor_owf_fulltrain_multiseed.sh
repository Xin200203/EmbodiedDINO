#!/usr/bin/env bash

set -uo pipefail

BASTION="${OWF_MONITOR_BASTION:-xxy@10.176.56.73:10246}"
TARGET="${OWF_MONITOR_TARGET:-xxy@10.176.56.76}"
INTERVAL_SECONDS="${OWF_MONITOR_INTERVAL_SECONDS:-180}"
RETRY_SECONDS="${OWF_MONITOR_RETRY_SECONDS:-300}"
LOG_FILE="${OWF_MONITOR_LOG:-.omx/logs/owf_fulltrain_multiseed.log}"

mkdir -p "$(dirname "$LOG_FILE")"

while true; do
    timestamp="$(date '+%Y-%m-%d %H:%M:%S %z')"
    if output="$(ssh \
            -o BatchMode=yes \
            -o ConnectTimeout=10 \
            -o ConnectionAttempts=1 \
            -o ServerAliveInterval=5 \
            -o ServerAliveCountMax=1 \
            -J "$BASTION" "$TARGET" '
root_base=/datadisk3/xxy/work_dirs/object_world_20260721
for seed in 7 42 123 2027 2050318485; do
    root="$root_base/fulltrain_seed${seed}_e4"
    if [ -s "$root/summary.tsv" ] && [ -s "$root/shuffled.complete" ]; then
        line="$(tail -n 1 "$root/summary.tsv")"
        printf "seed=%s complete %s\n" "$seed" "$line"
        continue
    fi
    line="$(grep "Epoch(train)" "$root/logs/train.log" 2>/dev/null | tail -n 1)"
    if [ -z "$line" ]; then
        line="initializing"
    fi
    printf "seed=%s %s\n" "$seed" "$line"
    error="$(grep -E "Traceback|CUDA out of memory|[Ll]oss: *nan" \
        "$root/logs/train.log" "$root/driver.log" 2>/dev/null | tail -n 1)"
    if [ -n "$error" ]; then
        printf "seed=%s ALERT %s\n" "$seed" "$error"
    fi
done
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
