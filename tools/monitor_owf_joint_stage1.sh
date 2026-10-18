#!/usr/bin/env bash

set -uo pipefail

BASTION="${OWF_JOINT_MONITOR_BASTION:-xxy@10.176.56.73:10246}"
TARGET="${OWF_JOINT_MONITOR_TARGET:-xxy@10.176.56.76}"
INTERVAL_SECONDS="${OWF_JOINT_MONITOR_INTERVAL_SECONDS:-180}"
RETRY_SECONDS="${OWF_JOINT_MONITOR_RETRY_SECONDS:-60}"
LOG_FILE="${OWF_JOINT_MONITOR_LOG:-.omx/logs/owf_joint_stage1.log}"

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
root=/datadisk3/xxy/work_dirs/object_world_joint_20260722
for spec in \
    seed2050318485_enabled_e16_b2a2 \
    seed7_enabled_e16_b2a2 \
    seed123_enabled_e16_b2a2 \
    seed2050318485_disabled_e16_b2a2 \
    seed7_disabled_e16_b2a2 \
    seed123_disabled_e16_b2a2; do
    log="$root/$spec/logs/train.log"
    if [ -s "$root/$spec/logs/train.resume.log" ]; then
        log="$root/$spec/logs/train.resume.log"
    fi
    line="$(grep -E "Epoch\((train|val)\)" "$log" 2>/dev/null | tail -n 1)"
    if [ -z "$line" ]; then
        line=initializing
    fi
    printf "%s %s\n" "$spec" "$line"
    error="$(grep -E "Traceback|CUDA out of memory|[Ll]oss: *nan|grad_norm: *nan" \
        "$log" 2>/dev/null | tail -n 1)"
    if [ -n "$error" ]; then
        printf "%s ALERT %s\n" "$spec" "$error"
    fi
    checkpoint="$(cat "$root/$spec/last_checkpoint" 2>/dev/null)"
    printf "%s state pilot_complete=%s resume_complete=%s resume_attempts=%s last_checkpoint=%s\n" \
        "$spec" \
        "$([ -f "$root/$spec/train.complete" ] && printf yes || printf no)" \
        "$([ -f "$root/$spec/train.resume.complete" ] && printf yes || printf no)" \
        "$(cat "$root/$spec/resume128.attempts" 2>/dev/null || printf 0)" \
        "${checkpoint:-missing}"
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
