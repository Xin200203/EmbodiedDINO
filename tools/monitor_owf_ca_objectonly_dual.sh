#!/usr/bin/env bash

set -uo pipefail

BASTION="${OWF_CA_OBJECTONLY_BASTION:-xxy@10.176.56.73:10246}"
HOST76="${OWF_CA_OBJECTONLY_HOST76:-xxy@10.176.56.76}"
HOST184="${OWF_CA_OBJECTONLY_HOST184:-184}"
INTERVAL_SECONDS="${OWF_CA_OBJECTONLY_MONITOR_INTERVAL_SECONDS:-120}"
RETRY_SECONDS="${OWF_CA_OBJECTONLY_MONITOR_RETRY_SECONDS:-30}"
LOG_FILE="${OWF_CA_OBJECTONLY_MONITOR_LOG:-.omx/logs/owf_ca_objectonly_dual.log}"
FAST_ROOT=/datadisk3/xxy/work_dirs/object_world_ca_objectonly_20260725/seed2050318485_e4_fast76
FORMAL_ROOT=/home/nebula/xxy/work_dirs/object_world_ca_objectonly_formal_20260725/seed2050318485_enabled_e128_b4a1

mkdir -p "$(dirname "$LOG_FILE")"

probe_fast() {
    ssh -J "$BASTION" -o BatchMode=yes -o ConnectTimeout=10 \
        -o ConnectionAttempts=1 "$HOST76" "
root='$FAST_ROOT'
line=\"\$(grep 'Epoch(train)' \"\$root/logs/train.log\" 2>/dev/null | tail -n 1)\"
if [ -z \"\$line\" ]; then line=initializing; fi
printf 'fast76 %s\\n' \"\$line\"
if [ -s \"\$root/summary.tsv\" ]; then tail -n 4 \"\$root/summary.tsv\"; fi
error=\"\$(grep -E 'Traceback|CUDA out of memory|loss: *nan|grad_norm: *nan' \"\$root/logs/train.log\" 2>/dev/null | tail -n 1)\"
if [ -n \"\$error\" ]; then printf 'fast76 ALERT %s\\n' \"\$error\"; fi
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
"
}

probe_formal() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 \
        "$HOST184" "
root='$FORMAL_ROOT'
line=\"\$(grep -E 'Epoch\\((train|val)\\)' \"\$root/logs/train.log\" 2>/dev/null | tail -n 1)\"
if [ -z \"\$line\" ]; then line=initializing; fi
printf 'formal184 %s\\n' \"\$line\"
error=\"\$(grep -E 'Traceback|CUDA out of memory|loss: *nan|grad_norm: *nan' \"\$root/logs/train.log\" 2>/dev/null | tail -n 1)\"
if [ -n \"\$error\" ]; then printf 'formal184 ALERT %s\\n' \"\$error\"; fi
checkpoint=\"\$(cat \"\$root/last_checkpoint\" 2>/dev/null)\"
printf 'formal184 checkpoint=%s\\n' \"\${checkpoint:-missing}\"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
"
}

while true; do
    timestamp="$(date '+%Y-%m-%d %H:%M:%S %z')"
    ok=true
    fast="$(probe_fast 2>&1)" || ok=false
    formal="$(probe_formal 2>&1)" || ok=false
    {
        printf '%s status=%s\n' "$timestamp" "$([ "$ok" = true ] && printf ok || printf partial)"
        printf '%s\n%s\n' "$fast" "$formal"
    } >>"$LOG_FILE"
    if [ "$ok" = true ]; then
        sleep "$INTERVAL_SECONDS"
    else
        sleep "$RETRY_SECONDS"
    fi
done
