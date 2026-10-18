#!/usr/bin/env bash

set -uo pipefail

HOST184="${OWF_CLS198_PAIR_HOST184:-10.177.69.184}"
HOST76="${OWF_CLS198_PAIR_HOST76:-10.176.56.76}"
INTERVAL_SECONDS="${OWF_CLS198_PAIR_MONITOR_INTERVAL_SECONDS:-120}"
RETRY_SECONDS="${OWF_CLS198_PAIR_MONITOR_RETRY_SECONDS:-30}"
LOG_FILE="${OWF_CLS198_PAIR_MONITOR_LOG:-.omx/logs/owf_cls198_objectonly_pair.log}"

ROOT184=/home/nebula/xxy/work_dirs/object_world_cls198_objectonly_20260724
ROOT76=/datadisk3/xxy/work_dirs/object_world_cls198_objectonly_20260724
SPEC_ON=seed2050318485_enabled_e128_b2a2
SPEC_OFF=seed2050318485_disabled_e128_b2a2
SPEC_OFF_184=seed2050318485_disabled_e128_b2a2_samehost184

mkdir -p "$(dirname "$LOG_FILE")"

probe() {
    local host="$1"
    local root="$2"
    local spec="$3"
    local label="$4"
    ssh \
        -o BatchMode=yes \
        -o ConnectTimeout=10 \
        -o ConnectionAttempts=1 \
        -o ServerAliveInterval=5 \
        -o ServerAliveCountMax=1 \
        "$host" "
run='$root/$spec'
log=\"\$run/logs/train.log\"
line=\"\$(grep -E 'Epoch\\((train|val)\\)' \"\$log\" 2>/dev/null | tail -n 1)\"
if [ -z \"\$line\" ]; then line=initializing; fi
printf '$label %s\\n' \"\$line\"
error=\"\$(grep -E 'Traceback|CUDA out of memory|FileNotFoundError|loss: *nan|grad_norm: *nan' \"\$log\" 2>/dev/null | tail -n 1)\"
if [ -n \"\$error\" ]; then printf '$label ALERT %s\\n' \"\$error\"; fi
checkpoint=\"\$(cat \"\$run/last_checkpoint\" 2>/dev/null)\"
printf '$label state complete=%s checkpoint=%s\\n' \
  \"\$([ -f \"\$run/train.complete\" ] && printf yes || printf no)\" \
  \"\${checkpoint:-missing}\"
"
}

while true; do
    timestamp="$(date '+%Y-%m-%d %H:%M:%S %z')"
    ok=true
    output=""
    if value="$(probe "$HOST184" "$ROOT184" "$SPEC_ON" object_on 2>&1)"; then
        output+="$value"$'\n'
    else
        ok=false
        output+="object_on unreachable: $value"$'\n'
    fi
    if value="$(probe "$HOST76" "$ROOT76" "$SPEC_OFF" object_off 2>&1)"; then
        output+="$value"$'\n'
    else
        ok=false
        output+="object_off unreachable: $value"$'\n'
    fi
    if value="$(probe "$HOST184" "$ROOT184" "$SPEC_OFF_184" object_off_184 2>&1)"; then
        output+="$value"$'\n'
    else
        ok=false
        output+="object_off_184 unreachable: $value"$'\n'
    fi
    {
        printf '%s status=%s\n' "$timestamp" "$([ "$ok" = true ] && printf ok || printf partial)"
        printf '%s' "$output"
    } >>"$LOG_FILE"
    if [ "$ok" = true ]; then
        sleep "$INTERVAL_SECONDS"
    else
        sleep "$RETRY_SECONDS"
    fi
done
