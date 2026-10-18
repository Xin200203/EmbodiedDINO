#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="${OWF_JOINT_REPO_ROOT:-/datadisk3/xxy/AutoSeg3D_v5_object_world_20260719}"
PYTHON_BIN="${OWF_JOINT_PYTHON:-/datadisk1/xxy/miniconda3/envs/ESAM/bin/python}"
OUTPUT_ROOT="${OWF_JOINT_OUTPUT_BASE:-/datadisk3/xxy/work_dirs/object_world_joint_20260722}"
INTERVAL_SECONDS="${OWF_JOINT_SUPERVISOR_INTERVAL_SECONDS:-300}"
MAX_RESUME_ATTEMPTS="${OWF_JOINT_MAX_RESUME_ATTEMPTS:-3}"
SUPERVISOR_LOG="$OUTPUT_ROOT/supervisor.log"

mkdir -p "$OUTPUT_ROOT"

launch_resume() {
    spec="$1"
    pilot_session="$2"
    session="$3"
    gpu="$4"
    seed="$5"
    scale="$6"
    output_dir="$OUTPUT_ROOT/$spec"
    pilot_log="$output_dir/logs/train.log"
    resume_log="$output_dir/logs/train.resume.log"
    attempts_file="$output_dir/resume128.attempts"

    if [[ -s "$pilot_log" ]] && grep -Eq \
            'CUDA out of memory|Traceback|loss: *nan|grad_norm: *nan' \
            "$pilot_log"; then
        if [[ ! -f "$output_dir/resume128.blocked" ]]; then
            printf '%s BLOCKED pilot error spec=%s\n' \
                "$(date --iso-8601=seconds)" "$spec" >>"$SUPERVISOR_LOG"
            touch "$output_dir/resume128.blocked"
        fi
        return
    fi

    if [[ -f "$output_dir/train.resume.complete" ]]; then
        return
    fi
    if [[ -f "$output_dir/resume128.blocked" ]]; then
        return
    fi
    if tmux has-session -t "$session" 2>/dev/null; then
        return
    fi
    if [[ ! -f "$output_dir/train.complete" || ! -s "$output_dir/epoch_16.pth" ]]; then
        if ! tmux has-session -t "$pilot_session" 2>/dev/null; then
            printf '%s ALERT pilot session missing before completion spec=%s session=%s\n' \
                "$(date --iso-8601=seconds)" "$spec" "$pilot_session" \
                >>"$SUPERVISOR_LOG"
        fi
        return
    fi

    if [[ ! -s "$output_dir/last_checkpoint" ]]; then
        printf '%s BLOCKED missing last_checkpoint spec=%s\n' \
            "$(date --iso-8601=seconds)" "$spec" >>"$SUPERVISOR_LOG"
        touch "$output_dir/resume128.blocked"
        return
    fi
    resume_checkpoint="$(tr -d '\r\n' <"$output_dir/last_checkpoint")"
    if [[ "$resume_checkpoint" != /* ]]; then
        resume_checkpoint="$output_dir/$resume_checkpoint"
    fi
    if [[ ! -s "$resume_checkpoint" ]]; then
        printf '%s BLOCKED invalid resume checkpoint spec=%s checkpoint=%s\n' \
            "$(date --iso-8601=seconds)" "$spec" "$resume_checkpoint" \
            >>"$SUPERVISOR_LOG"
        touch "$output_dir/resume128.blocked"
        return
    fi

    attempts=0
    if [[ -s "$attempts_file" ]]; then
        attempts="$(tr -dc '0-9' <"$attempts_file")"
        attempts="${attempts:-0}"
    fi
    if ((attempts >= MAX_RESUME_ATTEMPTS)); then
        printf '%s BLOCKED resume attempts exhausted spec=%s attempts=%s\n' \
            "$(date --iso-8601=seconds)" "$spec" "$attempts" \
            >>"$SUPERVISOR_LOG"
        touch "$output_dir/resume128.blocked"
        return
    fi
    if [[ -s "$resume_log" ]] && grep -Eq \
            'CUDA out of memory|Traceback|loss: *nan|grad_norm: *nan' \
            "$resume_log"; then
        printf '%s ALERT previous resume failed spec=%s attempt=%s\n' \
            "$(date --iso-8601=seconds)" "$spec" "$attempts" \
            >>"$SUPERVISOR_LOG"
    fi

    attempts=$((attempts + 1))
    printf '%s\n' "$attempts" >"$attempts_file"
    driver_log="$output_dir/logs/resume_driver.attempt${attempts}.log"
    tmux new-session -d -s "$session" \
        "env OWF_JOINT_REPO_ROOT=$REPO_ROOT OWF_JOINT_PYTHON=$PYTHON_BIN OWF_JOINT_OUTPUT_ROOT=$output_dir OWF_JOINT_GPU=$gpu OWF_JOINT_SEED=$seed OWF_JOINT_OUTPUT_SCALE=$scale OWF_JOINT_MAX_EPOCHS=128 OWF_JOINT_VAL_INTERVAL=16 OWF_JOINT_BATCH_SIZE=2 OWF_JOINT_ACCUMULATIVE_COUNTS=2 OWF_JOINT_RESUME=1 bash $REPO_ROOT/tools/run_owf_joint_stage1.sh >$driver_log 2>&1"
    printf '%s attempt=%s session=%s\n' \
        "$(date --iso-8601=seconds)" "$attempts" "$session" \
        >"$output_dir/resume128.launched"
    printf '%s LAUNCHED resume spec=%s session=%s gpu=%s seed=%s scale=%s attempt=%s checkpoint=%s\n' \
        "$(date --iso-8601=seconds)" "$spec" "$session" "$gpu" "$seed" "$scale" \
        "$attempts" "$resume_checkpoint" \
        >>"$SUPERVISOR_LOG"
}

while true; do
    launch_resume seed2050318485_enabled_e16_b2a2 \
        owf-joint-s2050318485-on-e16-b2a2 \
        owf-joint-s2050318485-on-e128-b2a2 0 2050318485 1.0
    launch_resume seed7_enabled_e16_b2a2 \
        owf-joint-s7-on-e16-b2a2 \
        owf-joint-s7-on-e128-b2a2 1 7 1.0
    launch_resume seed123_enabled_e16_b2a2 \
        owf-joint-s123-on-e16-b2a2 \
        owf-joint-s123-on-e128-b2a2 2 123 1.0
    launch_resume seed2050318485_disabled_e16_b2a2 \
        owf-joint-s2050318485-off-e16-b2a2 \
        owf-joint-s2050318485-off-e128-b2a2 3 2050318485 0.0
    launch_resume seed7_disabled_e16_b2a2 \
        owf-joint-s7-off-e16-b2a2 \
        owf-joint-s7-off-e128-b2a2 4 7 0.0
    launch_resume seed123_disabled_e16_b2a2 \
        owf-joint-s123-off-e16-b2a2 \
        owf-joint-s123-off-e128-b2a2 5 123 0.0
    sleep "$INTERVAL_SECONDS"
done
