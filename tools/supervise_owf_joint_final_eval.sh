#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="${OWF_JOINT_REPO_ROOT:-/datadisk3/xxy/AutoSeg3D_v5_object_world_20260719}"
PYTHON_BIN="${OWF_JOINT_PYTHON:-/datadisk1/xxy/miniconda3/envs/ESAM/bin/python}"
TRAIN_ROOT="${OWF_JOINT_OUTPUT_BASE:-/datadisk3/xxy/work_dirs/object_world_joint_20260722}"
CONFIG="${OWF_JOINT_CONFIG:-configs/codex_owf/AutoSeg3D_scannet200_stage1_v5ab1_owf_joint.py}"
INTERVAL_SECONDS="${OWF_JOINT_FINAL_EVAL_INTERVAL_SECONDS:-120}"
MAX_ATTEMPTS="${OWF_JOINT_FINAL_EVAL_MAX_ATTEMPTS:-2}"
SUPERVISOR_LOG="$TRAIN_ROOT/final_eval_supervisor.log"

mkdir -p "$TRAIN_ROOT"

launch_eval() {
    spec="$1"
    session="$2"
    gpu="$3"
    seed="$4"
    scale="$5"
    train_dir="$TRAIN_ROOT/$spec"
    checkpoint="$train_dir/epoch_128.pth"
    eval_dir="$train_dir/final_eval_e128_independent"
    log_dir="$eval_dir/logs"
    log_file="$log_dir/test.log"
    complete_marker="$eval_dir/eval.complete"
    attempts_file="$eval_dir/eval.attempts"

    if [[ -f "$complete_marker" ]]; then
        return
    fi
    if [[ ! -f "$train_dir/train.resume.complete" || ! -s "$checkpoint" ]]; then
        return
    fi
    if tmux has-session -t "$session" 2>/dev/null; then
        return
    fi

    attempts=0
    if [[ -s "$attempts_file" ]]; then
        attempts="$(tr -dc '0-9' <"$attempts_file")"
        attempts="${attempts:-0}"
    fi
    if ((attempts >= MAX_ATTEMPTS)); then
        return
    fi

    attempts=$((attempts + 1))
    mkdir -p "$log_dir"
    printf '%s\n' "$attempts" >"$attempts_file"
    {
        printf 'launched=%s\n' "$(date --iso-8601=seconds)"
        printf 'host=%s\n' "$(hostname)"
        printf 'repo=%s\n' "$REPO_ROOT"
        printf 'python=%s\n' "$PYTHON_BIN"
        printf 'config=%s\n' "$CONFIG"
        printf 'checkpoint=%s\n' "$checkpoint"
        printf 'gpu=%s\n' "$gpu"
        printf 'seed=%s\n' "$seed"
        printf 'output_scale=%s\n' "$scale"
        printf 'attempt=%s\n' "$attempts"
        sha256sum "$REPO_ROOT/$CONFIG" "$checkpoint"
    } >"$eval_dir/run_manifest.attempt${attempts}.txt"

    tmux new-session -d -s "$session" \
        "cd '$REPO_ROOT' && env PYTHONUNBUFFERED=1 PYTHONPATH='$REPO_ROOT' OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='$gpu' OWF_JOINT_SEED='$seed' OWF_JOINT_MODE=real OWF_JOINT_OUTPUT_SCALE='$scale' OWF_JOINT_MAX_EPOCHS=128 OWF_JOINT_VAL_INTERVAL=16 OWF_JOINT_BATCH_SIZE=2 OWF_JOINT_ACCUMULATIVE_COUNTS=2 OWF_JOINT_WORK_DIR='$train_dir' '$PYTHON_BIN' tools/test.py '$CONFIG' '$checkpoint' --work-dir '$eval_dir' >'$log_file' 2>&1 && touch '$complete_marker'"
    printf '%s launched spec=%s session=%s gpu=%s attempt=%s\n' \
        "$(date --iso-8601=seconds)" "$spec" "$session" "$gpu" "$attempts" \
        >>"$SUPERVISOR_LOG"
}

while true; do
    launch_eval seed2050318485_enabled_e16_b2a2 \
        owf-final-eval-s2050318485-on 0 2050318485 1.0
    launch_eval seed7_enabled_e16_b2a2 \
        owf-final-eval-s7-on 1 7 1.0
    launch_eval seed123_enabled_e16_b2a2 \
        owf-final-eval-s123-on 2 123 1.0
    launch_eval seed2050318485_disabled_e16_b2a2 \
        owf-final-eval-s2050318485-off 3 2050318485 0.0
    launch_eval seed7_disabled_e16_b2a2 \
        owf-final-eval-s7-off 4 7 0.0
    launch_eval seed123_disabled_e16_b2a2 \
        owf-final-eval-s123-off 5 123 0.0
    sleep "$INTERVAL_SECONDS"
done
