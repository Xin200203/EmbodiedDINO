#!/usr/bin/env bash

set -euo pipefail

: "${OWF_JOINT_REPO_ROOT:?set OWF_JOINT_REPO_ROOT}"
: "${OWF_JOINT_PYTHON:?set OWF_JOINT_PYTHON}"
: "${OWF_JOINT_OUTPUT_ROOT:?set OWF_JOINT_OUTPUT_ROOT}"

CONFIG="${OWF_JOINT_CONFIG:-configs/codex_owf/AutoSeg3D_scannet200_stage1_v5cls1_owf_objectonly_joint.py}"
GPU="${OWF_JOINT_GPU:-0}"
SEED="${OWF_JOINT_SEED:-2050318485}"
OUTPUT_SCALE="${OWF_JOINT_OUTPUT_SCALE:?set OWF_JOINT_OUTPUT_SCALE}"
WAIT_SECONDS="${OWF_JOINT_EVAL_WAIT_SECONDS:-120}"
CHECKPOINT="$OWF_JOINT_OUTPUT_ROOT/epoch_128.pth"
EVAL_DIR="$OWF_JOINT_OUTPUT_ROOT/final_eval_e128_independent"
LOG_DIR="$EVAL_DIR/logs"
LOG_FILE="$LOG_DIR/test.log"
COMPLETE_MARKER="$EVAL_DIR/eval.complete"

mkdir -p "$EVAL_DIR"
exec 9>"$EVAL_DIR/eval.lock"
if ! flock -n 9; then
    echo "another final-evaluation supervisor owns $EVAL_DIR" >&2
    exit 0
fi

while [[ ! -f "$OWF_JOINT_OUTPUT_ROOT/train.complete" || ! -s "$CHECKPOINT" ]]; do
    train_log="$OWF_JOINT_OUTPUT_ROOT/logs/train.log"
    if [[ -s "$train_log" ]] && grep -Eq \
            'Traceback|CUDA out of memory|FileNotFoundError|loss: *nan|grad_norm: *nan' \
            "$train_log"; then
        echo "training failed before final evaluation: $train_log" >&2
        exit 3
    fi
    sleep "$WAIT_SECONDS"
done

if [[ -f "$COMPLETE_MARKER" ]]; then
    exit 0
fi

cd "$OWF_JOINT_REPO_ROOT"
mkdir -p "$LOG_DIR"
{
    printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
    printf 'host=%s\n' "$(hostname)"
    printf 'repo=%s\nconfig=%s\ncheckpoint=%s\n' \
        "$OWF_JOINT_REPO_ROOT" "$CONFIG" "$CHECKPOINT"
    printf 'gpu=%s\nseed=%s\noutput_scale=%s\n' \
        "$GPU" "$SEED" "$OUTPUT_SCALE"
    sha256sum "$CONFIG" oneformer3d/mink_unet.py \
        oneformer3d/mixformer3d.py "$CHECKPOINT"
} >"$EVAL_DIR/run_manifest.txt"

env \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED="$SEED" \
    PYTHONPATH="$OWF_JOINT_REPO_ROOT" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    CUDA_VISIBLE_DEVICES="$GPU" \
    OWF_JOINT_SEED="$SEED" \
    OWF_JOINT_MODE=real \
    OWF_JOINT_OUTPUT_SCALE="$OUTPUT_SCALE" \
    OWF_JOINT_MAX_EPOCHS=128 \
    OWF_JOINT_VAL_INTERVAL=16 \
    OWF_JOINT_BATCH_SIZE=2 \
    OWF_JOINT_ACCUMULATIVE_COUNTS=2 \
    OWF_JOINT_WORK_DIR="$OWF_JOINT_OUTPUT_ROOT" \
    "$OWF_JOINT_PYTHON" tools/test.py "$CONFIG" "$CHECKPOINT" \
        --work-dir "$EVAL_DIR" >"$LOG_FILE" 2>&1

metric="$(grep 'all_ap:' "$LOG_FILE" | tail -n 1)"
if [[ -z "$metric" ]]; then
    echo "fresh evaluation completed without an all_ap metric: $LOG_FILE" >&2
    exit 4
fi
printf '%s\n' "$metric" >"$EVAL_DIR/metric.txt"
touch "$COMPLETE_MARKER"
