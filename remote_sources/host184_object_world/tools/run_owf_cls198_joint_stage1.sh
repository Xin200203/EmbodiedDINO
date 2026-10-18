#!/usr/bin/env bash

set -euo pipefail

: "${OWF_JOINT_REPO_ROOT:?set OWF_JOINT_REPO_ROOT}"
: "${OWF_JOINT_PYTHON:?set OWF_JOINT_PYTHON}"
: "${OWF_JOINT_OUTPUT_ROOT:?set OWF_JOINT_OUTPUT_ROOT}"

CONFIG="${OWF_JOINT_CONFIG:-configs/codex_owf/AutoSeg3D_scannet200_stage1_v5cls1_owf_joint.py}"
GPU="${OWF_JOINT_GPU:-0}"
SEED="${OWF_JOINT_SEED:-2050318485}"
OUTPUT_SCALE="${OWF_JOINT_OUTPUT_SCALE:-1.0}"
MAX_EPOCHS="${OWF_JOINT_MAX_EPOCHS:-16}"
VAL_INTERVAL="${OWF_JOINT_VAL_INTERVAL:-16}"
BATCH_SIZE="${OWF_JOINT_BATCH_SIZE:-2}"
ACCUMULATIVE_COUNTS="${OWF_JOINT_ACCUMULATIVE_COUNTS:-2}"
RESUME="${OWF_JOINT_RESUME:-0}"
INIT_CHECKPOINT="${OWF_JOINT_INIT_CHECKPOINT:-/home/nebula/xxy/3D_Reconstruction/work_dirs/ESAM_sv_scannet200_CA_dino/best_all_ap_50%_epoch_115.pth}"

mkdir -p "$OWF_JOINT_OUTPUT_ROOT/logs"
cd "$OWF_JOINT_REPO_ROOT"

if [[ ! -f "$CONFIG" ]]; then
    echo "missing config: $OWF_JOINT_REPO_ROOT/$CONFIG" >&2
    exit 2
fi
if [[ ! -x "$OWF_JOINT_PYTHON" ]]; then
    echo "python is not executable: $OWF_JOINT_PYTHON" >&2
    exit 2
fi
CHECKPOINT_FOR_MANIFEST="$INIT_CHECKPOINT"
if [[ "$RESUME" == "1" ]]; then
    if [[ ! -s "$OWF_JOINT_OUTPUT_ROOT/last_checkpoint" ]]; then
        echo "missing resume pointer: $OWF_JOINT_OUTPUT_ROOT/last_checkpoint" >&2
        exit 2
    fi
    CHECKPOINT_FOR_MANIFEST="$(tr -d '\r\n' <"$OWF_JOINT_OUTPUT_ROOT/last_checkpoint")"
    if [[ "$CHECKPOINT_FOR_MANIFEST" != /* ]]; then
        CHECKPOINT_FOR_MANIFEST="$OWF_JOINT_OUTPUT_ROOT/$CHECKPOINT_FOR_MANIFEST"
    fi
    if [[ ! -s "$CHECKPOINT_FOR_MANIFEST" ]]; then
        echo "resume checkpoint is missing: $CHECKPOINT_FOR_MANIFEST" >&2
        exit 2
    fi
elif [[ ! -s "$INIT_CHECKPOINT" ]]; then
    echo "missing initialization checkpoint: $INIT_CHECKPOINT" >&2
    exit 2
fi
if [[ "$RESUME" != "1" ]] && compgen -G "$OWF_JOINT_OUTPUT_ROOT/epoch_*.pth" >/dev/null; then
    echo "refusing to overwrite existing checkpoints in $OWF_JOINT_OUTPUT_ROOT" >&2
    exit 2
fi

manifest="$OWF_JOINT_OUTPUT_ROOT/run_manifest.txt"
log_file="$OWF_JOINT_OUTPUT_ROOT/logs/train.log"
complete_marker="$OWF_JOINT_OUTPUT_ROOT/train.complete"
if [[ "$RESUME" == "1" ]]; then
    attempt_tag="$(date '+%Y%m%d_%H%M%S')"
    manifest="$OWF_JOINT_OUTPUT_ROOT/run_manifest.resume.${attempt_tag}.txt"
    log_file="$OWF_JOINT_OUTPUT_ROOT/logs/train.resume.${attempt_tag}.log"
    complete_marker="$OWF_JOINT_OUTPUT_ROOT/train.resume.complete"
    ln -sfn "$(basename "$manifest")" \
        "$OWF_JOINT_OUTPUT_ROOT/run_manifest.resume.txt"
    ln -sfn "$(basename "$log_file")" \
        "$OWF_JOINT_OUTPUT_ROOT/logs/train.resume.log"
fi
{
    printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
    printf 'host=%s\n' "$(hostname)"
    printf 'repo=%s\n' "$OWF_JOINT_REPO_ROOT"
    printf 'config=%s\n' "$CONFIG"
    printf 'python=%s\n' "$OWF_JOINT_PYTHON"
    printf 'gpu=%s\n' "$GPU"
    printf 'seed=%s\n' "$SEED"
    printf 'output_scale=%s\n' "$OUTPUT_SCALE"
    printf 'max_epochs=%s\n' "$MAX_EPOCHS"
    printf 'val_interval=%s\n' "$VAL_INTERVAL"
    printf 'batch_size=%s\n' "$BATCH_SIZE"
    printf 'accumulative_counts=%s\n' "$ACCUMULATIVE_COUNTS"
    printf 'resume=%s\n' "$RESUME"
    printf 'checkpoint_for_manifest=%s\n' "$CHECKPOINT_FOR_MANIFEST"
    sha256sum \
        "$CONFIG" \
        oneformer3d/mink_unet.py \
        oneformer3d/mixformer3d.py \
        "$CHECKPOINT_FOR_MANIFEST"
} >"$manifest"

args=(
    "$OWF_JOINT_PYTHON" tools/train.py "$CONFIG"
    --work-dir "$OWF_JOINT_OUTPUT_ROOT"
    --cfg-options
    "randomness.seed=$SEED"
)
if [[ "$RESUME" == "1" ]]; then
    args+=(--resume)
fi

echo "[$(date --iso-8601=seconds)] launch seed=$SEED scale=$OUTPUT_SCALE gpu=$GPU max_epochs=$MAX_EPOCHS batch=$BATCH_SIZE accum=$ACCUMULATIVE_COUNTS resume=$RESUME"
env \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED="$SEED" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}" \
    PYTHONPATH="$OWF_JOINT_REPO_ROOT" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    CUDA_VISIBLE_DEVICES="$GPU" \
    OWF_JOINT_SEED="$SEED" \
    OWF_JOINT_MODE=real \
    OWF_JOINT_OUTPUT_SCALE="$OUTPUT_SCALE" \
    OWF_JOINT_MAX_EPOCHS="$MAX_EPOCHS" \
    OWF_JOINT_VAL_INTERVAL="$VAL_INTERVAL" \
    OWF_JOINT_BATCH_SIZE="$BATCH_SIZE" \
    OWF_JOINT_ACCUMULATIVE_COUNTS="$ACCUMULATIVE_COUNTS" \
    OWF_JOINT_WORK_DIR="$OWF_JOINT_OUTPUT_ROOT" \
    "${args[@]}" \
    >"$log_file" 2>&1

touch "$complete_marker"
echo "[$(date --iso-8601=seconds)] complete seed=$SEED scale=$OUTPUT_SCALE"
