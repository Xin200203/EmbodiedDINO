#!/usr/bin/env bash

set -euo pipefail

cleanup_children() {
    local children
    children="$(jobs -pr)"
    if [[ -n "$children" ]]; then
        kill $children 2>/dev/null || true
    fi
}
trap cleanup_children EXIT INT TERM

: "${OWF_REPO_ROOT:?set OWF_REPO_ROOT}"
: "${OWF_PYTHON:?set OWF_PYTHON}"
: "${OWF_BASE_CHECKPOINT:?set OWF_BASE_CHECKPOINT}"
: "${OWF_DINO_CACHE:?set OWF_DINO_CACHE}"
: "${OWF_OUTPUT_ROOT:?set OWF_OUTPUT_ROOT}"

CONFIG="${OWF_CONFIG:-configs/codex_owf/AutoSeg3D_scannet200_stage1_v5ab1_owf_fullval.py}"
TRAIN_ANN="${OWF_TRAIN_ANN:-scannet200_mv_oneformer3d_infos_train.pkl}"
GPU="${OWF_GPU:-1}"
EVAL_GPU_LIST_RAW="${OWF_EVAL_GPU_LIST:-$GPU}"
SEED="${OWF_SEED:-2050318485}"
EPOCHS="${OWF_MAX_EPOCHS:-4}"

mkdir -p "$OWF_OUTPUT_ROOT/logs"
SUMMARY="$OWF_OUTPUT_ROOT/summary.tsv"
printf 'seed\tphase\ttrain_ann\tmetric_line\tcheckpoint\n' >"$SUMMARY"

common_env() {
    env \
        PYTHONUNBUFFERED=1 \
        PYTHONPATH="$OWF_REPO_ROOT" \
        OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}" \
        CUDA_VISIBLE_DEVICES="${OWF_ACTIVE_GPU:-$GPU}" \
        OWF_PLACEMENT=pre_memory \
        OWF_INCLUDE_DINO=1 \
        OWF_INCLUDE_S2=0 \
        OWF_INCLUDE_XYZ=0 \
        OWF_ORACLE_PURITY=0 \
        OWF_PERSISTENT_ORACLE=0 \
        OWF_INIT_GATE=0.05 \
        OWF_LR=0.001 \
        OWF_MAX_EPOCHS="$EPOCHS" \
        OWF_VAL_INTERVAL="$EPOCHS" \
        OWF_DINO_CACHE="$OWF_DINO_CACHE" \
        OWF_CHECKPOINT="$OWF_BASE_CHECKPOINT" \
        "$@"
}

cd "$OWF_REPO_ROOT"
train_dir="$OWF_OUTPUT_ROOT/train"
train_log="$OWF_OUTPUT_ROOT/logs/train.log"
checkpoint="$train_dir/epoch_${EPOCHS}.pth"

if [[ ! -s "$checkpoint" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] full-train seed=${SEED} ann=${TRAIN_ANN}"
    PYTHONHASHSEED="$SEED" OWF_MODE=real OWF_OUTPUT_SCALE=1.0 \
        OWF_WORK_DIR="$train_dir" common_env \
        "$OWF_PYTHON" tools/train.py "$CONFIG" \
        --work-dir "$train_dir" \
        --cfg-options \
            "randomness.seed=${SEED}" \
            "train_dataloader.dataset.ann_file=${TRAIN_ANN}" \
        >"$train_log" 2>&1
fi

train_metric="$(grep 'Epoch(val)' "$train_log" | grep 'all_ap:' | tail -1 | tr '\t' ' ')"
printf '%s\ttrain_val\t%s\t%s\t%s\n' \
    "$SEED" "$TRAIN_ANN" "$train_metric" "$checkpoint" >>"$SUMMARY"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${train_metric}"

evaluate_variant() {
    local variant="$1"
    local gpu="$2"
    local mode
    local output_scale
    local log
    local work_dir
    local metric
    case "$variant" in
        no_write)
            mode=real
            output_scale=0.0
            ;;
        aligned)
            mode=real
            output_scale=1.0
            ;;
        shuffled)
            mode=shuffled
            output_scale=1.0
            ;;
    esac
    log="$OWF_OUTPUT_ROOT/logs/${variant}.log"
    work_dir="$OWF_OUTPUT_ROOT/${variant}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] full-val ${variant} seed=${SEED}"
    PYTHONHASHSEED="$SEED" OWF_MODE="$mode" \
        OWF_OUTPUT_SCALE="$output_scale" OWF_ACTIVE_GPU="$gpu" common_env \
        "$OWF_PYTHON" tools/test.py "$CONFIG" "$checkpoint" \
        --work-dir "$work_dir" \
        --cfg-options "randomness.seed=${SEED}" \
        >"$log" 2>&1
    metric="$(grep 'all_ap:' "$log" | tail -1 | tr '\t' ' ')"
    printf '%s\n' "$metric" >"$OWF_OUTPUT_ROOT/${variant}.complete"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${metric}"
}

IFS=',' read -r -a EVAL_GPUS <<<"$EVAL_GPU_LIST_RAW"
if [[ "${#EVAL_GPUS[@]}" -eq 0 ]]; then
    echo "OWF_EVAL_GPU_LIST did not contain a GPU" >&2
    exit 2
fi
EVAL_VARIANTS=(no_write aligned shuffled)
for ((start = 0; start < ${#EVAL_VARIANTS[@]}; start += ${#EVAL_GPUS[@]})); do
    pids=()
    batch_variants=()
    for ((slot = 0; slot < ${#EVAL_GPUS[@]}; slot++)); do
        index=$((start + slot))
        if ((index >= ${#EVAL_VARIANTS[@]})); then
            break
        fi
        gpu="${EVAL_GPUS[$slot]//[[:space:]]/}"
        if [[ -z "$gpu" ]]; then
            echo "empty GPU entry in OWF_EVAL_GPU_LIST=${EVAL_GPU_LIST_RAW}" >&2
            exit 2
        fi
        evaluate_variant "${EVAL_VARIANTS[$index]}" "$gpu" &
        pids+=("$!")
        batch_variants+=("${EVAL_VARIANTS[$index]}")
    done
    batch_failed=0
    for ((slot = 0; slot < ${#pids[@]}; slot++)); do
        if ! wait "${pids[$slot]}"; then
            echo "full-val evaluation failed: ${batch_variants[$slot]}" >&2
            batch_failed=1
        fi
    done
    if ((batch_failed)); then
        exit 1
    fi
done

for variant in "${EVAL_VARIANTS[@]}"; do
    metric="$(cat "$OWF_OUTPUT_ROOT/${variant}.complete")"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$SEED" "$variant" "$TRAIN_ANN" "$metric" "$checkpoint" \
        >>"$SUMMARY"
done
