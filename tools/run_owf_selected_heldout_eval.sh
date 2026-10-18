#!/usr/bin/env bash

set -euo pipefail

: "${OWF_REPO_ROOT:?set OWF_REPO_ROOT}"
: "${OWF_PYTHON:?set OWF_PYTHON}"
: "${OWF_BASE_CHECKPOINT:?set OWF_BASE_CHECKPOINT}"
: "${OWF_DINO_CACHE:?set OWF_DINO_CACHE}"
: "${OWF_SELECTED_CHECKPOINT:?set OWF_SELECTED_CHECKPOINT}"
: "${OWF_SELECTED_SEED:?set OWF_SELECTED_SEED}"
: "${OWF_OUTPUT_ROOT:?set OWF_OUTPUT_ROOT}"

CONFIG="${OWF_CONFIG:-configs/codex_owf/AutoSeg3D_scannet200_stage1_v5ab1_owf_fullval.py}"
ANN_FILE="${OWF_ANN_FILE:-scannet200_mv_oneformer3d_infos_val_complement262_ocwe20260720.pkl}"
GPU="${OWF_GPU:-1}"

mkdir -p "$OWF_OUTPUT_ROOT/logs"
SUMMARY="$OWF_OUTPUT_ROOT/heldout_summary.tsv"
printf 'seed\tvariant\tann_file\tmetric_line\tcheckpoint\n' >"$SUMMARY"

common_env() {
    env \
        PYTHONUNBUFFERED=1 \
        PYTHONPATH="$OWF_REPO_ROOT" \
        OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}" \
        CUDA_VISIBLE_DEVICES="$GPU" \
        OWF_PLACEMENT=pre_memory \
        OWF_INCLUDE_DINO=1 \
        OWF_INCLUDE_S2=0 \
        OWF_INCLUDE_XYZ=0 \
        OWF_ORACLE_PURITY=0 \
        OWF_PERSISTENT_ORACLE=0 \
        OWF_INIT_GATE=0.05 \
        OWF_LR=0.001 \
        OWF_MAX_EPOCHS=4 \
        OWF_VAL_INTERVAL=4 \
        OWF_DINO_CACHE="$OWF_DINO_CACHE" \
        OWF_CHECKPOINT="$OWF_BASE_CHECKPOINT" \
        "$@"
}

cd "$OWF_REPO_ROOT"
for variant in no_write aligned; do
    if [[ "$variant" == no_write ]]; then
        output_scale=0.0
    else
        output_scale=1.0
    fi
    log="$OWF_OUTPUT_ROOT/logs/${variant}.log"
    work_dir="$OWF_OUTPUT_ROOT/${variant}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${variant} seed=${OWF_SELECTED_SEED}"
    PYTHONHASHSEED="$OWF_SELECTED_SEED" OWF_MODE=real \
        OWF_OUTPUT_SCALE="$output_scale" common_env \
        "$OWF_PYTHON" tools/test.py "$CONFIG" "$OWF_SELECTED_CHECKPOINT" \
        --work-dir "$work_dir" \
        --cfg-options \
            "randomness.seed=${OWF_SELECTED_SEED}" \
            "test_dataloader.dataset.ann_file=${ANN_FILE}" \
        >"$log" 2>&1
    metric="$(grep 'all_ap:' "$log" | tail -1 | tr '\t' ' ')"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$OWF_SELECTED_SEED" "$variant" "$ANN_FILE" "$metric" \
        "$OWF_SELECTED_CHECKPOINT" >>"$SUMMARY"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${metric}"
done
