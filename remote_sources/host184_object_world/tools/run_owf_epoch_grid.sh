#!/usr/bin/env bash

set -euo pipefail

: "${OWF_REPO_ROOT:?set OWF_REPO_ROOT}"
: "${OWF_PYTHON:?set OWF_PYTHON}"
: "${OWF_CHECKPOINT_DIR:?set OWF_CHECKPOINT_DIR}"
: "${OWF_GRID_ROOT:?set OWF_GRID_ROOT}"
: "${OWF_DINO_CACHE:?set OWF_DINO_CACHE}"

CONFIG="${OWF_CONFIG:-configs/codex_owf/AutoSeg3D_scannet200_stage1_v5ab1_owf_subset50.py}"
GPU="${OWF_GPU:-0}"
EPOCHS="${OWF_GRID_EPOCHS:-1 2 3 4}"
SCALE="${OWF_OUTPUT_SCALE:-1.0}"

mkdir -p "$OWF_GRID_ROOT/logs"
SUMMARY="$OWF_GRID_ROOT/summary.tsv"
printf 'epoch\tscale\tmetric_line\n' >"$SUMMARY"

cd "$OWF_REPO_ROOT"
for epoch in $EPOCHS; do
    checkpoint="$OWF_CHECKPOINT_DIR/epoch_${epoch}.pth"
    log="$OWF_GRID_ROOT/logs/epoch_${epoch}.log"
    work_dir="$OWF_GRID_ROOT/epoch_${epoch}"
    test -f "$checkpoint"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] start epoch=${epoch} scale=${SCALE}"
    env \
        PYTHONUNBUFFERED=1 \
        PYTHONPATH="$OWF_REPO_ROOT" \
        OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}" \
        CUDA_VISIBLE_DEVICES="$GPU" \
        OWF_MODE=real \
        OWF_PLACEMENT=pre_memory \
        OWF_INCLUDE_DINO=1 \
        OWF_INCLUDE_S2="${OWF_GRID_INCLUDE_S2:-0}" \
        OWF_INCLUDE_XYZ="${OWF_GRID_INCLUDE_XYZ:-0}" \
        OWF_ORACLE_PURITY=0 \
        OWF_PERSISTENT_ORACLE=0 \
        OWF_OUTPUT_SCALE="$SCALE" \
        OWF_DINO_CACHE="$OWF_DINO_CACHE" \
        "$OWF_PYTHON" tools/test.py "$CONFIG" "$checkpoint" \
        --work-dir "$work_dir" >"$log" 2>&1
    metric_line="$(grep 'all_ap:' "$log" | tail -1 | tr '\t' ' ')"
    printf '%s\t%s\t%s\n' "$epoch" "$SCALE" "$metric_line" >>"$SUMMARY"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] done epoch=${epoch}: ${metric_line}"
done
