#!/usr/bin/env bash

set -euo pipefail

: "${OWF_REPO_ROOT:?set OWF_REPO_ROOT}"
: "${OWF_PYTHON:?set OWF_PYTHON}"
: "${OWF_CHECKPOINT:?set OWF_CHECKPOINT}"
: "${OWF_GRID_ROOT:?set OWF_GRID_ROOT}"
: "${OWF_DINO_CACHE:?set OWF_DINO_CACHE}"

CONFIG="${OWF_CONFIG:-configs/codex_owf/AutoSeg3D_scannet200_stage1_v5ab1_owf_subset50.py}"
GPU="${OWF_GPU:-0}"
SCALE="${OWF_OUTPUT_SCALE:-1.0}"
REPEATS="${OWF_GRID_REPEATS:-2}"
INCLUDE_S2="${OWF_GRID_INCLUDE_S2:-0}"
INCLUDE_XYZ="${OWF_GRID_INCLUDE_XYZ:-0}"

mkdir -p "$OWF_GRID_ROOT/logs"
SUMMARY="$OWF_GRID_ROOT/summary.tsv"
printf 'repeat\tvariant\tscale\tmetric_line\n' >"$SUMMARY"

run_eval() {
    local repeat="$1"
    local variant="$2"
    local mode="$3"
    local tag="r${repeat}_${variant}"
    local log="$OWF_GRID_ROOT/logs/${tag}.log"
    local work_dir="$OWF_GRID_ROOT/${tag}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] start ${tag}"
    env \
        PYTHONUNBUFFERED=1 \
        PYTHONPATH="$OWF_REPO_ROOT" \
        OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}" \
        CUDA_VISIBLE_DEVICES="$GPU" \
        OWF_MODE="$mode" \
        OWF_PLACEMENT=pre_memory \
        OWF_INCLUDE_DINO=1 \
        OWF_INCLUDE_S2="$INCLUDE_S2" \
        OWF_INCLUDE_XYZ="$INCLUDE_XYZ" \
        OWF_ORACLE_PURITY=0 \
        OWF_PERSISTENT_ORACLE=0 \
        OWF_OUTPUT_SCALE="$SCALE" \
        OWF_DINO_CACHE="$OWF_DINO_CACHE" \
        "$OWF_PYTHON" tools/test.py "$CONFIG" "$OWF_CHECKPOINT" \
        --work-dir "$work_dir" >"$log" 2>&1
    local metric_line
    metric_line="$(grep 'all_ap:' "$log" | tail -1 | tr '\t' ' ')"
    printf '%s\t%s\t%s\t%s\n' \
        "$repeat" "$variant" "$SCALE" "$metric_line" >>"$SUMMARY"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] done ${tag}: ${metric_line}"
}

cd "$OWF_REPO_ROOT"
for repeat in $(seq 1 "$REPEATS"); do
    run_eval "$repeat" no_injection stats-only
    run_eval "$repeat" aligned real
    run_eval "$repeat" shuffled shuffled
done
