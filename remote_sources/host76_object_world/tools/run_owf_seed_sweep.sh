#!/usr/bin/env bash

set -euo pipefail

: "${OWF_REPO_ROOT:?set OWF_REPO_ROOT}"
: "${OWF_PYTHON:?set OWF_PYTHON}"
: "${OWF_BASE_CHECKPOINT:?set OWF_BASE_CHECKPOINT}"
: "${OWF_DINO_CACHE:?set OWF_DINO_CACHE}"
: "${OWF_SWEEP_ROOT:?set OWF_SWEEP_ROOT}"

CONFIG="${OWF_CONFIG:-configs/codex_owf/AutoSeg3D_scannet200_stage1_v5ab1_owf_subset50.py}"
GPU="${OWF_GPU:-1}"
SEEDS="${OWF_SEEDS:-7 123 42 2027}"
EPOCHS="${OWF_MAX_EPOCHS:-4}"

mkdir -p "$OWF_SWEEP_ROOT/logs"
SUMMARY="$OWF_SWEEP_ROOT/summary.tsv"
printf 'seed\tphase\tmetric_line\tcheckpoint\n' >"$SUMMARY"

common_env() {
    env \
        PYTHONUNBUFFERED=1 \
        PYTHONPATH="$OWF_REPO_ROOT" \
        OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}" \
        CUDA_VISIBLE_DEVICES="$GPU" \
        OWF_MODE=real \
        OWF_PLACEMENT=pre_memory \
        OWF_INCLUDE_DINO=1 \
        OWF_INCLUDE_S2=0 \
        OWF_INCLUDE_XYZ=0 \
        OWF_ORACLE_PURITY=0 \
        OWF_PERSISTENT_ORACLE=0 \
        OWF_OUTPUT_SCALE=1.0 \
        OWF_INIT_GATE=0.05 \
        OWF_LR=0.001 \
        OWF_MAX_EPOCHS="$EPOCHS" \
        OWF_VAL_INTERVAL="$EPOCHS" \
        OWF_DINO_CACHE="$OWF_DINO_CACHE" \
        OWF_CHECKPOINT="$OWF_BASE_CHECKPOINT" \
        "$@"
}

cd "$OWF_REPO_ROOT"
for seed in $SEEDS; do
    work_dir="$OWF_SWEEP_ROOT/seed${seed}"
    train_log="$OWF_SWEEP_ROOT/logs/seed${seed}_train.log"
    eval_log="$OWF_SWEEP_ROOT/logs/seed${seed}_aligned_eval.log"
    checkpoint="$work_dir/epoch_${EPOCHS}.pth"

    if [[ ! -s "$checkpoint" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] train seed=${seed}"
        PYTHONHASHSEED="$seed" OWF_WORK_DIR="$work_dir" common_env \
            "$OWF_PYTHON" tools/train.py "$CONFIG" \
            --work-dir "$work_dir" \
            --cfg-options "randomness.seed=${seed}" \
            >"$train_log" 2>&1
    fi

    train_metric="$(grep 'Epoch(val)' "$train_log" | grep 'all_ap:' | tail -1 | tr '\t' ' ')"
    printf '%s\ttrain_val\t%s\t%s\n' \
        "$seed" "$train_metric" "$checkpoint" >>"$SUMMARY"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] fresh aligned eval seed=${seed}"
    PYTHONHASHSEED="$seed" common_env \
        "$OWF_PYTHON" tools/test.py "$CONFIG" "$checkpoint" \
        --work-dir "$OWF_SWEEP_ROOT/seed${seed}_aligned_eval" \
        --cfg-options "randomness.seed=${seed}" \
        >"$eval_log" 2>&1
    eval_metric="$(grep 'all_ap:' "$eval_log" | tail -1 | tr '\t' ' ')"
    printf '%s\tfresh_aligned\t%s\t%s\n' \
        "$seed" "$eval_metric" "$checkpoint" >>"$SUMMARY"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] seed=${seed}: ${eval_metric}"
done

