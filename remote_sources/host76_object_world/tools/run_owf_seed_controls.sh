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
SUMMARY="$OWF_SWEEP_ROOT/controls.tsv"
printf 'seed\tvariant\tmetric_line\tcheckpoint\n' >"$SUMMARY"

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
        OWF_MAX_EPOCHS="$EPOCHS" \
        OWF_VAL_INTERVAL="$EPOCHS" \
        OWF_DINO_CACHE="$OWF_DINO_CACHE" \
        OWF_CHECKPOINT="$OWF_BASE_CHECKPOINT" \
        "$@"
}

cd "$OWF_REPO_ROOT"
for seed in $SEEDS; do
    checkpoint="$OWF_SWEEP_ROOT/seed${seed}/epoch_${EPOCHS}.pth"
    [[ -s "$checkpoint" ]] || {
        echo "missing checkpoint: $checkpoint" >&2
        exit 1
    }

    for variant in no_write shuffled; do
        log="$OWF_SWEEP_ROOT/logs/seed${seed}_${variant}_eval.log"
        work_dir="$OWF_SWEEP_ROOT/seed${seed}_${variant}_eval"
        if [[ "$variant" == no_write ]]; then
            mode=real
            output_scale=0.0
        else
            mode=shuffled
            output_scale=1.0
        fi

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] seed=${seed} ${variant}"
        PYTHONHASHSEED="$seed" OWF_MODE="$mode" \
            OWF_OUTPUT_SCALE="$output_scale" common_env \
            "$OWF_PYTHON" tools/test.py "$CONFIG" "$checkpoint" \
            --work-dir "$work_dir" \
            --cfg-options "randomness.seed=${seed}" \
            >"$log" 2>&1
        metric="$(grep 'all_ap:' "$log" | tail -1 | tr '\t' ' ')"
        printf '%s\t%s\t%s\t%s\n' \
            "$seed" "$variant" "$metric" "$checkpoint" >>"$SUMMARY"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${metric}"
    done
done

if [[ -n "${OWF_ORIGINAL_CHECKPOINT:-}" ]]; then
    seed="${OWF_ORIGINAL_SEED:-2050318485}"
    checkpoint="$OWF_ORIGINAL_CHECKPOINT"
    [[ -s "$checkpoint" ]] || {
        echo "missing original checkpoint: $checkpoint" >&2
        exit 1
    }

    for variant in no_write aligned shuffled; do
        log="$OWF_SWEEP_ROOT/logs/seed${seed}_${variant}_eval.log"
        work_dir="$OWF_SWEEP_ROOT/seed${seed}_${variant}_eval"
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

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] seed=${seed} ${variant}"
        PYTHONHASHSEED="$seed" OWF_MODE="$mode" \
            OWF_OUTPUT_SCALE="$output_scale" common_env \
            "$OWF_PYTHON" tools/test.py "$CONFIG" "$checkpoint" \
            --work-dir "$work_dir" \
            --cfg-options "randomness.seed=${seed}" \
            >"$log" 2>&1
        metric="$(grep 'all_ap:' "$log" | tail -1 | tr '\t' ' ')"
        printf '%s\t%s\t%s\t%s\n' \
            "$seed" "$variant" "$metric" "$checkpoint" >>"$SUMMARY"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${metric}"
    done
fi
