#!/usr/bin/env bash

set -euo pipefail

: "${TRANSFER_REPO_ROOT:?set TRANSFER_REPO_ROOT}"
: "${TRANSFER_PYTHON:?set TRANSFER_PYTHON}"
: "${TRANSFER_OUTPUT_ROOT:?set TRANSFER_OUTPUT_ROOT}"

GPU="${TRANSFER_GPU:-0}"
SEEDS="${TRANSFER_SEEDS:-7 123}"

mkdir -p "$TRANSFER_OUTPUT_ROOT/logs"
SUMMARY="$TRANSFER_OUTPUT_ROOT/summary.tsv"
printf 'dataset\tvariant\tseed\tmetric_line\tcheckpoint\n' >"$SUMMARY"

cd "$TRANSFER_REPO_ROOT"
for seed in $SEEDS; do
    for variant in host ours; do
        if [[ "$variant" == host ]]; then
            config="configs/codex_v5/AutoSeg3D_v5ab0_eval_scenenn.py"
            checkpoint="ckpts/v5ab0_epoch_96.pth"
        else
            config="configs/codex_v5/AutoSeg3D_v5ab1_eval_scenenn.py"
            checkpoint="ckpts/epoch_96.pth"
        fi

        log="$TRANSFER_OUTPUT_ROOT/logs/scenenn_${variant}_seed${seed}.log"
        work_dir="$TRANSFER_OUTPUT_ROOT/scenenn_${variant}_seed${seed}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SceneNN ${variant} seed=${seed}"
        env \
            PYTHONHASHSEED="$seed" \
            PYTHONPATH="$TRANSFER_REPO_ROOT" \
            OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}" \
            CUDA_VISIBLE_DEVICES="$GPU" \
            "$TRANSFER_PYTHON" tools/test.py "$config" "$checkpoint" \
            --work-dir "$work_dir" \
            --cfg-options "randomness.seed=${seed}" \
            >"$log" 2>&1

        metric="$(grep 'all_ap:' "$log" | tail -1 | tr '\t' ' ')"
        printf 'SceneNN\t%s\t%s\t%s\t%s\n' \
            "$variant" "$seed" "$metric" "$checkpoint" >>"$SUMMARY"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${metric}"
    done
done
