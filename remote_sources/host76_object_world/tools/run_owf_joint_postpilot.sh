#!/usr/bin/env bash

set -euo pipefail

: "${OWF_JOINT_REPO_ROOT:?set OWF_JOINT_REPO_ROOT}"
: "${OWF_JOINT_PYTHON:?set OWF_JOINT_PYTHON}"
: "${OWF_JOINT_OUTPUT_ROOT:?set OWF_JOINT_OUTPUT_ROOT}"

CONFIG="${OWF_JOINT_CONFIG:-configs/codex_owf/AutoSeg3D_scannet200_stage1_v5ab1_owf_joint.py}"
ANALYZER="${OWF_JOINT_ANALYZER:-$OWF_JOINT_REPO_ROOT/tools/analyze_representation_dumps.py}"
TRAIN_ROOT="${OWF_JOINT_TRAIN_ROOT:-/datadisk3/xxy/work_dirs/object_world_joint_20260722}"
SEED="${OWF_JOINT_SEED:-2050318485}"
GPU="${OWF_JOINT_GPU:-6}"
WAIT_SECONDS="${OWF_JOINT_WAIT_SECONDS:-120}"
FULL_ANN="${OWF_JOINT_FULL_ANN:-scannet200_mv_oneformer3d_infos_val.pkl}"
REPR_ANN="${OWF_JOINT_REPR_ANN:-scannet200_mv_oneformer3d_infos_val_subset50.pkl}"

ENABLED_DIR="$TRAIN_ROOT/seed${SEED}_enabled_e16_b2a2"
DISABLED_DIR="$TRAIN_ROOT/seed${SEED}_disabled_e16_b2a2"
ENABLED_CHECKPOINT="$ENABLED_DIR/epoch_16.pth"
DISABLED_CHECKPOINT="$DISABLED_DIR/epoch_16.pth"

mkdir -p "$OWF_JOINT_OUTPUT_ROOT/logs" "$OWF_JOINT_OUTPUT_ROOT/fullval"

while [[ ! -f "$ENABLED_DIR/train.complete" || \
         ! -s "$ENABLED_CHECKPOINT" || \
         ! -f "$DISABLED_DIR/train.complete" || \
         ! -s "$DISABLED_CHECKPOINT" ]]; do
    for log in "$ENABLED_DIR/logs/train.log" "$DISABLED_DIR/logs/train.log"; do
        if [[ -s "$log" ]] && grep -Eq \
                'CUDA out of memory|Traceback|loss: *nan|grad_norm: *nan' "$log"; then
            echo "pilot failed before post-pilot analysis: $log" >&2
            exit 3
        fi
    done
    printf '[%s] waiting for paired epoch-16 checkpoints\n' \
        "$(date --iso-8601=seconds)"
    sleep "$WAIT_SECONDS"
done

cd "$OWF_JOINT_REPO_ROOT"
for required in "$CONFIG" "$ANALYZER" "$ENABLED_CHECKPOINT" \
        "$DISABLED_CHECKPOINT" oneformer3d/mink_unet.py \
        oneformer3d/mixformer3d.py; do
    if [[ ! -s "$required" ]]; then
        echo "missing required artifact: $required" >&2
        exit 2
    fi
done

{
    printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
    printf 'host=%s\nrepo=%s\nconfig=%s\nseed=%s\ngpu=%s\n' \
        "$(hostname)" "$OWF_JOINT_REPO_ROOT" "$CONFIG" "$SEED" "$GPU"
    printf 'full_ann=%s\nrepr_ann=%s\n' "$FULL_ANN" "$REPR_ANN"
    sha256sum "$CONFIG" "$ANALYZER" "$ENABLED_CHECKPOINT" \
        "$DISABLED_CHECKPOINT" oneformer3d/mink_unet.py \
        oneformer3d/mixformer3d.py
} >"$OWF_JOINT_OUTPUT_ROOT/run_manifest.txt"

run_test() {
    local variant="$1"
    local checkpoint="$2"
    local mode="$3"
    local scale="$4"
    local ann_file="$5"
    local work_dir="$6"
    local log="$7"
    shift 7

    env \
        PYTHONUNBUFFERED=1 \
        PYTHONHASHSEED="$SEED" \
        PYTHONPATH="$OWF_JOINT_REPO_ROOT" \
        OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}" \
        CUDA_VISIBLE_DEVICES="$GPU" \
        OWF_JOINT_SEED="$SEED" \
        OWF_JOINT_MODE="$mode" \
        OWF_JOINT_OUTPUT_SCALE="$scale" \
        OWF_JOINT_MAX_EPOCHS=16 \
        OWF_JOINT_VAL_INTERVAL=16 \
        OWF_JOINT_BATCH_SIZE=2 \
        OWF_JOINT_ACCUMULATIVE_COUNTS=2 \
        OWF_JOINT_WORK_DIR="$work_dir" \
        "$@" \
        "$OWF_JOINT_PYTHON" tools/test.py "$CONFIG" "$checkpoint" \
        --work-dir "$work_dir" \
        --cfg-options \
            "randomness.seed=$SEED" \
            "test_dataloader.dataset.ann_file=$ann_file" \
            "test_dataloader.dataset.pipeline.6.keys=['points','sp_pts_mask','pts_instance_mask','img_paths','dino_feat_maps','cam_info','dino_cache_paths']" \
        >"$log" 2>&1

    local metric
    metric="$(grep 'all_ap:' "$log" | tail -n 1 | tr '\t' ' ')"
    if [[ -z "$metric" ]]; then
        echo "missing metric for $variant in $log" >&2
        return 4
    fi
    printf '%s\n' "$metric"
}

SUMMARY="$OWF_JOINT_OUTPUT_ROOT/fullval_summary.tsv"
printf 'variant\tcheckpoint\tmode\tscale\tmetric_line\n' >"$SUMMARY"

FULL_NAMES=(disabled_no_write enabled_no_write enabled_aligned enabled_shuffled)
FULL_CHECKPOINTS=(
    "$DISABLED_CHECKPOINT" "$ENABLED_CHECKPOINT"
    "$ENABLED_CHECKPOINT" "$ENABLED_CHECKPOINT")
FULL_MODES=(real real real shuffled)
FULL_SCALES=(0.0 0.0 1.0 1.0)

for index in "${!FULL_NAMES[@]}"; do
    name="${FULL_NAMES[$index]}"
    checkpoint="${FULL_CHECKPOINTS[$index]}"
    mode="${FULL_MODES[$index]}"
    scale="${FULL_SCALES[$index]}"
    work_dir="$OWF_JOINT_OUTPUT_ROOT/fullval/$name"
    log="$OWF_JOINT_OUTPUT_ROOT/logs/fullval_${name}.log"
    done_file="$OWF_JOINT_OUTPUT_ROOT/fullval_${name}.complete"
    if [[ -s "$done_file" ]]; then
        metric="$(cat "$done_file")"
    else
        mkdir -p "$work_dir"
        echo "[$(date --iso-8601=seconds)] fullval $name"
        metric="$(run_test \
            "$name" "$checkpoint" "$mode" "$scale" "$FULL_ANN" \
            "$work_dir" "$log")"
        printf '%s\n' "$metric" >"$done_file"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$checkpoint" "$mode" "$scale" "$metric" >>"$SUMMARY"
done

REPR_ROOT="$OWF_JOINT_OUTPUT_ROOT/representation"
mkdir -p "$REPR_ROOT/logs" "$REPR_ROOT/dumps"
REPR_NAMES=(ocwe_no_write ocwe_aligned ocwe_shuffled)
REPR_MODES=(real real shuffled)
REPR_SCALES=(0.0 1.0 1.0)

for index in "${!REPR_NAMES[@]}"; do
    name="${REPR_NAMES[$index]}"
    mode="${REPR_MODES[$index]}"
    scale="${REPR_SCALES[$index]}"
    work_dir="$REPR_ROOT/eval_$name"
    dump_dir="$REPR_ROOT/dumps/$name"
    log="$REPR_ROOT/logs/$name.log"
    done_file="$REPR_ROOT/$name.complete"
    if [[ ! -s "$done_file" ]]; then
        mkdir -p "$work_dir" "$dump_dir"
        echo "[$(date --iso-8601=seconds)] representation $name"
        metric="$(run_test \
            "$name" "$ENABLED_CHECKPOINT" "$mode" "$scale" "$REPR_ANN" \
            "$work_dir" "$log" \
            OWF_REPR_DUMP_DIR="$dump_dir" \
            OWF_REPR_MAX_SCENES=50 \
            OWF_REPR_MAX_FRAMES=1000 \
            OWF_REPR_FRAME_STRIDE=4 \
            OWF_REPR_MAX_DUMPS_PER_VARIANT=150 \
            OWF_REPR_MAX_DUMPS_PER_SCENE=3 \
            OWF_REPR_INCLUDE_POINT_FEATURES=0 \
            OWF_REPR_INCLUDE_DINO_FEATURES=0)"
        printf '%s\n' "$metric" >"$done_file"
    fi
done

mkdir -p "$REPR_ROOT/analysis"
"$OWF_JOINT_PYTHON" "$ANALYZER" \
    --dump "ocwe_no_write=$REPR_ROOT/dumps/ocwe_no_write" \
    --dump "ocwe_aligned=$REPR_ROOT/dumps/ocwe_aligned" \
    --dump "ocwe_shuffled=$REPR_ROOT/dumps/ocwe_shuffled" \
    --output-dir "$REPR_ROOT/analysis" \
    --require-min-frames-per-variant 150 \
    --require-min-scenes-per-variant 50 \
    --require-gt-every-frame \
    --require-retrieval-queries \
    >"$REPR_ROOT/analysis.log" 2>&1

touch "$OWF_JOINT_OUTPUT_ROOT/postpilot.complete"
echo "[$(date --iso-8601=seconds)] post-pilot analysis complete"
