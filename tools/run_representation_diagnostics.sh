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
: "${OWF_DINO_CACHE:?set OWF_DINO_CACHE}"
: "${OWF_HOST_CHECKPOINT:?set OWF_HOST_CHECKPOINT}"
: "${OWF_OURS_CHECKPOINT:?set OWF_OURS_CHECKPOINT}"
: "${OWF_OCWE_CHECKPOINT:?set OWF_OCWE_CHECKPOINT}"
: "${OWF_OUTPUT_ROOT:?set OWF_OUTPUT_ROOT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYZER_SCRIPT="${OWF_ANALYZER_SCRIPT:-$SCRIPT_DIR/analyze_representation_dumps.py}"
GPU="${OWF_GPU:-1}"
GPU_LIST_RAW="${OWF_GPU_LIST:-$GPU}"
SEED="${OWF_SEED:-2050318485}"
ANN_FILE="${OWF_ANN_FILE:-scannet200_mv_oneformer3d_infos_val_subset50.pkl}"
HOST_CONFIG="${OWF_HOST_CONFIG:-configs/codex_v5/AutoSeg3D_scannet200_stage1_v5ab0_autoseg_clean_sameinit.py}"
OURS_CONFIG="${OWF_OURS_CONFIG:-configs/codex_v5/AutoSeg3D_scannet200_stage1_v5ab1_offline_dino_injection_only.py}"
OCWE_CONFIG="${OWF_OCWE_CONFIG:-configs/codex_owf/AutoSeg3D_scannet200_stage1_v5ab1_owf_subset50.py}"
MAX_SCENES="${OWF_REPR_MAX_SCENES:-50}"
MAX_FRAMES="${OWF_REPR_MAX_FRAMES:-1000}"
FRAME_STRIDE="${OWF_REPR_FRAME_STRIDE:-4}"
MAX_DUMPS="${OWF_REPR_MAX_DUMPS_PER_VARIANT:-150}"
MAX_DUMPS_PER_SCENE="${OWF_REPR_MAX_DUMPS_PER_SCENE:-3}"
REQUIRE_VALID="${OWF_REPR_REQUIRE_VALID:-1}"

mkdir -p "$OWF_OUTPUT_ROOT/logs" "$OWF_OUTPUT_ROOT/dumps"
SUMMARY="$OWF_OUTPUT_ROOT/eval_summary.tsv"
printf 'variant\tmetric_line\tcheckpoint\n' >"$SUMMARY"

resolve_repo_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s/%s\n' "$OWF_REPO_ROOT" "$path"
    fi
}

file_fingerprint() {
    local path="$1"
    if [[ -f "$path" ]]; then
        stat --printf='%n\t%s\t%Y\n' "$path"
    else
        printf '%s\tmissing\n' "$path"
    fi
}

run_variant() {
    local name="$1"
    local config="$2"
    local checkpoint="$3"
    local mode="$4"
    local output_scale="$5"
    local gpu="$6"
    local pack_index="$7"
    local pack_keys="$8"
    local work_dir="$OWF_OUTPUT_ROOT/eval_${name}"
    local dump_dir="$OWF_OUTPUT_ROOT/dumps/${name}"
    local log="$OWF_OUTPUT_ROOT/logs/${name}.log"
    local done_file="$OWF_OUTPUT_ROOT/${name}.complete"
    local signature_file="$OWF_OUTPUT_ROOT/${name}.signature"
    local expected_signature="$OWF_OUTPUT_ROOT/${name}.signature.expected"
    local config_path
    config_path="$(resolve_repo_path "$config")"

    {
        printf 'variant=%s\nmode=%s\noutput_scale=%s\nseed=%s\n' \
            "$name" "$mode" "$output_scale" "$SEED"
        printf 'ann_file=%s\nmax_scenes=%s\nmax_frames=%s\nframe_stride=%s\n' \
            "$ANN_FILE" "$MAX_SCENES" "$MAX_FRAMES" "$FRAME_STRIDE"
        printf 'dino_cache=%s\n' "$OWF_DINO_CACHE"
        printf 'max_dumps=%s\nmax_dumps_per_scene=%s\n' \
            "$MAX_DUMPS" "$MAX_DUMPS_PER_SCENE"
        printf 'diagnostic_pack_index=%s\ndiagnostic_pack_keys=%s\n' \
            "$pack_index" "$pack_keys"
        file_fingerprint "$config_path"
        file_fingerprint "$checkpoint"
        file_fingerprint "$OWF_REPO_ROOT/oneformer3d/mink_unet.py"
        file_fingerprint "$OWF_REPO_ROOT/oneformer3d/mixformer3d.py"
    } >"$expected_signature"

    if [[ -s "$done_file" && -s "$signature_file" && \
          -n "$(find "$dump_dir" -name '*.pt' -print -quit 2>/dev/null)" && \
          -z "$(cmp -s "$expected_signature" "$signature_file" || printf mismatch)" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] skip completed ${name}"
    else
        if [[ -e "$dump_dir" || -e "$work_dir" || -e "$done_file" ]]; then
            local stale_dir="$OWF_OUTPUT_ROOT/stale/$(date '+%Y%m%d_%H%M%S')_${name}"
            mkdir -p "$stale_dir"
            [[ -e "$dump_dir" ]] && mv "$dump_dir" "$stale_dir/dumps"
            [[ -e "$work_dir" ]] && mv "$work_dir" "$stale_dir/work_dir"
            [[ -e "$done_file" ]] && mv "$done_file" "$stale_dir/complete"
            [[ -e "$signature_file" ]] && mv "$signature_file" "$stale_dir/signature"
        fi
        mkdir -p "$work_dir" "$dump_dir"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] representation eval ${name}"
        (
            cd "$OWF_REPO_ROOT"
            env \
                PYTHONUNBUFFERED=1 \
                PYTHONPATH="$OWF_REPO_ROOT" \
                OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}" \
                CUDA_VISIBLE_DEVICES="$gpu" \
                PYTHONHASHSEED="$SEED" \
                OWF_MODE="$mode" \
                OWF_PLACEMENT=pre_memory \
                OWF_INCLUDE_DINO=1 \
                OWF_INCLUDE_S2=0 \
                OWF_INCLUDE_XYZ=0 \
                OWF_ORACLE_PURITY=0 \
                OWF_PERSISTENT_ORACLE=0 \
                OWF_OUTPUT_SCALE="$output_scale" \
                OWF_INIT_GATE=0.05 \
                OWF_LR=0.001 \
                OWF_MAX_EPOCHS=4 \
                OWF_VAL_INTERVAL=4 \
                OWF_DINO_CACHE="$OWF_DINO_CACHE" \
                OWF_REPR_DUMP_DIR="$dump_dir" \
                OWF_REPR_MAX_SCENES="$MAX_SCENES" \
                OWF_REPR_MAX_FRAMES="$MAX_FRAMES" \
                OWF_REPR_FRAME_STRIDE="$FRAME_STRIDE" \
                OWF_REPR_MAX_DUMPS_PER_VARIANT="$MAX_DUMPS" \
                OWF_REPR_MAX_DUMPS_PER_SCENE="$MAX_DUMPS_PER_SCENE" \
                OWF_REPR_INCLUDE_POINT_FEATURES=0 \
                OWF_REPR_INCLUDE_DINO_FEATURES=0 \
                "$OWF_PYTHON" tools/test.py "$config_path" "$checkpoint" \
                --work-dir "$work_dir" \
                --cfg-options \
                    "randomness.seed=${SEED}" \
                    "test_dataloader.dataset.ann_file=${ANN_FILE}" \
                    "test_dataloader.dataset.pipeline.${pack_index}.keys=${pack_keys}"
        ) >"$log" 2>&1
        grep 'all_ap:' "$log" | tail -1 >"$done_file"
        cp "$expected_signature" "$signature_file"
    fi
    rm -f "$expected_signature"
    metric="$(cat "$done_file")"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${name}: ${metric}"
}

IFS=',' read -r -a GPUS <<<"$GPU_LIST_RAW"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
    echo "OWF_GPU_LIST did not contain a GPU" >&2
    exit 2
fi

NAMES=(host embodieddino ocwe_no_write ocwe_aligned ocwe_shuffled)
CONFIGS=("$HOST_CONFIG" "$OURS_CONFIG" "$OCWE_CONFIG" "$OCWE_CONFIG" "$OCWE_CONFIG")
CHECKPOINTS=(
    "$OWF_HOST_CHECKPOINT" "$OWF_OURS_CHECKPOINT" "$OWF_OCWE_CHECKPOINT"
    "$OWF_OCWE_CHECKPOINT" "$OWF_OCWE_CHECKPOINT")
MODES=(real real real real shuffled)
SCALES=(0.0 0.0 0.0 1.0 1.0)
PACK_INDEXES=(4 6 6 6 6)
PACK_KEYS=(
    "['points','sp_pts_mask','pts_instance_mask']"
    "['points','sp_pts_mask','pts_instance_mask','img_paths','dino_feat_maps','cam_info','dino_cache_paths']"
    "['points','sp_pts_mask','pts_instance_mask','img_paths','dino_feat_maps','cam_info','dino_cache_paths']"
    "['points','sp_pts_mask','pts_instance_mask','img_paths','dino_feat_maps','cam_info','dino_cache_paths']"
    "['points','sp_pts_mask','pts_instance_mask','img_paths','dino_feat_maps','cam_info','dino_cache_paths']")

for ((start = 0; start < ${#NAMES[@]}; start += ${#GPUS[@]})); do
    pids=()
    batch_names=()
    for ((slot = 0; slot < ${#GPUS[@]}; slot++)); do
        index=$((start + slot))
        if ((index >= ${#NAMES[@]})); then
            break
        fi
        gpu="${GPUS[$slot]//[[:space:]]/}"
        if [[ -z "$gpu" ]]; then
            echo "empty GPU entry in OWF_GPU_LIST=${GPU_LIST_RAW}" >&2
            exit 2
        fi
        run_variant \
            "${NAMES[$index]}" "${CONFIGS[$index]}" \
            "${CHECKPOINTS[$index]}" "${MODES[$index]}" \
            "${SCALES[$index]}" "$gpu" \
            "${PACK_INDEXES[$index]}" "${PACK_KEYS[$index]}" &
        pids+=("$!")
        batch_names+=("${NAMES[$index]}")
    done
    batch_failed=0
    for ((slot = 0; slot < ${#pids[@]}; slot++)); do
        if ! wait "${pids[$slot]}"; then
            echo "representation eval failed: ${batch_names[$slot]}" >&2
            batch_failed=1
        fi
    done
    if ((batch_failed)); then
        exit 1
    fi
done

printf 'variant\tmetric_line\tcheckpoint\n' >"$SUMMARY"
for ((index = 0; index < ${#NAMES[@]}; index++)); do
    metric="$(cat "$OWF_OUTPUT_ROOT/${NAMES[$index]}.complete")"
    printf '%s\t%s\t%s\n' \
        "${NAMES[$index]}" "$metric" "${CHECKPOINTS[$index]}" >>"$SUMMARY"
done

ANALYZER_VALIDATION_ARGS=()
if [[ "$REQUIRE_VALID" == "1" ]]; then
    ANALYZER_VALIDATION_ARGS=(
        --require-min-frames-per-variant "$MAX_DUMPS_PER_SCENE"
        --require-min-scenes-per-variant "$MAX_SCENES"
        --require-gt-every-frame
        --require-retrieval-queries)
fi

"$OWF_PYTHON" "$ANALYZER_SCRIPT" \
    --dump "host=$OWF_OUTPUT_ROOT/dumps/host" \
    --dump "embodieddino=$OWF_OUTPUT_ROOT/dumps/embodieddino" \
    --dump "ocwe_no_write=$OWF_OUTPUT_ROOT/dumps/ocwe_no_write" \
    --dump "ocwe_aligned=$OWF_OUTPUT_ROOT/dumps/ocwe_aligned" \
    --dump "ocwe_shuffled=$OWF_OUTPUT_ROOT/dumps/ocwe_shuffled" \
    --output-dir "$OWF_OUTPUT_ROOT/analysis" \
    "${ANALYZER_VALIDATION_ARGS[@]}" \
    >"$OWF_OUTPUT_ROOT/analysis.log" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] representation analysis complete"
