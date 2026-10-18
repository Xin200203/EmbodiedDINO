#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${OWF_JOINT_REPO_ROOT:-/home/nebula/xxy/AutoSeg3D_v5_object_world_20260719}"
PYTHON_BIN="${OWF_JOINT_PYTHON:-/home/nebula/miniconda3/envs/ESAM/bin/python}"
CONFIG="${OWF_JOINT_CONFIG:-configs/codex_owf/AutoSeg3D_scannet200_stage1_v5cls1_owf_objectonly_joint.py}"
ENABLED_ROOT="${OWF_CLS198_ENABLED_ROOT:-/home/nebula/xxy/work_dirs/object_world_cls198_objectonly_20260724/seed2050318485_enabled_e128_b2a2}"
DISABLED_ROOT="${OWF_CLS198_DISABLED_ROOT:-/home/nebula/xxy/work_dirs/object_world_cls198_objectonly_20260724/seed2050318485_disabled_e128_b2a2_samehost184}"
INIT_CHECKPOINT="${OWF_JOINT_INIT_CHECKPOINT:-/home/nebula/xxy/3D_Reconstruction/work_dirs/ESAM_sv_scannet200_CA_dino/best_all_ap_50%_epoch_115.pth}"
WAIT_SECONDS="${OWF_CLS198_QUEUE_WAIT_SECONDS:-120}"

mkdir -p "$DISABLED_ROOT/logs"
exec 9>"$DISABLED_ROOT/queue.lock"
if ! flock -n 9; then
    echo "another disabled-run queue owns $DISABLED_ROOT" >&2
    exit 0
fi

while [[ ! -f "$ENABLED_ROOT/final_eval_e128_independent/eval.complete" ||
         ! -s "$ENABLED_ROOT/final_eval_e128_independent/metric.txt" ||
         ! -s "$ENABLED_ROOT/epoch_128.pth" ]]; do
    train_log="$ENABLED_ROOT/logs/train.log"
    eval_log="$ENABLED_ROOT/final_eval_e128_independent/logs/test.log"
    if [[ -s "$train_log" ]] && grep -Eq \
            'Traceback|CUDA out of memory|FileNotFoundError|loss: *nan|grad_norm: *nan' \
            "$train_log"; then
        echo "enabled training failed before queued control: $train_log" >&2
        exit 3
    fi
    if [[ -s "$eval_log" ]] && grep -q 'Traceback' "$eval_log"; then
        echo "enabled final evaluation failed before queued control: $eval_log" >&2
        exit 3
    fi
    sleep "$WAIT_SECONDS"
done

if [[ -f "$DISABLED_ROOT/train.complete" ]]; then
    exit 0
fi

exec env \
    OWF_JOINT_REPO_ROOT="$REPO_ROOT" \
    OWF_JOINT_PYTHON="$PYTHON_BIN" \
    OWF_JOINT_OUTPUT_ROOT="$DISABLED_ROOT" \
    OWF_JOINT_CONFIG="$CONFIG" \
    OWF_JOINT_GPU=0 \
    OWF_JOINT_SEED=2050318485 \
    OWF_JOINT_OUTPUT_SCALE=0.0 \
    OWF_JOINT_MAX_EPOCHS=128 \
    OWF_JOINT_VAL_INTERVAL=16 \
    OWF_JOINT_BATCH_SIZE=2 \
    OWF_JOINT_ACCUMULATIVE_COUNTS=2 \
    OWF_JOINT_RESUME=0 \
    OWF_JOINT_INIT_CHECKPOINT="$INIT_CHECKPOINT" \
    bash "$REPO_ROOT/tools/run_owf_cls198_joint_stage1.sh"
