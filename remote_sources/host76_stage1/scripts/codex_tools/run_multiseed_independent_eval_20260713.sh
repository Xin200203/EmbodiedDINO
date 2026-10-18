#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/datadisk1/xxy/AutoSeg3D_v5_stage1_clean_20260604}"
WORK_ROOT="${WORK_ROOT:-/datadisk3/xxy/work_dirs/autoseg3d_v5_stage1_clean_20260619}"
PYTHON_BIN="${PYTHON_BIN:-/datadisk1/xxy/miniconda3/envs/ESAM/bin/python}"
STAMP="20260713"

cd "${REPO}"

launch_eval() {
    local name="$1"
    local gpu="$2"
    local config="$3"
    local checkpoint="$4"
    local out_dir="${WORK_ROOT}/eval_${name}_independent_${STAMP}"

    mkdir -p "${out_dir}"
    if grep -q "Epoch(test) \[312/312\].*all_ap:" "${out_dir}/run.log" 2>/dev/null; then
        echo "[skip] ${name}: completed result already exists"
        return
    fi
    if [[ -f "${out_dir}/pid" ]] && kill -0 "$(cat "${out_dir}/pid")" 2>/dev/null; then
        echo "[skip] ${name}: PID $(cat "${out_dir}/pid") is already running"
        return
    fi

    nohup env \
        CUDA_VISIBLE_DEVICES="${gpu}" \
        PYTHONPATH="${REPO}:${PYTHONPATH:-}" \
        "${PYTHON_BIN}" "${REPO}/tools/test.py" \
        "${REPO}/configs/codex_v5/${config}" \
        "${checkpoint}" \
        --work-dir "${out_dir}" \
        >"${out_dir}/run.log" 2>&1 &
    echo "$!" >"${out_dir}/pid"
    echo "[launch] ${name}: GPU ${gpu}, PID $!, output ${out_dir}"
}

launch_eval \
    "v5ab0_seed7_e128" 1 \
    "AutoSeg3D_scannet200_stage1_v5ab0_autoseg_clean_sameinit_seed7.py" \
    "${WORK_ROOT}/v5ab0_autoseg_clean_stage1_seed7/epoch_128.pth"

launch_eval \
    "v5ab1_seed7_e128" 2 \
    "AutoSeg3D_scannet200_stage1_v5ab1_offline_dino_injection_only_seed7.py" \
    "${WORK_ROOT}/v5ab1_offline_dino_injection_only_stage1_seed7/epoch_128.pth"

launch_eval \
    "v5ab0_seed123_e128" 3 \
    "AutoSeg3D_scannet200_stage1_v5ab0_autoseg_clean_sameinit_seed123.py" \
    "${WORK_ROOT}/v5ab0_autoseg_clean_stage1_seed123/epoch_128.pth"

launch_eval \
    "v5ab1_seed123_e112" 5 \
    "AutoSeg3D_scannet200_stage1_v5ab1_offline_dino_injection_only_seed123.py" \
    "${WORK_ROOT}/v5ab1_offline_dino_injection_only_stage1_seed123/epoch_112.pth"
