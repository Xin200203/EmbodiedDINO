#!/usr/bin/env bash
while [ ! -f /datadisk1/xxy/AutoSeg3D_v5_stage1_clean_20260604/metrics/clip_cache_full_summary_20260707.json ]; do sleep 300; done
if grep -q '"num_failed": 0' /datadisk1/xxy/AutoSeg3D_v5_stage1_clean_20260604/metrics/clip_cache_full_summary_20260707.json; then
  touch /datadisk3/xxy/dino_cache/autoseg3d_v5_stage1_clean_20260604/scannet200_mv_fast/CLIP_CACHE_READY
fi
