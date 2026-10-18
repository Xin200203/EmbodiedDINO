#!/usr/bin/env python3
"""Run SAM automatic and FastSAM on identical sampled ScanNet frames and save proposal statistics."""
import argparse, csv, json, math, os, pickle, re, time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from ultralytics import FastSAM


def load_pkl(path):
    obj = pickle.load(open(path, 'rb'))
    if isinstance(obj, dict) and 'data_list' in obj:
        return obj['data_list']
    return obj


def choose_frame_indices(n, max_frames):
    if max_frames <= 0 or n <= max_frames:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, num=max_frames).round().astype(int).tolist()))


def load_rgb(path):
    return np.array(Image.open(path).convert('RGB'))


def filter_masks(masks, min_area=300):
    out=[]
    for m in masks:
        mb=np.asarray(m).astype(bool)
        area=int(mb.sum())
        if area>=min_area:
            out.append(mb)
    out.sort(key=lambda x:int(x.sum()), reverse=True)
    return out


def mask_stats(masks, hw):
    h,w=hw
    areas=np.array([int(m.sum()) for m in masks], dtype=np.float64)
    if len(masks)==0:
        return dict(num_masks=0, area_mean=0, area_median=0, area_p10=0, area_p90=0, small_lt1k=0, small_lt2k=0, coverage=0, overlap_pixels_mean=0, overlap_pixels_p95=0, overlap_pair_iou_mean=0, overlap_pair_iou_max=0, nested_pair_frac=0)
    stack=np.stack(masks, axis=0).astype(np.uint8)
    cover_count=stack.sum(axis=0)
    covered=cover_count>0
    pair_ious=[]; nested=0; pairs=0
    n=len(masks)
    # cap pair work for extreme frames but keep deterministic top-area masks.
    max_pair_n=min(n, 160)
    for i in range(max_pair_n):
        mi=masks[i]; ai=areas[i]
        for j in range(i+1, max_pair_n):
            mj=masks[j]; aj=areas[j]
            inter=int(np.logical_and(mi,mj).sum())
            if inter==0:
                continue
            union=ai+aj-inter
            pair_ious.append(float(inter/max(union,1)))
            if inter/max(min(ai,aj),1) > 0.8:
                nested += 1
            pairs += 1
    vals=cover_count[covered]
    return dict(
        num_masks=int(n), area_mean=float(areas.mean()), area_median=float(np.median(areas)), area_p10=float(np.percentile(areas,10)), area_p90=float(np.percentile(areas,90)),
        small_lt1k=float((areas<1000).mean()), small_lt2k=float((areas<2000).mean()), coverage=float(covered.mean()),
        overlap_pixels_mean=float(vals.mean()) if vals.size else 0, overlap_pixels_p95=float(np.percentile(vals,95)) if vals.size else 0,
        overlap_pair_iou_mean=float(np.mean(pair_ious)) if pair_ious else 0, overlap_pair_iou_max=float(np.max(pair_ious)) if pair_ious else 0,
        nested_pair_frac=float(nested/max(pairs,1)) if pairs else 0)


def parse_scene_frame(path):
    m=re.search(r'(scene\d{4}_\d{2}).*/(\d+)\.jpg', str(path))
    return (m.group(1), int(m.group(2))) if m else ('unknown', -1)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ann-file', default='/datadisk1/xxy/data/data-ESAM/scannet200-mv_fast/runs_mv/fastsam_everything_stride20_img640/scannet200_mv_oneformer3d_infos_val_subset50.pkl')
    ap.add_argument('--data-root', default='/datadisk1/xxy/data/data-ESAM/scannet200-mv_fast/runs_mv/fastsam_everything_stride20_img640')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--max-scenes', type=int, default=10)
    ap.add_argument('--max-frames-per-scene', type=int, default=2)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--sam-checkpoint', default='/datadisk1/xxy/ESAM/data/sam_vit_h_4b8939.pth')
    ap.add_argument('--fastsam-checkpoint', default='/datadisk1/xxy/data/data-ESAM/scannet200-mv_fast/FastSAM-x.pt')
    ap.add_argument('--min-area', type=int, default=300)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    data=load_pkl(args.ann_file)[:args.max_scenes]
    frames=[]
    for item in data:
        for i in choose_frame_indices(len(item['img_paths']), args.max_frames_per_scene):
            frames.append(os.path.join(args.data_root, item['img_paths'][i]))
    sam=sam_model_registry['vit_h'](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    sam_gen=SamAutomaticMaskGenerator(model=sam, points_per_side=32, points_per_batch=64, pred_iou_thresh=0.88, stability_score_thresh=0.95, min_mask_region_area=args.min_area, output_mode='binary_mask')
    fast=FastSAM(args.fastsam_checkpoint)
    rows=[]
    for idx,img_path in enumerate(frames):
        scene,frame_id=parse_scene_frame(img_path)
        rgb=load_rgb(img_path)
        t0=time.time(); anns=sam_gen.generate(rgb); sam_time=time.time()-t0
        sam_masks=filter_masks([a['segmentation'] for a in anns], min_area=args.min_area)
        t0=time.time(); res=fast(img_path, device=args.device, retina_masks=True, imgsz=640, conf=0.1, iou=0.9, verbose=False); fast_time=time.time()-t0
        if res and res[0].masks is not None:
            fast_masks=filter_masks(res[0].masks.data.detach().cpu().numpy().astype(bool), min_area=args.min_area)
        else:
            fast_masks=[]
        for model,masks,tm in [('sam_auto',sam_masks,sam_time),('fastsam_everything',fast_masks,fast_time)]:
            st=mask_stats(masks, rgb.shape[:2])
            st.update(model=model, scene=scene, frame_id=frame_id, image=img_path, runtime_sec=tm, index=idx)
            rows.append(st)
        print(f'[frame {idx+1}/{len(frames)}] {scene}/{frame_id} sam={len(sam_masks)} fast={len(fast_masks)} sam_time={sam_time:.2f}s fast_time={fast_time:.2f}s', flush=True)
    with open(out/'proposal_stats.csv','w',newline='') as f:
        fields=list(rows[0].keys()); w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    def summar(model):
        rs=[r for r in rows if r['model']==model]
        keys=['num_masks','area_median','area_mean','small_lt1k','small_lt2k','coverage','overlap_pixels_mean','overlap_pixels_p95','overlap_pair_iou_mean','nested_pair_frac','runtime_sec']
        d={'n':len(rs)}
        for k in keys:
            vals=[r[k] for r in rs]
            d[k+'_mean']=float(np.mean(vals)); d[k+'_median']=float(np.median(vals))
        return d
    pairs=[]
    for i in range(0,len(rows),2):
        s=rows[i]; f=rows[i+1]
        pairs.append(dict(scene=s['scene'], frame_id=s['frame_id'], sam_masks=s['num_masks'], fastsam_masks=f['num_masks'], ratio=s['num_masks']/max(f['num_masks'],1), sam_area_median=s['area_median'], fastsam_area_median=f['area_median'], sam_small_lt1k=s['small_lt1k'], fastsam_small_lt1k=f['small_lt1k'], sam_overlap=s['overlap_pixels_mean'], fastsam_overlap=f['overlap_pixels_mean']))
    with open(out/'sam_vs_fastsam_pairs.csv','w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(pairs[0].keys())); w.writeheader(); w.writerows(pairs)
    summary={'ann_file':args.ann_file,'data_root':args.data_root,'frames':len(frames),'sam_auto':summar('sam_auto'),'fastsam_everything':summar('fastsam_everything'), 'pair_ratio_mean':float(np.mean([p['ratio'] for p in pairs])), 'pair_ratio_median':float(np.median([p['ratio'] for p in pairs])), 'pairs':pairs}
    with open(out/'summary.json','w') as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
