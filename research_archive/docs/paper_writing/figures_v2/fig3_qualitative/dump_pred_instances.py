#!/usr/bin/env python3
"""Dump per-scene predicted instances + per-scene AP, for the qualitative figure.

WHY A METRIC SUBCLASS AND NOT A VISUALIZATION HOOK
--------------------------------------------------
The tensors the paper's AP is computed from are the ones the metric consumes.
Dumping from a separate hook would re-derive predictions through a second code
path and could silently disagree with the reported numbers.  This subclass
writes them out on the way into the very same eval call, so a panel and its AP
cannot come from different predictions.

WHAT COMES OUT
--------------
  <out>/npz/<scene>.npz
      pred_label int16 (N,)   -1 = unclaimed, else an index into pred_score
      pred_score float32 (K,) confidence per predicted instance
      gt_inst    int16 (N,)   RAW GT instance id (not the evaluator's remap)
      gt_sem     int32 (N,)   RAW GT semantic id
      n_pred     int          K before overlap resolution
  <out>/per_scene_ap.json     {scene: {ap, ap50, ap25, n_pred, n_gt}}
  <out>/scene_info.json       {scene: dataset data_list entry (paths)}

Predicted masks MAY OVERLAP.  A render needs exactly one colour per point, so a
point is given to its highest-scoring claimant; iterating score-descending and
only writing where still unclaimed does that in one pass.  `n_pred` keeps the
true instance count so an instance that lost every one of its points to overlap
is still visible in the stats rather than silently vanishing.

xyz is deliberately NOT dumped here.  GT masks are per RAW point, while the
tensors flowing through the model are downsampled/transformed, so the only
trustworthy xyz is the raw point file -- which the renderer loads itself and
asserts against N.  Dumping a transformed xyz would look right and be wrong.

RUN
    python dump_pred_instances.py --config <dumped work_dir cfg> \
        --checkpoint <epoch_96.pth> --out <dir> [--stride 5]
"""

import argparse
import copy
import json
import os
import os.path as osp

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmdet3d.registry import DATASETS, METRICS

import oneformer3d  # noqa: F401  -- registers models, datasets, metrics
from oneformer3d.unified_metric import UnifiedSegMetric
from oneformer3d.instance_seg_eval import (instance_cat_agnostic_eval,
                                           instance_seg_eval)

_STATE = {'out': None, 'scenes': None}


@METRICS.register_module(force=True)
class DumpingUnifiedSegMetric(UnifiedSegMetric):
    """UnifiedSegMetric that also writes each scene's prediction to disk."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._idx = 0
        self._names = []
        self._announced = False

    # -- scene identity ---------------------------------------------------
    # data_samples reach a metric as plain dicts (Evaluator.process calls
    # to_dict()), so metainfo keys sit at the top level.  Which key carries the
    # scene differs between mmdet3d dataset classes, so try the plausible ones
    # and fall back to dataloader order -- deterministic here because test uses
    # batch_size=1 and a non-shuffling sampler.
    def _scene_of(self, ds):
        for k in ('lidar_path', 'pts_path', 'scan_id', 'scene_name',
                  'sample_idx', 'lidar_idx'):
            v = ds.get(k)
            if isinstance(v, str) and v:
                base = osp.basename(v)
                for suf in ('.bin', '.npy', '.pth', '.ply'):
                    base = base.replace(suf, '')
                if not self._announced:
                    print(f'[scene-id] using data_sample["{k}"] -> {base}',
                          flush=True)
                    self._announced = True
                return base
        seq = _STATE['scenes']
        if seq is not None and self._idx < len(seq):
            if not self._announced:
                print('[scene-id] no usable key in data_sample; falling back '
                      f'to dataloader order (keys were: {sorted(ds.keys())})',
                      flush=True)
                self._announced = True
            return seq[self._idx]
        return f'sample{self._idx:05d}'

    def process(self, data_batch, data_samples):
        out = _STATE['out']
        for ds in data_samples:
            scene = self._scene_of(ds)
            self._names.append(scene)
            self._idx += 1
            if out is not None:
                self._dump(out, scene, ds)
        super().process(data_batch, data_samples)

    @staticmethod
    def _dump(out, scene, ds):
        pred = ds['pred_pts_seg']
        masks = np.asarray(_np(pred['pts_instance_mask'][0]))
        scores = np.asarray(_np(pred['instance_scores']), dtype=np.float32)
        if masks.ndim != 2:
            raise RuntimeError(
                f'{scene}: expected (K,N) instance masks, got {masks.shape}')
        k_n, n_pt = masks.shape
        label = np.full(n_pt, -1, np.int16)
        for k in np.argsort(-scores):          # high score first ...
            m = masks[k].astype(bool)
            label[m & (label < 0)] = k         # ... so it wins every overlap
        ann = ds['eval_ann_info']
        np.savez_compressed(
            osp.join(out, 'npz', f'{scene}.npz'),
            pred_label=label,
            pred_score=scores,
            gt_inst=np.asarray(ann['pts_instance_mask']).astype(np.int16),
            gt_sem=np.asarray(ann['pts_semantic_mask']).astype(np.int32),
            n_pred=np.int32(k_n))

    # -- per-scene AP -----------------------------------------------------
    def compute_metrics(self, results):
        logger = None
        self.valid_class_ids = self.dataset_meta['seg_valid_class_ids']
        classes = self.metric_meta['classes']
        num_stuff = len(self.stuff_class_inds)
        per_scene = {}
        for i, (ann, pred) in enumerate(results):
            scene = self._names[i] if i < len(self._names) else f'i{i}'
            sem, inst = self.map_inst_markup(
                ann['pts_semantic_mask'].copy(),
                ann['pts_instance_mask'].copy(),
                self.valid_class_ids[num_stuff:], num_stuff)
            p_mask = [torch.tensor(pred['pts_instance_mask'][0])]
            p_lab = [torch.tensor(pred['instance_labels'])]
            p_sc = [torch.tensor(pred['instance_scores'])]
            fn = (instance_seg_eval if p_lab[0].max() > 0
                  else instance_cat_agnostic_eval)
            try:
                r = fn([sem], [inst], p_mask, p_lab, p_sc,
                       valid_class_ids=self.valid_class_ids[num_stuff:],
                       class_labels=classes[num_stuff:-1], logger=logger)
                per_scene[scene] = {
                    'ap': float(r['all_ap']),
                    'ap50': float(r['all_ap_50%']),
                    'ap25': float(r['all_ap_25%']),
                    'n_pred': int(len(p_sc[0])),
                    'n_gt': int(len(np.unique(inst[inst >= 0]))),
                }
            except Exception as e:                      # noqa: BLE001
                # A scene with no GT instance of a valid class makes AP
                # undefined.  Record it instead of aborting 60 scenes' worth of
                # work -- and keep the reason, so it is not read as a zero.
                per_scene[scene] = {'error': f'{type(e).__name__}: {e}'}
            if (i + 1) % 10 == 0:
                print(f'[per-scene AP] {i + 1}/{len(results)}', flush=True)
        with open(osp.join(_STATE['out'], 'per_scene_ap.json'), 'w') as f:
            json.dump(per_scene, f, indent=1, sort_keys=True)
        ok = [v['ap'] for v in per_scene.values() if 'ap' in v]
        print(f'[per-scene AP] {len(ok)} scenes ok, '
              f'mean of per-scene AP = {np.mean(ok) if ok else float("nan"):.4f} '
              '(NOT comparable to dataset AP -- different pooling)', flush=True)
        return super().compute_metrics(results)


def _np(v):
    return v.cpu().numpy() if hasattr(v, 'cpu') else v


def scene_key(info):
    for k in ('lidar_points', 'point_cloud'):
        d = info.get(k) or {}
        for kk in ('lidar_path', 'pts_path', 'lidar_idx'):
            v = d.get(kk)
            if isinstance(v, str) and v:
                return osp.basename(v).split('.')[0]
    for k in ('scan_id', 'scene_name', 'sample_idx', 'lidar_path', 'pts_path'):
        v = info.get(k)
        if isinstance(v, str) and v:
            return osp.basename(v).split('.')[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stride', type=int, default=0,
                    help='keep every Nth validation scene. Spreads the subset '
                         'over the whole scene-id range instead of taking a '
                         'prefix, at the same cost.')
    ap.add_argument('--scenes', default='',
                    help='comma list of scene names; overrides --stride')
    a = ap.parse_args()

    os.makedirs(osp.join(a.out, 'npz'), exist_ok=True)
    _STATE['out'] = a.out

    cfg = Config.fromfile(a.config)
    cfg.work_dir = osp.join(a.out, 'work')
    cfg.load_from = a.checkpoint
    cfg.resume = False
    for key in ('test_evaluator', 'val_evaluator'):
        if cfg.get(key) is not None:
            cfg[key]['type'] = 'DumpingUnifiedSegMetric'
    # Visualization hooks add cost and can write huge trees; not wanted here.
    if cfg.get('default_hooks', {}).get('visualization') is not None:
        cfg.default_hooks.visualization.draw = False

    # Subsetting has to happen in the CONFIG, before the runner exists: the
    # dataloader's sampler is sized at construction time, so calling
    # get_subset_ on an already-built dataset leaves the sampler indexing rows
    # that are no longer there.  So probe the dataset once to learn the scene
    # order, then hand the indices to the real build via cfg.
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    dcfg = copy.deepcopy(cfg.test_dataloader.dataset)
    while 'dataset' in dcfg and 'type' in dcfg.get('dataset', {}):
        dcfg = dcfg['dataset']              # unwrap Repeat/Concat style wrappers
    probe = DATASETS.build(copy.deepcopy(dcfg))
    names = [scene_key(probe.get_data_info(i)) or f'idx{i}'
             for i in range(len(probe))]
    print(f'[probe] {len(names)} scenes, first={names[:2]}', flush=True)

    if a.scenes:
        want = [s.strip() for s in a.scenes.split(',') if s.strip()]
        missing = [s for s in want if s not in names]
        if missing:
            raise SystemExit(f'not in the validation split: {missing}')
        idx = [names.index(s) for s in want]
    elif a.stride > 1:
        idx = list(range(0, len(names), a.stride))
    else:
        idx = None

    if idx is not None:
        target = cfg.test_dataloader.dataset
        while 'dataset' in target and 'type' in target.get('dataset', {}):
            target = target['dataset']
        target['indices'] = idx
        names = [names[i] for i in idx]
        print(f'[subset] {len(names)} of {len(probe)} scenes', flush=True)
    del probe
    _STATE['scenes'] = names

    runner = Runner.from_cfg(cfg)
    with open(osp.join(a.out, 'scene_order.json'), 'w') as f:
        json.dump(names, f, indent=1)

    runner.test()


if __name__ == '__main__':
    main()
