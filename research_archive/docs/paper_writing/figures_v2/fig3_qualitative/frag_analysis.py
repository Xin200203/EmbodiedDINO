#!/usr/bin/env python3
"""Rank scenes by how much VISIBLE fragmentation separates the two models.

A large AP delta does not have to be visible.  AP moves on score ranking and on
overlap near the threshold; a panel shows only one thing clearly -- whether an
object carries one colour or several.  So rank on that directly:

  split  : a GT object that at least two predicted instances each claim >= FRAC
           of.  This is over-segmentation, the signature of a merge that failed
           across frames, and it is what the eye sees.
  merged : a predicted instance that claims >= FRAC of at least two GT objects.
           Under-segmentation; visible as one colour spanning two things.

Structure, floor and unlabeled vertices are excluded exactly as the renderer
excludes them, so the count describes the panel that will actually be printed
rather than the raw scene.
"""

import argparse
import json
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from render_qualitative import (STRUCTURE_CLASSES, SEM_UNLABELED,  # noqa: E402
                                FLOOR_CLASS, class_tuple)


def counts(pred, gt, ids, frac):
    split = merged = 0
    per_gt = {}
    for g in ids:
        m = gt == g
        tot = m.sum()
        p, c = np.unique(pred[m], return_counts=True)
        keep = [(int(pi), int(ci)) for pi, ci in zip(p, c)
                if pi >= 0 and ci >= frac * tot]
        per_gt[int(g)] = len(keep)
        if len(keep) >= 2:
            split += 1
    for pi in np.unique(pred[pred >= 0]):
        m = pred == pi
        g, c = np.unique(gt[m], return_counts=True)
        hit = 0
        for gi, _ in zip(g, c):
            if gi < 0:
                continue
            if (m & (gt == gi)).sum() >= frac * (gt == gi).sum():
                hit += 1
        if hit >= 2:
            merged += 1
    return split, merged, per_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shortlist', required=True)
    ap.add_argument('--dump-host', required=True)
    ap.add_argument('--dump-ours', required=True)
    ap.add_argument('--frac', type=float, default=0.2)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    sl = json.load(open(a.shortlist))
    delta = {r['scene']: r['delta'] for r in sl['all']}
    names = class_tuple()
    idx = {n: i for i, n in enumerate(names)}
    struct = np.array([idx[n] for n in STRUCTURE_CLASSES if n in idx], np.int64)
    floor_i = idx[FLOOR_CLASS]

    rows = []
    for s in delta:
        fh, fo = (osp.join(d, 'npz', s + '.npz')
                  for d in (a.dump_host, a.dump_ours))
        if not (osp.exists(fh) and osp.exists(fo)):
            continue
        h, o = np.load(fh), np.load(fo)
        gi, gs = h['gt_inst'].astype(np.int64), h['gt_sem']
        shown = ~(np.isin(gs, struct) | (gs == floor_i) |
                  (gs == SEM_UNLABELED) | (gi < 0))
        g = np.where(shown, gi, -1)
        ids = np.unique(g[g >= 0])
        if len(ids) < 8:
            continue
        hs, hm, _ = counts(np.where(shown, h['pred_label'], -1), g, ids, a.frac)
        os_, om, _ = counts(np.where(shown, o['pred_label'], -1), g, ids, a.frac)
        rows.append({'scene': s, 'delta': round(delta[s], 4), 'n_obj': len(ids),
                     'split_host': hs, 'split_ours': os_,
                     'merge_host': hm, 'merge_ours': om,
                     'gain': (hs - os_) + (hm - om)})
    rows.sort(key=lambda r: (-r['gain'], -r['delta']))
    print(f'{"scene":<15}{"dAP":>8}{"objs":>5}{"splitH":>7}{"splitO":>7}'
          f'{"mergeH":>7}{"mergeO":>7}{"gain":>6}')
    for r in rows[:20]:
        print(f'{r["scene"]:<15}{r["delta"]:>+8.4f}{r["n_obj"]:>5}'
              f'{r["split_host"]:>7}{r["split_ours"]:>7}'
              f'{r["merge_host"]:>7}{r["merge_ours"]:>7}{r["gain"]:>+6}')
    if a.out:
        json.dump(rows, open(a.out, 'w'), indent=1)
        print(f'\n[write] {a.out}')


if __name__ == '__main__':
    main()
