#!/usr/bin/env python3
"""Vet shortlisted scenes for whether they can carry a readable panel.

A high AP delta does not make a good panel.  Two things kill one regardless:
  * ONE HUGE INSTANCE.  scene0458_00 has a +AP delta and renders as a single
    slab filling two thirds of the frame, because ScanNet labels a wall-sized
    surface as an object class.  `dominance` = the largest shown instance's
    share of shown object vertices; above ~0.35 the panel is that object.
  * NOTHING TO SEE.  Few shown instances, or most vertices unclaimed by either
    model, and the reader has no decomposition to compare.

Also prints the biggest shown instances by class name, which is how a structural
class that STRUCTURE_CLASSES still misses gets found.
"""

import argparse
import json
import os
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from render_qualitative import (STRUCTURE_CLASSES, SEM_UNLABELED,  # noqa: E402
                                FLOOR_CLASS, class_tuple)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shortlist', required=True)
    ap.add_argument('--dump-host', required=True)
    ap.add_argument('--dump-ours', required=True)
    ap.add_argument('--all', action='store_true', help='vet every ranked scene')
    ap.add_argument('--top-classes', type=int, default=4)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    sl = json.load(open(a.shortlist))
    scenes = ([r['scene'] for r in sl['all']] if a.all else sl['shortlist'])
    delta = {r['scene']: r['delta'] for r in sl['all']}
    names = class_tuple()
    idx = {n: i for i, n in enumerate(names)}
    struct = np.array([idx[n] for n in STRUCTURE_CLASSES if n in idx], np.int64)
    floor_i = idx[FLOOR_CLASS]

    rows = []
    for s in scenes:
        fh = osp.join(a.dump_host, 'npz', s + '.npz')
        fo = osp.join(a.dump_ours, 'npz', s + '.npz')
        if not (osp.exists(fh) and osp.exists(fo)):
            continue
        h, o = np.load(fh), np.load(fo)
        gi, gs = h['gt_inst'], h['gt_sem']
        shown = ~(np.isin(gs, struct) | (gs == floor_i) |
                  (gs == SEM_UNLABELED) | (gi < 0))
        if shown.sum() == 0:
            continue
        ids, cnt = np.unique(gi[shown], return_counts=True)
        order = np.argsort(-cnt)
        big = [(names[int(gs[gi == ids[j]][0])] if gs[gi == ids[j]].size else '?',
                round(float(cnt[j] / shown.sum()), 3)) for j in order[:a.top_classes]]
        rows.append({
            'scene': s, 'delta': round(delta.get(s, float('nan')), 4),
            'n_shown_inst': int(len(ids)),
            'dominance': round(float(cnt[order[0]] / shown.sum()), 3),
            'n_pred_host': int(h['n_pred']), 'n_pred_ours': int(o['n_pred']),
            'uncov_host': round(float((h['pred_label'][shown] < 0).mean()), 3),
            'uncov_ours': round(float((o['pred_label'][shown] < 0).mean()), 3),
            'biggest': big,
        })

    rows.sort(key=lambda r: (r['dominance'], -r['n_shown_inst']))
    print(f'{"scene":<15}{"dAP":>8}{"inst":>5}{"domin":>7}'
          f'{"predH":>6}{"predO":>6}{"uncH":>6}{"uncO":>6}  biggest shown')
    for r in rows:
        flag = ('  <-- ok' if r['dominance'] <= 0.35 and r['n_shown_inst'] >= 10
                else '')
        print(f'{r["scene"]:<15}{r["delta"]:>+8.4f}{r["n_shown_inst"]:>5}'
              f'{r["dominance"]:>7.3f}{r["n_pred_host"]:>6}{r["n_pred_ours"]:>6}'
              f'{r["uncov_host"]:>6.2f}{r["uncov_ours"]:>6.2f}  '
              f'{", ".join(f"{c}:{f}" for c, f in r["biggest"])}{flag}')
    if a.out:
        json.dump(rows, open(a.out, 'w'), indent=1)
        print(f'\n[write] {a.out}')


if __name__ == '__main__':
    main()
