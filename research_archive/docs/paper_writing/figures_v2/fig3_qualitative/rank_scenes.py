#!/usr/bin/env python3
"""Rank validation scenes by how much EmbodiedDINO gains over the matched host.

WHY RANKING AND NOT PICKING BY EYE
----------------------------------
A qualitative figure chosen by browsing renders is chosen by the author's taste,
which is unreportable.  Ranking by per-scene AP difference is a rule that can be
written into the caption, so a reader knows exactly how the scenes were selected
and can discount accordingly.  It is still selection -- it must be DISCLOSED,
not presented as a random sample.

The shortlist deliberately mixes ranks: the top of the ranking shows the effect
at its clearest, and a few median scenes show what the typical case looks like.
A figure built only from the top would be the strongest possible misreading of
an average 5.7-point gain.
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', required=True, help='host per_scene_ap.json')
    ap.add_argument('--ours', required=True, help='ours per_scene_ap.json')
    ap.add_argument('--top', type=int, default=8)
    ap.add_argument('--median', type=int, default=4)
    ap.add_argument('--min-gt', type=int, default=8,
                    help='skip scenes with very few GT instances: a 3-object '
                         'scene can show a big AP delta and still look empty')
    ap.add_argument('--out', default='shortlist.json')
    a = ap.parse_args()

    H = json.load(open(a.host))
    O = json.load(open(a.ours))
    rows, skipped = [], {'error': 0, 'few_gt': 0, 'missing': 0}
    for s, h in H.items():
        o = O.get(s)
        if o is None:
            skipped['missing'] += 1
            continue
        if 'ap' not in h or 'ap' not in o:
            skipped['error'] += 1
            continue
        if h.get('n_gt', 0) < a.min_gt:
            skipped['few_gt'] += 1
            continue
        rows.append({
            'scene': s, 'ap_host': h['ap'], 'ap_ours': o['ap'],
            'delta': o['ap'] - h['ap'], 'n_gt': h['n_gt'],
            'n_pred_host': h['n_pred'], 'n_pred_ours': o['n_pred'],
        })
    rows.sort(key=lambda r: -r['delta'])
    print(f'{len(rows)} comparable scenes; skipped {skipped}')
    d = [r['delta'] for r in rows]
    print(f'delta AP: mean {sum(d) / len(d):+.4f}  '
          f'median {sorted(d)[len(d) // 2]:+.4f}  '
          f'ours wins {sum(x > 0 for x in d)}/{len(d)}')

    mid = len(rows) // 2
    pick = rows[:a.top] + rows[mid:mid + a.median]
    print(f'\n{"rank":>4} {"scene":<16} {"host":>7} {"ours":>7} {"delta":>7} '
          f'{"n_gt":>5} {"pred h/o":>10}')
    for r in pick:
        i = rows.index(r)
        tag = 'TOP' if i < a.top else 'MED'
        print(f'{i:>4} {r["scene"]:<16} {r["ap_host"]:>7.4f} {r["ap_ours"]:>7.4f} '
              f'{r["delta"]:>+7.4f} {r["n_gt"]:>5} '
              f'{r["n_pred_host"]:>4}/{r["n_pred_ours"]:<4} {tag}')

    json.dump({'all': rows, 'shortlist': [r['scene'] for r in pick],
               'rule': f'ranked by per-scene class-agnostic AP(ours) - AP(host); '
                       f'top {a.top} plus {a.median} scenes at the median rank; '
                       f'scenes with fewer than {a.min_gt} GT instances excluded'},
              open(a.out, 'w'), indent=1)
    print(f'\n[write] {a.out}')


if __name__ == '__main__':
    main()
