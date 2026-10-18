#!/usr/bin/env python3
"""
S2 step 0 -- measure the per-scene frame-count distribution.

Why this must run FIRST: the truncation hook in
`ESAM/oneformer3d/loading.py:387` is guarded by

    if self.max_frames > 0 and len(pts_file_paths) > self.max_frames:

so a scene with fewer frames than `max_frames` is left untouched.  Pick the four
K values without knowing this distribution and you can easily choose a K that
truncates almost nothing, producing a flat curve that means nothing.

Run this on a machine that has the validation pkl (the repo checkout carries only
the preprocessing scripts, no data).

    python3 frame_count_histogram.py --pkl <path to *_infos_val.pkl>

Only the standard library is used, so no mmengine/mmdet3d import is required.
The pkl schema is auto-detected rather than assumed.
"""

import argparse
import pickle
import sys
from collections import Counter


def quantiles(xs, qs=(0, 5, 10, 25, 50, 75, 90, 95, 100)):
    s = sorted(xs)
    n = len(s)
    out = {}
    for q in qs:
        if n == 1:
            out[q] = s[0]
            continue
        pos = (q / 100) * (n - 1)
        lo, hi = int(pos), min(int(pos) + 1, n - 1)
        frac = pos - lo
        out[q] = s[lo] * (1 - frac) + s[hi] * frac
    return out


def find_frame_lists(item):
    """Return {key: length} for every value that looks like a per-frame list."""
    hits = {}

    def walk(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(obj, (list, tuple)):
            if obj and all(isinstance(e, str) for e in obj):
                hits[path] = len(obj)
            elif obj and all(isinstance(e, dict) for e in obj):
                hits[path] = len(obj)
    walk(item, "")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--key", default=None,
                    help="explicit frame-list key, e.g. 'pts_paths'; "
                         "omit to auto-detect the longest candidate")
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    items = data["data_list"] if isinstance(data, dict) and "data_list" in data else data
    if not isinstance(items, list) or not items:
        sys.exit(f"unexpected pkl structure: top-level type {type(data)}")

    print(f"scenes in pkl: {len(items)}")
    cands = find_frame_lists(items[0])
    if not cands:
        sys.exit("no per-frame list found in the first item; inspect the schema manually")

    print("\ncandidate per-frame fields (from the first scene):")
    for k, n in sorted(cands.items(), key=lambda kv: -kv[1]):
        print(f"  {k:44s} len={n}")

    key = args.key or max(cands, key=cands.get)
    print(f"\nusing key: {key}")

    def get(item, dotted):
        cur = item
        for part in dotted.split("."):
            if part:
                cur = cur[part]
        return cur

    counts = []
    for it in items:
        try:
            counts.append(len(get(it, key)))
        except (KeyError, TypeError):
            pass

    if not counts:
        sys.exit(f"key {key!r} not resolvable across scenes")

    print(f"scenes with a resolvable frame list: {len(counts)}")
    print(f"total frames: {sum(counts)}   mean: {sum(counts)/len(counts):.1f}")
    print("\nquantiles of per-scene frame count:")
    for q, v in quantiles(counts).items():
        print(f"  p{q:<3d} {v:8.1f}")

    print("\nhow many scenes each candidate K would actually truncate:")
    print(f"  {'K':>6}  {'scenes truncated':>17}  {'fraction':>9}  {'frames kept':>12}")
    for k in (5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200):
        trunc = sum(1 for c in counts if c > k)
        kept = sum(min(c, k) for c in counts)
        print(f"  {k:>6}  {trunc:>17}  {trunc/len(counts):>8.1%}  {kept:>12}")
    print(f"  {'all':>6}  {0:>17}  {0.0:>8.1%}  {sum(counts):>12}")

    print("\nPick four K so that EVERY K truncates a large majority of scenes and")
    print("the largest K is close to no truncation.  A K that truncates <50% of")
    print("scenes is not a useful point on the curve.")

    hist = Counter(min(c // 10 * 10, 200) for c in counts)
    print("\ncoarse histogram (bin = 10 frames, 200+ collapsed):")
    for b in sorted(hist):
        print(f"  {b:>4}-{b+9:<4} {'#' * min(hist[b], 60)} {hist[b]}")


if __name__ == "__main__":
    main()
