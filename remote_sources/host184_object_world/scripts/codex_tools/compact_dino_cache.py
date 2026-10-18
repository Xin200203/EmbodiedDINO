#!/usr/bin/env python3
"""Compact DINO cache files accidentally saved from batch-view tensors.

When a per-frame feature is a view into a batch tensor, torch.save stores the
whole underlying batch storage. This script rewrites each payload with
payload['feat'].contiguous().clone() so the file stores only one feature map.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch


def compact_one(path_str: str, max_expected_mb: float, dry_run: bool = False):
    path = Path(path_str)
    before = path.stat().st_size
    if before <= max_expected_mb * 1024 * 1024:
        return {'path': str(path), 'status': 'skip_small', 'before': before, 'after': before}
    try:
        payload = torch.load(path, map_location='cpu')
        if isinstance(payload, dict) and torch.is_tensor(payload.get('feat')):
            payload['feat'] = payload['feat'].contiguous().clone()
        elif torch.is_tensor(payload):
            payload = payload.contiguous().clone()
        else:
            return {'path': str(path), 'status': 'skip_unknown_payload', 'before': before, 'after': before}
        if dry_run:
            return {'path': str(path), 'status': 'dry_run', 'before': before, 'after': before}
        tmp = path.with_suffix(path.suffix + '.compact_tmp')
        torch.save(payload, tmp)
        tmp.replace(path)
        after = path.stat().st_size
        return {'path': str(path), 'status': 'compacted', 'before': before, 'after': after}
    except Exception as exc:
        return {'path': str(path), 'status': 'failed', 'before': before, 'after': before, 'error': repr(exc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-root', required=True)
    ap.add_argument('--summary', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max-expected-mb', type=float, default=6.0)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    root = Path(args.cache_root)
    files = [str(p) for p in root.rglob('*.pt')]
    t0 = time.time()
    status_counts = {}
    before_total = 0
    after_total = 0
    failures = []
    n = len(files)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(compact_one, p, args.max_expected_mb, args.dry_run) for p in files]
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            status_counts[res['status']] = status_counts.get(res['status'], 0) + 1
            before_total += int(res.get('before', 0))
            after_total += int(res.get('after', 0))
            if res['status'] == 'failed':
                failures.append(res)
            if i % 500 == 0 or i == n:
                print(json.dumps({'progress': [i, n], 'status_counts': status_counts}, ensure_ascii=False), flush=True)
    summary = {
        'cache_root': str(root),
        'num_files': n,
        'workers': args.workers,
        'status_counts': status_counts,
        'before_bytes_sum_seen': before_total,
        'after_bytes_sum_seen': after_total,
        'saved_bytes_sum_seen': before_total - after_total,
        'elapsed_sec': round(time.time() - t0, 3),
        'failures': failures[:50],
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} compaction failures')

if __name__ == '__main__':
    main()
