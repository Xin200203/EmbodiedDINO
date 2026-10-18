#!/usr/bin/env python3
"""Parse Phase-A granularity experiment logs.

Outputs a TSV with AP/AP50/AP25, error flag, and simple suppression statistics.
Designed for local artifact mirrors under artifacts/76_granlearn_phaseA_20260627/logs.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

METRIC_RE = re.compile(r"all_ap: ([0-9.]+)\s+all_ap_50%: ([0-9.]+)\s+all_ap_25%: ([0-9.]+)")
DROP_RE = re.compile(r"b0:(\d+)->(\d+)")
SEED_RE = re.compile(r"\[SEEDCALIB\].*mode=([^\s/]+).*accepted_edges=(\d+).*candidates=(\d+)")
ERR_RE = re.compile(r"Traceback|RuntimeError|AssertionError|CUDA error|Error")


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text(errors="ignore") if path.exists() else ""
    metric = list(METRIC_RE.finditer(text))
    ap = metric[-1].groups() if metric else ("NA", "NA", "NA")
    drops = []
    for m in DROP_RE.finditer(text):
        before, after = map(int, m.groups())
        if before > 0:
            drops.append((before, after, (before - after) / before))
    seed_events = []
    for m in SEED_RE.finditer(text):
        mode, accepted, cand = m.groups()
        seed_events.append((mode, int(accepted), int(cand)))
    accepted_sum = sum(x[1] for x in seed_events)
    cand_sum = sum(x[2] for x in seed_events)
    return {
        "run": path.name.replace(".test.partial.log", "").replace(".test.log", ""),
        "AP": ap[0],
        "AP50": ap[1],
        "AP25": ap[2],
        "error": "yes" if ERR_RE.search(text) else "no",
        "drop_events": len(drops),
        "avg_drop_frac": f"{(sum(d[2] for d in drops) / len(drops)):.4f}" if drops else "0.0000",
        "max_drop_frac": f"{max((d[2] for d in drops), default=0.0):.4f}",
        "seed_events": len(seed_events),
        "accepted_edges": accepted_sum,
        "candidate_edges": cand_sum,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir", type=Path)
    args = ap.parse_args()
    logs = sorted(args.log_dir.glob("*.log"))
    rows = [parse_log(p) for p in logs]
    cols = ["run", "AP", "AP50", "AP25", "error", "drop_events", "avg_drop_frac", "max_drop_frac", "seed_events", "accepted_edges", "candidate_edges"]
    print("\t".join(cols))
    for row in rows:
        print("\t".join(str(row[c]) for c in cols))


if __name__ == "__main__":
    main()
