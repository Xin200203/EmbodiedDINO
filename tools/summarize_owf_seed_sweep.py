#!/usr/bin/env python3

import argparse
import csv
import re


METRIC_RE = re.compile(
    r"all_ap:\s*([0-9.]+)\s+all_ap_50%:\s*([0-9.]+)\s+"
    r"all_ap_25%:\s*([0-9.]+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank OCWE seeds from aligned and matched-control logs.")
    parser.add_argument("summary_tsv")
    parser.add_argument("controls_tsv")
    return parser.parse_args()


def parse_metric(line):
    match = METRIC_RE.search(line)
    if not match:
        raise ValueError(f"cannot parse metric line: {line}")
    return tuple(100.0 * float(value) for value in match.groups())


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def format_metrics(values):
    return "/".join(f"{value:.2f}" for value in values)


def subtract(left, right):
    return tuple(a - b for a, b in zip(left, right))


def main():
    args = parse_args()
    aligned_rows = {
        int(row["seed"]): parse_metric(row["metric_line"])
        for row in read_rows(args.summary_tsv)
        if row["phase"] == "fresh_aligned"
    }
    controls = {}
    for row in read_rows(args.controls_tsv):
        controls.setdefault(int(row["seed"]), {})[row["variant"]] = (
            parse_metric(row["metric_line"]))
    for seed, variants in controls.items():
        if seed not in aligned_rows and "aligned" in variants:
            aligned_rows[seed] = variants["aligned"]

    ranked = []
    for seed, aligned in aligned_rows.items():
        variants = controls.get(seed, {})
        if "no_write" not in variants or "shuffled" not in variants:
            continue
        no_write = variants["no_write"]
        shuffled = variants["shuffled"]
        gain = subtract(aligned, no_write)
        correspondence = subtract(aligned, shuffled)
        passes = all(value >= 0.0 for value in gain) and all(
            value > 0.0 for value in correspondence)
        ranked.append((passes, gain, correspondence, seed, aligned, no_write,
                       shuffled))

    ranked.sort(
        key=lambda row: (
            row[0], row[1][0], row[1][1], row[1][2], row[2][0], row[2][1],
            row[2][2]),
        reverse=True)

    print("| Rank | Seed | Gate | No write | Aligned | Shuffled | "
          "Aligned-no write | Aligned-shuffled |")
    print("|---:|---:|:---:|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(ranked, start=1):
        passes, gain, correspondence, seed, aligned, no_write, shuffled = row
        print(
            f"| {rank} | {seed} | {'pass' if passes else 'fail'} | "
            f"{format_metrics(no_write)} | {format_metrics(aligned)} | "
            f"{format_metrics(shuffled)} | {format_metrics(gain)} | "
            f"{format_metrics(correspondence)} |")

    if not ranked:
        raise SystemExit("no seed has complete aligned/no-write/shuffled rows")
    if not ranked[0][0]:
        raise SystemExit("no seed passes the non-regression/correspondence gate")
    print(f"\nselected_seed={ranked[0][3]}")


if __name__ == "__main__":
    main()
