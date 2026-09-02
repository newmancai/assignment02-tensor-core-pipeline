#!/usr/bin/env python3
"""Turn independent 05_thin_gemm outputs into per-row median Markdown."""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict[tuple[str, int, int, int], tuple[float, ...]]:
    rows: dict[tuple[str, int, int, int], tuple[float, ...]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 12 or not fields[1].isdigit():
            continue
        layer, m, n, k = fields[:4]
        numeric = tuple(
            float(value.rstrip("%"))
            for value in fields[4:9] + fields[10:12]
        )
        # us, TFLOPS, GB/s, AI, roofTF, %TCpeak, %BWroof
        rows[(layer, int(m), int(n), int(k))] = numeric
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()

    runs = [load(path) for path in args.runs]
    if not runs or any(len(run) != 63 for run in runs):
        raise SystemExit(f"expected 63 rows per run, got {[len(run) for run in runs]}")
    keys = set(runs[0])
    if any(set(run) != keys for run in runs[1:]):
        raise SystemExit("run shape sets differ")

    by_layer: dict[str, list[tuple[str, int, int, int]]] = defaultdict(list)
    for key in runs[0]:
        by_layer[key[0]].append(key)

    for layer, layer_keys in by_layer.items():
        _, _, n, k = layer_keys[0]
        print(f"### `{layer}` (`N={n}, K={k}`)\n")
        print("| M | us | AI | theory roof TFLOPS | bound | TFLOPS | % TC peak | % memory roof |")
        print("|---:|---:|---:|---:|:---:|---:|---:|---:|")
        for key in sorted(layer_keys, key=lambda item: item[1]):
            columns = [statistics.median(run[key][i] for run in runs) for i in range(7)]
            us, tflops, _gbps, ai, roof, tc_pct, bw_pct = columns
            bound = "memory" if roof < 2250.0 else "compute"
            print(
                f"| {key[1]} | {us:.1f} | {ai:.1f} | {roof:.1f} | {bound} | "
                f"{tflops:.1f} | {tc_pct:.1f}% | {bw_pct:.1f}% |"
            )
        print()


if __name__ == "__main__":
    main()
