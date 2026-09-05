"""Strict standalone-run audit of state_matrix_probe.py; JSON/Markdown to stdout.

Uses only stdlib and generic local helpers from summarize_release.py. No GPU
imports, no input/output file writes, and no pooling with release-probe jobs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from summarize_release import (CHANNELS, NAMES, aggregate, canonical, comparison,
                               parse_text, shape, state, tensor_failure, valid_timing)


def expected_shapes():
    result = [shape(tokens=t, state_mode=s, lengths=[t] if packed else None)
              for t in (2048,4096,8192) for s in ("both","in","out","none") for packed in (False,True)]
    return result + [shape(tokens=t) for t in (2049,4095,8191)]


def summarize_text(text, source="<memory>"):
    rows, issues, ignored = parse_text(text)
    environments = [row for row in rows if row.get("kind") == "state_environment"]
    if len(environments) != 1:
        issues.append("expected exactly one state_environment; multiple jobs are not pooled")
    if any(row.get("kind") in ("environment", "performance", "complete") for row in rows):
        issues.append("foreign release-probe rows detected; do not mix job schemas")
    markers = [row for row in rows if row.get("kind") == "state_matrix_complete"]
    if len(markers) != 1 or any(markers[0].get(key) != value for key, value in
                               (("shapes",27),("correctness_rows",81),("performance_rows",243))):
        issues.append("state_matrix_complete missing/duplicate or counts differ from 27/81/243 contract")
    correct = [row for row in rows if row.get("kind") == "state_correctness"]
    perf = [row for row in rows if row.get("kind") == "state_performance"]
    ends = [row for row in rows if row.get("kind") == "state_shape_complete"]
    expected = expected_shapes()
    expected_correct = Counter((i, name) for i in range(27) for name in NAMES)
    if Counter((row.get("case"), row.get("name")) for row in correct) != expected_correct:
        issues.append("state_correctness requires each of 27 cases x 3 names exactly once")
    if Counter((row.get("case"), row.get("name"), row.get("repeat")) for row in perf) != Counter(
            (i,name,repeat) for i in range(27) for name in NAMES for repeat in range(3)):
        issues.append("state_performance requires each of 27 cases x 3 names x 3 repeats exactly once")
    if Counter(row.get("case") for row in ends) != Counter(range(27)):
        issues.append("state_shape_complete requires each of 27 cases exactly once")
    for row in correct + perf + ends:
        index = row.get("case")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 27 or row.get("shape") != expected[index]:
            issues.append(f"{row.get('kind')}: shape/case does not match schema contract")
    for row in correct:
        meta = row.get("shape", {})
        fields = {"out", "final_state"} if meta.get("state_mode") in ("both", "out") else {"out"}
        if not isinstance(row.get("tensors"), dict) or set(row["tensors"]) != fields:
            issues.append("correctness tensor field coverage mismatch")
    failed = any(tensor_failure(row) for row in correct)
    result = []
    for index, meta in enumerate(expected):
        selected = [row for row in perf if row.get("case") == index]
        channels = {}
        local = []
        for channel in CHANNELS:
            variants = {}
            for name in NAMES:
                by_repeat = defaultdict(list)
                for row in selected:
                    if row.get("name") != name:
                        continue
                    if not valid_timing(row.get(channel)) or row[channel]["count"] != (30 if channel == "cache_perturbed" else 60):
                        local.append(f"{name}/{channel}: invalid timing or sample count (expected 60/60/30)")
                    elif isinstance(row.get("repeat"), int) and not isinstance(row["repeat"], bool):
                        by_repeat[row["repeat"]].append(row)
                if set(by_repeat) != {0,1,2} or any(len(items) != 1 for items in by_repeat.values()):
                    local.append(f"{name}/{channel}: missing/duplicate repeat")
                samples = [dict(repeat=repeat, **items[0][channel]) for repeat, items in sorted(by_repeat.items()) if len(items) == 1]
                if samples:
                    variants[name] = aggregate(samples)
            channels[channel] = dict(variants=variants, release_auto_vs={
                baseline: comparison(variants.get("release_auto"), variants.get(baseline))
                for baseline in ("baseline_auto", "release_v128")})
        issues.extend(f"case {index}: {message}" for message in dict.fromkeys(local))
        result.append(dict(case=index, shape=meta, packed=meta["lengths"] is not None,
                           has_initial=meta["state_mode"] in ("both","in"),
                           has_final=meta["state_mode"] in ("both","out"), tail=meta["tokens"] % 16 != 0,
                           channels=channels, decisions={name: [dict(repeat=row.get("repeat"), decision=row.get("decision"))
                                                              for row in selected if row.get("name") == name] for name in NAMES}))
    # Group summaries describe shape-to-shape range, not a pooled latency or CI.
    ranges = []
    for has_initial in (True,False):
        for packed in (False,True):
            subset = [item for item in result if item["has_initial"] == has_initial and item["packed"] == packed and not item["tail"]]
            for channel in CHANNELS:
                values = [item["channels"][channel]["release_auto_vs"]["baseline_auto"].get("latency_reduction_pct") for item in subset]
                values = [value for value in values if value is not None]
                ranges.append(dict(has_initial=has_initial, packed=packed, channel=channel,
                                   shape_count=len(subset), min_shape_reduction_pct=min(values) if values else None,
                                   max_shape_reduction_pct=max(values) if values else None))
    status = state(issues, failed)
    if status != "PASS":
        for item in result:
            for data in item["channels"].values():
                for comp in data["release_auto_vs"].values():
                    comp["status"] = "UNVERIFIED"
    return dict(source=source, status=status, issues=list(dict.fromkeys(issues)), ignored_non_json_lines=ignored,
                environment=environments[0] if len(environments) == 1 else {}, complete_markers=markers,
                observed_counts=dict(shapes=len(ends), correctness=len(correct), performance=len(perf)),
                correctness_status_counts=dict(Counter(row.get("status", "MISSING") for row in correct)),
                non_reference_comparison_rows=sum(row.get("name") != "release_v128" for row in correct),
                reference_self_check_rows=sum(row.get("name") == "release_v128" for row in correct),
                shapes=result, shape_gain_ranges=ranges,
                caveats=["This is one state-matrix job; no timing is pooled with Job 19901 or other runs.",
                         "Each latency is the median of three round medians. p10/p90 and across-shape ranges are not confidence intervals.",
                         "81 PASS rows include 27 release_v128 self-checks and 54 non-reference comparisons against release_v128.",
                         "CUDA-event eager/graph/cache_perturbed are distinct measurement scopes. Cache eviction is pre-call, not cold K2 proof.",
                         "Tested BF16 state contracts, H12/B1 and one packed sequence; this does not establish arbitrary packed/multi-sequence performance."])


def markdown(report):
    lines = [f"# State matrix audit: {report['status']}", "", f"Source: `{report['source']}`", "",
             "Observed: `" + canonical(report["observed_counts"]) + "`; non-reference comparison rows: " + str(report["non_reference_comparison_rows"]) + ".", "",
             "| T | State | Packed | Timing | Old auto ms | Release auto ms | Release 128 ms | Reduction vs old | Reduction vs 128 | Worst paired vs old |",
             "|---:|---|---|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(value, percent=False):
        return "UNVERIFIED" if value is None else f"{value:.3f}%" if percent else f"{value:.6f}"
    for item in report["shapes"]:
        for channel, data in item["channels"].items():
            old, full = (data["release_auto_vs"][name] for name in ("baseline_auto","release_v128"))
            values = [fmt(data["variants"].get(name, {}).get("median_ms")) for name in NAMES]
            values += [fmt(old.get("latency_reduction_pct"), True), fmt(full.get("latency_reduction_pct"), True),
                       fmt(old.get("worst_paired_round_reduction_pct"), True)]
            lines.append(f"| {item['shape']['tokens']} | {item['shape']['state_mode']} | {item['packed']} | {channel} | " + " | ".join(values) + " |")
    lines += ["", "## Caveats", ""] + ["- " + item for item in report["caveats"] + report["issues"]]
    lines += ["", "JSON output preserves all rounds and per-shape dispatch decisions."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--format", choices=("json","markdown"), default="json")
    args = parser.parse_args()
    report = summarize_text(args.log.read_text(encoding="utf-8", errors="replace"), str(args.log))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) if args.format == "json" else markdown(report))


if __name__ == "__main__":
    main()
