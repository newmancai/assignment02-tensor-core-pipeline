"""Summarize one mainline_probe run per mixed JSONL/Slurm log, without GPUs.

Print JSON or Markdown to stdout. Inputs remain unchanged; distinct logs are
never pooled. p10/p90 are descriptive within-run quantiles, not confidence
intervals. Default completeness contract: three repeats per shape and mode.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import sys


CHANNELS = ("eager", "graph")
ORIGINAL = ("v16", "v32", "v64", "v128")
ENTRY_CASES = {"binding", "alignment", "parity", "stream_graph", "cpu_rejection", "multi_gpu"}


def shape_key(shape):
    return json.dumps(shape, sort_keys=True, separators=(",", ":"), allow_nan=False)


def finite_positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def entry_summary(rows):
    suites = [row for row in rows if row.get("suite") == "entry_hardening"]
    cases = [row for row in rows if row.get("case") in ENTRY_CASES and "kind" not in row]
    checks = [row for row in rows if "check" in row and "kind" not in row]
    failure_cases = sorted({str(row["case"]) for row in cases if row.get("status") == "FAIL"})
    skipped_cases = sorted({str(row["case"]) for row in cases if row.get("status") == "SKIP"})
    for suite in suites:
        failure_cases.extend(str(case) for case in suite.get("failures", []))
        skipped_cases.extend(str(case) for case in suite.get("skipped", []))
    failed_checks = [row for row in checks if row.get("status") == "FAIL"]
    skipped_checks = [row for row in checks if row.get("status") == "SKIP"]
    if not suites:
        state = "INCOMPLETE" if cases or checks else "NOT_RECORDED"
    elif len(suites) != 1:
        state = "INCOMPLETE"
    elif failure_cases or failed_checks or suites[0].get("status") != "PASS":
        state = "FAIL"
    elif skipped_cases or skipped_checks:
        state = "PASS_WITH_SKIPS"
    else:
        state = "PASS"
    return {
        "status": state,
        "suite_records": suites,
        "case_status_counts": dict(Counter(row.get("status", "UNSPECIFIED") for row in cases)),
        "cases": cases,
        "check_status_counts": dict(Counter(row.get("status", "UNSPECIFIED") for row in checks)),
        "skipped_cases": sorted(set(skipped_cases)),
        "skipped_checks": skipped_checks,
        "failure_cases": sorted(set(failure_cases)),
        "failed_checks": failed_checks,
    }


def summarize_text(text, *, source="<memory>", expected_repeats=3):
    if expected_repeats < 1:
        raise ValueError("expected_repeats must be positive")
    rows, issues = [], []
    ignored = 0
    for number, line in enumerate(text.splitlines(), 1):
        if not line.lstrip().startswith("{"):
            ignored += bool(line.strip())
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            issues.append(f"line {number}: invalid/truncated JSON object")
            continue
        if isinstance(row, dict):
            rows.append(row)
    environments = [row for row in rows if row.get("kind") == "environment"]
    complete_count = sum(row.get("kind") == "complete" for row in rows)
    if complete_count != 1:
        issues.append(f"expected one kind=complete, observed {complete_count}")
    if len(environments) != 1:
        issues.append(f"expected one environment; do not combine runs, observed {len(environments)}")
    environment = environments[0] if len(environments) == 1 else {}
    expected_names = set(environment.get("modes", {}).values()) | {"legacy16", "legacy128"}
    if not environment.get("modes"):
        issues.append("environment.modes is missing; variant coverage cannot be established")

    correctness = [row for row in rows if row.get("kind") == "correctness"]
    correctness_ends = [row for row in rows if row.get("kind") == "correctness_complete"]
    counts = Counter(row.get("status", "UNSPECIFIED") for row in correctness)
    invalid_pass = [row for row in correctness if row.get("status") == "PASS" and (
        not row.get("tensors") or any(
            tensor.get("bitwise") is not True or tensor.get("finite") is not True
            for tensor in row.get("tensors", {}).values()
        )
    )]
    correctness_keys = [(row.get("case"), row.get("mode")) for row in correctness]
    if len(set(correctness_keys)) != len(correctness_keys):
        issues.append("duplicate correctness case/mode records")
    if correctness or correctness_ends:
        if len(correctness_ends) != 1 or correctness_ends[0].get("comparison_rows") != len(correctness):
            issues.append("correctness_complete count does not match observed correctness rows")
    correctness_state = (
        "NOT_RECORDED" if not correctness and not correctness_ends else
        "FAIL" if invalid_pass or any(row.get("status") != "PASS" for row in correctness) else
        "PASS" if correctness_ends and correctness else "INCOMPLETE"
    )

    groups = {}
    for row in rows:
        if row.get("kind") not in ("performance", "shape_complete"):
            continue
        try:
            key = shape_key(row["shape"])
        except (KeyError, TypeError, ValueError):
            issues.append("performance/shape_complete row has invalid shape metadata")
            continue
        group = groups.setdefault(key, {"shape": row["shape"], "case_indices": set(), "complete": 0, "runs": {}})
        group["case_indices"].add(row.get("case"))
        if row["kind"] == "shape_complete":
            group["complete"] += 1
            continue
        name, repeat = row.get("name"), row.get("repeat")
        if not isinstance(name, str) or not isinstance(repeat, int) or isinstance(repeat, bool):
            issues.append(f"{key}: invalid name/repeat")
            continue
        if not all(isinstance(row.get(channel), dict) and finite_positive(row[channel].get("median_ms"))
                   for channel in CHANNELS):
            issues.append(f"{key}/{name}/{repeat}: invalid latency")
            continue
        runs = group["runs"].setdefault(name, {})
        if repeat in runs:
            issues.append(f"{key}/{name}: duplicate repeat {repeat}; first row retained, never averaged")
        else:
            runs[repeat] = row
    if not groups:
        issues.append("no performance shapes")

    results = []
    regressions = defaultdict(list)
    for key, group in groups.items():
        if group["complete"] != 1:
            issues.append(f"{key}: expected one shape_complete, observed {group['complete']}")
        if len(group["case_indices"]) != 1:
            issues.append(f"{key}: repeated shape across case indices; refusing to claim independent coverage")
        missing = expected_names - set(group["runs"])
        if missing:
            issues.append(f"{key}: missing variants {sorted(missing)}")
        expected_ids = set(range(expected_repeats))
        for name, runs in group["runs"].items():
            if set(runs) != expected_ids:
                issues.append(f"{key}/{name}: expected repeats {sorted(expected_ids)}, observed {sorted(runs)}")
        result = {"shape": group["shape"], "case_indices": sorted(group["case_indices"], key=str),
                  "shape_complete_count": group["complete"], "channels": {}}
        for channel in CHANNELS:
            variants = {}
            for name, runs in group["runs"].items():
                samples = [{"repeat": repeat, **runs[repeat][channel]} for repeat in sorted(runs)]
                values = [sample["median_ms"] for sample in samples]
                variants[name] = {
                    "median_ms": statistics.median(values),
                    "repeat_count": len(samples),
                    "min_repeat_median_ms": min(values),
                    "max_repeat_median_ms": max(values),
                    "rounds": samples,
                }
            originals = [name for name in ORIGINAL if name in variants]
            best = min(originals, key=lambda name: variants[name]["median_ms"]) if len(originals) == 4 else None
            baseline_names = {"same_binary_v16": "v16", "legacy16": "legacy16",
                              "same_binary_v128": "v128", "best_original_slice": best}
            for name, variant in variants.items():
                variant["comparisons"] = {}
                for label, baseline in baseline_names.items():
                    if baseline is None or baseline not in variants:
                        variant["comparisons"][label] = {"status": "MISSING_BASELINE"}
                        continue
                    ref = variants[baseline]
                    paired = []
                    for repeat in sorted(set(group["runs"][name]) & set(group["runs"][baseline])):
                        candidate_ms = group["runs"][name][repeat][channel]["median_ms"]
                        baseline_ms = group["runs"][baseline][repeat][channel]["median_ms"]
                        paired.append({"repeat": repeat, "candidate_ms": candidate_ms,
                                       "baseline_ms": baseline_ms,
                                       "latency_reduction_pct": 100 * (1 - candidate_ms / baseline_ms)})
                    gain = 100 * (1 - variant["median_ms"] / ref["median_ms"])
                    worst = min(item["latency_reduction_pct"] for item in paired) if paired else None
                    comparison = {
                        "status": "AVAILABLE", "baseline_name": baseline,
                        "baseline_median_ms": ref["median_ms"],
                        "latency_reduction_pct": gain, "speedup": ref["median_ms"] / variant["median_ms"],
                        "paired_rounds": paired,
                        "worst_paired_latency_reduction_pct": worst,
                        "worst_paired_regression_pct": max(0.0, -worst) if worst is not None else None,
                    }
                    variant["comparisons"][label] = comparison
                    regressions[(name, channel, label)].append({
                        "shape": group["shape"], "baseline_name": baseline,
                        "latency_reduction_pct": gain,
                        "worst_paired_latency_reduction_pct": worst,
                    })
            result["channels"][channel] = {"best_original_slice": best, "variants": variants}
        results.append(result)
    worst_cases = []
    for (name, channel, baseline), records in sorted(regressions.items()):
        worst = min(records, key=lambda item: item["latency_reduction_pct"])
        paired_records = [item for item in records if item["worst_paired_latency_reduction_pct"] is not None]
        worst_paired = min(paired_records, key=lambda item: item["worst_paired_latency_reduction_pct"]) if paired_records else None
        worst_cases.append({
            "name": name, "channel": channel, "baseline": baseline, "shape_count": len(records),
            "worst_shape": worst,
            "worst_shape_regression_pct": max(0.0, -worst["latency_reduction_pct"]),
            "worst_paired_shape": worst_paired,
        })
    entry = entry_summary(rows)
    if entry["status"] == "INCOMPLETE":
        issues.append("entry_hardening records exist without exactly one terminal suite record")
    failed = correctness_state == "FAIL" or entry["status"] == "FAIL"
    status = "INCOMPLETE" if issues else "FAIL" if failed else "UNVERIFIED" if correctness_state != "PASS" else "PASS"
    if status == "PASS" and entry["status"] == "PASS_WITH_SKIPS":
        status = "PASS_WITH_SKIPS"
    return {
        "source": source, "status": status, "complete_count": complete_count,
        "failure_detected": failed, "expected_repeats": expected_repeats,
        "issues": issues, "ignored_non_json_lines": ignored, "environment": environment,
        "correctness": {"status": correctness_state, "row_count": len(correctness),
                        "status_counts": dict(counts), "invalid_pass_rows": len(invalid_pass),
                        "tensor_comparison_count": sum(len(row.get("tensors", {})) for row in correctness),
                        "terminal_records": correctness_ends},
        "entry_hardening": entry, "shapes": results, "worst_cases": worst_cases,
        "measurement_notes": [
            "median_ms is the median of per-repeat medians; no samples pooled across binaries or logs",
            "positive latency_reduction_pct means faster; negative means a regression",
            "best_original_slice is selected from same-binary v16/v32/v64/v128 aggregate medians per shape/channel",
            "paired_rounds compare the same repeat number, not simultaneous GPU executions",
            "p10_ms/p90_ms are within-run descriptive quantiles, never confidence intervals",
            "worst_cases retains the worst shape and paired repeat; no aggregate speedup hides regressions",
        ],
    }


def markdown(summary):
    lines = [f"# {summary['source']}", "", f"Status: **{summary['status']}**; kind=complete: {summary['complete_count']}.", "",
             f"Correctness: {summary['correctness']['status']}; {summary['correctness']['row_count']} rows; "
             f"status counts {summary['correctness']['status_counts']}; "
             f"entry hardening: {summary['entry_hardening']['status']}.", ""]
    if summary["issues"]:
        lines += ["Issues:", ""] + [f"- {issue}" for issue in summary["issues"]] + [""]
    for shape in summary["shapes"]:
        lines += [f"## Shape {json.dumps(shape['shape'], sort_keys=True)}", "",
                  "| Mode | Timing | Median ms | vs V16 | vs legacy16 | vs V128 | vs best original | Worst paired regression vs best |", 
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for channel, data in shape["channels"].items():
            for name, variant in sorted(data["variants"].items()):
                comparisons = variant["comparisons"]
                fields = []
                for baseline in ("same_binary_v16", "legacy16", "same_binary_v128", "best_original_slice"):
                    value = comparisons[baseline].get("latency_reduction_pct")
                    fields.append("N/A" if value is None else f"{value:+.3f}%")
                worst = comparisons["best_original_slice"].get("worst_paired_regression_pct")
                lines.append(f"| {name} | {channel} | {variant['median_ms']:.6f} | " + " | ".join(fields) +
                             f" | {'N/A' if worst is None else f'{worst:.3f}%'} |")
        lines += ["", "Per-repeat median ms (p10/p90 remain descriptive and are preserved in JSON):", ""]
        for channel, data in shape["channels"].items():
            lines.append(f"- {channel}, best original: {data['best_original_slice']}")
            for name, variant in sorted(data["variants"].items()):
                times = ", ".join(f"r{row['repeat']}={row['median_ms']:.6f}" for row in variant["rounds"])
                lines.append(f"  - {name}: {times}")
        lines.append("")
    entry = summary["entry_hardening"]
    lines += ["## Entry hardening", "", f"Case statuses: {entry['case_status_counts']}", "",
              f"Skipped cases: {entry['skipped_cases']}; failures: {entry['failure_cases']}", ""]
    for row in entry["skipped_checks"]:
        lines.append(f"- SKIP {row.get('check')}: {row.get('reason', 'reason not recorded')}")
    lines += ["", "Notes:", ""] + [f"- {note}" for note in summary["measurement_notes"]]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", help="one run per log; '-' reads stdin (default)")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--expected-repeats", type=int, default=3)
    args = parser.parse_args()
    if args.expected_repeats < 1:
        parser.error("--expected-repeats must be positive")
    logs = args.logs or ["-"]
    if logs.count("-") > 1:
        parser.error("stdin can only be read once")
    summaries = [summarize_text(sys.stdin.read() if path == "-" else Path(path).read_text(),
                               source=path, expected_repeats=args.expected_repeats) for path in logs]
    if args.format == "json":
        print(json.dumps({"runs": summaries}, indent=2, allow_nan=False))
    else:
        print("\n\n---\n\n".join(markdown(summary) for summary in summaries))
    return 0 if all(summary["status"] in ("PASS", "PASS_WITH_SKIPS") for summary in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
