"""Audit release_probe.py JSONL/Slurm logs; never pool independent GPU runs.

Pure standard library, no GPU imports, stdout only. See --help for sidecar logs.
The schema contract is the 2026-09-05 release probe (40 correctness / 11 perf cases).
p10/p90 and the three round medians are descriptive, never confidence intervals.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import statistics


CHANNELS = ("eager", "graph", "cache_perturbed")
NAMES = ("baseline_auto", "release_auto", "release_v128")
MODES = ("auto", "force16", "off")
ENTRY_CASES = {"binding", "alignment", "parity", "stream_graph", "cpu_rejection", "multi_gpu"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def shape(**description):
    return dict(tokens=description.pop("tokens"), heads=description.pop("heads", 12),
                batch=description.pop("batch", 1), lengths=description.pop("lengths", None),
                fp32=description.pop("fp32", False), state_mode=description.pop("state_mode", "both"),
                gate=description.pop("gate", None), **description)


def expected_shapes(kind="correctness", sanitizer=False):
    if sanitizer:
        descriptions = [dict(tokens=2049), dict(tokens=2048, state_mode="out", lengths=[2048])]
    elif kind == "performance":
        descriptions = [dict(tokens=t) for t in (2048, 3072, 4096, 6144, 8192, 16384)]
        descriptions += [dict(tokens=8192, lengths=[8192]), dict(tokens=8192, batch=2),
                         dict(tokens=8192, lengths=[1024]*8), dict(tokens=4096, fp32=True),
                         dict(tokens=8192, state_mode="out")]
    else:
        descriptions = [dict(tokens=t) for t in (1,17,2047,2048,2049,3072,4095,4096,4097,6144,8191,8192,8193,16384)]
        descriptions += [dict(tokens=t, lengths=[t] if packed else None, state_mode=s)
                         for t in (2048,8192) for packed in (False,True) for s in ("in","out","none")]
        descriptions += [dict(tokens=4096, fp32=True, state_mode=s) for s in ("both","in","out")]
        descriptions += [dict(tokens=8192, heads=h) for h in (24,48,96)]
        descriptions += [dict(tokens=8192, batch=b) for b in (2,4)]
        descriptions += [dict(tokens=8192, lengths=v) for v in ([8192], [1024]*8, [16,32,512,1024,2512,4096], [0,8192])]
        descriptions += [dict(tokens=8192, gate=g) for g in (-8.,12.)]
    return [shape(**item) for item in descriptions]


def state(issues=(), failed=False, skipped=False):
    return "FAIL" if failed else "UNVERIFIED" if issues else "PASS_WITH_SKIPS" if skipped else "PASS"


def parse_text(text):
    rows, issues, ignored = [], [], 0
    for number, line in enumerate(text.splitlines(), 1):
        if not line.lstrip().startswith("{"):
            ignored += bool(line.strip())
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            issues.append(f"line {number}: invalid/truncated JSON")
            continue
        if not isinstance(row, dict):
            issues.append(f"line {number}: expected object")
        else:
            rows.append(row)
    return rows, issues, ignored


def select_main(rows):
    """Environment delimits probe invocations; sanitizer=true never selects main."""
    sessions = []
    for row in rows:
        if row.get("kind") == "environment":
            sessions.append([row])
        elif sessions and row.get("kind"):
            sessions[-1].append(row)
    candidates = []
    for session in sessions:
        classified_other = any(row.get("kind") == "profile_complete" or
                               (row.get("kind") in ("complete", "correctness_complete") and
                                row.get("sanitizer") is True) for row in session)
        if not classified_other:
            candidates.append(session)
    issues = []
    if len(candidates) != 1:
        issues.append(f"expected exactly one main environment/session, observed {len(candidates)}; not pooling")
    main = candidates[0] if len(candidates) == 1 else []
    ends = [row for row in main if row.get("kind") == "complete"]
    if len(ends) != 1 or ends[0].get("sanitizer") is not False:
        issues.append("main complete(sanitizer=false) missing, ambiguous, or duplicated")
    return main, issues, len(sessions)


def tensor_failure(row):
    tensors = row.get("tensors")
    return (row.get("status") != "PASS" or not isinstance(tensors, dict) or not tensors or
            any(not isinstance(t, dict) or t.get("bitwise") is not True or t.get("finite") is not True
                for t in tensors.values()))


def correctness_summary(rows, sanitizer=False):
    checks = [row for row in rows if row.get("kind") == "correctness"]
    ends = [row for row in rows if row.get("kind") == "correctness_complete"]
    expected = expected_shapes(sanitizer=sanitizer)
    keys = [(row.get("case"), row.get("mode")) for row in checks]
    issues = []
    if Counter(keys) != Counter((i, mode) for i in range(len(expected)) for mode in MODES):
        issues.append(f"expected {len(expected)} cases x 3 modes exactly once; observed {len(checks)} rows")
    for row in checks:
        case = row.get("case")
        if not isinstance(case, int) or isinstance(case, bool) or not 0 <= case < len(expected) or row.get("shape") != expected[case]:
            issues.append(f"case {case}: shape differs from release_probe contract")
            continue
        fields = {"out", "final_state"} if expected[case]["state_mode"] in ("both", "out") else {"out"}
        if not isinstance(row.get("tensors"), dict) or set(row["tensors"]) != fields:
            issues.append(f"case {case}: missing/unexpected output tensor checks")
    if (len(ends) != 1 or ends[0].get("sanitizer") is not sanitizer or
            ends[0].get("comparison_rows") != len(checks)):
        issues.append("correctness_complete marker absent/duplicated or comparison_rows/sanitizer mismatch")
    return dict(status=state(issues, any(tensor_failure(row) for row in checks)), issues=issues,
                expected_rows=len(expected)*3, observed_rows=len(checks),
                status_counts=dict(Counter(row.get("status", "MISSING") for row in checks)),
                tensor_comparisons=sum(len(row.get("tensors", {})) for row in checks if isinstance(row.get("tensors"), dict)),
                markers=ends)


def indexed_checks(rows, kind, field, count):
    checks = [row for row in rows if row.get("kind") == kind]
    issues = []
    if Counter(row.get(field) for row in checks) != Counter(range(count)):
        issues.append(f"expected {kind} {field}=0..{count-1} exactly once")
    if any(not isinstance(row.get("tensors"), dict) or set(row["tensors"]) != {"out", "final_state"} for row in checks):
        issues.append("missing output/final_state tensor check")
    return dict(status=state(issues, any(tensor_failure(row) for row in checks)), issues=issues, rows=checks)


def positive(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def valid_timing(value):
    return (isinstance(value, dict) and all(positive(value.get(key)) for key in ("median_ms", "p10_ms", "p90_ms"))
            and value["p10_ms"] <= value["median_ms"] <= value["p90_ms"]
            and isinstance(value.get("count"), int) and not isinstance(value["count"], bool) and value["count"] > 0)


def aggregate(samples):
    values = [item["median_ms"] for item in samples]
    return dict(median_ms=statistics.median(values), min_round_median_ms=min(values),
                max_round_median_ms=max(values), rounds=samples)


def comparison(candidate, baseline):
    if candidate is None or baseline is None:
        return dict(status="UNVERIFIED", reason="missing candidate or baseline")
    by_repeat = {item["repeat"]: item["median_ms"] for item in baseline["rounds"]}
    pairs = [dict(repeat=item["repeat"], latency_reduction_pct=100*(1-item["median_ms"]/by_repeat[item["repeat"]]))
             for item in candidate["rounds"] if item["repeat"] in by_repeat]
    worst = min((item["latency_reduction_pct"] for item in pairs), default=None)
    return dict(status="PASS", latency_reduction_pct=100*(1-candidate["median_ms"]/baseline["median_ms"]),
                speedup=baseline["median_ms"]/candidate["median_ms"], paired_rounds=pairs,
                worst_paired_round_reduction_pct=worst,
                worst_paired_round_regression_pct=max(0, -worst) if worst is not None else None)


def performance_summary(rows, repeats=3, concurrent=False):
    kind = "concurrent" if concurrent else "performance"
    expected = [shape(tokens=8192)] if concurrent else expected_shapes("performance")
    names = NAMES[:2] if concurrent else NAMES
    channels = ("pair",) if concurrent else CHANNELS
    expected_repeats = 3 if concurrent else repeats  # concurrent() is explicitly fixed at 3.
    records = [row for row in rows if row.get("kind") == kind]
    issues, groups = [], defaultdict(list)
    for row in records:
        try:
            key = canonical(row.get("shape"))
        except (TypeError, ValueError):
            issues.append("invalid/non-finite shape")
            continue
        groups[key].append(row)
    expected_keys = {canonical(item) for item in expected}
    if set(groups) != expected_keys:
        issues.append(f"expected {len(expected)} {kind} shapes, observed {len(groups)}; missing/unexpected shape coverage")
    result = []
    for index, meta in enumerate(expected):
        data = groups.get(canonical(meta), [])
        local = []
        keys = [(row.get("name"), row.get("repeat")) for row in data]
        if Counter(keys) != Counter((name, repeat) for name in names for repeat in range(expected_repeats)):
            local.append("missing/duplicated/unexpected variant or repeat; no averaging duplicates")
        if concurrent:
            if any(row.get("requests") != 2 for row in data):
                local.append("concurrent interval must contain exactly two requests")
        else:
            if any(row.get("case") != index for row in data):
                local.append("case index does not match schema shape")
            ends = [row for row in rows if row.get("kind") == "shape_complete" and row.get("shape") == meta]
            if len(ends) != 1 or ends[0].get("case") != index:
                local.append("shape_complete missing/duplicated/mismatched")
        channel_results = {}
        for channel in channels:
            variants = {}
            for name in names:
                selected = [row for row in data if row.get("name") == name]
                by_repeat = defaultdict(list)
                for row in selected:
                    if not valid_timing(row.get(channel)):
                        local.append(f"{name}/{channel}: invalid timing values/count/quantile order")
                    elif isinstance(row.get("repeat"), int) and not isinstance(row["repeat"], bool):
                        by_repeat[row["repeat"]].append(row)
                samples = [dict(repeat=repeat, **items[0][channel]) for repeat, items in sorted(by_repeat.items()) if len(items) == 1]
                if samples:
                    variants[name] = aggregate(samples)
            comparisons = {name: comparison(variants.get("release_auto"), variants.get(name)) for name in names if name != "release_auto"}
            channel_results[channel] = dict(variants=variants, release_auto_vs=comparisons)
        for reason in dict.fromkeys(local):
            issues.append(f"{canonical(meta)}: {reason}")
        result.append(dict(shape=meta, status=state(local), channels=channel_results,
                           decisions={name: [dict(repeat=row.get("repeat"), decision=row.get("decision")) for row in data if row.get("name") == name] for name in names}))
    # A computed ratio is descriptive even when coverage is incomplete; never call it verified.
    if issues:
        for item in result:
            if item["status"] != "PASS":
                for channel in item["channels"].values():
                    for value in channel["release_auto_vs"].values():
                        value["status"] = "UNVERIFIED"
    return dict(status=state(issues), issues=issues, observed_rows=len(records),
                expected_rows=len(expected)*len(names)*expected_repeats, shapes=result)


def entry_summary(rows):
    suites = [row for row in rows if row.get("suite") == "entry_hardening"]
    cases = [row for row in rows if "kind" not in row and row.get("case") in ENTRY_CASES]
    checks = [row for row in rows if "kind" not in row and "check" in row]
    issues = []
    if len(suites) != 1:
        issues.append("expected one entry_hardening terminal suite record")
    if Counter(row["case"] for row in cases) != Counter(ENTRY_CASES):
        issues.append("entry case coverage missing/duplicated")
    failed = any(row.get("status") == "FAIL" for row in cases + checks + suites)
    failed = failed or any(row.get("failures") for row in suites)
    if any(row.get("status") not in ("PASS", "SKIP") for row in cases) or any(row.get("status") != "PASS" for row in suites):
        issues.append("entry case/suite status missing or unexpected")
    skips = [row for row in cases + checks if row.get("status") == "SKIP"]
    return dict(status=state(issues, failed, bool(skips or any(row.get("skipped") for row in suites))),
                issues=issues, suites=suites, cases=cases, skips=skips,
                failed_checks=[row for row in checks if row.get("status") == "FAIL"])


def exit_summary(rows, kind, key, targets):
    result = {}
    for target in targets:
        found = [row for row in rows if row.get("kind") == kind and row.get(key) == target]
        codes = [row.get("exit_code") for row in found]
        invalid = len(codes) != 1 or not isinstance(codes[0], int) or isinstance(codes[0], bool)
        result[target] = dict(status=state(["missing/duplicate/noninteger exit"] if invalid else [],
                                           any(isinstance(code, int) and not isinstance(code, bool) and code != 0 for code in codes)),
                              records=found)
    return result


def sidecar_summary(text, kind, target, candidate_sha256):
    if text is None:
        return dict(status="UNVERIFIED", issues=["sidecar log not supplied; exit code alone does not verify its recorded coverage"])
    rows, issues, _ = parse_text(text)
    environments = [row for row in rows if row.get("kind") == "environment"]
    if len(environments) != 1 or not candidate_sha256 or environments[0].get("candidate_sha256") != candidate_sha256:
        issues.append("sidecar candidate SHA256 missing or inconsistent with main")
    failed = False
    extra = {}
    if kind == "sanitizer":
        correct = correctness_summary(rows, sanitizer=True)
        extra["correctness"] = correct
        issues.extend(correct["issues"])
        failed = correct["status"] == "FAIL"
        ends = [row for row in rows if row.get("kind") == "complete"]
        if len(ends) != 1 or ends[0].get("sanitizer") is not True:
            issues.append("sanitizer complete(sanitizer=true) missing/duplicated")
        errors = [int(value) for value in re.findall(r"ERROR SUMMARY:\s*(\d+)\s+errors?", text)]
        extra["error_summary_counts"] = errors
        if len(errors) != 1:
            issues.append("expected one Compute Sanitizer ERROR SUMMARY")
        failed = failed or any(errors)
    else:
        ends = [row for row in rows if row.get("kind") == "profile_complete"]
        extra["markers"] = ends
        if len(ends) != 1 or ends[0].get("mode") != target:
            issues.append("profile_complete mode missing/duplicated/mismatched")
    return dict(status=state(issues, failed), issues=issues, **extra)


def summarize_text(text, *, source="<memory>", repeats=3, sanitizer_logs=None, profile_logs=None):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rows, issues, ignored = parse_text(text)
    main, session_issues, session_count = select_main(rows)
    issues.extend(session_issues)
    environment = main[0] if main else {}
    correct = correctness_summary(main)
    chain = indexed_checks(main, "state_chain", "step", 3)
    perf = performance_summary(main, repeats)
    concurrent = performance_summary(main, concurrent=True)
    concurrent_correct = indexed_checks(main, "concurrent_correctness", "case", 2)
    entry = entry_summary(rows)
    sanitizer = exit_summary(rows, "sanitizer_exit", "tool", ("memcheck", "synccheck"))
    profiles = exit_summary(rows, "profile_exit", "variant", ("baseline", "release"))
    for kind, reports, logs in (("sanitizer", sanitizer, sanitizer_logs or {}), ("profile", profiles, profile_logs or {})):
        for target, report in reports.items():
            report["sidecar"] = sidecar_summary(logs.get(target), kind, target, environment.get("candidate_sha256"))
    components = [correct, chain, perf, concurrent, concurrent_correct, entry]
    components += list(sanitizer.values()) + list(profiles.values())
    components += [report["sidecar"] for report in list(sanitizer.values()) + list(profiles.values())]
    statuses = [item["status"] for item in components]
    if "UNVERIFIED" in statuses:
        issues.append("one or more required coverage/sidecar/exit checks are UNVERIFIED")
    return dict(source=source, status=state(issues, "FAIL" in statuses, "PASS_WITH_SKIPS" in statuses),
                issues=issues, ignored_non_json_lines=ignored, probe_session_count=session_count,
                environment=environment, main_complete_verified=not session_issues,
                correctness=correct, state_chain=chain, performance=perf, concurrent=concurrent,
                concurrent_correctness=concurrent_correct, entry_hardening=entry,
                sanitizer=sanitizer, profiles=profiles,
                interpretation=["Median of per-round medians; p10/p90 are within-round quantiles, not confidence intervals.",
                                "Positive latency_reduction_pct means faster; paired rounds retain the worst observed regression.",
                                "cache_perturbed zeros 256 MiB before the start event; K1 may still warm K2 workspace, so this is not cold-K2 proof.",
                                "Concurrent pair is a joined two-request/two-stream latency, not per-request latency or serving throughput.",
                                "release_auto is guarded; its name is not proof P4 ran for every shape. Keep dispatch decisions and inspect the guard."])


def markdown(report):
    lines = [f"# Release audit: {report['status']}", "", f"Source: `{report['source']}`", "",
             f"Main complete: {report['main_complete_verified']}; correctness: {report['correctness']['status']} "
             f"({report['correctness']['observed_rows']}/{report['correctness']['expected_rows']}); "
             f"state chain: {report['state_chain']['status']}; concurrent correctness: {report['concurrent_correctness']['status']}; "
             f"entry hardening: {report['entry_hardening']['status']}.", "",
             "| Shape | Timing | Old auto ms | Release auto ms | Release 128 ms | Reduction vs old | Reduction vs 128 | Worst paired vs old |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    def number(value, suffix=""):
        return "UNVERIFIED" if value is None else f"{value:.6f}{suffix}" if not suffix else f"{value:.3f}{suffix}"
    for item in report["performance"]["shapes"]:
        for channel, data in item["channels"].items():
            variants = data["variants"]
            old = data["release_auto_vs"]["baseline_auto"]
            full = data["release_auto_vs"]["release_v128"]
            values = [number(variants.get(name, {}).get("median_ms")) for name in NAMES]
            values += [number(old.get("latency_reduction_pct"), "%"), number(full.get("latency_reduction_pct"), "%"),
                       number(old.get("worst_paired_round_reduction_pct"), "%")]
            lines.append(f"| `{canonical(item['shape'])}` | {channel} ({item['status']}) | " + " | ".join(values) + " |")
    lines += ["", "## Concurrent two-request pair", ""]
    for item in report["concurrent"]["shapes"]:
        data = item["channels"]["pair"]
        lines.append(f"Status: {item['status']}; " + "; ".join(f"{name}: {number(value['median_ms'])} ms" for name, value in data["variants"].items()) + ".")
        lines.append("Comparison (including all paired rounds): `" + canonical(data["release_auto_vs"]["baseline_auto"]) + "`")
    lines += ["", "## Instrumentation and entry skips", ""]
    for label in ("sanitizer", "profiles"):
        for target, item in report[label].items():
            lines.append(f"- {label}/{target}: exit {item['status']}, sidecar {item['sidecar']['status']}; `" + canonical(item) + "`")
    lines += ["- Entry skips: `" + canonical(report["entry_hardening"]["skips"]) + "`", "", "## Caveats", ""]
    lines += ["- " + item for item in report["interpretation"]]
    lines += ["- " + item for item in report["issues"]]
    for section in ("correctness", "state_chain", "performance", "concurrent", "concurrent_correctness", "entry_hardening"):
        lines += [f"- {section}: {issue}" for issue in report[section]["issues"]]
    lines += ["", "All per-round median/p10/p90/count values and dispatch decisions are retained in JSON output."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="One main Slurm log; runs are never pooled")
    parser.add_argument("--repeats", type=int, default=3, help="Expected main performance repeats (concurrent remains 3)")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--memcheck-log", type=Path)
    parser.add_argument("--synccheck-log", type=Path)
    parser.add_argument("--baseline-profile-log", type=Path)
    parser.add_argument("--release-profile-log", type=Path)
    args = parser.parse_args()
    read = lambda path: path.read_text(encoding="utf-8", errors="replace") if path else None
    report = summarize_text(read(args.log), source=str(args.log), repeats=args.repeats,
                            sanitizer_logs={"memcheck": read(args.memcheck_log), "synccheck": read(args.synccheck_log)},
                            profile_logs={"baseline": read(args.baseline_profile_log), "release": read(args.release_profile_log)})
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) if args.format == "json" else markdown(report))


if __name__ == "__main__":
    main()
