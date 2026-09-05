"""Pure-stdlib audit of one matched_probe.py mixed JSONL log; stdout only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


NAMES = ("release_none", "release_zero", "release_zero_each", "release_nonzero", "legacy_none", "legacy_zero")
ZERO_NAMES = tuple(name for name in NAMES if name != "release_nonzero")
CHANNELS = ("eager", "graph", "cache_perturbed")
CONTRASTS = (("release_zero", "release_none"), ("release_zero_each", "release_none"),
             ("release_nonzero", "release_zero"), ("release_zero_each", "release_zero"),
             ("release_none", "legacy_none"), ("release_zero", "legacy_zero"),
             ("legacy_zero", "legacy_none"), ("release_zero_each", "legacy_none"))


def expected_shapes():
    descriptions = [dict(tokens=t, gate=g) for t in (2048,4096,8192) for g in (None,-8.)]
    descriptions += [dict(tokens=8192,lengths=[8192]), dict(tokens=2049), dict(tokens=8191), dict(tokens=8192,state_mode="none")]
    return [dict(dict(batch=1,heads=12,fp32=False,lengths=None,state_mode="both",gate=None), **description) for description in descriptions]


def valid_timing(value, channel):
    def positive(item):
        return isinstance(item, (int,float)) and not isinstance(item,bool) and math.isfinite(item) and item > 0
    return (isinstance(value,dict) and all(positive(value.get(k)) for k in ("median_ms","p10_ms","p90_ms"))
            and value["p10_ms"] <= value["median_ms"] <= value["p90_ms"]
            and value.get("count") == (30 if channel == "cache_perturbed" else 60))


def comparison(candidate, baseline):
    if candidate is None or baseline is None:
        return dict(status="UNVERIFIED")
    old = {row["repeat"]: row["median_ms"] for row in baseline["rounds"]}
    pairs = [dict(repeat=row["repeat"], latency_reduction_pct=100*(1-row["median_ms"]/old[row["repeat"]]),
                  candidate_minus_baseline_us=1000*(row["median_ms"]-old[row["repeat"]]))
             for row in candidate["rounds"] if row["repeat"] in old]
    worst = min((row["latency_reduction_pct"] for row in pairs), default=None)
    return dict(status="PASS", latency_reduction_pct=100*(1-candidate["median_ms"]/baseline["median_ms"]),
                candidate_minus_baseline_us=1000*(candidate["median_ms"]-baseline["median_ms"]),
                paired_rounds=pairs, worst_paired_reduction_pct=worst,
                worst_paired_regression_pct=max(0,-worst) if worst is not None else None)


def summarize_text(text, source="<memory>"):
    rows, issues = [], []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.lstrip().startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            issues.append(f"line {number}: malformed/truncated JSON")
            continue
        if isinstance(row,dict):
            rows.append(row)
    environments = [r for r in rows if r.get("kind") == "environment"]
    if len(environments) != 1 or environments[0].get("experiment") != "matched-zero-state":
        issues.append("expected one matched-zero-state environment; do not pool jobs/profiles")
    ends = [r for r in rows if r.get("kind") == "matched_complete"]
    if len(ends) != 1 or ends[0].get("shapes") != 10:
        issues.append("matched_complete(shapes=10) missing/duplicated/mismatched")
    expected = expected_shapes()
    perf = [r for r in rows if r.get("kind") == "performance"]
    if Counter((r.get("case"),r.get("name"),r.get("repeat")) for r in perf) != Counter(
            (case,name,repeat) for case in range(10) for name in NAMES for repeat in range(3)):
        issues.append("expected exactly 10 shapes x 6 arms x 3 rounds")
    shape_ends = [r for r in rows if r.get("kind") == "shape_complete"]
    if Counter(r.get("case") for r in shape_ends) != Counter(range(10)):
        issues.append("expected exactly one shape_complete per case")
    correctness = {}
    failed = False
    for kind in ("correctness", "post_timing_correctness", "nonzero_correctness"):
        checks = [r for r in rows if r.get("kind") == kind]
        if kind == "nonzero_correctness":
            actual_keys = Counter(r.get("case") for r in checks)
            wanted = Counter(range(10))
        else:
            actual_keys = Counter((r.get("case"), r.get("name")) for r in checks)
            wanted = Counter((case,name) for case in range(10) for name in ZERO_NAMES)
        if actual_keys != wanted:
            issues.append(f"{kind}: missing/duplicated/mismatched case/arm checks")
        for row in checks:
            case = row.get("case")
            if not isinstance(case,int) or isinstance(case,bool) or not 0 <= case < 10:
                issues.append(f"{kind}: invalid case")
                continue
            fields = {"out"} if expected[case]["state_mode"] == "none" else {"out","final_state"}
            tensors = row.get("tensors", {})
            if not isinstance(tensors,dict) or set(tensors) != fields:
                issues.append(f"{kind}/{case}: missing/unexpected tensor checks")
            failed = failed or row.get("status") == "FAIL" or (isinstance(tensors,dict) and any(
                isinstance(t,dict) and (t.get("bitwise") is False or t.get("finite") is False) for t in tensors.values()))
            if row.get("status") != "PASS" or not isinstance(tensors,dict) or any(
                    not isinstance(t,dict) or t.get("bitwise") is not True or t.get("finite") is not True for t in tensors.values()):
                issues.append(f"{kind}/{case}: PASS/bitwise/finite not established")
            if kind == "correctness" and row.get("reference_sanity") is not (row.get("name") == "legacy_none"):
                issues.append(f"{kind}/{case}: incorrect/missing reference_sanity label")
        self_count = sum(r.get("name") == "legacy_none" for r in checks) if kind != "nonzero_correctness" else 0
        correctness[kind] = dict(rows=len(checks), status_counts=dict(Counter(r.get("status","MISSING") for r in checks)),
                                 self_comparisons=self_count, nonself_comparisons=len(checks)-self_count)
    for row in perf + shape_ends + [r for r in rows if r.get("kind") == "correctness"]:
        case = row.get("case")
        if not isinstance(case,int) or isinstance(case,bool) or not 0 <= case < 10 or row.get("shape") != expected[case]:
            issues.append(f"{row.get('kind')}: invalid shape/case contract")
    shapes = []
    for case, shape in enumerate(expected):
        channels = {}
        selected = [r for r in perf if r.get("case") == case]
        for channel in CHANNELS:
            variants = {}
            for name in NAMES:
                by_repeat = defaultdict(list)
                for row in selected:
                    if row.get("name") != name:
                        continue
                    if not valid_timing(row.get(channel),channel):
                        issues.append(f"case {case}/{name}/{channel}: invalid timing/sample count")
                    elif isinstance(row.get("repeat"),int) and not isinstance(row["repeat"],bool):
                        by_repeat[row["repeat"]].append(row)
                if set(by_repeat) != {0,1,2} or any(len(items) != 1 for items in by_repeat.values()):
                    issues.append(f"case {case}/{name}/{channel}: invalid repeat coverage")
                samples = [dict(repeat=repeat,**items[0][channel]) for repeat,items in sorted(by_repeat.items()) if len(items)==1]
                if samples:
                    values = [sample["median_ms"] for sample in samples]
                    variants[name] = dict(median_ms=statistics.median(values), min_round_median_ms=min(values),
                                          max_round_median_ms=max(values), rounds=samples)
            channels[channel] = dict(variants=variants, contrasts={f"{candidate}_vs_{baseline}": comparison(variants.get(candidate),variants.get(baseline))
                                                                  for candidate,baseline in CONTRASTS})
        shapes.append(dict(case=case,shape=shape,has_final=shape["state_mode"] != "none",channels=channels,
                           actual_initial_state={name: "none" if name.endswith("_none") else "nonzero_bf16" if name=="release_nonzero" else "zero_bf16" for name in NAMES}))
    status = "FAIL" if failed else "UNVERIFIED" if issues else "PASS"
    if status != "PASS":
        for item in shapes:
            for data in item["channels"].values():
                for comp in data["contrasts"].values():
                    comp["status"] = "UNVERIFIED"
    exits = [r for r in rows if r.get("kind") == "profile_exit"]
    return dict(source=source,status=status,issues=list(dict.fromkeys(issues)),
                environment=environments[0] if len(environments)==1 else {}, completion=ends,
                observed_counts=dict(shapes=len(shape_ends),performance=len(perf)),correctness=correctness,
                shapes=shapes,profile_exit_records=exits,
                limitations=["Main status audits the matched measurement schema, not promotion or every protocol recommendation.",
                             "GPU Event latency is not CPU allocation wall time. Graph zero_each is captured zero fill plus forward, not Python allocation per replay.",
                             "cache_perturbed zeros 256 MiB before the timed call; this is not cold-K2 proof.",
                             "No per-row dispatch/kernel identity, zero-buffer immutability snapshots, synchronized host wall time, or post-timing nonzero check are logged.",
                             "state_mode is input-case metadata: zero/nonzero arms intentionally have initial state even when metadata says none.",
                             "50 pre and 50 post zero-semantic checks each contain 10 legacy_none self-checks; nonzero has 10 separate old-V128 comparisons.",
                             "Medians are medians of 3 round medians. p10/p90 and across-shape ranges are not confidence intervals."])


def markdown(report):
    lines=[f"# Matched audit: {report['status']}","",f"Source: `{report['source']}`","",
           "| Case | Scope | None ms | Reused zero ms | Created zero ms | Nonzero ms | Reuse reduction | Create reduction | Create−reuse µs |",
           "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "UNVERIFIED" if value is None else f"{value:.6f}"
    for item in report["shapes"]:
        for channel,data in item["channels"].items():
            values=[fmt(data["variants"].get(name,{}).get("median_ms")) for name in NAMES[:4]]
            values += [fmt(data["contrasts"][f"{name}_vs_release_none"].get("latency_reduction_pct")) for name in ("release_zero","release_zero_each")]
            values += [fmt(data["contrasts"]["release_zero_each_vs_release_zero"].get("candidate_minus_baseline_us"))]
            lines.append(f"| {item['case']} | {channel} | "+" | ".join(values)+" |")
    lines += ["", "Case definitions: `"+json.dumps([dict(case=x["case"],shape=x["shape"]) for x in report["shapes"]],sort_keys=True)+"`",
              "", "Correctness: `"+json.dumps(report["correctness"],sort_keys=True)+"`", ""]
    lines += ["- "+item for item in report["limitations"]+report["issues"]]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log",type=Path)
    parser.add_argument("--format",choices=("json","markdown"),default="json")
    args=parser.parse_args()
    report=summarize_text(args.log.read_text(encoding="utf-8",errors="replace"),str(args.log))
    print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False) if args.format=="json" else markdown(report))


if __name__=="__main__":
    main()
