"""Fail-closed audit of one final clean_probe.py JSONL/Slurm log.

Pure stdlib, stdout only. PASS covers the main clean probe, not unprovided
sanitizer/profile/entry sidecars. CLI exit: 0=PASS, 2=FAIL or UNVERIFIED.
This schema includes both 3-step chains and 80 post-timing checks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import statistics


NAMES = ("p4_auto", "phase1_auto", "v128")
MODES = ("auto", "force16", "off")
COUNTS = {"eager":60, "graph":60, "cache_perturbed":30, "wall_sync":20}


def shape(**description):
    return dict(dict(heads=12,batch=1,lengths=None,fp32=False,state_mode="both",gate=None), **description)


def correctness_shapes():
    d=[dict(tokens=t) for t in (1,17,2047,2048,2049,3072,4095,4096,4097,6144,8191,8192,8193,16384)]
    d += [dict(tokens=t,lengths=[t] if packed else None,state_mode=s) for t in (2048,8192)
          for packed in (False,True) for s in ("in","out","none")]
    d += [dict(tokens=4096,fp32=True,state_mode=s) for s in ("both","in","out")]
    d += [dict(tokens=8192,heads=h) for h in (24,48,96)]
    d += [dict(tokens=8192,batch=b) for b in (2,4)]
    d += [dict(tokens=8192,lengths=v) for v in ([8192],[1024]*8,[16,32,512,1024,2512,4096],[0,8192])]
    d += [dict(tokens=8192,gate=g) for g in (-8.,12.)]
    return [shape(**item) for item in d]


def extra_shapes():
    return [shape(tokens=t,state_mode=s,lengths=[t] if packed else None) for t,s,packed in
            ((2049,"out",False),(4095,"out",False),(8191,"out",False),(2049,"none",False),
             (2049,"out",True),(4095,"both",True),(8191,"in",False))]


def performance_shapes():
    d=[dict(tokens=t,state_mode=s,lengths=[t] if packed else None) for t in (2048,4096,8192)
       for s in ("both","in","out","none") for packed in (False,True)]
    d += [dict(tokens=t,state_mode=s) for t in (2049,4095,8191) for s in ("both","out")]
    d += [dict(tokens=t,state_mode=s) for t in (3072,6144) for s in ("both","out")]
    d += [dict(tokens=2047,state_mode="out"),dict(tokens=8193,state_mode="out"),
          dict(tokens=4096,fp32=True),dict(tokens=8192,batch=2),dict(tokens=8192,heads=24),
          dict(tokens=8192,lengths=[1024]*8)]
    return [shape(**item) for item in d]


def status(issues=(), failed=False):
    return "FAIL" if failed else "UNVERIFIED" if issues else "PASS"


def is_index(value, count):
    return type(value) is int and 0 <= value < count


def parse(text):
    rows,issues=[],[]
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant {value}")
    for line_number,line in enumerate(text.splitlines(),1):
        if not line.lstrip().startswith("{"):
            continue
        try:
            row=json.loads(line,parse_constant=reject_constant)
        except (ValueError,json.JSONDecodeError):
            issues.append(f"line {line_number}: malformed/truncated/non-finite JSON")
            continue
        if not isinstance(row,dict):
            issues.append(f"line {line_number}: expected JSON object")
        else:
            rows.append(row)
    return rows,issues


def check_tensors(row, fields):
    tensors=row.get("tensors")
    failed=row.get("status")=="FAIL"
    issues=[]
    if row.get("status")!="PASS":
        issues.append("status is not PASS")
    if not isinstance(tensors,dict) or set(tensors)!=fields:
        issues.append("missing/unexpected tensor fields")
    if isinstance(tensors,dict):
        for value in tensors.values():
            if not isinstance(value,dict):
                issues.append("invalid tensor check")
                continue
            failed=failed or value.get("bitwise") is False or value.get("finite") is False
            if value.get("bitwise") is not True or value.get("finite") is not True:
                issues.append("bitwise/finite not established")
    return issues,failed


def fields_for(meta):
    return {"out","final_state"} if meta["state_mode"] in ("both","out") else {"out"}


def case_checks(rows, kind, shapes, names, name_field, sanitizer=None):
    checks=[row for row in rows if row.get("kind")==kind]
    valid=[]; issues=[]; failed=False
    for row in checks:
        case=row.get("case"); name=row.get(name_field)
        if not is_index(case,len(shapes)) or name not in names:
            issues.append("invalid case/name")
            continue
        valid.append((case,name))
        if row.get("shape")!=shapes[case]:
            issues.append(f"case {case}: shape mismatch")
        if sanitizer is not None and row.get("sanitizer") is not sanitizer:
            issues.append(f"case {case}: sanitizer flag mismatch/missing")
        tensor_issues,tensor_failed=check_tensors(row,fields_for(shapes[case]))
        issues += [f"case {case}/{name}: {item}" for item in tensor_issues]
        failed=failed or tensor_failed
    if Counter(valid)!=Counter((case,name) for case in range(len(shapes)) for name in names):
        issues.append("missing/duplicate case/variant coverage")
    return dict(status=status(issues,failed),issues=issues,observed_rows=len(checks),
                expected_rows=len(shapes)*len(names),
                status_counts=dict(Counter(row.get("status","MISSING") for row in checks)),
                self_comparison_rows=sum(row.get("name")=="v128" for row in checks))


def chain_checks(rows, kind):
    checks=[row for row in rows if row.get("kind")==kind]
    issues=[]; failed=False; indices=[]
    for row in checks:
        step=row.get("step")
        if not is_index(step,3):
            issues.append("invalid chain step")
        else:
            indices.append(step)
            if kind=="first_prefill_chain" and row.get("initial_present") is not (step>0):
                issues.append(f"step {step}: expected initial_present={step>0}")
        tensor_issues,tensor_failed=check_tensors(row,{"out","final_state"})
        issues += tensor_issues; failed=failed or tensor_failed
    if Counter(indices)!=Counter(range(3)):
        issues.append("expected exactly 3 unique chain steps")
    return dict(status=status(issues,failed),issues=issues,rows=checks)


def timing_valid(value,count):
    def positive(item):
        return type(item) in (int,float) and math.isfinite(item) and item>0
    return (isinstance(value,dict) and type(value.get("count")) is int and value["count"]==count
            and all(positive(value.get(key)) for key in ("median_ms","p10_ms","p90_ms"))
            and value["p10_ms"]<=value["median_ms"]<=value["p90_ms"])


def contrast(candidate,baseline):
    if not candidate or not baseline:
        return dict(status="UNVERIFIED")
    by_repeat={row["repeat"]:row["median_ms"] for row in baseline["rounds"]}
    pairs=[dict(repeat=row["repeat"],latency_reduction_pct=100*(1-row["median_ms"]/by_repeat[row["repeat"]]),
                speedup=by_repeat[row["repeat"]]/row["median_ms"],
                candidate_minus_baseline_us=1000*(row["median_ms"]-by_repeat[row["repeat"]]))
           for row in candidate["rounds"] if row["repeat"] in by_repeat]
    gains=[row["latency_reduction_pct"] for row in pairs]
    return dict(status="PASS",latency_reduction_pct=100*(1-candidate["median_ms"]/baseline["median_ms"]),
                speedup=baseline["median_ms"]/candidate["median_ms"],
                candidate_minus_baseline_us=1000*(candidate["median_ms"]-baseline["median_ms"]),
                paired_rounds=pairs,paired_min_reduction_pct=min(gains) if gains else None,
                paired_max_reduction_pct=max(gains) if gains else None,
                worst_paired_regression_pct=max(0,-min(gains)) if gains else None)


def summarize_series(records,names,counts):
    issues=[]; channels={}
    keys=[]
    for row in records:
        if row.get("name") not in names or not is_index(row.get("repeat"),3):
            issues.append("invalid variant/repeat")
        else:
            keys.append((row["name"],row["repeat"]))
    if Counter(keys)!=Counter((name,repeat) for name in names for repeat in range(3)):
        issues.append("missing/duplicated variant/repeat")
    for channel,count in counts.items():
        variants={}
        for name in names:
            by_repeat=defaultdict(list)
            for row in records:
                if row.get("name")!=name:
                    continue
                if not timing_valid(row.get(channel),count):
                    issues.append(f"{name}/{channel}: invalid timing/count (expected {count})")
                elif is_index(row.get("repeat"),3):
                    by_repeat[row["repeat"]].append(row)
            samples=[dict(repeat=repeat,**items[0][channel]) for repeat,items in sorted(by_repeat.items()) if len(items)==1]
            if samples:
                values=[item["median_ms"] for item in samples]
                variants[name]=dict(median_ms=statistics.median(values),min_round_median_ms=min(values),
                                    max_round_median_ms=max(values),rounds=samples)
        comparisons={baseline:contrast(variants.get("phase1_auto"),variants.get(baseline))
                     for baseline in names if baseline!="phase1_auto"}
        channels[channel]=dict(variants=variants,phase1_auto_vs=comparisons)
    if issues:
        for channel in channels.values():
            for comparison in channel["phase1_auto_vs"].values():
                comparison["status"]="UNVERIFIED"
    return dict(status=status(issues),issues=list(dict.fromkeys(issues)),channels=channels)


def performance_summary(rows):
    shapes=performance_shapes()
    records=[row for row in rows if row.get("kind")=="performance"]
    ends=[row for row in rows if row.get("kind")=="shape_complete"]
    issues=[]; results=[]
    if len(records)!=360:
        issues.append(f"expected 360 performance records, observed {len(records)}")
    if any(not is_index(row.get("case"),40) for row in records+ends):
        issues.append("invalid case index")
    for case,meta in enumerate(shapes):
        data=[row for row in records if type(row.get("case")) is int and row["case"]==case]
        markers=[row for row in ends if type(row.get("case")) is int and row["case"]==case]
        local=[]
        if len(markers)!=1 or markers[0].get("shape")!=meta:
            local.append("shape_complete missing/duplicated/mismatched")
        if any(row.get("shape")!=meta for row in data):
            local.append("performance shape mismatch")
        series=summarize_series(data,NAMES,COUNTS)
        local += series["issues"]
        decisions={name:[dict(repeat=row.get("repeat"),decision=row.get("decision")) for row in data if row.get("name")==name] for name in NAMES}
        slices={name:[row["decision"].get("value_slice") if isinstance(row.get("decision"),dict) else None
                      for row in data if row.get("name")==name] for name in NAMES}
        if any(type(value) is not int or value not in (16,32,64,128) for values in slices.values() for value in values):
            local.append("invalid/missing ValueSlice decision")
        if any(value!=128 for value in slices["v128"]):
            local.append("v128 control did not select V128")
        normalized=[value if type(value) is int else None for value in slices["p4_auto"]+slices["phase1_auto"]]
        if len(set(normalized))!=1:
            local.append("P4/new auto or rounds disagree on ValueSlice; not an isolated same-policy comparison")
        series.update(case=case,shape=meta,issues=local,status=status(local),decisions=decisions,
                      geometry_scope="candidate_domain" if case<34 else "out_of_envelope_control",
                      has_initial=meta["state_mode"] in ("both","in"),has_final=meta["state_mode"] in ("both","out"),
                      packed=meta["lengths"] is not None)
        if local:
            for channel in series["channels"].values():
                for comp in channel["phase1_auto_vs"].values():
                    comp["status"]="UNVERIFIED"
        issues += [f"case {case}: {item}" for item in local]
        results.append(series)
    terminal=[row for row in rows if row.get("kind")=="performance_complete"]
    if len(terminal)!=1 or terminal[0].get("shapes")!=40 or terminal[0].get("rows")!=360:
        issues.append("performance_complete(40 shapes/360 rows) missing/duplicated/mismatched")
    return dict(status=status(issues),issues=issues,observed_rows=len(records),shapes=results,markers=terminal)


def concurrent_summary(rows):
    records=[row for row in rows if row.get("kind")=="concurrent"]
    checks=[row for row in rows if row.get("kind")=="concurrent_correctness"]
    issues=[]; failed=False; states={}; keys=[]
    if len(records)!=12 or any(row.get("state_mode") not in ("both","out") for row in records):
        issues.append("expected 12 concurrent records over both/out")
    for mode in ("both","out"):
        data=[row for row in records if row.get("state_mode")==mode]
        series=summarize_series(data,NAMES[:2],{"pair":30})
        if any(row.get("requests")!=2 for row in data):
            series["issues"].append("pair must contain exactly two requests")
            series["status"]="UNVERIFIED"
        issues += [f"{mode}: {item}" for item in series["issues"]]
        states[mode]=series
    for row in checks:
        mode=row.get("state_mode"); case=row.get("case")
        if mode not in ("both","out") or not is_index(case,2):
            issues.append("invalid concurrent correctness key")
        else:
            keys.append((mode,case))
        tensor_issues,tensor_failed=check_tensors(row,{"out","final_state"})
        issues += tensor_issues; failed=failed or tensor_failed
    if Counter(keys)!=Counter((mode,case) for mode in ("both","out") for case in range(2)):
        issues.append("expected 4 unique concurrent correctness rows")
    return dict(status=status(issues,failed),issues=issues,observed_timing_rows=len(records),
                correctness_rows=checks,states=states,
                scope="Joined two-request/two-stream interval; not per-request latency or serving throughput")


def summarize_text(text,source="<memory>"):
    rows,issues=parse(text)
    environments=[row for row in rows if row.get("kind")=="environment"]
    failed=any(row.get("status")=="FAIL" for row in rows)
    ancillary=[row for row in rows if row.get("suite")=="entry_hardening" or row.get("kind") in ("sanitizer_exit","profile_exit")]
    if any(type(row.get("exit_code")) is int and row["exit_code"]!=0 for row in ancillary):
        failed=True
    report=dict(source=source,status="UNVERIFIED",issues=issues,ancillary_records=ancillary)
    # Refuse to produce combined timing statistics when invocations are mixed.
    if len(environments)!=1 or environments[0].get("experiment")!="clean-phase1":
        issues.append("expected one clean-phase1 environment; do not pool jobs, profiles or other experiments")
        report["status"]=status(issues,failed)
        return report
    environment=environments[0]
    for field in ("candidate_sha256","baseline_sha256","wrapper_sha256"):
        if not isinstance(environment.get(field),str) or not re.fullmatch(r"[0-9a-f]{64}",environment[field]):
            issues.append(f"missing/invalid {field}")
    complete=[row for row in rows if row.get("kind")=="clean_complete"]
    if len(complete)!=1 or complete[0].get("sanitizer") is not False:
        issues.append("clean_complete(sanitizer=false) missing/duplicated; sanitizer/profile cannot complete main")
    if any(row.get("kind")=="profile_complete" for row in rows):
        issues.append("profile invocation present in main measurement log")
    normal_kinds={"correctness","extra_correctness","state_chain","first_prefill_chain","graph_correctness",
                  "post_correctness","performance","shape_complete","performance_complete","concurrent","concurrent_correctness"}
    if len(complete)==1:
        end_index=rows.index(complete[0])
        if any(row.get("kind") in normal_kinds for row in rows[end_index+1:]):
            issues.append("main records appear after clean_complete")
    correct=case_checks(rows,"correctness",correctness_shapes(),MODES,"mode")
    correct_ends=[row for row in rows if row.get("kind")=="correctness_complete"]
    if len(correct_ends)!=1 or correct_ends[0].get("sanitizer") is not False or correct_ends[0].get("comparison_rows")!=120:
        correct["issues"].append("correctness_complete(sanitizer=false,comparison_rows=120) missing/duplicated/mismatched")
        correct["status"]=status(correct["issues"],correct["status"]=="FAIL")
    sections=dict(correctness=correct,
                  extra_correctness=case_checks(rows,"extra_correctness",extra_shapes(),MODES[:2],"mode",False),
                  state_chain=chain_checks(rows,"state_chain"),
                  first_prefill_chain=chain_checks(rows,"first_prefill_chain"),
                  graph_correctness=case_checks(rows,"graph_correctness",performance_shapes(),NAMES,"name"),
                  post_correctness=case_checks(rows,"post_correctness",performance_shapes(),NAMES[:2],"name"),
                  performance=performance_summary(rows),concurrent=concurrent_summary(rows))
    for name,section in sections.items():
        issues += [f"{name}: {item}" for item in section["issues"]]
        failed=failed or section["status"]=="FAIL"
    report.update(environment=environment,complete_markers=complete,**sections,status=status(issues,failed),
                  scope_notes=["PASS audits this main clean probe only. Entry/sanitizer/profile sidecars require independent audits; observed external FAIL/nonzero exits still veto PASS.",
                               "120 helper comparisons, 14 extra comparisons, original state_chain3, first_prefill_chain3, graph120 (40 V128 self-checks), post80, concurrent12 timings/4 checks are distinct contracts.",
                               "Each reported latency is the median of 3 round medians; p10/p90 and paired min/max are descriptive, not confidence intervals.",
                               "No pooled jobs. All contrasts are within this binary pair/job and this measurement scope.",
                               "candidate_domain is metadata-based intended geometry, not proof of the selected compiled Phase1 subvariant.",
                               "Cache perturbation is pre-call, not cold-K2 proof. Synced host wall is not asynchronous throughput; concurrent pair is not half a request latency."])
    if report["status"]!="PASS":
        for item in sections["performance"]["shapes"]:
            for channel in item["channels"].values():
                for comp in channel["phase1_auto_vs"].values():
                    comp["status"]="UNVERIFIED"
        for item in sections["concurrent"]["states"].values():
            for channel in item["channels"].values():
                for comp in channel["phase1_auto_vs"].values():
                    comp["status"]="UNVERIFIED"
    return report


def markdown(report):
    lines=[f"# Clean main-probe audit: {report['status']}","",f"Source: `{report['source']}`","",
           "| Case / shape | Scope | P4 ms | Phase1 ms | V128 ms | Gain vs P4 | Gain vs V128 | Paired min/max vs P4 |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(value,percent=False):
        return "UNVERIFIED" if value is None else f"{value:.3f}%" if percent else f"{value:.6f}"
    for item in report.get("performance",{}).get("shapes",[]):
        meta=item["shape"]
        label=f"{item['case']}: T{meta['tokens']} H{meta['heads']} B{meta['batch']} {meta['state_mode']} lengths={meta['lengths']} fp32={meta['fp32']}"
        for channel,data in item["channels"].items():
            old,full=(data["phase1_auto_vs"][name] for name in ("p4_auto","v128"))
            values=[fmt(data["variants"].get(name,{}).get("median_ms")) for name in NAMES]
            values += [fmt(old.get("latency_reduction_pct"),True),fmt(full.get("latency_reduction_pct"),True),
                       fmt(old.get("paired_min_reduction_pct"),True)+" / "+fmt(old.get("paired_max_reduction_pct"),True)]
            lines.append(f"| {label} | {channel} | "+" | ".join(values)+" |")
    lines += ["","## Concurrent pair","", "```json",json.dumps(report.get("concurrent",{}),ensure_ascii=False,indent=2),"```",""]
    lines += ["- "+item for item in report.get("scope_notes",[])+report["issues"]]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log",type=Path)
    parser.add_argument("--format",choices=("json","markdown"),default="json")
    args=parser.parse_args()
    report=summarize_text(args.log.read_text(encoding="utf-8",errors="replace"),str(args.log))
    print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False) if args.format=="json" else markdown(report))
    return 0 if report["status"]=="PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
