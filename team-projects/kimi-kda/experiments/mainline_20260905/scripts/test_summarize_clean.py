"""Pure in-memory clean-probe fixtures, including fail-closed counterexamples."""
import contextlib
import copy
import io
import json
import unittest
from unittest.mock import patch

import summarize_clean as subject


def tensors(meta=None):
    fields={"out","final_state"} if meta is None else subject.fields_for(meta)
    return {field:dict(bitwise=True,finite=True) for field in fields}


def timing(value,count):
    return dict(median_ms=value,p10_ms=.9*value,p90_ms=1.1*value,count=count)


def fixture():
    rows=[dict(kind="environment",experiment="clean-phase1",candidate_sha256="a"*64,
               baseline_sha256="b"*64,wrapper_sha256="c"*64)]
    for case,meta in enumerate(subject.correctness_shapes()):
        for mode in subject.MODES:
            rows.append(dict(kind="correctness",case=case,shape=meta,mode=mode,tensors=tensors(meta),status="PASS"))
    rows += [dict(kind="state_chain",step=step,tensors=tensors(),status="PASS") for step in range(3)]
    rows.append(dict(kind="correctness_complete",comparison_rows=120,sanitizer=False))
    for case,meta in enumerate(subject.extra_shapes()):
        for mode in subject.MODES[:2]:
            rows.append(dict(kind="extra_correctness",case=case,shape=meta,mode=mode,sanitizer=False,tensors=tensors(meta),status="PASS"))
    rows += [dict(kind="first_prefill_chain",step=step,initial_present=step>0,tensors=tensors(),status="PASS") for step in range(3)]
    for case,meta in enumerate(subject.performance_shapes()):
        for name in subject.NAMES:
            rows.append(dict(kind="graph_correctness",case=case,shape=meta,name=name,tensors=tensors(meta),status="PASS"))
        for name,value in zip(subject.NAMES,(1.,.9,2.)):
            for repeat in range(3):
                times={channel:timing(value,count) for channel,count in subject.COUNTS.items()}
                rows.append(dict(kind="performance",case=case,shape=meta,name=name,repeat=repeat,
                                 decision=dict(value_slice=128 if name=="v128" else 16),**times))
        for name in subject.NAMES[:2]:
            rows.append(dict(kind="post_correctness",case=case,shape=meta,name=name,tensors=tensors(meta),status="PASS"))
        rows.append(dict(kind="shape_complete",case=case,shape=meta))
    rows.append(dict(kind="performance_complete",shapes=40,rows=360))
    for mode in ("both","out"):
        for name,value in zip(subject.NAMES[:2],(2.,1.9)):
            for repeat in range(3):
                rows.append(dict(kind="concurrent",state_mode=mode,name=name,repeat=repeat,requests=2,pair=timing(value,30)))
        rows += [dict(kind="concurrent_correctness",state_mode=mode,case=case,tensors=tensors(),status="PASS") for case in range(2)]
    rows.append(dict(kind="clean_complete",sanitizer=False))
    return rows


def encode(rows):
    return "Slurm header\n"+"\n".join(json.dumps(row) for row in rows)+"\nSlurm footer\n"


class CleanTests(unittest.TestCase):
    def setUp(self):
        self.rows=fixture()

    def report(self):
        return subject.summarize_text(encode(self.rows))

    def test_full_contract_and_references(self):
        report=self.report()
        self.assertEqual(report["status"],"PASS",report["issues"])
        for name,count in (("correctness",120),("extra_correctness",14),("graph_correctness",120),("post_correctness",80)):
            self.assertEqual(report[name]["observed_rows"],count)
        self.assertEqual(report["graph_correctness"]["self_comparison_rows"],40)
        self.assertEqual(len(report["state_chain"]["rows"]),3)
        self.assertEqual(len(report["first_prefill_chain"]["rows"]),3)
        self.assertEqual(report["performance"]["observed_rows"],360)
        self.assertEqual(report["concurrent"]["observed_timing_rows"],12)
        self.assertEqual(len(report["concurrent"]["correctness_rows"]),4)

    def test_medians_and_paired_min_max(self):
        row=next(r for r in self.rows if r["kind"]=="performance" and r["case"]==0 and r["name"]=="phase1_auto" and r["repeat"]==2)
        row["eager"]=timing(1.2,60)
        data=self.report()["performance"]["shapes"][0]["channels"]["eager"]
        self.assertAlmostEqual(data["variants"]["phase1_auto"]["median_ms"],.9)
        self.assertAlmostEqual(data["phase1_auto_vs"]["p4_auto"]["latency_reduction_pct"],10)
        self.assertAlmostEqual(data["phase1_auto_vs"]["p4_auto"]["paired_min_reduction_pct"],-20)
        self.assertAlmostEqual(data["phase1_auto_vs"]["p4_auto"]["paired_max_reduction_pct"],10)
        self.assertAlmostEqual(data["phase1_auto_vs"]["v128"]["latency_reduction_pct"],55)

    def test_scope_domains_not_pooled(self):
        shapes=self.report()["performance"]["shapes"]
        self.assertEqual(sum(x["geometry_scope"]=="candidate_domain" for x in shapes),34)
        self.assertEqual(sum(x["geometry_scope"]=="out_of_envelope_control" for x in shapes),6)

    def test_concurrent_keeps_joined_pair(self):
        data=self.report()["concurrent"]["states"]["out"]["channels"]["pair"]
        self.assertEqual(data["variants"]["p4_auto"]["median_ms"],2.)
        self.assertAlmostEqual(data["phase1_auto_vs"]["p4_auto"]["latency_reduction_pct"],5)

    def test_missing_main_terminal(self):
        self.rows.pop()
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_sanitizer_terminal_does_not_complete_main(self):
        self.rows[-1]["sanitizer"]=True
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_missing_helper_terminal(self):
        self.rows=[r for r in self.rows if r["kind"]!="correctness_complete"]
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_wrong_performance_terminal(self):
        next(r for r in self.rows if r["kind"]=="performance_complete")["rows"]=132
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_duplicate_terminal(self):
        self.rows.append(copy.deepcopy(self.rows[-1]))
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_records_after_terminal(self):
        self.rows.insert(1,self.rows.pop())
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_missing_extra(self):
        self.rows.remove(next(r for r in self.rows if r["kind"]=="extra_correctness"))
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_first_prefill_starts_truly_absent(self):
        next(r for r in self.rows if r["kind"]=="first_prefill_chain")["initial_present"]=True
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_original_chain_cannot_substitute_prefill_chain(self):
        self.rows=[r for r in self.rows if r["kind"]!="first_prefill_chain"]
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_missing_post_timing(self):
        self.rows.remove(next(r for r in self.rows if r["kind"]=="post_correctness"))
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_missing_graph_reference_self_row(self):
        self.rows.remove(next(r for r in self.rows if r["kind"]=="graph_correctness" and r["name"]=="v128"))
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_duplicate_round(self):
        self.rows.append(copy.deepcopy(next(r for r in self.rows if r["kind"]=="performance")))
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_missing_metric(self):
        del next(r for r in self.rows if r["kind"]=="performance")["wall_sync"]
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_wrong_metric_count(self):
        next(r for r in self.rows if r["kind"]=="performance")["wall_sync"]["count"]=30
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_shape_mismatch(self):
        row=next(r for r in self.rows if r["kind"]=="performance")
        row["shape"]=dict(row["shape"],tokens=123)
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_bitwise_failure(self):
        next(r for r in self.rows if r["kind"]=="correctness")["tensors"]["out"]["bitwise"]=False
        self.assertEqual(self.report()["status"],"FAIL")

    def test_missing_tensor_field(self):
        del next(r for r in self.rows if r["kind"]=="correctness")["tensors"]["final_state"]
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_v128_must_be_forced128(self):
        next(r for r in self.rows if r["kind"]=="performance" and r["name"]=="v128")["decision"]["value_slice"]=16
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_policy_changes_not_hidden(self):
        next(r for r in self.rows if r["kind"]=="performance" and r["name"]=="phase1_auto")["decision"]["value_slice"]=32
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_missing_concurrent_correctness(self):
        self.rows.remove(next(r for r in self.rows if r["kind"]=="concurrent_correctness"))
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_wrong_concurrent_request_count(self):
        next(r for r in self.rows if r["kind"]=="concurrent")["requests"]=1
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_foreign_schema_and_multiple_jobs_not_pooled(self):
        self.rows+=copy.deepcopy(self.rows)
        report=self.report()
        self.assertEqual(report["status"],"UNVERIFIED")
        self.assertNotIn("performance",report)

    def test_missing_binary_hash(self):
        del self.rows[0]["candidate_sha256"]
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_malformed_and_nonfinite_json(self):
        for suffix in ('{"kind":','{"kind":"bad","value":NaN}'):
            self.assertEqual(subject.summarize_text(encode(self.rows)+suffix)["status"],"UNVERIFIED")

    def test_observed_external_nonzero_vetoes_pass(self):
        self.rows.append(dict(kind="sanitizer_exit",tool="memcheck",exit_code=99))
        self.assertEqual(self.report()["status"],"FAIL")

    def test_cli_fail_closed_and_stdout(self):
        self.rows.pop()
        output=io.StringIO()
        with patch("sys.argv",["summarize_clean.py","fixture.log"]), patch.object(subject.Path,"read_text",return_value=encode(self.rows)), contextlib.redirect_stdout(output):
            code=subject.main()
        self.assertEqual(code,2)
        self.assertEqual(json.loads(output.getvalue())["status"],"UNVERIFIED")


if __name__=="__main__":
    unittest.main()
