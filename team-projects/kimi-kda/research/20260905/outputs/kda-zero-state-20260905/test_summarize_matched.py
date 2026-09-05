"""In-memory fixtures; no GPU or remote access."""
import copy
import json
import unittest

import summarize_matched as subject


def fixture():
    rows=[dict(kind="environment",experiment="matched-zero-state",binary_sha256="fixture")]
    for case,shape in enumerate(subject.expected_shapes()):
        fields=("out",) if shape["state_mode"]=="none" else ("out","final_state")
        tensors={field:dict(bitwise=True,finite=True) for field in fields}
        for kind in ("correctness","post_timing_correctness"):
            for name in subject.ZERO_NAMES:
                row=dict(kind=kind,case=case,name=name,status="PASS",tensors=copy.deepcopy(tensors))
                if kind=="correctness":
                    row.update(shape=shape,reference_sanity=name=="legacy_none")
                rows.append(row)
        rows.append(dict(kind="nonzero_correctness",case=case,status="PASS",tensors=copy.deepcopy(tensors)))
        for name,value in zip(subject.NAMES,(1.,.8,.85,.8,1.1,1.)):
            for repeat in range(3):
                times={channel:dict(median_ms=value,p10_ms=value*.9,p90_ms=value*1.1,count=30 if channel=="cache_perturbed" else 60)
                       for channel in subject.CHANNELS}
                rows.append(dict(kind="performance",case=case,shape=shape,name=name,repeat=repeat,**times))
        rows.append(dict(kind="shape_complete",case=case,shape=shape))
    rows.append(dict(kind="matched_complete",shapes=10))
    return rows


class MatchedTests(unittest.TestCase):
    def setUp(self):
        self.rows=fixture()

    def report(self):
        return subject.summarize_text("Slurm\n"+"\n".join(json.dumps(row) for row in self.rows))

    def test_complete_and_references(self):
        report=self.report()
        self.assertEqual(report["status"],"PASS")
        self.assertEqual(report["correctness"]["correctness"]["nonself_comparisons"],40)
        self.assertEqual(report["correctness"]["post_timing_correctness"]["self_comparisons"],10)
        self.assertEqual(report["correctness"]["nonzero_correctness"]["nonself_comparisons"],10)
        self.assertAlmostEqual(report["shapes"][0]["channels"]["eager"]["contrasts"]["release_zero_vs_release_none"]["latency_reduction_pct"],20)
        self.assertIn("not confidence intervals",subject.markdown(report))

    def test_missing_complete(self):
        self.rows.pop()
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_duplicate_round(self):
        self.rows.append(copy.deepcopy(next(r for r in self.rows if r["kind"]=="performance")))
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_missing_post_check(self):
        self.rows.remove(next(r for r in self.rows if r["kind"]=="post_timing_correctness"))
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_nonzero_failure(self):
        next(r for r in self.rows if r["kind"]=="nonzero_correctness")["tensors"]["out"]["bitwise"]=False
        self.assertEqual(self.report()["status"],"FAIL")

    def test_reference_label(self):
        next(r for r in self.rows if r["kind"]=="correctness")["reference_sanity"]=True
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_missing_final_field(self):
        del next(r for r in self.rows if r["kind"]=="correctness")["tensors"]["final_state"]
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_no_output_state_arm_metadata(self):
        shape=self.report()["shapes"][9]
        self.assertEqual(shape["shape"]["state_mode"],"none")
        self.assertFalse(shape["has_final"])
        self.assertEqual(shape["actual_initial_state"]["release_zero"],"zero_bf16")

    def test_wrong_count(self):
        next(r for r in self.rows if r["kind"]=="performance")["cache_perturbed"]["count"]=60
        self.assertEqual(self.report()["status"],"UNVERIFIED")

    def test_multiple_jobs(self):
        self.rows+=copy.deepcopy(self.rows)
        self.assertEqual(self.report()["status"],"UNVERIFIED")


if __name__=="__main__":
    unittest.main()
