"""In-memory state-matrix fixtures; no GPU or remote work."""
import copy
import json
import unittest

import summarize_state_matrix as subject


def fixture():
    rows = [dict(kind="state_environment", candidate_sha256="fixture")]
    for case, shape in enumerate(subject.expected_shapes()):
        for name, value in zip(subject.NAMES, (1., .8, 1.2)):
            fields = ("out","final_state") if shape["state_mode"] in ("both","out") else ("out",)
            rows.append(dict(kind="state_correctness", case=case, shape=shape, name=name, status="PASS",
                             tensors={field: dict(bitwise=True, finite=True) for field in fields}))
            for repeat in range(3):
                times = {channel: dict(median_ms=value, p10_ms=value*.9, p90_ms=value*1.1,
                                       count=30 if channel == "cache_perturbed" else 60) for channel in subject.CHANNELS}
                rows.append(dict(kind="state_performance", case=case, shape=shape, name=name, repeat=repeat, **times))
        rows.append(dict(kind="state_shape_complete", case=case, shape=shape))
    rows.append(dict(kind="state_matrix_complete", shapes=27, correctness_rows=81, performance_rows=243))
    return rows


class StateMatrixTests(unittest.TestCase):
    def setUp(self):
        self.rows = fixture()

    def summarize(self):
        return subject.summarize_text("Slurm header\n" + "\n".join(json.dumps(row) for row in self.rows))

    def test_complete(self):
        report = self.summarize()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["non_reference_comparison_rows"], 54)
        self.assertEqual(report["reference_self_check_rows"], 27)
        self.assertAlmostEqual(report["shapes"][0]["channels"]["eager"]["release_auto_vs"]["baseline_auto"]["latency_reduction_pct"], 20)
        self.assertIn("not confidence intervals", subject.markdown(report))

    def test_missing_terminal(self):
        self.rows.pop()
        self.assertEqual(self.summarize()["status"], "UNVERIFIED")

    def test_wrong_terminal_counts(self):
        self.rows[-1]["performance_rows"] = 99
        self.assertEqual(self.summarize()["status"], "UNVERIFIED")

    def test_missing_or_duplicate_round(self):
        self.rows.append(copy.deepcopy(next(row for row in self.rows if row["kind"] == "state_performance")))
        self.assertEqual(self.summarize()["status"], "UNVERIFIED")

    def test_wrong_sample_count(self):
        next(row for row in self.rows if row["kind"] == "state_performance")["cache_perturbed"]["count"] = 60
        self.assertEqual(self.summarize()["status"], "UNVERIFIED")

    def test_tensor_false(self):
        next(row for row in self.rows if row["kind"] == "state_correctness")["tensors"]["out"]["bitwise"] = False
        self.assertEqual(self.summarize()["status"], "FAIL")

    def test_missing_output_field(self):
        del next(row for row in self.rows if row["kind"] == "state_correctness")["tensors"]["final_state"]
        self.assertEqual(self.summarize()["status"], "UNVERIFIED")

    def test_forbid_job_pooling(self):
        self.rows += copy.deepcopy(self.rows)
        self.assertEqual(self.summarize()["status"], "UNVERIFIED")

    def test_forbid_release_schema(self):
        self.rows.append(dict(kind="complete", sanitizer=False))
        self.assertEqual(self.summarize()["status"], "UNVERIFIED")

    def test_shape_contract(self):
        row = next(row for row in self.rows if row["kind"] == "state_performance")
        row["shape"] = dict(row["shape"], tokens=123)
        self.assertEqual(self.summarize()["status"], "UNVERIFIED")


if __name__ == "__main__":
    unittest.main()
