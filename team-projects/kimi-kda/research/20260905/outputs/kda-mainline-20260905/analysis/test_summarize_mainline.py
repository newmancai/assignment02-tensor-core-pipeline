"""CPU-only in-memory fixtures for completion and regression reporting."""

import json
import unittest

from summarize_mainline import markdown, summarize_text


MODES = {16: "v16", 32: "v32", 64: "v64", 128: "v128", 216: "v16_prefetch2"}


def fixture(*, shapes=1, complete=True, correctness=True, skipped=False):
    rows = []
    if skipped:
        rows += [
            {"check": "alias_default", "status": "SKIP", "reason": "not supplied"},
            {"check": "multi_gpu", "status": "SKIP", "reason": "one device"},
            {"case": "binding", "status": "PASS"},
            {"case": "multi_gpu", "status": "SKIP"},
            {"suite": "entry_hardening", "status": "PASS", "failures": [], "skipped": ["multi_gpu"]},
        ]
    rows.append({"kind": "environment", "modes": MODES, "candidate_sha256": "fixture-only"})
    if correctness:
        rows += [
            {"kind": "correctness", "case": 0, "mode": 216, "name": "v16_prefetch2", "status": "PASS",
             "tensors": {"out": {"bitwise": True, "finite": True}}},
            {"kind": "correctness_complete", "comparison_rows": 1},
        ]
    medians = {"v16": 1.4, "v32": 1.2, "v64": 1.0, "v128": 2.0,
               "legacy16": 1.5, "legacy128": 2.1, "v16_prefetch2": 0.9}
    for index in range(shapes):
        shape = {"tokens": 8192 + index * 16, "heads": 12, "batch": 1,
                 "lengths": None, "fp32": False, "state_mode": "both", "gate": None}
        for repeat in range(3):
            for name, median in medians.items():
                if name == "v16_prefetch2":
                    median = (0.8, 0.9, 1.6)[repeat] if index == 0 else 1.2
                row = {"kind": "performance", "case": index, "shape": shape,
                       "name": name, "repeat": repeat}
                for channel, factor in (("eager", 1), ("graph", 0.8)):
                    row[channel] = {"median_ms": median * factor, "min_ms": median * factor * 0.8,
                                    "p10_ms": median * factor * 0.9, "p90_ms": median * factor * 1.1, "count": 60}
                rows.append(row)
        rows.append({"kind": "shape_complete", "case": index, "shape": shape})
    if complete:
        rows.append({"kind": "complete"})
    return rows


def encode(rows):
    return "Sat Sep 5 UTC\nsrun: running fixture\n" + "\n".join(json.dumps(row) for row in rows)


class SummaryTests(unittest.TestCase):
    def test_complete_and_four_baselines(self):
        result = summarize_text(encode(fixture()))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["correctness"]["status_counts"], {"PASS": 1})
        self.assertEqual(result["complete_count"], 1)
        self.assertEqual(result["ignored_non_json_lines"], 2)
        eager = result["shapes"][0]["channels"]["eager"]
        self.assertEqual(eager["best_original_slice"], "v64")
        candidate = eager["variants"]["v16_prefetch2"]
        self.assertEqual(candidate["median_ms"], 0.9)
        self.assertEqual(len(candidate["rounds"]), 3)
        expected = {"same_binary_v16": 100 * (1 - 0.9 / 1.4), "legacy16": 40,
                    "same_binary_v128": 55, "best_original_slice": 10}
        for baseline, gain in expected.items():
            self.assertAlmostEqual(candidate["comparisons"][baseline]["latency_reduction_pct"], gain)
        self.assertAlmostEqual(candidate["comparisons"]["best_original_slice"]["worst_paired_regression_pct"], 60)
        self.assertIn("p90_ms", candidate["rounds"][0])
        self.assertAlmostEqual(result["shapes"][0]["channels"]["graph"]["variants"]["v16_prefetch2"]["median_ms"], 0.72)

    def test_worst_shape_is_not_hidden(self):
        result = summarize_text(encode(fixture(shapes=2)))
        worst = next(row for row in result["worst_cases"] if row["name"] == "v16_prefetch2"
                     and row["channel"] == "eager" and row["baseline"] == "best_original_slice")
        self.assertAlmostEqual(worst["worst_shape_regression_pct"], 20)
        self.assertEqual(worst["worst_shape"]["shape"]["tokens"], 8208)
        self.assertEqual(worst["worst_paired_shape"]["shape"]["tokens"], 8192)

    def test_no_complete_never_passes(self):
        result = summarize_text(encode(fixture(complete=False)))
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["correctness"]["status"], "PASS")

    def test_missing_repeat_incomplete(self):
        rows = [row for row in fixture() if not (row.get("kind") == "performance"
                and row.get("name") == "v16_prefetch2" and row.get("repeat") == 2)]
        result = summarize_text(encode(rows))
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertTrue(any("expected repeats" in issue for issue in result["issues"]))

    def test_missing_shape_completion_incomplete(self):
        result = summarize_text(encode([row for row in fixture() if row.get("kind") != "shape_complete"]))
        self.assertEqual(result["status"], "INCOMPLETE")

    def test_missing_original_slice_has_no_oracle(self):
        rows = [row for row in fixture() if row.get("name") != "v32"]
        result = summarize_text(encode(rows))
        self.assertEqual(result["status"], "INCOMPLETE")
        channel = result["shapes"][0]["channels"]["eager"]
        self.assertIsNone(channel["best_original_slice"])
        comparison = channel["variants"]["v16_prefetch2"]["comparisons"]["best_original_slice"]
        self.assertEqual(comparison["status"], "MISSING_BASELINE")

    def test_duplicate_repeat_not_pooled(self):
        rows = fixture()
        rows.append(next(row for row in rows if row.get("kind") == "performance"))
        result = summarize_text(encode(rows))
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["shapes"][0]["channels"]["eager"]["variants"]["v16"]["repeat_count"], 3)

    def test_correctness_count_mismatch(self):
        rows = fixture()
        next(row for row in rows if row.get("kind") == "correctness_complete")["comparison_rows"] = 2
        self.assertEqual(summarize_text(encode(rows))["status"], "INCOMPLETE")

    def test_false_tensor_invalidates_pass_label(self):
        rows = fixture()
        next(row for row in rows if row.get("kind") == "correctness")["tensors"]["out"]["finite"] = False
        result = summarize_text(encode(rows))
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(result["failure_detected"])

    def test_failure_without_complete_still_incomplete(self):
        rows = fixture(complete=False)
        next(row for row in rows if row.get("kind") == "correctness")["status"] = "FAIL"
        result = summarize_text(encode(rows))
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertTrue(result["failure_detected"])

    def test_correctness_not_recorded_is_not_pass(self):
        result = summarize_text(encode(fixture(correctness=False)))
        self.assertEqual(result["status"], "UNVERIFIED")
        self.assertEqual(result["correctness"]["status"], "NOT_RECORDED")

    def test_skip_preserved_not_counted_as_pass(self):
        result = summarize_text(encode(fixture(skipped=True)))
        self.assertEqual(result["status"], "PASS_WITH_SKIPS")
        entry = result["entry_hardening"]
        self.assertEqual(entry["case_status_counts"], {"PASS": 1, "SKIP": 1})
        self.assertEqual(entry["skipped_cases"], ["multi_gpu"])
        self.assertEqual(len(entry["skipped_checks"]), 2)

    def test_entry_requires_terminal_suite(self):
        rows = [row for row in fixture(skipped=True) if row.get("suite") != "entry_hardening"]
        self.assertEqual(summarize_text(encode(rows))["status"], "INCOMPLETE")

    def test_entry_failure_overrides_suite_pass(self):
        rows = fixture(skipped=True)
        rows.insert(0, {"case": "alignment", "status": "FAIL", "returncode": -6})
        result = summarize_text(encode(rows))
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["entry_hardening"]["failure_cases"], ["alignment"])

    def test_nonpositive_latency_does_not_form_a_speedup(self):
        rows = fixture()
        next(row for row in rows if row.get("kind") == "performance")["eager"]["median_ms"] = 0
        result = summarize_text(encode(rows))
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertTrue(any("invalid latency" in issue for issue in result["issues"]))

    def test_truncated_json_and_multiple_runs(self):
        self.assertEqual(summarize_text(encode(fixture()) + '\n{"kind":')["status"], "INCOMPLETE")
        self.assertEqual(summarize_text(encode(fixture() + fixture()))["status"], "INCOMPLETE")

    def test_markdown_shows_rounds_and_limit(self):
        rendered = markdown(summarize_text(encode(fixture(skipped=True))))
        self.assertIn("r2=1.600000", rendered)
        self.assertIn("never confidence intervals", rendered)
        self.assertIn("SKIP multi_gpu", rendered)
        self.assertIn("vs legacy16", rendered)


if __name__ == "__main__":
    unittest.main()
