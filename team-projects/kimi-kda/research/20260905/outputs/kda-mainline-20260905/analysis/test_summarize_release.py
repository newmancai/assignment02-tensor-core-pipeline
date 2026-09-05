"""In-memory schema fixtures; no files, GPUs, or optional dependencies required."""
import copy
import json
import unittest

import summarize_release as subject


def encode(rows):
    return "Slurm fixture header\n" + "\n".join(json.dumps(row) for row in rows) + "\nSlurm footer\n"


def tensors(meta=None):
    fields = ("out", "final_state") if meta is None or meta["state_mode"] in ("both", "out") else ("out",)
    return {key: dict(bitwise=True, finite=True) for key in fields}


def timing(value):
    return dict(median_ms=value, p10_ms=value*.9, p90_ms=value*1.1, count=60)


def correct_rows(sanitizer=False):
    rows = [dict(kind="environment", candidate_sha256="fixture-sha")]
    for index, meta in enumerate(subject.expected_shapes(sanitizer=sanitizer)):
        for mode in subject.MODES:
            rows.append(dict(kind="correctness", case=index, mode=mode, shape=meta, tensors=tensors(meta), status="PASS"))
    rows.append(dict(kind="correctness_complete", comparison_rows=6 if sanitizer else 120, sanitizer=sanitizer))
    return rows


def fixture():
    rows = [dict(case=name, status="PASS") for name in sorted(subject.ENTRY_CASES)]
    rows.append(dict(suite="entry_hardening", status="PASS", failures=[], skipped=[]))
    rows += correct_rows()
    rows += [dict(kind="state_chain", step=i, tensors=tensors(), status="PASS") for i in range(3)]
    for index, meta in enumerate(subject.expected_shapes("performance")):
        for repeat in range(3):
            for name, value in zip(subject.NAMES, (1., .8, 1.2)):
                row = dict(kind="performance", case=index, shape=meta, repeat=repeat, name=name, decision={"value_slice": 16})
                row.update({channel: timing(value*(1+repeat*.1)) for channel in subject.CHANNELS})
                rows.append(row)
        rows.append(dict(kind="shape_complete", case=index, shape=meta))
    for repeat in range(3):
        for name, value in (("baseline_auto", 2.), ("release_auto", 1.9)):
            rows.append(dict(kind="concurrent", shape=subject.shape(tokens=8192), name=name,
                             repeat=repeat, requests=2, pair=timing(value*(1+repeat*.1))))
    rows += [dict(kind="concurrent_correctness", case=i, tensors=tensors(), status="PASS") for i in range(2)]
    rows.append(dict(kind="complete", sanitizer=False))
    rows += [dict(kind="sanitizer_exit", tool=tool, exit_code=0) for tool in ("memcheck", "synccheck")]
    rows += [dict(kind="profile_exit", variant=name, exit_code=0) for name in ("baseline", "release")]
    sanitizer = encode(correct_rows(True)+[dict(kind="complete", sanitizer=True)]) + "========= ERROR SUMMARY: 0 errors\n"
    logs = dict(sanitizer_logs={name: sanitizer for name in ("memcheck", "synccheck")},
                profile_logs={name: encode([dict(kind="environment", candidate_sha256="fixture-sha"),
                                           dict(kind="profile_complete", mode=name)]) for name in ("baseline", "release")})
    return rows, logs


class ReleaseSummaryTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.logs = fixture()

    def summarize(self):
        return subject.summarize_text(encode(self.rows), **self.logs)

    def test_complete_contract_and_gain(self):
        result = self.summarize()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["correctness"]["observed_rows"], 120)
        self.assertEqual(result["performance"]["observed_rows"], 99)
        eager = result["performance"]["shapes"][0]["channels"]["eager"]
        self.assertAlmostEqual(eager["variants"]["release_auto"]["median_ms"], .88)
        self.assertEqual(len(eager["variants"]["release_auto"]["rounds"]), 3)
        self.assertAlmostEqual(eager["release_auto_vs"]["baseline_auto"]["latency_reduction_pct"], 20)
        self.assertAlmostEqual(eager["release_auto_vs"]["release_v128"]["latency_reduction_pct"], 100/3)
        self.assertIn("not confidence intervals", " ".join(result["interpretation"]))
        self.assertIn("cache_perturbed", subject.markdown(result))

    def test_only_sanitizer_complete_cannot_complete_main(self):
        self.rows = correct_rows(True)+[dict(kind="complete", sanitizer=True)]
        result = self.summarize()
        self.assertEqual(result["status"], "UNVERIFIED")
        self.assertFalse(result["main_complete_verified"])
        self.assertEqual(result["correctness"]["observed_rows"], 0)

    def test_main_missing_complete_not_repaired_by_sanitizer_session(self):
        self.rows = [r for r in self.rows if r.get("kind") != "complete"]
        self.rows += correct_rows(True)+[dict(kind="complete", sanitizer=True)]
        result = self.summarize()
        self.assertFalse(result["main_complete_verified"])
        self.assertEqual(result["status"], "UNVERIFIED")

    def test_separate_sanitizer_session_not_pooled(self):
        self.rows += correct_rows(True)+[dict(kind="complete", sanitizer=True)]
        result = self.summarize()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["correctness"]["observed_rows"], 120)

    def test_two_main_runs_refuse_pooling(self):
        self.rows += copy.deepcopy(self.rows)
        self.assertEqual(self.summarize()["status"], "UNVERIFIED")

    def test_missing_correctness_terminal(self):
        self.rows = [r for r in self.rows if r.get("kind") != "correctness_complete"]
        self.assertEqual(self.summarize()["correctness"]["status"], "UNVERIFIED")

    def test_wrong_correctness_count(self):
        next(r for r in self.rows if r.get("kind") == "correctness_complete")["comparison_rows"] = 6
        self.assertEqual(self.summarize()["correctness"]["status"], "UNVERIFIED")

    def test_missing_state_chain_and_concurrent_correctness(self):
        self.rows = [r for r in self.rows if not (r.get("kind") == "state_chain" and r["step"] == 2)
                     and not (r.get("kind") == "concurrent_correctness" and r["case"] == 1)]
        result = self.summarize()
        self.assertEqual(result["state_chain"]["status"], "UNVERIFIED")
        self.assertEqual(result["concurrent_correctness"]["status"], "UNVERIFIED")

    def test_actual_mismatch_fails(self):
        next(r for r in self.rows if r.get("kind") == "correctness")["tensors"]["out"]["bitwise"] = False
        self.assertEqual(self.summarize()["status"], "FAIL")

    def test_missing_and_duplicate_repeat_unverified(self):
        row = next(r for r in self.rows if r.get("kind") == "performance")
        self.rows.append(copy.deepcopy(row))
        result = self.summarize()
        self.assertEqual(result["performance"]["status"], "UNVERIFIED")
        self.assertEqual(result["performance"]["shapes"][0]["channels"]["eager"]["release_auto_vs"]["baseline_auto"]["status"], "UNVERIFIED")

    def test_worst_paired_round_preserved_not_hidden_by_median(self):
        row = next(r for r in self.rows if r.get("kind") == "performance" and r["case"] == 0 and r["name"] == "release_auto" and r["repeat"] == 2)
        row["eager"] = timing(1.32)
        comp = self.summarize()["performance"]["shapes"][0]["channels"]["eager"]["release_auto_vs"]["baseline_auto"]
        self.assertAlmostEqual(comp["latency_reduction_pct"], 20)
        self.assertAlmostEqual(comp["worst_paired_round_regression_pct"], 10)

    def test_concurrent_is_pair_not_divided_by_two(self):
        data = self.summarize()["concurrent"]["shapes"][0]["channels"]["pair"]
        self.assertAlmostEqual(data["variants"]["baseline_auto"]["median_ms"], 2.2)
        self.assertAlmostEqual(data["release_auto_vs"]["baseline_auto"]["latency_reduction_pct"], 5)

    def test_nonzero_real_exit_beats_success_sidecar(self):
        next(r for r in self.rows if r.get("kind") == "sanitizer_exit")["exit_code"] = 99
        self.assertEqual(self.summarize()["status"], "FAIL")

    def test_missing_profile_exit_unverified_even_with_profile_complete(self):
        self.rows = [r for r in self.rows if r.get("kind") != "profile_exit"]
        self.assertEqual(self.summarize()["profiles"]["release"]["status"], "UNVERIFIED")

    def test_missing_sidecars_are_not_pass(self):
        result = subject.summarize_text(encode(self.rows))
        self.assertEqual(result["status"], "UNVERIFIED")
        self.assertEqual(result["sanitizer"]["memcheck"]["status"], "PASS")
        self.assertEqual(result["sanitizer"]["memcheck"]["sidecar"]["status"], "UNVERIFIED")

    def test_sanitizer_nonzero_error_summary_fails(self):
        self.logs["sanitizer_logs"]["memcheck"] = self.logs["sanitizer_logs"]["memcheck"].replace("0 errors", "1 error")
        self.assertEqual(self.summarize()["status"], "FAIL")

    def test_profile_wrong_binary_unverified(self):
        self.logs["profile_logs"]["release"] = self.logs["profile_logs"]["release"].replace("fixture-sha", "other-sha")
        self.assertEqual(self.summarize()["status"], "UNVERIFIED")

    def test_entry_skips_explicit(self):
        self.rows.append(dict(check="alias_default", status="SKIP", reason="alias not supplied"))
        result = self.summarize()
        self.assertEqual(result["status"], "PASS_WITH_SKIPS")
        self.assertEqual(len(result["entry_hardening"]["skips"]), 1)

    def test_missing_entry_case_unverified(self):
        self.rows = [r for r in self.rows if r.get("case") != "multi_gpu"]
        self.assertEqual(self.summarize()["entry_hardening"]["status"], "UNVERIFIED")

    def test_truncated_json_unverified(self):
        self.assertEqual(subject.summarize_text(encode(self.rows)+'{"kind":', **self.logs)["status"], "UNVERIFIED")


if __name__ == "__main__":
    unittest.main()
