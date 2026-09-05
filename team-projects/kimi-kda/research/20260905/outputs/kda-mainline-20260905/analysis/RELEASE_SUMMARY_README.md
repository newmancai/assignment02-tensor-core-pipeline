# Clean release log audit

`summarize_release.py` uses only the Python standard library, never imports the GPU probe, reads inputs without changing them, and emits JSON (default) or Markdown to stdout. It implements the `release_probe.py` schema as of 2026-09-05: 40 correctness cases × 3 modes, 3 recurrent state-chain steps, 11 performance shapes × 3 variants × 3 rounds, and 2 concurrent correctness checks. A changed probe schema requires a corresponding explicit audit update.

Run from the workspace root:

```sh
python3 outputs/kda-mainline-20260905/analysis/summarize_release.py \
  outputs/kda-mainline-20260905/release_19901.log \
  --memcheck-log outputs/kda-mainline-20260905/release_19901_memcheck.log \
  --synccheck-log outputs/kda-mainline-20260905/release_19901_synccheck.log \
  --baseline-profile-log outputs/kda-mainline-20260905/release_19901_baseline_ncu.log \
  --release-profile-log outputs/kda-mainline-20260905/release_19901_release_ncu.log \
  --format markdown
```

Omit `--format markdown` for all per-round median/p10/p90/count values, dispatch decisions, and paired-round comparisons in structured JSON. `--repeats N` changes only the expected single-request performance rounds; the probe's concurrent test is hardcoded to 3 rounds.

The main log is required; all four sidecar arguments are optional to run the tool, but omitted sidecars explicitly produce `UNVERIFIED` coverage. The report's top-level status is `PASS`, `PASS_WITH_SKIPS`, `UNVERIFIED`, or `FAIL`. Read that status; normal CLI completion does not by itself mean the audited experiment passed.

- Main `complete` and `correctness_complete` must explicitly have `sanitizer=false`. Each sanitizer sidecar must instead contain exactly 6 correctness rows, both `sanitizer=true` terminal markers, and a zero-error summary. An environment record delimits a probe invocation, so a sanitizer run cannot supply the main run's completion or correctness rows.
- Main-log `sanitizer_exit` and `profile_exit` records check actual recorded exit codes separately from sidecar completion. A nonzero exit cannot be repaired by a successful-looking sidecar. Sidecar candidate SHA256 must match the main run. Profile verification confirms recorded invocation completion, not the validity of a performance-causality interpretation.
- Entry-suite skips remain visible, including check-level alias skips that are absent from the suite's own `skipped` list. Missing suite or case coverage is `UNVERIFIED`.
- Duplicated or missing cases/variants/repeats/terminal markers are not pooled or silently averaged. Multiple main runs must be audited separately.
- The reported latency is the median of the three round medians. Positive reduction means faster: `100 × (1 − release_auto / baseline)`. Paired repeat comparisons retain the worst observed regression. Within-round p10/p90 values are **not confidence intervals**.
- A concurrent measurement is one joined interval for two requests, not half that time and not a serving-throughput claim. Cache perturbation occurs before the first timed event, not between K1 and K2; this does not establish cold K2 workspace behavior. `release_auto` does not imply every shape ran P4.

Run the in-memory fixtures (no GPU, remote access, or fixture files):

```sh
python3 -m unittest discover \
  -s outputs/kda-mainline-20260905/analysis \
  -p 'test_summarize_release.py' -v
```

Initial validation: 20 fixtures passed, including false main-completion from sanitizer logs, duplicate sessions/repeats, missing markers and exits, nonzero sanitizer errors, SHA mismatch, explicit entry skips, and an aggregate gain hiding a losing paired round. Job 19901 with all four sidecars audited as `PASS_WITH_SKIPS` (alias and multi-GPU checks skipped); no remote/GPU work was done by this summarizer.
