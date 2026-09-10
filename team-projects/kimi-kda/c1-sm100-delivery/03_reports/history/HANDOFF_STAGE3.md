# FlashKDA Hardware-Explicit Typed IR - Stage 3 Handoff

## Outcome

Stage 3 closed the first proof-carrying production activation loop. The frozen
typed winners are now reachable through explicit public
`recurrent_kda(..., backend="evolution")` dispatch. `backend="auto"` is
unchanged. The adapter preserves output identity and public in-place state
updates, caches prepared routes per CUDA stream, and falls back to CAKE for
unqualified shapes.

The certificate is `evidence/b300_stage3_activation_certificate.json`.

## Scope-identical B300 result

All measurements use the same public `recurrent_kda` call scope, rotating
initial-state pools, cold-L2 CUPTI timing, and paired CAKE/evolution order. Each
backend has two blocks of 20 dry runs plus 100 measured iterations.

| Shape | CAKE public | Deployment public | Speedup | Bootstrap 95% | Decision |
|---|---:|---:|---:|---:|---|
| H96 fixed 8192 | 763.767 us | 631.717 us | 1.2090x | [1.2066, 1.2114] | activate |
| H96 mixed | 570.614 us | 570.531 us | 1.0001x | [0.9993, 1.0008] | retain CAKE |
| H96 uniform | 638.723 us | 532.148 us | 1.2003x | [1.1971, 1.2027] | activate |
| H64 fixed 8192 | 747.988 us | 592.963 us | 1.2614x | [1.2584, 1.2642] | activate |
| H64 mixed | 397.683 us | 385.379 us | 1.0319x | [1.0312, 1.0328] | activate |
| H64 uniform | 430.370 us | 366.434 us | 1.1745x | [1.1703, 1.1773] | activate |

- Six-shape output and final-state correctness: PASS.
- Qualified activation count: 5.
- Retained CAKE count: 1.
- Guarded deployment geomean across all six shapes: 1.142037x.
- Minimum activated-shape 95% lower bound: 1.031211x.

These values are production-API-scope evidence, unlike the directional Stage 2
prepared-launch comparison.

## Implementation

Remote checkout:

`<REMOTE_HOME>/flashinfer-cake-sm103-20260908`

Modified files:

- `flashinfer/kda.py`: explicit `evolution` public backend; auto unchanged.
- `flashinfer/kda_evolution.py`: strict contract, five-shape guard, CAKE
  fallback, distinct final-state scratch/copy-back, per-stream metadata and
  prepared-route cache, dynamic state rebinding.
- `benchmarks/bench_recurrent_kda_prefill.py`: public evolution option and
  physical-route recording.
- `benchmarks/bench_flash_kda_evolution.py`: public correctness path.
- `benchmarks/bench_flash_kda_stage3_public_pair.py`: paired confidence harness.
- `tests/kda/test_flashkda_blackwell_evolution.py`: activation-policy and state
  ABI regressions.

Local review copies live under `stage3_remote/`.

## Evidence and accounting

- `evidence/b300_stage3_public_paired.json`: authoritative scope-identical
  correctness, samples, confidence intervals, and decisions.
- `evidence/b300_stage3_public_paired_accounting.json`: final successful paired
  allocation.
- `evidence/b300_stage3_activation_certificate.json`: source-to-IR, typed
  semantic delta, implementation hashes, benchmark hashes, policy, and all
  attempt-ledger links.
- `evidence/attempts/*stage3*.json`: seven append-only experiment segments,
  including environment failure 23100, wiring failure 23102, uncached adapter
  calibration 23103, and final cached adapter qualification 23104.

Total Stage 3 recorded attempts: 111.751755 GPU-s and 23.899655 kJ estimated
allocation energy. Failed and superseded measurements are included, including
the two Python-header provisioning failures and the final GPU pytest success.

## Validation

- Typed-IR local suite: 20/20 PASS.
- Remote evolution non-GPU suite: 189 PASS, 6 GPU tests deselected.
- B300 evolution GPU pytest: 6/6 PASS.
- B300 public paired six-shape correctness: PASS.
- Remote `git diff --check`: PASS.

A broader `test_recurrent_kda_prefill.py` collection produced 293 PASS and 143
skips, plus three CuTe-DSL-only import failures because `cutlass` is absent from
the remote Python environment. Those tests do not enter the CAKE/evolution path
and are not regressions from this change; the failure was retained in the task
log rather than counted as a clean full-suite pass.

## Next stage

Do not broaden `auto` in this patch. The explicit backend has now passed the
existing B300 GPU pytest coverage and is ready for maintainer review. After the
adapter lands, a separate policy patch may route the five qualified shapes from
`auto`; keeping that change separate makes rollback and review unambiguous.
Then extend proof-carrying activation to new schedule search (warp-role
allocation, completion topology, and stage/resource co-optimization) one
qualified shape family at a time.
