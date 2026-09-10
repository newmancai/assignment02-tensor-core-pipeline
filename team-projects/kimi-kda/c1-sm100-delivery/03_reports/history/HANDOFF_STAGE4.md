# FlashKDA Hardware-Explicit Typed IR - Stage 4 Handoff

## Outcome

Stage 4 moved the evolution loop from H64/H96 proxy workloads to the official
Kimi-K3 TP8 H12 recurrent-KDA operator profile. Route counterfactuals first
showed that the existing dispatcher already chose the best available route
family on all six shapes. Intra-route evolution then found a new, qualified
physical policy: for H12 BT16 prepare-chain workloads with more than 128 total
chunks, use 9 chunks per prepare CTA instead of 4. H12 workloads at or below
128 chunks retain 1 chunk per CTA.

The proof-carrying result is
`evidence/b300_stage4_h12_activation_certificate.json`.

## Why this is a Runtime Profile Agent result

The loop was workload-bound, counterfactual, and failure-aware:

1. Bind the experiment to Kimi-K3's H12/D128/BF16 TP8 contract.
2. Reproduce the public CAKE route and physical receipt for every shape.
3. Eliminate route-selection changes by forcing all available route families.
4. Search legal intra-route work assignments.
5. Reject an invalid short-carrier expansion after it exposed a real memory
   safety precondition; retain the failed allocation in the ledger.
6. Refine the winning neighborhood and qualify it with interleaved public-API
   measurements, correctness, bootstrap intervals, and explicit fallback.
7. Promote the discovered knob into typed IR as
   `RecurrencePlan.prepare_chunks_per_cta`, with verifier, lowering, mutation,
   and tests.

## Kimi-K3 TP8 profile

All six cases use 12 heads, QK/V dimension 128, BF16 activations, FP32
`A_log`/`dt_bias`, fused QK L2 normalization, fused gating, beta logits, an
initial state, and gate lower bound -5. The six sequence layouts are fixed-512,
fixed-8192, packed 128x8, packed 512x32, packed 1024x8, and the official mixed
vector `[1300, 547, 2048, 963, 271, 3063]`.

## Scope-identical B300 qualification

The authoritative comparison uses one prepared public `recurrent_kda` object
per shape and changes only the H12 long-BT16 chunks-per-CTA policy. Each arm has
two blocks of 20 dry runs and 100 measured iterations, CUPTI timing, cold L2,
256 rotating initial-state slots, and order
baseline/candidate/candidate/baseline.

| Shape | Baseline | Candidate | Speedup | Bootstrap 95% | Decision |
|---|---:|---:|---:|---:|---|
| H12 packed 512x32 | 178.945 us | 178.625 us | 1.0018x | [1.0005, 1.0028] | unaffected |
| H12 packed 128x8 | 26.336 us | 26.368 us | 0.9988x | [0.9964, 1.0012] | unaffected |
| H12 fixed 512 | 49.504 us | 49.568 us | 0.9987x | [0.9955, 1.0026] | retain cpc=1 |
| H12 fixed 8192 | 516.803 us | 498.433 us | 1.0369x | [1.0344, 1.0391] | activate cpc=9 |
| H12 packed mixed | 256.081 us | 236.961 us | 1.0807x | [1.0789, 1.0830] | activate cpc=9 |
| H12 packed 1024x8 | 107.360 us | 106.048 us | 1.0124x | [0.9998, 1.0160] | unaffected |

- Correctness: PASS for outputs and final states.
- Directional counterfactual at cpc=9: exact zero max-absolute delta.
- Activated-shape geomean: 1.058545x.
- Guarded six-shape geomean: 1.019146x.
- Activated-shape minimum 95% lower bound: 1.034432x.

The post-change public medians are 179.233, 26.464, 49.728, 497.922,
237.264, and 106.065 microseconds in preset order.

## Implementation and review artifact

Remote branch: `codex/kda-h12-physical-stage4`

Remote commit: `986c1eb94cc2aa185befbd4bf59fb40a0a04eaa4`

The production patch changes one constant in `flashinfer/kda_prefill.py` and
the corresponding policy assertion in
`tests/kda/test_recurrent_kda_prefill.py`. The portable patch is
`output/stage4_patches/0001-perf-kda-tune-H12-long-prepare-grid-on-Blackwell.patch`.

The local typed IR additionally exposes `prepare_chunks_per_cta`, rejects it on
non-prepare recurrence, carries it through lowering, and provides a typed
mutation. Typed-IR tests are 21/21 PASS; the targeted FlashInfer policy test is
1/1 PASS; remote `git diff --check` passes.

## Evidence and accounting

- `evidence/b300_stage4_h12_cake.json`: reproduced H12 public baseline.
- `evidence/b300_stage4_h12_route_search.json`: route-family counterfactuals.
- `evidence/b300_stage4_h12_physical_search.json`: coarse physical search.
- `evidence/b300_stage4_h12_cpc_refine.json`: local refinement around the winner.
- `evidence/b300_stage4_h12_cpc_pair_v2.json`: authoritative paired result.
- `evidence/b300_stage4_h12_cpc9_public.json`: deployed-policy public medians.
- `evidence/attempts/*stage4*.json`: ten append-only experiment records.

Stage 4 used 49.754904 recorded B300 GPU-seconds and an estimated 11.092740 kJ,
including three failed attempts. One failure found a route-recorder blind spot;
the other two established that the short H12 carrier is not a freely
interchangeable specialization.

## Scientific claim boundary

This is a new incremental improvement over an existing CAKE-generated H12
route on B300. It does not yet reproduce or exceed CAKE's 2.05x B200 result
against official FlashKDA, and it is not yet an end-to-end Kimi-K3 serving
result. Those comparisons use different baselines, hardware, and scope.

## Stage 5

The next falsifiable target is to replace the H12-specific constant with a
held-out-tested, architecture-normalized prepare-grid policy derived from total
chunks, SM waves, and sequence skew. Qualification should add B200, unseen H12
sequence distributions, prepare/chain CUPTI attribution, and an end-to-end
Kimi-K3 SGLang measurement. Only after these pass should the paper claim
generalization beyond a two-shape activation.
