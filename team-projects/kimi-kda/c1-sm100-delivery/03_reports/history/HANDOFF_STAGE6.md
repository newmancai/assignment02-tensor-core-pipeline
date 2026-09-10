# FlashKDA Multi-Agent Runtime Profile Evolution - Stage 6 Handoff

## Outcome

Stage 6 turns the Stage 4/5 workflow into the first executable slice of
Multi-Agent Runtime Profile Evolution (MARPE). Specialized agents may propose,
criticize, or merge typed schedules, but deterministic verification,
correctness, measurement, and activation remain the only authorities.

This stage does **not** claim multi-agent emergence. That claim requires an
equal-budget B300 ablation against single-agent, role-play, homogeneous pooled,
and no-communication controls.

## Implemented vertical slice

- `kda_ir/runtime_profile.py`
  - workload features: total/max chunks, effective sequence parallelism,
    max-chunk fraction, chunk CV;
  - B300-normalized prepare-grid receipt: rectangular/scheduled CTAs, SM waves,
    tail utilization, wave-quantization status.
- `kda_ir/multi_agent.py`
  - typed Agent roles and contributions;
  - verify-before-merge coordination;
  - structural schedule deduplication with contributor provenance;
  - deterministic `REJECT / MEASURE / ACTIVATE` evidence gate.
- `tests/test_multi_agent_runtime.py`
  - consensus cannot bypass `KIR404`;
  - duplicate typed schedules are measured once;
  - CI crossing one cannot activate;
  - profile/grid arithmetic is regression-tested.
- `analyze_runtime_profiles.py`
  - joins Stage 4 development and Stage 5 held-out evidence with runtime
    profile receipts;
  - emits `evidence/b300_stage6_runtime_profile_analysis.json`.

Local typed-IR suite: **25/25 PASS**.

## Research synthesis

`RESEARCH_RETROSPECTIVE_MULTI_AGENT.md` reviews Agent-System Interface, MASAI,
Google AI co-scientist, AlphaEvolve, Astra, CudaForge, Meta KernelFalcon and
KernelAgent, KernelArc, and KernelBench-Verified. The central synthesis is:

> Multi-agent value comes from independent search width, role-specific context,
> executable feedback, and controlled memory. For production kernels, these
> mechanisms must remain upstream of a deterministic proof and deployment
> plane.

## Stage 4/5 mechanism re-analysis

There are six affected B300 observations across development and held-out sets;
all six have bootstrap 95% lower bounds above one. In this small post-hoc set:

- total chunks vs log-speedup Pearson: about -0.06;
- prepare waves removed vs log-speedup Pearson: about -0.02;
- effective sequence parallelism vs log-speedup Pearson: about 0.81.

These are descriptive values, not a causal model. The stronger physical
hypothesis is that cpc changes prepare duration while the recurrent chain
changes the percentage denominator. Fixed-8192, packed-skew, and packed-mixed
all save roughly 18-19 microseconds despite different relative speedups.

## Preregistered next B300 experiment

`stage3_remote/benchmarks/bench_flash_kda_h12_profile_matrix.py` fixes four
profiles at exactly 512 chunks / 8192 tokens before the first Stage 6 run:

1. fixed `(8192,)`;
2. balanced packed `(4000, 4192)`;
3. four-way packed `(1952, 2016, 2080, 2144)`;
4. skew packed `(64, 64, 64, 8000)`.

It searches cpc `{3, 4, 6, 9, 11, 12}` under the public CAKE route, with output
and final-state correctness, CUPTI cold-L2 timing, 256 state rotations, 20 dry
runs, and 100 measured iterations. The Slurm wrapper is
`stage6_b300_h12_profile_matrix.sbatch`.

The screen ran as Slurm job `23168`. cpc=9 was the fastest of
`{3, 4, 6, 9, 11, 12}` on all four profiles, with all correctness checks and
route receipts passing. A same-object paired confirmation ran as job `23169`
with order baseline/candidate/candidate/baseline, two 100-sample blocks per arm,
CUPTI cold-L2 timing, and 256 state rotations.

| Profile | Max chunks | Baseline | cpc=9 | Speedup | Bootstrap 95% |
|---|---:|---:|---:|---:|---:|
| fixed 8192 | 512 | 518.851 us | 501.315 us | 1.0350x | [1.0335, 1.0361] |
| balanced 2-way | 262 | 312.147 us | 294.946 us | 1.0583x | [1.0564, 1.0598] |
| balanced 4-way | 134 | 207.795 us | 189.905 us | 1.0942x | [1.0913, 1.0972] |
| packed skew | 500 | 507.972 us | 490.307 us | 1.0360x | [1.0334, 1.0380] |

- Output/final-state correctness: PASS for every screened and paired row.
- Route: invariant `bt16_prepare_chain_m64` with identical physical variants.
- Four-profile speedup geomean: `1.055612x`.
- Minimum paired 95% lower bound: `1.033364x`.
- Absolute saving: `17.20-17.89 us`, CV `1.42%`.
- Effective-sequence-parallelism vs log-speedup Pearson: `0.99796`.
- Stage 6 accounting: `9.024826` B300 GPU-s and `2.222875 kJ`.

Certificate: `evidence/b300_stage6_h12_profile_certificate.json`.

Interpretation: the nearly constant absolute saving and increasing relative
speedup as the chain critical path shortens support a prepare/chain mixture
hypothesis. This is not yet causal phase attribution, a new untouched held-out
deployment test, or a multi-agent-versus-single-agent result.

## Next falsifiable steps

1. Add prepare-only and chain-only CUPTI activity attribution.
2. Promote wave quantization into a second typed knob only after a case where
   it changes the actual launch grid is measured.
3. Implement canonical candidate IDs and conclusions-only event memory.
4. Run a fixed-budget multi-agent ablation; keep total tokens, candidate count,
   and B300 GPU-seconds equal across arms.
5. Update the deployment policy only after a new untouched held-out
   qualification. Until then, retain the Stage 5 cpc=9 policy and CAKE fallback.
