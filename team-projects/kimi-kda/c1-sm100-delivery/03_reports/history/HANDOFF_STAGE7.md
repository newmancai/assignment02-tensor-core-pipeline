# FlashKDA MARPE - Stage 7 Handoff

## Outcome

Stage 7 closes the first proof-carrying Runtime Profile Agent loop and moves the
H12 `cpc=9` explanation from correlation to two-level physical evidence:

1. public-call CUPTI activity attributes 97.49%-98.86% of the full-span gain to
   the prepare launch across four matched-work profiles;
2. a representative Nsight Compute run supports a CTA work-decomposition and
   scheduling-wave mechanism while ruling out a changed per-CTA resource
   footprint.

This is a single-B300 result. It does not establish multi-agent superiority,
cross-device transfer, a new deployment policy, or end-to-end Kimi-K3 gains.

## Phase experiment

Slurm job `23176` ran four profiles with fixed total work (512 chunks / 8192
tokens) and different recurrent critical paths. Every public call emitted the
same two ordered physical kernels, prepare then chain, and output/final-state
correctness passed.

| Profile | Prepare cpc4 (us) | Prepare cpc9 (us) | Prepare saving (us) | Chain cpc4 (us) | Full saving (us) |
|---|---:|---:|---:|---:|---:|
| Fixed 8192 | 83.425 | 65.248 | 18.177 | 433.748 | 18.545 |
| Balanced 2-way | 83.040 | 65.009 | 18.032 | 227.585 | 18.240 |
| Balanced 4-way | 83.073 | 65.089 | 17.984 | 122.513 | 18.448 |
| Packed skew | 83.296 | 65.280 | 18.016 | 423.905 | 18.383 |

Prepare saving has mean `18.052 us` and CV `0.41%`. Chain changes by only
`0.09-0.59 us`. Baseline chain time correlates with maximum sequence chunks at
`r=0.9999995` and is descriptively fit by
`T_chain(us)=11.99+0.824*max_chunks`; cpc=9 retains nearly the same slope.

Authoritative evidence:

- `evidence/b300_stage7_h12_phase_activity.json`
- `evidence/b300_stage7_h12_phase_activity_accounting.json`

## Nsight Compute mechanism evidence

Job `23177` failed before profiling because the target script lacked an explicit
interpreter; this attempt remains in the append-only ledger. The corrected job
`23195` completed. On the representative fixed-8192 prepare launch:

| Counter | cpc=4 | cpc=9 | Interpretation |
|---|---:|---:|---|
| Grid size | 1536 CTA | 684 CTA | -55.47% scheduling units |
| Occupancy-aware waves/SM | 2.08 | 0.92 | crosses below one resident wave |
| Block size | 128 | 128 | unchanged |
| Registers/thread | 77 (80 allocated) | 77 (80 allocated) | unchanged |
| Shared memory/block | 45,056 B | 45,056 B | unchanged |
| Executed instructions | 17,479,856 | 16,787,432 | -3.96% |
| Active warps | 29.52% | 28.56% | nearly unchanged |
| NCU duration | 82.240 us | 67.648 us | -17.74%, explanatory only |
| SM / DRAM / L2 throughput | 30.39 / 17.22 / 13.81% | 37.52 / 20.84 / 16.95% | higher active-period utilization |

The best supported model is therefore:

```text
public latency ~= prepare(total work, cpc, CTA waves, resources)
               + chain(max dependency length, effective parallelism)
               + launch gap
```

`cpc=9` reduces prepare scheduling units and crosses a wave boundary without
changing the per-CTA resource footprint; `max_chunks` controls the residual
recurrent critical path. NCU counter replay means its duration is not a timing
gate. CUPTI public-path timing remains authoritative.

Evidence:

- `evidence/b300_stage7_h12_prepare_ncu_summary.json`
- `evidence/stage7_ncu/prepare-cpc4.csv`
- `evidence/stage7_ncu/prepare-cpc9.csv`
- `evidence/b300_stage7_h12_prepare_ncu_accounting.json`

## Autonomous closed loop

`kda_ir/autonomous_agent.py` implements a deterministic controller around
proposal sources and the measurement oracle:

- content-addressed canonical typed candidates;
- verify-before-measure and structural deduplication;
- equal low-budget screen, top-K qualification, correctness/confidence gate;
- conclusions-only memory with profile/version identity;
- explicit GPU budget and evidence-plateau stopping;
- activation only with public-scope evidence and fallback.

`run_autonomous_profile_agent.py` replays 24 Stage 6 candidate measurements and
four paired receipts. It independently selects and activates cpc=9 on all four
profiles. This proves the loop implementation, not a multi-agent advantage.

Evidence:

- `evidence/stage7_autonomous_agent_replay.json`
- `evidence/stage7_conclusions_memory.json`
- `evidence/b300_stage7_certificate.json`

Local typed-IR suite: **27/27 PASS**.

## Accounting

Stage 7 retains three attempts, including the failed profiler launch:

- `17.311264` B300 GPU-seconds;
- `3.548480 kJ` estimated allocation energy.

The Stage 4-7 ledger totals 16 attempts, `81.224` GPU-seconds, and `18.005 kJ`.
Telemetry includes compilation, setup, counter replay, synchronization, and idle
gaps; it is not kernel-only energy.

## Upstream research and paper artifacts

The retrospective and paper now connect MARPE to NVIDIA CUDA/Blackwell execution
and occupancy guidance, CUTLASS task schedulers, CUPTI, Nsight Compute, Stream-K,
TVM/Ansor/TensorIR/MetaSchedule, Triton, FlashAttention-3, KDA lineage, and recent
agentic kernel systems. The bibliography contains 30 entries, including primary
NVIDIA documentation.

- `RESEARCH_RETROSPECTIVE_MULTI_AGENT.md`
- `paper/runtime_profile_evolution/main.tex`
- `paper/runtime_profile_evolution/references.bib`
- `paper/runtime_profile_evolution/figures/marpe-closed-loop.tex`

## Next falsifiable steps

1. Run an equal-budget four-arm search ablation: best solo, homogeneous pool,
   role chain, and strategy specialists with conclusions-only memory.
2. Add a second orthogonal typed knob only where it changes the physical launch,
   then test whether cross-lineage composition beats each parent.
3. Repeat matched-work phase tests at a second total-work level to separate
   fixed launch overhead from wave/tail scaling.
4. Keep the Stage 5 deployment rule and CAKE fallback until a new untouched
   qualification supports any policy change.
5. Treat transfer to B200/B300 replicas and end-to-end serving as separate
   claims with new certificates.
