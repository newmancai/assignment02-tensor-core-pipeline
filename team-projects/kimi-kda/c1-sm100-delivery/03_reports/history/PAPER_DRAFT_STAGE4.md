# Runtime Profile Evolution: Proof-Carrying Post-Compilation Optimization for Recurrent KDA

*Working paper draft, September 2026. Results are preliminary and claim-bounded.*

## Abstract

Modern GPU kernel systems can synthesize strong schedule portfolios, yet a
deployed model profile may still expose physical launch parameters that are
suboptimal on a particular accelerator. We introduce **Runtime Profile
Evolution (RPE)**, a proof-carrying loop that binds optimization to a real model
contract, represents physical schedules in typed IR, rejects illegal mutations,
measures legal counterfactuals through the public API, and activates only
candidates with correctness and confidence evidence. On an NVIDIA B300, using
the Kimi-K3 tensor-parallel-8 recurrent-KDA profile (12 heads, head dimension
128, BF16), RPE discovers that long H12 BT16 prepare-chain workloads should
assign 9 rather than 4 chunks to each prepare CTA. The change improves the two
affected preset shapes by 1.0369x and 1.0807x, with bootstrap 95% lower bounds
of 1.0344x and 1.0789x. With fallback on four unaffected shapes, the guarded
six-shape geometric mean is 1.0191x. On six shapes fixed before a held-out run,
all four affected cases support the policy, with speedups from 1.0040x to
1.0371x and no affected-shape regression. Failed candidates and all 54.89
allocated GPU-seconds are retained in an append-only ledger. These results establish an
incremental improvement over a CAKE-generated B300 route; they do not yet
constitute an end-to-end Kimi-K3 result or a reproduction of CAKE's B200
comparison against official FlashKDA.

## 1. Introduction

Kernel optimization systems usually stop at one of two boundaries. Compilers
and generators produce a kernel portfolio before deployment, while runtime
dispatchers choose among frozen variants after deployment. Real serving
profiles create a third opportunity: measured production shapes can reveal
that a correct physical route contains a suboptimal work-assignment parameter,
even when the route family itself is already optimal.

CAKE demonstrates that agent-guided CUDA kernel engineering can accumulate a
typed hardware-explicit IR, verification rules, cost models, and reusable
tactics. Its KDA study reports a 2.05x geometric-mean speedup over official
FlashKDA on six B200 BF16 shapes and validates end-to-end Kimi-K3 serving. Our
question is complementary: **can the system continue learning after a strong
CAKE route has been selected, using the exact deployed model profile, without
weakening correctness or silently moving the measurement boundary?**

RPE treats runtime profile optimization as a sequence of proof-carrying state
transitions. A candidate must retain the model/operator contract, pass typed
verification, identify its exact physical receipt, survive public-API
correctness, and show a positive confidence lower bound under an interleaved
measurement protocol. Failures remain first-class evidence and can introduce
new typing constraints.

This draft makes four preliminary contributions:

1. A workload identity boundary derived from Kimi-K3's TP8 H12 recurrent-KDA
   contract, rather than a convenient proxy shape.
2. A typed `prepare_chunks_per_cta` schedule dimension with verifier, lowering,
   and structure-preserving mutation support.
3. A counterfactual-to-activation workflow that first eliminates route-level
   alternatives and then searches intra-route work assignment.
4. A B300 case study with exact correctness, paired confidence intervals,
   guarded fallback, preregistered held-out cases, and full success/failure
   resource accounting.

## 2. Background and Positioning

### 2.1 Recurrent KDA

KDA combines token-ordered recurrent state updates with matrix operations and
fused gate/beta transformations. The recurrence constrains legal parallelism:
tokens may be chunked and preparation may run independently, but state-chain
consumption must preserve sequence order. Kimi-K3 uses 96 global KDA heads;
tensor parallelism of eight gives 12 local heads per GPU.

### 2.2 CAKE

[CAKE](https://arxiv.org/html/2608.12629v1) introduces a hardware-explicit
typed IR and an agentic optimization workflow whose accumulated knowledge can
improve later kernels. The present work starts from a CAKE-generated FlashInfer
route and studies continued, profile-conditioned physical evolution. Therefore
our current percentage gains are *incremental over CAKE on B300*, not directly
comparable to CAKE's B200 gains over official FlashKDA.

### 2.3 Runtime Profile Evolution

RPE differs from ordinary autotuning in its evidence boundary. The unit being
optimized is not an isolated kernel launch alone, but an exact public operator
call with state mutation, route receipt, workload identity, fallback decision,
and experiment ledger. A mutation becomes deployable only together with its
proof bundle.

## 3. System Design

### 3.1 Typed physical schedule

The IR represents tile shape, warp roles, storage spaces and lifetimes,
split-phase barriers, completion joins, recurrence topology, and resource use.
Stage 4 adds `prepare_chunks_per_cta` to the recurrence plan. The verifier
requires a positive value and rejects non-default prepare assignment on direct
recurrence. Lowering emits the value as part of the prepare-grid plan.

### 3.2 Profile binding

The H12 preset fixes six shape/layout cases and the following common contract:
12 heads; QK and value dimensions of 128; BF16 Q/K/V/gate/beta and state; FP32
`A_log` and `dt_bias`; QK L2 normalization; gate and beta processing in the
kernel; lower bound -5; and explicit initial/final state behavior.

### 3.3 Evolution loop

The loop has five gates:

1. **Route counterfactual:** force every registered route family and reject
   slower, unsupported, or incorrect alternatives.
2. **Typed mutation:** vary a physical parameter without changing mathematical
   semantics.
3. **Coarse and refined search:** identify a promising region using public-API
   CUPTI measurements with cold L2 and rotating states.
4. **Paired qualification:** run baseline/candidate/candidate/baseline blocks on
   the same prepared object and bootstrap median speedup.
5. **Guarded activation:** activate only affected shapes whose 95% confidence
   lower bound exceeds one; otherwise retain the baseline.

### 3.4 Failure knowledge

An attempted expansion of the H12 short state carrier to longer sequences
caused an illegal memory access. RPE records the failed allocation and promotes
the observed precondition into the search-space boundary: the short carrier is
legal only up to its compiled token extent. This is not treated as a noisy
benchmark sample.

## 4. Experimental Methodology

### 4.1 Hardware and software

Experiments use one NVIDIA B300 SXM6 AC, compute capability 10.3, with 148 SMs,
PyTorch 2.10.0+cu130, and the FlashInfer CAKE checkout identified by the Stage 4
certificate. Exact source, benchmark, result, patch, and accounting hashes are
stored with the artifact.

### 4.2 Measurement protocol

The authoritative comparison measures the public recurrent-KDA API including
in-place state update. CUPTI timing uses cold L2 and 256 rotating initial-state
slots. Each arm receives two blocks, each with 20 dry runs and 100 measured
iterations. The order is reversed in the second half. Confidence intervals use
10,000 bootstrap draws over per-iteration samples.

### 4.3 Correctness

Candidates are compared against the existing dispatcher for both output and
final recurrent state. The selected cpc=9 counterfactual has zero maximum
absolute difference on both affected shapes. The paired public test also passes
the existing BF16 tolerance.

## 5. Results

| H12 shape | Baseline (us) | Candidate (us) | Speedup | 95% CI | Policy |
|---|---:|---:|---:|---:|---|
| packed 512x32 | 178.945 | 178.625 | 1.0018x | [1.0005, 1.0028] | unchanged |
| packed 128x8 | 26.336 | 26.368 | 0.9988x | [0.9964, 1.0012] | unchanged |
| fixed 512 | 49.504 | 49.568 | 0.9987x | [0.9955, 1.0026] | retain cpc=1 |
| fixed 8192 | 516.803 | 498.433 | 1.0369x | [1.0344, 1.0391] | cpc=9 |
| packed mixed | 256.081 | 236.961 | 1.0807x | [1.0789, 1.0830] | cpc=9 |
| packed 1024x8 | 107.360 | 106.048 | 1.0124x | [0.9998, 1.0160] | unchanged |

The affected-shape geometric mean is 1.0585x. Counting unaffected shapes as
1.0x under guarded fallback gives 1.0191x across all six. The improvement comes
without a new device kernel: it changes the prepare grid's work assignment for
long H12 BT16 workloads while retaining the same generated prepare and chain
physical receipts.

### 5.1 Preregistered held-out evaluation

Six additional H12 shapes were fixed in code before their first GPU run. The
64- and 128-chunk controls remain statistically indistinguishable from 1.0x,
consistent with the policy boundary. All four shapes above the 128-chunk
threshold support cpc=9.

| Held-out shape | Total chunks | Speedup | 95% CI |
|---|---:|---:|---:|
| fixed 4096 | 256 | 1.0040x | [1.0026, 1.0058] |
| fixed 16384 | 1024 | 1.0122x | [1.0113, 1.0138] |
| packed skew | 512 | 1.0371x | [1.0347, 1.0397] |
| packed irregular | 418 | 1.0239x | [1.0205, 1.0268] |

The affected held-out geometric mean is 1.0192x; the guarded geometric mean
over all six held-out cases is 1.0128x. This supports transfer beyond the two
development shapes but remains a single-device study.

## 6. Discussion

The route counterfactual is as important as the winner. Direct N32, direct N16,
and BT16 forcing show that the existing route selector is already strong for
this profile. The residual opportunity lies one level lower, in the amount of
work assigned to each prepare CTA. This suggests that a schedule portfolio and
a runtime profile agent should cooperate: the portfolio supplies verified
physical kernels, while the agent evolves workload-to-launch policies.

The result also exposes a limitation of fixed head-count heuristics. Both long
winning shapes have approximately 512 total 16-token chunks, despite very
different sequence layouts. Their shared optimum at cpc=9 suggests that total
work and SM-wave geometry may be more predictive than head count alone. This is
a hypothesis, not yet a general rule.

## 7. Limitations

1. The result covers one B300 and six H12 preset shapes.
2. Held-out evidence contains only six preregistered shapes on the same device;
   a larger stratified workload matrix is still required.
3. The current production patch uses a measured constant rather than an
   architecture-normalized analytical policy.
4. No B200 reproduction or direct official-FlashKDA baseline is included.
5. No end-to-end Kimi-K3/SGLang latency or throughput result is yet available.
6. Bootstrap samples come from repeated executions within one allocation; a
   multi-day/multi-device replication is still required.

## 8. Roadmap to a Stronger-than-CAKE Result

A credible superiority claim must be multi-axis rather than a larger isolated
speedup number. We will test whether RPE can add post-compilation gains on top
of CAKE while preserving CAKE's strengths:

1. Derive a total-chunk/SM-wave policy and evaluate preregistered held-out H12
   distributions.
2. Replicate on B200 and B300, then compare against official FlashKDA and the
   frozen CAKE dispatcher under matched hardware and API scope.
3. Attribute prepare and chain time, occupancy, memory traffic, and launch
   overhead with CUPTI/Nsight counters.
4. Run Kimi-K3 in SGLang and report end-to-end prefill throughput and latency.
5. Expand from one knob to warp-role allocation, completion topology, stage
   count, and resource co-optimization, with verifier-generated failure rules.
6. Report clean-start search cost, successful and failed GPU-seconds, energy,
   held-out transfer, and rollback behavior.

## 9. Reproducibility

The Stage 4 and Stage 5 handoffs name every benchmark and evidence file. Their
certificates record source, result, patch, and accounting hashes; the attempts
directory contains every GPU allocation, including failures. The portable
production patch contains two changed lines and applies independently of the
research harness.

## 10. Conclusion

Runtime Profile Evolution found a small but statistically strong improvement
inside an already optimized CAKE-generated KDA route and converted it into
typed, verified, deployable knowledge. The present result is deliberately
narrow. Its larger promise is a continuous optimization agent that learns from
real serving profiles after kernel generation, while carrying enough evidence
to make every activation reviewable and reversible.

## References

1. CAKE: Coding Agent for Kernel Evolution,
   [arXiv:2608.12629](https://arxiv.org/abs/2608.12629).
2. Moonshot AI, [FlashKDA](https://github.com/MoonshotAI/FlashKDA).
3. Moonshot AI, [Kimi-K3 model configuration](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json).
