# Stage 11: FlashKDA SM80 MMA → SM100 typed migration review

## Decision

Do not reopen the already-falsified Phase-6 instruction-only swap.  Keep the
optimized `mma.sync` path as the incumbent.  The only technically distinct
SM100 hypothesis worth future GPU budget is a cross-phase dataflow rewrite
that keeps compatible values in TMEM across K2 Phases 1/3/4/6.

This is not a blanket rejection of Blackwell Tensor Cores.  It is a scoped
negative for one exact candidate on one exact B300/SM103a profile.

## Evidence already available

- Official FlashKDA emits 3,640 static recurrence HMMA instructions and no
  TCGEN/UTCMMA in the archived SASS receipt.
- Kimi-K3 TP8 has 12 local heads.  The official recurrence therefore launches
  only 12 CTAs on a 148-SM B300; measured SM/DRAM throughput is 2.64%/1.24%.
- At the most naturally compatible Phase-6 shape, `m128n128k16`, the optimistic
  L0 probe gives `mma.sync/tcgen05 = 0.919742x` after 64-way amortization.
- The V16 core probe is positive (`1.500761x`) but falls to `0.777985x` in the
  conservative integration envelope.  This preserves a dataflow-rewrite
  hypothesis while rejecting a bare opcode swap.
- ValueSlice attacks the observed limiter directly by expanding K2 from 12 to
  96 CTAs and achieved about 27% lower forward latency; later phase-prefetch
  work improves the same incumbent further.  These gains are not tcgen05 gains.

## What changed in the IR

`kda_ir/mma_migration.py` now carries the `typed_mma_migration_v2` contract.
It freezes each site's logical contraction, dtype/major mode and rounding-DAG
digest; distinguishes instruction-only, padding/packing, pipeline remap and
CHUNK-retile tiers; and verifies tcgen05 tile legality, single-thread issue,
TMEM lifetime, commit/wait/readback ordering, padding safety and resource
coverage.  V2 adds launch topology, cross-site TMEM carriers, event-interval
alias checking and a namespaced full-SHA comparison baseline.  Full SHA-256
candidate identities prevent semantic duplicates.

The archived direct-swap result is stored as a `MigrationFailureCard`, not a
verifier ban.  A second `MigrationKnowledgeClaim` layer marks it only as a
`single_path_observation` and `scoped_prior`; planned counterfactuals do not
count as corroboration. Its applicability key names B300/SM103a, H12, CHUNK16,
Phase-6, V128, grid12, inner64 and L0. Its reopen conditions explicitly permit
a cross-phase TMEM-resident rewrite, a grid96 thin-N ValueSlice route, a
saturated/hoisted-allocation mechanism control, or a changed hardware/toolchain
receipt. Performance conclusions are statically forbidden from becoming
verifier rules.

## Relation to the small assignment

Modules 1–4.3 supply the lowering vocabulary: MMA atoms, fragment/layout
mapping, descriptors/swizzles, memory-proxy fences, TMA, tcgen05/TMEM,
mbarriers and staged pipelines.  Module 4.5 supplies a dispatch lesson: the
Kimi-K3 `in_proj_qkvgfab` projection is routed by real `(M,N,K)` shape, and
small `M` can favor skinny CUDA-core kernels over Tensor Cores.

That does not make the projection GEMM and recurrent KDA the same operator.
The transferable principle is that peak Tensor-Core throughput is not a
decision rule; operator shape, independent work, protocol overhead and runtime
profile must all be part of the receipt.

## Multi-agent use

For this migration space, agents should own orthogonal evidence lines:
baseline archaeology, ISA mapping, pipeline/resources, numerical adversary,
typed implementation and deterministic measurement judgment.  They exchange
only verified conclusions and scoped failure cards.  Stage 9's arm isolation,
round barriers, sandboxing and accounting can be reused if the cross-phase
space becomes large enough; the technical migration tiers must not be confused
with the experimental agent arms.

No repeat of the direct-swap GPU job is justified: that exact hypothesis
already failed its entry gate. New GPU budget must change a declared typed
subtree and compare against the strongest structure-matched incumbent.

## Cross-path challenge result

The local-optimum concern was tested with an already implemented, materially
different path rather than another opcode microbenchmark. B300 job 23866
compared the commit-verified official FlashKDA SM80 path with the CAKE frozen
SM100 full-stack path across six Kimi-K3 TP8 H12 profiles. All output and final
state comparisons passed. The SM100 path was 2.126x--3.397x faster than the
official raw kernel and 2.202x--3.645x faster than the public-semantics-adapted
peer; fixed T8192 was 2.789x/2.794x respectively.

This refutes the blanket proposition "SM100 migration is not valuable." It
does not refute the scoped direct-swap failure and does not attribute the
full-stack gain solely to tcgen05: CAKE also changes fusion, work partition,
roles, residency and pipelines. The next causal ladder is therefore shared
P3/P4 lifecycle, ValueSlice thin-N tcgen05, cross-phase residency, and finally
role/resource ablations against their strongest structure-matched incumbents.

## Current causal campaign

The next experiment is frozen as a complete 2x2 rather than another isolated
winner search. Factor A is P3/P4 cross-phase TMEM lifecycle reuse; factor B is
ValueSlice V16 (`m128n16k16`, H12 x 8 slices = grid 96); AB tests their
interaction. A or B losing does not cancel AB. The interaction estimate is
`(AB-A)-(B-control)`, and every cell retains CHUNK16, token recurrence order,
the archived rounding DAG, public output/final-state semantics, SM103a and the
same strongest optimized SM80 comparator. CAKE full-stack timing is explicitly
forbidden as causal evidence for a cell.

The A typed fixture contains two math sites, one allocation plan, serialized
reuse of accumulator TMEM columns `[0,16)`, and a separate BF16 rounded-U
carrier in `[16,32)`. It records the source contraction as
`16x16 @ 16x128` and separately proves the physical tcgen05 form through
`(A@B)^T = B^T@A^T`, yielding `m128n16k16`; the physical form is not allowed to
silently replace the logical contract. The verifier accepts non-overlapping
reuse but still rejects a true lifetime overlap or an invalid transpose shape.
The B CUDA entry fixes D128/CHUNK16/V16/grid96 and validates FP32 output plus
bit-exact BF16 state across both mbarrier parity phases before timing.

The minimum A implementation boundary is now source-grounded: P3 computes
`U0^T[128,16] @ INV^T[16,16]`, rounds that result to BF16, and P4 consumes it
as `BF16(U1^T)[128,16] @ Mqk^T[16,16]`. Both paths must share this exact CPU
oracle order. The tcgen path allocates once, serializes P3/P4 over accumulator
columns `[0,16)`, and keeps the rounded carrier separate. A direct TMEM-carrier
implementation is not yet claimed compile-ready: this repository has no
compiled `tcgen05.st` carrier precedent, so its store/load lane mapping must be
validated on SM103a before it can replace a conservative SMEM carrier. The
isolated slice also does not cover beta/(v-u), P1/P5/P6, or the production CUTE
warp-specialized layout.

GPU attempts 9900 and 9901 did not produce performance observations. The
currently visible queue exposed only an RTX 5090 (SM120): job 9900 compiled the
SM103a binary but could not execute it, while job 9901 established that this
consumer target does not assemble the `tcgen05`/TMEM instruction family. These
are environment/compile attempt records, not negative evidence about V16 on
B300. The Slurm entry now exits before compiling when the assigned device is
not compute capability 10.3. B300 correctness and timing remain pending until
an SM103a node is visible again.

That pending screen was subsequently executed through the correct
`<B300_LOGIN_HOST>` endpoint as job 24111. All L0 output and L1 BF16 state checks
passed for inner counts 1/2/4. At V16/grid96, one-shot speedups were 0.751x
(L0) and 0.807x (L1). After 64-way protocol amortization, preferred-layout L0
rose to 1.615x while the scalar-rematerialization L1 remained 0.779x. This is
a resolved mechanism observation: the thin-N primitive has headroom, but the
current representation transfer consumes more than that headroom. It remains
an inconclusive challenge—not completion of cell B—because the probe does not
implement the production ValueSlice pipeline or eliminate that transfer.
Numerically, the amortized tcgen core saves 0.256 us/phase, while its current
L1 path loses 0.328 us/phase overall. The next lifecycle implementation must
therefore remove at least 0.328 us/phase from the tcgen path (22.1% of its L1
time), or equivalently eliminate about 56.2% of the measured differential
integration overhead, before it can beat the matched HMMA L1 arm.
