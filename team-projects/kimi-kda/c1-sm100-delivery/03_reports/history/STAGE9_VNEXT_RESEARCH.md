# Stage 9 vNext: diversity-preserving multi-agent profile search

> Scope: improvements inside the frozen Stage9 cpc-policy experiment only.
> The niche grammar below must not be presented as the general framework search
> space. Open migration work follows `FRAMEWORK_SCOPE_CORRECTION_20260910.md`.

Date: 2026-09-10

## Result of the v5 audit

The v5 pilot proves that isolated agents, typed verification, measured feedback,
shared conclusions, and two-parent synthesis execute end to end.  It does not
yet show useful multi-agent search diversity:

- round 0 produced four proposal slots but one unique semantic policy;
- all four nominal roles used only `total_chunks`;
- round 1 produced three unique policies, but capacity and resource again
  duplicated each other;
- the compound improved over the best parent by only 0.02834% on held-out;
- the current action language changes only `chunks_per_cta`, while the resource
  receipt is fixed.  A distinct "resource" agent therefore has no resource knob
  to manipulate and is a role label without a distinct search space.

The root cause is role/grammar mismatch, not insufficient agent count.

## Recommended architecture

Use three competing explorer niches and one post-measurement synthesizer:

1. `wave_capacity`: only `total_chunks` predicates; searches CTA-wave
   transitions and CPC leaves.
2. `tail_skew`: only `max_chunks` and `num_sequences`; searches longest-chain
   and imbalance effects.
3. `mixed_interaction`: must combine `total_chunks` with one tail/skew feature;
   searches cases where grid size alone is insufficient.
4. `critic_synthesizer`: does not spend a round-0 proposal.  After verified
   measurements, it attacks the best competing explanations and may compose
   parents from different niches.

These are machine-checkable semantic contracts, not prompt-only personas.  The
resource role should return only after the candidate IR gains an independently
variable resource action such as warps/CTA, stages, SMEM footprint, or launch
bounds, or after the study spans receipts whose resident capacity differs.

## Competition and selection

Use a small MAP-Elites-style archive rather than global top-k convergence.  A
niche is keyed by tree depth, predicate-feature family, and CPC regime.  Keep
the best correct candidate in every niche, then allocate the next measurements
across niches before spending extra slots within one niche.

The first expansion around every measured elite is deterministic:

- move a threshold to nearby observed values or their midpoints;
- move one CPC leaf to an adjacent allowed value;
- project a collapsed tree into a role-specific feature family while retaining
  its shape and quantile-aligned thresholds.

An LLM call is justified only for a new structural hypothesis, repair, or
cross-niche composition that the deterministic operators cannot express.  This
makes tokens pay for reasoning rather than integer enumeration.  The new local
implementation in `kda_ir/stage9_search.py` produces 10 wave, 3 tail/skew, and
4 mixed-interaction typed candidates from the v5 round-0 incumbent, with no
model calls.

Selection remains lexicographic: correctness first, then paired
uniform-profile log speedup.  Parent and child must be measured in the same GPU
allocation and interleaved order; compound credit is child-minus-best-parent,
not child-minus-baseline.

## Communication and memory

Agents keep private working context.  The round barrier publishes only:

- canonical candidate and niche IDs;
- correctness and route identity;
- paired per-profile deltas and uncertainty;
- a short causal conclusion;
- a failure fingerprint or inherited-parent reference.

Short-term memory is a tabu set of attempted semantic IDs, failure
fingerprints, and current niche elites.  Long-term memory contains compact wins
and traps only after deterministic validation; raw chain of thought, full chat
history, counters, and scheduler state are excluded.  Retention should be
bounded per operator/hardware/route key so a win on another route cannot become
false evidence here.

## Sandbox and control-plane boundary

Keep the existing Atrex-backed isolated request adapter and durable event
receipts.  Proposal agents receive no repository, shell, profiler, or held-out
access.  Only the deterministic evaluator owns B300 execution.  The policy
verifier, candidate deduplication, archive update, budget accounting, and
activation gate remain ordinary code rather than agent decisions.

## Open-source decisions

- Retain PIKE/Atrex as already vendored: PIKE supplies branch-search patterns;
  Atrex supplies process adapters and durable execution evidence.
- Adopt the ideas, but not a new dependency, from KernelPro: semantic profiler
  feedback, progressive widening, asymmetric branching, and dead-end pruning.
  These directly address v5 convergence.
- Adopt the compressed attempt-memory shape from StitchCUDA, but do not vendor
  its implementation.  Its code is PolyForm Noncommercial 1.0.0 and its fixed
  planner-coder-verifier workflow targets raw KernelBench programs rather than
  this typed dispatcher policy.
- Do not vendor KernelMem at present: its dual-level memory is relevant, but the
  repository exposes no license file and its main loop assumes generated CUDA
  source plus NCU/NSYS access.
- Do not add LangGraph or AutoGen.  Their orchestration/checkpoint abstractions
  duplicate the smaller control plane already present and do not solve semantic
  diversity.
- AKO4X is MIT-licensed and close to FlashInfer, but its published roadmap says
  parallel subagents are not implemented yet.  Treat its campaign archive and
  adapter seam as references, not a runtime dependency.
- Do not replace the archive with OpenEvolve wholesale.  Its current upstream
  discussion acknowledges that archive sampling behaves more like a ranked
  buffer than strict feature-bin MAP-Elites sampling; the required niche map is
  small enough to implement and test locally.

Primary references:

- KernelPro: https://arxiv.org/abs/2606.26453
- StitchCUDA: https://github.com/UMN-APEX-Lab/StitchCUDA
- KernelSkill/KernelMem: https://arxiv.org/abs/2603.10085 and
  https://github.com/0satan0/KernelMem
- AKO4X: https://github.com/TongmingLAIC/AKO4X
- OpenEvolve archive discussion:
  https://github.com/algorithmicsuperintelligence/openevolve/discussions/445
- LangGraph supervisor memory:
  https://github.com/langchain-ai/langgraph-supervisor-py

## Next experiment, in order

Steps 1--2 were executed in B300 job 23823.  All 17 semantic candidates passed
correctness and collapsed to 16 development behaviors.  Each niche winner was
then remeasured against the v5 parent with 100-repeat interleaved pairs:

- `wave_capacity`: 1.000346x, but its eight-profile CPC behavior is exactly the
  same as the parent, and every per-profile interval crosses one;
- `tail_skew`: 0.978989x;
- `mixed_interaction`: 0.988022x.

The run used 30.206 allocated GPU-seconds and an estimated 9.603 kJ.  This is a
useful negative result: forced semantic diversity works mechanically, but the
current development profiles favor the capacity-only behavior.  It does not
justify more LLM calls or synthesis within the same tree grammar.

The next useful experiment is therefore one of:

1. extend the typed action IR with a genuinely independent resource or
   scheduling knob, then reinstate a resource-search niche;
2. test the existing capacity rule on a different device/resource receipt;
3. if the paper still needs a causal multi-agent claim, run multiple equal-budget
   trajectories against a sequential single-agent and homogeneous best-of-N.
   Until then, retain the v5 feasibility-only claim.

The production dispatcher remains Stage 10.  Stage 9 vNext is a research path,
not an activation candidate until it passes fresh held-out confidence gates.
