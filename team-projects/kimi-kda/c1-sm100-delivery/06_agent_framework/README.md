# KDA typed-IR vertical slice

The current mainline delivery, beginning with the exact question
“FlashKDA 官方 kernel 当前使用 SM80 MMA，分析迁移到 SM100 是否值得”, is
`SM100_MAINLINE_DELIVERY.md`.

The course-aligned final report and defense structure is
`C1_FINAL_ASSIGNMENT_OUTLINE_20260910.md`. Ready-to-adapt prose for the required
“reproduction and measurement -> analysis (conclusion + evidence) -> challenge”
sequence is `C1_REPRODUCTION_ANALYSIS_CHALLENGE_CONTENT_20260910.md`.

For the latest prospective capacity-policy qualification, proof-carrying
shadow resolver, research synthesis, and next experiment, see
`HANDOFF_STAGE10.md` and
`RESEARCH_RETROSPECTIVE_MULTI_AGENT.md`. The publication-facing final paper
and its reproducibility summary are indexed by `PAPER_FINALIZATION.md`.
The source-to-ISA review of the original FlashKDA SM80 MMA path, its scoped
negative `tcgen05` result, and the new typed migration/failure-memory contract
are recorded in `STAGE11_MMA_MIGRATION_REVIEW.md`.
The Stage 12 cross-route design grammar, compound portfolio coordinator,
layout canonicalizer, finite-envelope closure certificates, and new B300
measurements are summarized in
`../03_reports/STAGE12_TYPED_IR_SEARCH_CLOSURE_20260911.md`.
The single-file continuation point for the complete Stage 11 state and B300
job 24111 is `HANDOFF_STAGE11.md`; the formerly open P3/P4 implementation now
has compiled, exact-validation and timing evidence in
`../04_evidence/agent_rounds/round10_p34/`.
The audited open-source execution/search stack and its exact integration
boundary are recorded in `OPEN_SOURCE_MULTI_AGENT_STACK.md`; pinned source
snapshots and licenses live under `third_party/`.

This prototype tests one thesis: an optimization agent should mutate a typed
physical schedule, not unconstrained CUDA/PTX text. Invalid barrier, memory,
resource, and recurrence plans are rejected before backend code generation.

The current slice deliberately supports only what is needed to express three
FlashKDA schedule families:

- the official direct physical schedule;
- the B300 value-sliced schedule;
- a CAKE-class BT16 prepare/chain schedule.

It contains seventeen small layers:

1. `kda_ir.model`: typed roles, buffers, barriers, operations, resources, and
   recurrence decomposition;
2. `kda_ir.verify`: named static rules (`KIRxxx`) that collect all violations;
3. `kda_ir.mutate`: bounded structural transformations exposed to an agent;
4. `kda_ir.lower`: a verified, backend-neutral launch plan.
5. `kda_ir.evolve`: an agent-facing proposal protocol, verifier-filtered
   evolution result, resource Pareto frontier, and recurring-failure ledger.
6. `kda_ir.cutlass_ts`: an inspectable construction plan for NVIDIA CUTLASS
   experimental Task Scheduling, including typed pipeline protocols.
7. `kda_ir.flashinfer_catalog`: a reader and contract checker for FlashInfer's
   frozen generated-KDA metadata.
8. `kda_ir.frozen_source` and `kda_ir.toolchain`: exact contract recovery from
   annotated generated CUDA plus feature-safe SM100a/SM103a compile plans.
9. `kda_ir.cutlass_runtime`: lazy, EULA-bound instantiation of real CUTLASS
   `CooperativeGroup` and `PipelineConfig` objects from a verified plan.
10. `kda_ir.kda_first_edge`: exact recovery of the frozen Q/K TMA edge and an
    explicit `SplitPhaseFanout` lowering boundary for `qk_full`/`smem_free`.
11. `CompletionJoin`: a typed consumer-completion token that must dominate
    stage recycling; the verifier checks participants, owner, arrival count,
    operation events, and wait-before-recycle ordering.
12. source-grounded import: frozen role ranges and barrier-use events are
    recovered with source lines, then converted into a verified Q/K schedule
    slice with transitive completion certificates.
13. `kda_ir.runtime_policy`: an exact resource-receipt-bound resident-grid
    shadow resolver with named runtime-policy guards and explicit fallback.
14. `kda_ir.mma_migration`: a site-level SM80→SM100 migration contract with
    frozen math/rounding semantics, tcgen05/TMEM lifecycle checks, full
    candidate hashes, and scoped negative-result memory.
15. `kda_ir.design_ir`: finite cross-route architecture axes plus typed SSA
    dataflow and conservative layout canonicalization that cannot cross a
    BF16 rounding boundary.
16. `kda_ir.portfolio_search`: compatible multi-role patch composition and a
    quality-diversity elite archive keyed by physical mechanism.
17. `kda_ir.closure`: scope-explicit finite-envelope certificates that list
    uncovered semantic designs instead of promoting a local plateau to a
    global-optimum claim.

Stage 11 additionally freezes a two-factor causal ladder in
`evidence/stage11_tcgen05_causal_ladder_preregistration.json`: P3/P4 shared
TMEM lifecycle (A), V16/grid96 thin-N execution (B), and their interaction.
The ladder prevents one-path negatives or CAKE full-stack results from being
promoted into an opcode-level rule.

The frontier intentionally avoids pretending that a static resource heuristic
or the materialization byte model is a trustworthy latency model. Verified
candidates still require compilation, correctness, and B300 measurement before
performance ranking. The current suite contains 108 CPU tests, including the
explicit rule that a TMEM accumulator layout cannot masquerade as an A-operand
layout without a verified transform.

Run the CPU-only vertical-slice checks:

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

Inspect two lowering plans:

```bash
PYTHONPATH=. python3 demo.py
```

Inspect a real FlashInfer generated-kernel catalog:

```bash
PYTHONPATH=. python3 inspect_flashinfer.py \
  /path/to/flashkda_generated_variant_metadata.json \
  --arch sm_103a --schedule cake-class
```

Recover the exact role/resource/pipeline contract retained in one frozen CAKE
source and cross-check it against its metadata record:

```bash
PYTHONPATH=. python3 inspect_frozen_source.py \
  /path/to/flashkda_generated_bf16_fused_m128_<id>.cu \
  /path/to/flashkda_generated_variant_metadata.json
```

See `UPSTREAM_REVIEW.md` for the upstream findings, the boundary between this
IR and CUTLASS Task Scheduling, and the next experiment.

Inspect and optionally retain a source-to-IR certificate:

```bash
PYTHONPATH=. python3 inspect_frozen_source.py \
  /path/to/flashkda_generated_bf16_fused_m128_<id>.cu \
  /path/to/flashkda_generated_variant_metadata.json \
  --output evidence/source_to_ir_<id>.json
```

## Next implementation boundary

The directly representable `qk_raw_full` TMA-to-prep edge is now recovered as a
five-stage, 16384-byte transaction consumed by one four-warp (128-thread) prep
instance. `SplitPhaseFanout` now preserves the separate `qk_full` ready and
`smem_free` recycle semantics. CUTLASS TS can reuse its native split-consumer
state, while exact elect-one arrivals need `LeaderArrivalSplitConsumer`; a
cooperative-arrival rewrite remains a searchable, benchmarkable alternative.
Arrival ownership is now typed as `elect_one` or `cooperative`, rather than
inferred from an integer barrier count. The verifier rejects inconsistent
mode/count combinations (KIR163--KIR166), and lowering emits a typed
`LeaderArrivalSplitConsumerPlan` whenever native cooperative Task Scheduling
would change the arrival protocol. Multi-consumer fan-outs must additionally
name a `CompletionJoin` (KIR167--KIR174); cooperative arrivals do not remove
that completion-dominance requirement.

The cooperative form now has a real B300 execution smoke in
`smoke_split_consumer.py`. It launches 4 producer warps, 4 wait-only compute
warps, and 1 release-owning recycler warp over five stages. The producer writes
stage values 1 through 5 into SMEM; every compute lane observes their sum as 15
after the full-barrier waits. The checked single-pass run explored 3071
interleavings with zero deadlocks/races and returned
`marker_values=[15] count=288` on B300. Because it does not wrap the five-stage ring, it is not
evidence that stage reuse is safe without a completion join.
The machine-readable run record is
`evidence/b300_cooperative_split_consumer.json`; a CPU regression test keeps
its schedule, marker, and backend-capability contract internally consistent.

The exact elect-one protocol now also executes on B300 through public CUTLASS
pipeline primitives. `smoke_leader_split_consumer.py` initializes both phase
barriers with count one, elects one producer/recycler lane, and makes the
recycler wait for all wait-only consumers through the same `CompletionJoin`
semantics represented in the IR. A ten-iteration run crosses the
five-stage ring boundary and returns `marker_values=[55] count=288`; its record
is `evidence/b300_leader_split_consumer.json`. This proves the synchronization
primitive, not yet the complete FlashKDA math body or Task Scheduling adapter.

This runtime path requires the two codegen capabilities checked by KIR710 and
KIR711. See `UPSTREAM_REVIEW.md` for the minimal upstream failures and fixes.

## Source-grounded Q/K milestone

The real `a1418fd1ae` frozen variant is now imported rather than approximated.
The importer preserves its exact 32-warp placement, 117376-byte SMEM contract,
240 TMEM columns, five-stage ring, and source digest. It recovers `aux_mma` as
the `qk_full` publisher, `compute` and `mma` as consumers, four elected compute
warp leaders as `smem_free` arrivals, and `prep` as the reuse waiter. This
required a third typed arrival mode, `elect_one_per_warp`.

Barrier calls inside role bodies, including SM100 `elect_commit`, become source
events. The importer constructs a transitive source-order graph and requires a
path from every `qk_full` consumer to the `smem_free` wait before emitting the
schedule. KIR713 rejects an untypable arrival count and KIR714 rejects a broken
completion chain. The retained machine-readable result is
`evidence/source_to_ir_a1418fd1ae.json`.

## Stage 2: typed deltas, B300 calibration, and deployment tactics

`inspect_evolution_corpus.py` lifts the 29 frozen Blackwell evolution sources
out of stringly-typed filenames. The resulting corpus has 29 unique semantic
fingerprints: seven scalar-tile schedules, 21 value-tile schedules, and one
value64 split. Each fingerprint records topology, value rows, chunk contract,
token extent, persistent tasks, grid stride, and whether an explicit tile
schedule is required.

The retained B300 SM103a evidence contains six-shape correctness and CUPTI
cold-L2 timing at 20 dry runs and 100 measured iterations. Against the CAKE
public route, the prepared evolution candidates show a directional 1.153x
geometric-mean speedup (1.002x to 1.290x); five of six shapes have a bootstrap
95% lower bound above one. This is not a production API speedup claim:
`backend=auto` still resolves all six shapes to CAKE, so public-adapter
activation and scope-identical remeasurement are the next gate.

`account_b300_run.py` records allocation wall time, visible GPU count,
GPU-seconds, power, utilization, memory, and Slurm provenance. Its append-only
attempt ledger retains failed compiler/runtime experiments as part of the
compute budget. The retained successful manifests account for 55.32
single-B300 GPU-seconds, 1,980 CUPTI samples, and an estimated 12.73 kJ of
allocation energy. `calibrate_b300.py` joins these records with typed
candidates, correctness, bootstrap confidence, and activate/retain tactics in
`evidence/b300_stage2_calibration.json`.
