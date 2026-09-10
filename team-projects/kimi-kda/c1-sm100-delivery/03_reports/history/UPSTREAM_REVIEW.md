# Upstream review: CAKE, CUTLASS Task Scheduling, and FlashInfer

Reviewed on 2026-09-08 against the CAKE paper, current NVIDIA CUTLASS Python
DSL documentation and the installed CUTLASS DSL 4.7.1 package on `<B300_LOGIN_HOST>`,
plus FlashInfer's frozen generated-KDA metadata.

## Conclusion

The original thesis is valid, but it is not itself "beyond CAKE": CAKE already
describes agent search over a typed, hardware-explicit schedule IR and promotes
recurring failures into verifier rules, IR primitives, or cost-model updates.
Its 2.05x FlashKDA result demonstrates a physical-schedule replacement rather
than mere parameter tuning.

The new opportunity is a practical systems boundary:

1. keep a compact, agent-facing KDA schedule IR with stable diagnostics;
2. use CUTLASS experimental Task Scheduling as an optional checked lowering
   substrate instead of inventing another CUDA execution framework;
3. preserve FlashInfer's frozen source/metadata contract as the deployable
   output, so downstream users do not depend on the search system;
4. close the loop with compile, sanitizer, numerical and B300 measurements,
   promoting only recurring and general failures into the language.

This is an inference from the public interfaces. The CAKE paper does not state
that CUTLASS Task Scheduling is CAKE's implementation, so the two must not be
conflated.

## What changed after review

The first prototype had the correct structural boundary but an inaccurate
barrier abstraction. A scalar arrival count cannot represent Blackwell
pipelines: TMA completion is governed by transaction bytes, and UMMA/async
producer-consumer combinations have different protocols and signaling scopes.

The vertical slice now models:

- six concrete pipeline kinds (`TmaAsync`, `TmaUmma`, `UmmaAsync`, and peers);
- explicit acquire/commit/wait/release lifecycle and role permissions;
- transaction-byte contracts for TMA pipelines;
- CTA/task-warp signaling policies and cluster shape;
- barrier storage in the shared-memory resource check;
- a CUTLASS Task Scheduling construction plan with `PipelineConfig` factory
  names verified against the installed package;
- a reader/checker for FlashInfer's real generated variant catalog.

The frozen catalog bridge is intentionally one-way. Artifact metadata exposes
architecture, thread count, shared-memory footprint, launch shape, ABI, routes,
PDL and source hashes. It does not expose warp roles, buffer lifetimes, barrier
phases or recurrence dependencies, so reconstructing a CAKE schedule from the
artifact would be false precision.

## Architecture boundary

```text
agent proposal
    -> KDA typed schedule IR
    -> static verifier (stable KIR diagnostics)
    -> CUTLASS Task Scheduling construction plan
    -> existing numerical work body
    -> compile / sanitizer / correctness / B300 measurement
    -> frozen FlashInfer source + selector metadata

recurring general failure ---------------------> verifier rule or IR primitive
```

The IR owns the agent-safe mutation surface and KDA-specific recurrence
semantics. CUTLASS Task Scheduling owns executable resource/task/pipeline
machinery. FlashInfer metadata owns the productization boundary. These are
three contracts, not three names for one abstraction.

## Licensing boundary

The experimental CUTLASS Python DSL package is distributed under the NVIDIA
EULA and the installed sources carry `LicenseRef-NvidiaProprietary`; it should
not be treated as ordinary Apache-licensed CUTLASS C++ source. This repository
therefore records a neutral construction plan and does not copy NVIDIA
implementation code. Any direct runtime dependency or redistribution needs an
explicit legal/product decision.

## Next falsifiable experiment

Implement one D128 BF16 direct-M128 schedule by instantiating Task Scheduling
resources, tasks and pipelines around the current trusted math body. The first
milestone is not a speedup claim. It is:

- generated kernel compiles for SM100/SM103;
- verifier-approved lifecycle agrees with runtime construction;
- bitwise or agreed numerical oracle passes;
- frozen export metadata matches threads, shared memory and launch shape;
- B300 measurements can compare schedule changes without changing the math.

Only after that should the search space admit structural mutations such as
role reassignment, pipeline-kind replacement, stage/interleave changes, value
slicing and prepare/chain recurrence decomposition.

## First exact-ingest result

The frozen `direct_m128` body `07f44c1e56` retains enough annotations to recover
an exact physical contract without guessing: 1024 threads, 227968 bytes of
shared memory, 256 TMEM columns, six warp roles, 19 barrier groups and 77 staged
barriers. The same body is referenced by both SM100a and SM103a deployment
variants, and every source-versus-metadata check passes.

Both architecture-specific objects compile with CUDA 13.0.88 when using
explicit feature targets:

```text
-gencode=arch=compute_100a,code=sm_100a
-gencode=arch=compute_103a,code=sm_103a
```

Using the superficially equivalent `-arch=sm_100a` emitted `.target sm_100` in
this environment and failed at `ptxas` on `tcgen05`, `setmaxnreg`, and related
feature-specific instructions. The prototype now emits only the checked
compute/code pair. This is the first concrete example of a dynamic failure
becoming a reusable toolchain rule.

`cuobjdump` adds a second resource layer that the export metadata does not
contain. Both objects use 64 registers per thread and 1024 bytes of compiler
static shared memory; the SM100a object has a 16-byte stack while SM103a has no
stack allocation. The launch contract's 227968 bytes are dynamic shared memory,
so the verifier must check the 228992-byte total against the device limit.

## First real Task Scheduling objects

The verified plans now instantiate real CUTLASS Task Scheduling objects on the
B300 environment. The value-sliced fixture produces `TmaAsync(1 -> 96 threads)`
and `AsyncAsync(96 -> 96)` configs; the prepare/chain fixture produces
`TmaUmma(1 -> 1)` and `UmmaAsync(1 -> 96)`. Stage counts, transaction bytes,
signaling policies and four-dimensional VMNK layouts survive the lowering.

This also exposed an important correction to the next milestone. A Task takes
a captured `ScheduleResult`, and work is expressed through CuTe DSL resource
methods decorated as producer or consumer work. Task Scheduling cannot simply
wrap an opaque frozen CUDA/PTX math body. The safe migration path is therefore:

1. keep the frozen CUDA kernel as the numerical and performance oracle;
2. port one resource/work edge at a time into CuTe DSL;
3. compile and compare after each edge;
4. assemble the complete Task graph only after the individual work bodies are
   equivalent.

The next vertical slice is the first `TmaAsync` KDA input edge, not a one-shot
rewrite of the entire direct-M128 body.

## First real KDA edge and the new primitive

Exact source tracing shows that the first Q/K path is not a direct
`TmaUmma` edge. Each stage issues two TMA transfers (8192-byte Q and 8192-byte
K) into `qk_raw_full`; one of five four-warp prep instances consumes the raw
stage, performs the transforms, and publishes `qk_full`. Both the four-warp
compute role and the one-warp UMMA role wait on that publication. Stage reuse
is authorized independently through `smem_free`.

That topology yields two concrete results:

- `qk_raw_full` lowers exactly to a five-stage `TmaAsync` configuration with a
  16384-byte transaction and a 128-thread async consumer group;
- `qk_full` is represented as typed
  `SplitPhaseFanout(ready, consumers, reuse_waiter)`. CUTLASS TS already supports
  wait-only and release-only tasks by automatically enabling
  `advance_on_wait`, so that portion can be reused directly.
- exact lowering still needs a narrower `LeaderArrivalSplitConsumer` backend
  primitive. Arrival ownership is variant-specific and must be recovered from
  source: the real `a1418fd1ae` kernel uses one elected `aux_mma` lane for
  `qk_full`, but four elected compute-warp leaders for `smem_free`.

The cooperative-arrival form is not discarded. Its arrival counts are a legal,
typed schedule mutation whose synchronization overhead can be measured on
B300; it is simply not mislabeled as a fully native or exact lowering. A
multi-consumer fan-out still needs a completion join before stage reuse.

This is a stronger branch than parameter search: the IR now detects a place
where an existing scheduling abstraction would change the physical protocol,
preserves it as an explicit unlowered edge, and turns the gap into the next
language feature rather than silently emitting a plausible but wrong schedule.

## Executable cooperative fan-out and compiler rules

The native cooperative-arrival alternative now compiles and executes on B300,
including a real SMEM payload crossing the `qk_full` barrier. Its physical
roles are 4 producer warps, 4 wait-only compute warps, and one recycler warp;
the latter alone releases each of five stages. CUTLASS TS automatically enables
`advance_on_wait`, the single-pass checker explores 3071 states with no
deadlock or race, and all 288 marker outputs equal 15 (the sum of stage payloads
1 through 5). This run does not wrap the five-stage ring, so it does not prove
that the cooperative form can safely reuse stages without a completion join.

Reaching that result exposed two general codegen defects in the installed DSL,
both now represented as backend diagnostics instead of being hidden in the
smoke:

- KIR710: legacy staged control flow eagerly replaces a frozen dataclass with
  a mutable `SimpleNamespace` proxy before executing the region. A pipeline
  carrying live barrier pointers therefore loses methods such as
  `producer_acquire`. Delaying proxy installation until immutable
  reconstruction preserves the typed object during region execution.
- KIR711: ordinary All-thread pipeline paths initialized `do_wait`,
  `do_release`, and `do_commit` as runtime `Boolean(True)`. This creates
  unnecessary dynamic regions that try to carry Python metadata. The default
  paths must be emitted unconditionally; only actual CTA-leader predicates
  should create runtime branches.

The distinction matters for the agent architecture. KIR710/711 are not CUDA
syntax failures and not schedule illegality; they are backend-capability
failures detected before an agent spends search budget on candidates that the
installed compiler cannot faithfully stage. The schedule remains portable,
while the compatibility profile determines whether executable custom work may
be lowered.

The exact-backend boundary is now typed as well. `SplitPhaseFanout` records
whether ready/free arrivals are `elect_one`, `elect_one_per_warp`, or
`cooperative`; KIR163--KIR166 and KIR175--KIR176
check those modes against their initialized counts and owning warp roles.
Lowering produces a `LeaderArrivalSplitConsumerPlan` for the producer, every
wait-only consumer, the reuse waiter, both phase counts, and the
advance-without-release behavior. `CompletionJoin` separately records every
participant, the release-owning waiter, arrival mode/count, and operation-level
arrive/wait events. KIR167--KIR174 reject a missing join, mismatched ownership,
incorrect cardinality, or recycle that is not dominated by the join wait. This
separates two legitimate arrival search nodes while preserving the same stage
reuse obligation for both.

The raw backend primitive is now executable too. A B300 smoke initializes the
ready/free mbarriers with count one, elects a single producer and recycler lane,
and uses a consumer-completion rendezvous before stage release. Ten iterations
exercise one full wrap of the five-stage ring; all 288 output lanes observe 55,
the sum of payloads 1 through 10. The rendezvous is semantically important:
electing one recycler is safe only if its completion dominates every wait-only
consumer. That condition is now a first-class `CompletionJoin` referenced by
the fan-out instead of an implicit boolean assumption. Integration into Task Scheduling remains
the next backend step; complete FlashKDA numerical equivalence is still a
separate milestone.

## Source-grounded IR: the next step beyond CAKE

CAKE's central representation choice remains sound: roles, resources,
synchronization, pipelines, and instruction forms should be visible before
CUDA generation. Its reported KDA experiment, however, deliberately treats
official FlashKDA as an implementation-hidden timing baseline. That is useful
for clean-start evaluation but leaves a complementary opportunity: import a
production kernel into typed IR with a checkable mapping back to source, then
search explicit semantic deltas rather than starting from raw text or a blank
schedule.

CUTLASS Task Scheduling reinforces this need. Its official validation guide
states that synchronization hidden inside `@cute.jit` work bodies is opaque to
TS and remains the author's responsibility. Frozen KDA uses exactly such
hardware-specific events: ordinary mbarrier waits/arrivals and SM100
`elect_commit` signals participate in one completion chain.

The `a1418fd1ae` source import now recovers:

- exact warp ranges for all six roles, including `aux_mma=[10,11]`,
  `mma=[9,9]`, and `prep=[12,31]`;
- `qk_full`: five stages, `aux_mma` publication, `compute`/`mma` consumers,
  elect-one count 1;
- `smem_free`: four elected compute-warp arrivals and a `prep` reuse wait,
  represented as `elect_one_per_warp` count 4;
- a source-line-backed transitive path from each qk consumer through the
  intermediate barrier graph to `smem_free` reuse.

This changes the role of the IR. It is not only a safer synthesis language; it
is also a semantic microscope for expert kernels. An agent can now propose a
change as a typed delta against recovered ground truth, and the verifier can
distinguish a deliberate schedule replacement from accidental loss of an
existing dependency. KIR713 and KIR714 are the first import-side rules: one
rejects barrier counts that cannot be assigned a hardware arrival type, while
the other rejects an import where any qk consumer lacks a transitive completion
path to stage reuse.

The current certificate proves structural source-order reachability. A later
stage must additionally prove that role-local stage expressions and phase bits
refer to the same logical ring slot before treating the certificate as a full
formal happens-before proof.

## Primary sources

- CAKE paper: <https://arxiv.org/html/2608.12629>
- CUTLASS Task Scheduling introduction:
  <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/ts_general/ts_introduction.html>
- CUTLASS Task Scheduling pipelines:
  <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/ts_general/ts_pipelines.html>
- CUTLASS Task Scheduling memory API:
  <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/ts_api/memory.html>
- CUTLASS Task Scheduling validation and explicit verification gaps:
  <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/ts_general/ts_validation.html>
- CUTLASS Python DSL license:
  <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/license.html>
