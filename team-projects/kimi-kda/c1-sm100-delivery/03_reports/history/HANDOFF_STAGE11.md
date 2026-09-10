# Stage 11 handoff: Kimi KDA SM80 MMA to SM100 causal migration

## Research-scope update (2026-09-10)

For an executable continuation guide, start with
`HANDOFF_SM100_OPEN_RESEARCH.md`. It consolidates the scope correction,
current state, file/tool entry points, next actions, and failure handling.
The mainline deliverable now explicitly includes evidence-linked multi-path
Typed IR, path comparisons, and a retrospective that answers the migration
question. Concurrency and the earlier paper's other extensions follow as
subsequent challenges; they do not replace this mainline result.

The active question is exactly: **FlashKDA 官方 kernel 当前使用 SM80 MMA，分析迁移到 SM100 是否值得**.
The user has explicitly reopened the research space. Read
`SM100_MIGRATION_OPEN_RESEARCH.md` and `SM100_OPEN_AGENT_PROTOCOL.md` before
selecting the next implementation. CHUNK16, the existing rounding DAG,
P3/P4, V16, and the lifecycle/thin-N 2x2 are conditions of particular
experiments, not global restrictions on this question. The minimum P3/P4
probe below remains one candidate; it is no longer a mandatory predecessor
of every other idea. Preserve public semantics and declare numerical
compatibility requirements explicitly for each experiment. Preserve all
archived evidence and preregistrations under their original scope.

The new documents contain source research and proposed experiments, not new
GPU results or an implemented replacement for the Stage9 runner.

## Current conclusion

The project must not summarize the migration as either "tcgen05 is useless"
or "V16 already accelerates FlashKDA." The supported conclusion is narrower:

- The Phase-6 V128 instruction-only replacement is negative on B300:
  `mma.sync/tcgen05 = 0.919742x` in the optimistic grid12/L0/inner64 scope.
- A materially different V16/grid96 mechanism screen is positive only when
  operands are already in each path's preferred layout: B300 job 24111 gives
  `1.615021x` at L0/inner64.
- The current scalar-rematerialization envelope loses at `0.778523x` for
  L1/inner64. Thus the open mechanism is cross-phase representation residency,
  not another bare opcode swap.
- CAKE's full-stack SM100 path is a positive existence proof, with 2.126x to
  3.397x raw speedup and 2.655x geometric mean over six correctness-passing H12
  profiles. It changes too many physical subtrees to attribute that gain to
  tcgen05 alone.

The next implementation must remove at least `0.328 us/phase` from the current
V16 tcgen L1 path (22.1% of its L1 time) to beat the matched HMMA L1 arm.

## What is implemented

### Typed migration and knowledge

- `kda_ir/mma_migration.py`
  - typed_mma_migration_v2;
  - target architecture, logical/physical MNK, instruction family and resource
    contracts;
  - explicit `transpose_output_swap_operands` equivalence;
  - TMEM protocol and event lifetimes;
  - cross-site carriers and rounding-boundary preservation;
  - namespaced full-SHA baseline identities.
- `kda_ir/mma_migration_fixtures.py`
  - P3/P4 shared-lifecycle candidate;
  - source logical form `16x16 @ 16x128`;
  - physical tcgen05 form `m128n16k16` through `(A@B)^T=B^T@A^T`;
  - serialized accumulator reuse and an explicit rounded-BF16 carrier.
- `kda_ir/mma_knowledge.py`
  - attempt, single-path, multi-path, causal and transfer maturity;
  - performance decisions cannot become verifier rules;
  - single-path negatives remain reopenable scoped priors.

The CPU suite currently passes 87/87 tests.

### Multi-agent causal protocol

`evidence/stage11_tcgen05_causal_ladder_preregistration.json` freezes a 2x2:

| Cell | Lifecycle A | Thin-N B | State |
|---|---:|---:|---|
| 00 | no | no | V128 direct-swap control measured negative |
| 10 | yes | no | typed/design complete; CUDA not implemented |
| 01 | no | V16/grid96 | mechanism probe measured; production cell incomplete |
| 11 | yes | V16/grid96 | waits for A and B implementation freezes |

Rules: A/B mutation subtrees are disjoint; a negative A or B does not cancel
AB; CAKE full-stack results cannot be used as a tcgen05 main-effect estimate;
the interaction is `(AB-A)-(B-control)`.

### B300 V16 mechanism screen

Source:

- `assignment02-github/team-projects/kimi-kda/experiments/stage11_v16_tcgen_probe/`

Job 24111 ran on `<B300_LOGIN_HOST>`, partition `gpu`, NVIDIA B300 SXM6 AC,
compute capability 10.3. All L0 output and L1 BF16 state checks passed for
inner 1/2/4. Timing at V16/grid96:

| Scope | inner | mma us/phase | tcgen us/phase | mma/tcgen |
|---|---:|---:|---:|---:|
| L0 preferred layout | 1 | 6.160480 | 8.208480 | 0.750502x |
| L1 scalar reformat | 1 | 8.280320 | 10.256160 | 0.807351x |
| L0 preferred layout | 64 | 0.672010 | 0.416100 | 1.615021x |
| L1 scalar reformat | 64 | 1.152415 | 1.480258 | 0.778523x |

Evidence:

- `evidence/stage11_v16_b300/stage11_v16_tcgen_probe_24111.csv`
- `evidence/stage11_v16_b300/stage11_v16_tcgen_24111.log`
- `evidence/stage11_v16_b300/stage11_v16_tcgen_probe.sass`
- `evidence/stage11_v16_b300/stage11_v16_tcgen_24111_accounting.json`
- `evidence/stage11_mma_migration_memory.json`

The V16 challenge is deliberately recorded as `inconclusive`, because the
standalone probe does not reproduce production ValueSlice warp specialization,
TMA overlap, P3/P4/P6 residency or public forward timing.

## Next exact implementation

Implement the minimum A path before attempting the full AB kernel:

```text
P3: U0^T[128,16] @ INV^T[16,16] -> U1^T[128,16]
round: U1_bf16^T = BF16(U1^T)
P4: U1_bf16^T[128,16] @ Mqk^T[16,16] -> O^T[128,16]
```

Requirements:

1. HMMA, tcgen05 and CPU oracle use identical inputs and the same intermediate
   BF16 rounding point.
2. Start with a safe SMEM rounded carrier if necessary, but allocate TMEM once
   and serialize P3/P4 reuse of the accumulator columns.
3. Isolate the rounded-carrier materialization cost rather than hiding it.
4. Validate output bitwise where the source contract requires BF16 identity.
5. Only after that works, attempt a direct TMEM carrier and integrate P6.
6. Compare any production candidate against the optimized HMMA ValueSlice +
   phase-prefetch incumbent, not the weak official V128 baseline.

No CUDA source for this P3/P4 probe has been completed yet. The unresolved ISA
detail is a compiled and lane-verified `tcgen05.st/ld` packed-BF16 carrier on
SM103a; do not claim it works until B300 compilation and correctness pass.

## Multi-agent interpretation for the paper

The old system was mainly parallel delegation and should remain a baseline,
not an emergence claim. Stage 9 later implemented typed proposal isolation,
canonical deduplication, conclusions-only memory, deterministic evaluation and
equal candidate budgets. Its diverse B300 search produced no new winner inside
the existing decision-tree grammar (wave 1.000346x, tail/skew 0.978989x, mixed
0.988022x), showing that the action grammar—not merely the number of agents—was
the limiting factor.

The defensible novelty claim is not "we invented Typed IR." It is a
source-grounded, proof-carrying Typed IR used simultaneously as the optimization
language, verifier boundary, multi-agent communication protocol and scoped
knowledge store for stateful KDA/profile optimization.

## Resume checklist

At the start of the next conversation:

1. Read this file and `STAGE11_MMA_MIGRATION_REVIEW.md`.
2. Treat `evidence/stage11_mma_migration_memory.json` as the machine-readable
   claim state.
3. Do not rerun V128 direct swap or job 24111 unchanged.
4. Continue from the minimum P3->BF16->P4 lifecycle probe.
5. Use `ssh <B300_LOGIN_HOST>`; the old `221.238.13.179:10029` endpoint is a different
   RTX 5090 cluster and must not be used for B300 conclusions.
