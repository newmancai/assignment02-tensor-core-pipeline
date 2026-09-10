# Current mainline status

Updated: 2026-09-10.

## Completion state

| Mainline layer | State | Evidence |
|---|---|---|
| Official reproduction and SM80-MMA confirmation | historical complete | archived official benchmark/SASS/NCU |
| Multiple migration paths | complete for the mainline decision | V128 negative, V16 layout crossover, CAKE positive full path, P3/P4 design, H3 implemented negative path |
| Evidence-linked Typed IR | complete at the claimed scope | `paths/index.json`, migration v2 probes, H3 IR, source-grounded partial CAKE IR |
| H3 algorithm screen | passed CPU entry gates | single chunk, multi-chunk recurrence and input-derived oracle |
| H3 matched-HMMA CUDA implementation | implemented and compiled for SM103a | isolated `FlashKDA-h3-pw`; build-attempt-2 log |
| H3 B300 correctness | passed | 8/8 cases, output and final state, fixed/tail/multichunk/packed |
| B300 H3 measurement | isolated-process 6xH12 + 3xH96 matrix complete | cold-L2 CUPTI; H12 geomean 0.9055x, H96 0.7581x |
| SM100 incremental conclusion for H3 | stopped | full-path practical gate failed; do not advance this layout to tcgen05 |
| CAKE full-stack B300 measurement | isolated-process paired matrix complete | all nine shapes positive; H12 geomean 2.482x, H96 2.253x |
| Mainline answer | closed at current scope | full-stack SM100 migration is worthwhile on measured B300 profiles; opcode-only/local migrations are conditional |

## New input-level result

`prototype_h3_input_oracle.py` derives C16 workspace values from common
q/k/v/g/beta, A_log and dt_bias inputs.  It compares official chunk ordering
and H3 P/W ordering to the same FP64 FLA-style token recurrence.

- 18 cases, failures: 0, entry gate: passed.
- Fixed lengths: 16, 17, 37, 65.
- Packed lengths: `[17,33,65]`, `[4,8,12]`.
- Includes tail, strong decay, zero/nonzero initial state.
- Three continuous-call T32+T33 checks match the corresponding T65 execution
  bitwise for both official and H3 routes because the split is C16-aligned.
- Worst H3 relative-L2: output 0.0056483, final state 0.00479087.
- Worst official relative-L2 against the same reference: output 0.0056483,
  final state 0.00477313.
- H3 and official are not generally BF16-equal; H3 remains an
  algorithm-equivalent path, not an official-bitwise path.

The CPU oracle approximates rather than reproduces CUDA `tanh.approx`,
`ex2.approx.ftz`, and warp reduction order.  Its result authorizes CUDA
implementation; it does not replace the first GPU correctness gate.

## Implemented CUDA path

The isolated `flash_kda_h3_C` extension is now built.  K1 reads V and emits BF16 P/W;
K2 consumes P/W and leaves P4/P6 epilogues unchanged.  The implemented
workspace ledger must be exactly:

```text
W 4096 + Qd 4096 + Kr 4096 + g_total 512 + P 4096 + Mqk 512
= 17,408 bytes per touched chunk/head
```

This is 3,584 bytes (+25.9%) above the official 13,824-byte workspace.  The
matched-HMMA full call is the decision scope; K1 and K2 timings are diagnostic.
The SM103a build succeeded after selecting the transposed LDSM atom required
by the P/W B operand.  The B300 GPU gate then passed all eight generated cases
for both output and final state.  H3 remains algorithm-equivalent rather than
official-bitwise.

The first same-process CUDA-event pilot measured H12/T8192 at 1.3753 ms
official versus 1.3481 ms H3.  It passed the preregistered advance rule but is
now retained only as a development observation.  A later audit found common
weak/global symbols in the two extensions and demonstrated that extension load
order changes absolute timing.  Formal comparison therefore uses one extension
per worker process with the CUPTI runtime from the CAKE checkout.

The isolated-process nine-case matrix confirms the stop decision.  All cases
remained correct.  H12 speedups are 0.7587x, 0.8908x, 1.0028x, 1.0219x,
0.9571x and 0.8316x, for a 0.9055x geomean.  H96 speedups are 0.8001x,
0.7624x and 0.7142x, for a 0.7581x geomean.  Thus the one fixed-T8192 winner
does not survive the cross-shape practical gate, and this H3 layout must not be
used as the parent of a tcgen05 migration.  The extra workspace and relocated
work are plausible explanations, not phase-attributed causal measurements.

## Full-stack SM100 confirmation

CAKE and official FlashKDA were rerun with exactly one implementation in each
worker process.  The formal run uses four balanced-order blocks, 20 warmup and
100 measured calls per block, cold-L2 CUPTI timing, identical shapes/seeds,
and the public in-place-state contract.  H12 speedups span 2.0775--2.7454x and
have a 2.4823x geomean.  H96 speedups span 1.8819--2.6222x and have a 2.2532x
geomean.  Every per-shape bootstrap interval is above 1.0.  Correctness is
linked to the prior public paired and Stage12 peer checks; the timing workers
do not claim a new independent correctness oracle.

CAKE's generated SM103a source contains `tcgen05.mma`, TMEM allocation and
load/store operations, but also changes fusion, warp roles, barriers,
pipelines, layouts and dispatch.  The measured gain therefore belongs to the
co-designed full SM100 path, not to `tcgen05` as a single isolated cause.

## Current answer

The paper's premise is supported in its defensible form: the official kernel
still executes legacy HMMA recurrence math on B300, and migrating to a
co-designed SM100 path is worthwhile on all nine measured H12/H96 profiles.
The evidence rejects the stronger shortcut that merely replacing the MMA
instruction, or applying any algebraically valid local rewrite, is sufficient.
Direct V128 tcgen05 and H3 P/W are negative; V16 shows a positive native-layout
core whose scalar materialization erases the gain.  The practical mainline
recommendation is to retain the proven full-stack CAKE route and its guarded
evolution variants, stop H3, and reserve component-level tcgen causality,
cross-GPU replication and concurrency for explicitly bounded follow-up work.
