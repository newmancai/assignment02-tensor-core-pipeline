# Round 1 review: evidence-linked migration paths

> Terminal update: H3 was implemented as an isolated matched-HMMA CUDA
> extension, compiled for SM103a, passed the B300 correctness gate, and ran a
> six-H12/three-H96 matrix.  An extension-symbol audit then replaced the
> same-process timing with isolated-process CUPTI measurements; the terminal
> stop decision is unchanged.  CAKE received the same isolated-process paired
> treatment over six H12 and three H96 profiles.

## Decision

Stop **H3 P/W precomputation at C16** after the matched-HMMA full-path matrix.
Do not migrate this H3 layout to tcgen05.  Continue to avoid describing a
CAKE-to-HMMA rewrite as a pure opcode ablation.

H3 supplied the intended information: it was numerically sound and gave a
small fixed-T8192 gain, but it lost across packed and H96 shapes.  In the
formal isolated-process CUPTI matrix, all nine cases remained correct; the
six-case H12 geomean was 0.9055x and the H96 geomean was 0.7581x.  The 25.9%
larger workspace and relocated work are mechanism hypotheses, not directly
phase-attributed causes.  This is sufficient to stop without an optimized-HMMA
comparison or tcgen05 child, because the parent already fails its practical gate.

## Independent proposals

### Algorithm/source: H3 then H3+H4

Use `R=INV*diag(beta)`, precompute `P=R*V` and `W=R*Kd`, and replace the
state-dependent `INV*(beta*(V-Kd*S))` with `P-W*S`.  At C16/D128/V128 this
moves a C2V contraction off the serial K2 path but adds W preparation and P/W
workspace.  If C16 is viable, retain numerical subblocks BC16 while trying a
physical/recurrence block BT32 using the block-triangular route already present
in the FLA source.  An explicit D-by-D affine scan was rejected as a first
prototype because exact segment composition costs D3 and D2V; its compact
low-rank form largely reduces to the same W/Y organization.

### SM100 implementation: H2-SS then H6

Form one logical contraction
`U^T[128,16] @ [Mqk^T|Kr][16,144]`.  A pure-SMEM operand version avoids the
unverified `tcgen05.st` carrier and reuses the validated allocation, descriptor,
commit/wait, TMEM-read and deallocation protocol in the Phase-6 probe.  The
first 16 columns must retain P4's BF16 contraction-and-add epilogue; the last
128 must retain P6's FP32 delta and gated state update.  A later H6 stage would
make the producer emit the tcgen-ready U layout and then separately validate a
packed-BF16 TMEM carrier.

### Full-cost path: native full-stack comparison and layout ablation

Separate three effects: official adapted path, the best constructible HMMA
implementation with the same mathematical/kernel boundary, and a native
tcgen05 implementation.  Time K1, K2 and full forward, including workspace and
layout work.  The official six-array workspace touches 13,824 bytes per
chunk/head across the K1/K2 boundary; at T8192 this is about 162 MiB for H12
and 1296 MiB for H96.  CAKE's archived result is a full-path existence proof,
but its TMEM, roles, barriers and layouts are co-designed, so a forced HMMA
replacement would not be a clean single-variable experiment.

## Cross-criticism

- H2's strongest claimed sharing mechanism is weaker than it first appears:
  official HMMA already retains the same U register fragment across P4 and P6.
  A logical n144 fusion may lower as n16+n128 or padded n160 and still require
  two distinct readback epilogues.  H2 remains a useful low-cost mechanism
  probe, but only against this register-resident matched baseline.
- H3's identity is exact over reals but changes the BF16 rounding tree.  Its
  main performance risk is moving work rather than removing it: W construction
  adds a C2D contraction and P/W adds traffic.  Its main numerical risk is
  cancellation and recurrence of state error over long weak-decay sequences.
- H3 can later feed H2/H6 by producing U directly in a consumer-ready layout,
  but combining them initially would lose H2's bitwise isolation and make
  attribution ambiguous.  Test H3 and H2 separately before a joint path.
- A CAKE matched-HMMA comparison can fairly compare native physical
  implementations under one public contract, but cannot generally identify a
  pure tcgen05 main effect because changing the accumulator also changes roles,
  layout, synchronization and occupancy.

## Results by path

| Path | Representation | Evidence scope | Result |
|---|---|---|---|
| Official SM80 MMA | historical baseline, partially linked | full forward, SASS/NCU | reproduced baseline |
| Phase-6 V128 direct tcgen05 | complete probe IR v2 | B300 V128/grid12/inner64/L0 | regression, 0.919742x |
| Phase-6 V16 thin-N | production IR incomplete | B300 grid96 microprobe | L0 1.615021x; L1 0.778523x |
| P3/P4 cross-phase TMEM | typed design | no CUDA implementation | unresolved carrier/lane mapping |
| CAKE full-stack SM100 | source-grounded partial IR plus full evidence | isolated-process B300 public path, 6xH12 + 3xH96 | all positive; H12 geomean 2.482x, H96 2.253x; not tcgen-causal |
| H3 P/W C16 | open-path IR v1 + compiled CUDA parent | isolated-process B300 public path, 6xH12 + 3xH96 | all correct; stop, H12 geomean 0.9055x, H96 0.7581x |

The H3 single-chunk screen ran 80 C16/D128/V128 cases.  Against a common FP64
reference, its worst relative-L2 was 0.00302 for U, 0.00361 for output, and
0.00344 for final state; the corresponding official-order maxima were 0.00344,
0.00405 and 0.00383.  All BF16-equality flags were false.

The recurrence screen ran 32 scenarios spanning 2/4/16/64 chunks, lengths
17/33/65, nonzero state, weak/strong decay, cancellation and two-call state
handoff.  The predeclared gate passed with no failures.  Worst H3 relative-L2
was 0.01013 for output and 0.01287 for final state, versus 0.01017 and 0.01304
for official ordering.  State handoff was bitwise stable for the identical
precomputed chunk sequence.  Those original workspace-boundary gaps were
subsequently narrowed by the 18-case input-level oracle (common q/k/v/g/beta
inputs, fixed/tail/packed) and the 8-case B300 output/final-state gate.  Exact
CUDA approximate-instruction order remains outside the CPU model and is why
the GPU gate is retained.

## Current answer to the main question

The evidence does not support a blanket yes or no.  A bare C16 Phase-6 opcode
swap is not worth integrating under its measured V128 condition, and the V16
screen shows why layout materialization can erase a faster tcgen05 core.  In
contrast, CAKE proves that a full SM100-specific KDA organization can be worth
shipping for all nine measured H12/H96 profiles.  The unresolved question is how
much of that value requires a full-stack rewrite and whether a smaller,
maintainable migration beats the optimized HMMA incumbent.  H3 now answers one
such attempt negatively: despite correct CUDA behavior and a 1.0219x fixed
T8192 result in the isolated matrix, its cross-shape cost is unacceptable.

## Non-blocking follow-up work

1. Preserve H3 as a negative, evidence-linked branch; do not add tcgen05.
2. If further mechanism work is required, use K1/K2 diagnostics only to explain
   the regression, not to reopen the stopped branch without a new hypothesis.
3. If component-level causality is pursued, the next independent low-cost
   probe is H2-SS m128n144k16, with the explicit caveat that official HMMA
   already retains U and that a logical n144 fusion
   may split or pad.  It cannot be presented as a full-forward win.

The B300 entry was rechecked on 2026-09-10.  H3 compile/correctness and the
terminal H3 and CAKE isolated-process matrices have been executed and archived.
The same-process timing files remain as diagnostic history and must not be used
as the formal magnitude evidence because exported extension symbols caused
load-order-sensitive absolute times.
