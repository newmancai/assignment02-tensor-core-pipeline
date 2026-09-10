# H3 matched-HMMA B300 measurement design

Status: **executed through the terminal matched-HMMA full-path gate.**  The
SM103a build and B300 correctness gate passed.  The H12/T8192 pilot advanced,
but the formal isolated-process six-H12/three-H96 CUPTI matrix triggered
`stop_h3`, with H12/H96 geomeans of 0.9055x/0.7581x.  This H3 layout does not
advance to tcgen05.

## Question and variants

This experiment asks whether moving `P = R V` and `W = R Kd` into K1, then
using `U = P - W S` in K2, improves the complete FlashKDA forward path.  It
does not test tcgen05 yet.

The three mandatory variants are:

| ID | Meaning | Required identity |
|---|---|---|
| `official_hmma` | pinned MoonshotAI FlashKDA | commit `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b`, CUTLASS `5c149f52a436782210263fb2f19b354443a61c6a` |
| `h3_matched_hmma` | H3 P/W using the same SM80 HMMA atoms and the official K1/K2 launch boundary | dedicated extension name and source/binary SHA-256 |
| `optimized_hmma` | guarded ValueSlice V16 + Phase-6 prefetch4 + Phase-1 lookahead incumbent | five-patch build recorded by `assignment02-github/team-projects/kimi-kda/experiments/mainline_20260905/data/BUILD_MANIFEST.json` |

`h3_matched_hmma` is matched at the instruction family and external contract,
not at total work: H3 intentionally adds `W` preparation and changes the BF16
rounding DAG.  `optimized_hmma` is a practical incumbent, not an opcode
control.  Its guard only selects the optimized path for BF16 public state,
N=1, H=12, and 2048--8192 tokens; every fallback must be recorded rather than
silently labelled optimized.

## Case set and semantic contract

Run the existing six H12 Stage11 cases, then the task's three H96 shapes:

- fixed 8192;
- packed mixed `(1300, 547, 2048, 963, 271, 3063)`;
- packed uniform `8 x 1024`.

All variants receive byte-identical q/k/v/g/beta/A_log/dt_bias and the same
nonzero initial state.  Use rotating, preinitialized state slots so every
timed call starts from identical state.  Validate both BF16 output and final
state against the input-level oracle before timing.  H3 is
`algorithm_equivalent_not_official_bitwise`; use the frozen H3 input-oracle
tolerance, never the Stage11 peer `1e-2` tolerance by default.  Record the
actual tolerance and per-target max-abs, relative-L2 and normalized-Linf.

For timing, expose one common in-place-state public contract.  Official and H3
write a distinct final-state buffer internally and therefore include the
same-stream copy-back in their `public_full` measurement.  Also retain a
diagnostic `raw_full` scope without copy-back, but do not use it for the
headline comparison.  Preallocate output, final-state scratch, workspace and
state rotations outside timing.  Build/JIT, allocation, metadata preparation,
packed sequence ordering and state-pool reset are outside timing.  Operations
that execute for every forward call, including beta transpose, layout packing,
workspace stores/loads and state copy-back, remain inside their applicable
scope.

Use cold-L2 CUPTI timing with the Stage11 pattern, 20 dry runs and 100 measured
iterations per block where the state-rotation budget permits.  Interleave the
three variants using a balanced rotating order across blocks; do not measure
all samples of one variant before the next.  Record raw samples and block
medians.  Do not pool absolute times across Slurm jobs.

## K1, K2 and full scopes

Each build must expose diagnostic launch controls equivalent to the existing
`BLOCK_LEVEL_K1` / `BLOCK_LEVEL_K2` split, without changing the compiled body
between a phase timing and full timing.

| Scope | Timed work | Output used for correctness? |
|---|---|---|
| `k1` | prepare kernel and all per-call producer transforms/stores | inspect P/W or official intermediates using a diagnostic oracle |
| `k2` | recurrence kernel from precomputed workspace, including its loads and final-state handling | yes; workspace and initial state are reset outside timing |
| `raw_full` | K1 + K2 and every required device transform, excluding public state copy-back | diagnostic only |
| `public_full` | `raw_full` plus same-stream state copy-back where required | yes; headline scope |

K1 and K2 timings are explanatory.  Only `public_full` decides whether H3 is
useful.  `k1 + k2` is not substituted for the measured full span because
launch gaps, stream ordering and transforms can differ.

## Workspace accounting

Report two byte counts separately:

1. `allocated_bytes`: size of the reusable allocation, including safe upper
   bounds and alignment slack;
2. logical `touched_bytes`: bytes addressed by the executed case.

For every workspace array record dtype, logical per-tile shape,
`bytes_per_tile`, exact `tiles_touched`, K1 read/write multiplicity and K2
read/write multiplicity.  Derive phase totals as

```text
k1_touched = sum(bytes_per_tile * tiles_touched * (k1_reads + k1_writes))
k2_touched = sum(bytes_per_tile * tiles_touched * (k2_reads + k2_writes))
full_touched = k1_touched + k2_touched
```

These are semantic device-memory requests, not measured HBM traffic: cache
hits, sector amplification and TMA transaction details can differ.  Optional
NCU DRAM/L2 counters go in a separate `measured_counters` object.

The official six-array model is 13,824 bytes per chunk/head written by K1 and
read by K2.  H3 should normally replace `kd` (4096 bytes) with `W`, replace the
512-byte `INV` with a 4096-byte `P`, and keep the other arrays, giving 17,408
bytes per chunk/head in the workspace and 34,816
logical workspace bytes touched across K1+K2.  Do not hard-code this expected
value into the runner: fail the preflight if the source-derived array ledger
and the implementation disagree.  Original input reads such as moving `v`
from K2 to K1 are recorded under `non_workspace_device_bytes`, not hidden in
the workspace number.

## Decision and stop rules

Stop before performance measurement if SM103a compilation fails, the H3 input
oracle fails for output or final state, a packed case crosses sequence
boundaries, or the workspace ledger does not match the implemented addresses.

After a correct H12 fixed-8192 pilot:

- stop H3 implementation if the 95% paired-bootstrap confidence interval for
  `official_hmma / h3_matched_hmma` on `public_full` has upper bound at or below
  1.0;
- otherwise run all H12 and H96 cases;
- call H3 practically competitive only if its H12 geomean speedup versus
  `official_hmma` exceeds 1.05, no case regresses by more than 2%, and it is not
  slower than `optimized_hmma` on the shapes where that optimized route is
  actually selected;
- if H3 improves K2 but loses in `public_full`, report that K1/workspace costs
  falsified the full-path proposal.  Do not proceed to tcgen05 on that H3
  layout.

The original same-process pilot and matrix are retained as development history,
but not as formal magnitude evidence.  Although Python used `RTLD_LOCAL`, both
extensions exported common weak/global CUDA symbols, and reversing load order
changed absolute timings.  The terminal runner therefore launches one
extension per worker process, uses four balanced blocks with 20 warmups and 100
samples each, and measures cold-L2 CUPTI public-full time.  Its result is
`evidence/h3_b300_isolated_process_cupti.json`; the source/binary identity is
recorded in `evidence/h3_source_manifest.json`.
