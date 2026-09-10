# H3 C16 matched-HMMA source patch draft

Status: source-audited design; not compiled and not a GPU result.  This draft
targets an isolated copy of `FlashKDA`; it must not be applied to the official
snapshot in place.

## Contract fixed by the existing H3 probes

For each C16 tile, retain the current KDA input/output/state interface but use

```text
P = BF16(INV @ BF16(diag(beta) * V))
W = BF16(INV @ BF16(diag(beta) * Kd))
U = BF16(P - FP32(W @ S))
```

and leave the current P4 output and P6 state-update epilogues unchanged.  This
is the already recorded `algorithm_equivalent_not_official_bitwise` path, not
an official bitwise-compatible build.  `beta` in these equations is the same
activated BF16 value used by current K2 Phase 3.  The inner `BF16(beta*X)` is
required by the saved CPU probe; rounding `INV*diag(beta)` into a stored R
would be a different rounding DAG.  Do not distribute or fuse later
output/state epilogues.

## Workspace replacement

Current per-tile arrays are:

```text
Kd 4096 | Qd 4096 | Kr 4096 | g_total 512 | INV 512 | Mqk 512
= 13,824 bytes
```

The minimal H3 replacement is:

```text
W 4096 | Qd 4096 | Kr 4096 | g_total 512 | P 4096 | Mqk 512
= 17,408 bytes
```

Thus `W` reuses the old Kd slot, `P` replaces the old INV slot but expands it
from 512 to 4096 bytes, and the net increase is 3,584 bytes/tile/head (25.9%).
All offsets remain 128-byte aligned.  In `utils.cuh`, introduce a separate
experimental `H3WorkspaceSizes` rather than silently changing the official
`WorkspaceSizes`:

```cpp
static constexpr int kW         = CHUNK * D * 2;
static constexpr int kQDecayed  = CHUNK * D * 2;
static constexpr int kKRestored = CHUNK * D * 2;
static constexpr int kGTotal    = D * 4;
static constexpr int kP         = CHUNK * D * 2;
static constexpr int kMqk       = CHUNK * CHUNK * 2;
```

The experimental `get_workspace_size_h3` must use the 17,408-byte total plus
the unchanged aligned tile-prefix buffer.  Keeping a distinct entry point
prevents an old caller from allocating the smaller official workspace.

## K1 prepare patch

1. Add `TmaLoadV` immediately after `TmaLoadK` in the template and kernel
   arguments.  Include another `cosize(QKLayout)*sizeof(BF16)` (4096 B at
   C16/D128) in `kTmaTransactionBytes`, create the V tensor from the same
   `[H,T,D]` layout, and issue its TMA load alongside Q/K.
2. Add a separate `v` array of `cosize(MMALayout)` BF16 elements to
   `SharedStorageK1`.  It cannot share the Phase-A/Phase-B union: INV does not
   exist until Phase B, while V must remain live for P.  This increases K1
   dynamic SMEM by approximately 4096 B.
3. After `neumann_inv_fused_1warp` and its synchronization, keep INV intact but
   stop exporting it.  Activate beta exactly as current K2 does, then form
   `BF16(beta[row]*X[row,col])` in the Kd/V operand registers before HMMA.
4. Reuse dead Phase-B buffers without increasing them further:
   - compute `W=R@Kd` into the now-dead `k_inv` storage;
   - after W is complete, compute `P=R@V` into the now-dead `k_decayed`
     storage;
   - retain `q_decayed`, `k_restored`, `g_total`, and `Mqk` for export.
5. Implement the two products as explicit 16x16 output tiles with the same
   `SM80_16x8x16_F32BF16BF16F32_TN` atom and FP32 accumulation used by K2.
   Eight warps can each own one D/16 column tile and calculate W then P using
   the beta-scaled B fragments.  Store each contraction to BF16 independently.
   Do not assume the existing `mma_m16n16_*_1warp` helper has the correct
   orientation for `INV[16,16] @ X[16,128]`; use `local_tile` and compile-time
   shape assertions.
6. Replace the six TMA stores with W, Qd, Kr, g_total, P, Mqk.  Preserve one
   `tma_store_arrive()` per issued store and the final `tma_store_wait<0>()`.

Tail behavior needs an explicit check.  Reuse the exact V tile addressing that
K2 currently uses.  Tail rows of W/P cannot affect state because the matching
Kr rows are zero, but the input-level correctness suite must still check
packed sequence boundaries rather than assuming padding.

## K2 recurrence patch

1. Rename `k_decayed` to `w` and `INV` to `p` in `InputStorage`; resize P from
   LMLayout to MMALayout.  Remove `v` and `beta` from the staged input storage.
2. Replace K2 template/arguments and load descriptors:

```text
remove: TmaLoadV, TmaLoadBeta, TmaLoadWsINV
keep:   TmaLoadWsW, TmaLoadWsQD, TmaLoadWsKR,
        TmaLoadWsGT, TmaLoadWsP, TmaLoadWsMqk
```

   Correspondingly, `kTmaTransactionBytes` changes from 17,984 B
   (`v + beta32 + Kd + Qd + Kr + GT + INV + Mqk`) to 17,408 B
   (`W + Qd + Kr + GT + P + Mqk`).
3. Phase 1 remains the dual W/Q projection against current state.  Rename the
   old `u_acc` result to `ws_acc`; Qd@S and its BF16 cast are unchanged.
4. Delete current Phase 2 V load and the Phase 3 beta/INV multiply.  Load the
   matching P fragment and form, elementwise:

```cpp
u_bf16 = BF16(p_bf16 - BF16_or_FP32(ws_acc));
```

   The exact cast sequence must match the CPU H3 probe: `W@S` accumulates in
   FP32; the subtraction is performed using the probe's BF16 boundary; U is
   finally BF16.  Add a small device oracle test before timing if the chosen
   C++ expression is ambiguous due to overloaded BF16 operators.
5. Leave Phase 4 (`Mqk@U`, FP32-to-BF16, then BF16 add), Phase 5 store, and
   Phase 6 (`Kr^T@U` plus gate/state update) byte-for-byte unchanged where
   possible.  This isolates H3 from H2/H6.

## Launch and public wrapper

Create experimental siblings `launch_fwd_h3` and `fwd_h3` rather than a macro
that changes the official entry silently.

- Build `tma_load_v` for K1 instead of K2.
- Replace workspace tensors/descriptors Kd/INV with W/P, both shaped
  `[H*total_tiles,16,128]` and using `TMAVOLayout`.
- Pass V to K1; remove V/beta/INV descriptors from K2.
- Kernel grids, 256 K1 threads, 192 K2 threads, three input stages, two output
  stages, state descriptors, and output ABI stay unchanged.
- Export `get_workspace_size_h3` so Python allocates 17,408 bytes per tile/head.

Keeping a separate extension name (for example `flash_kda_h3_C`) makes paired
official/H3 measurements possible in one environment without overwriting the
official binary.

## Timing scopes

Use four scopes and never present the middle two as full-forward speedup:

1. public `fwd` versus `fwd_h3`, CUDA events around the full call, identical
   tensors/state/order;
2. K1 kernel duration, obtained from Nsight Systems/CUPTI kernel records;
3. K2 kernel duration from the same trace;
4. full call including workspace allocation only as a separately labelled
   deployment scope (the normal kernel comparison should reuse preallocated
   correctly sized workspaces).

The primary matched-HMMA result is scope 1.  Scopes 2/3 explain whether H3
actually shortens recurrence or merely moves work.  Also record workspace
bytes and K1/K2 dynamic SMEM, registers, spills, and active blocks/SM.  K1 is
currently launch-bounded for eight blocks/SM; the extra 4 KiB and new HMMA work
may change occupancy.

## Lowest-cost compile path (no GPU execution)

In an isolated copy with the draft implemented, compile only the CUDA
translation unit first:

```bash
/usr/local/cuda-13.0/bin/nvcc -std=c++17 -O1 -c \
  csrc/smxx/fwd_launch_h3.cu \
  -Icutlass/include -Icutlass/tools/util/include -Icsrc \
  -gencode arch=compute_103a,code=sm_103a \
  --expt-relaxed-constexpr --expt-extended-lambda \
  -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ \
  -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ \
  -Xptxas=-v,--warn-on-spills -o /tmp/fwd_launch_h3.o
```

This checks templates, layouts and the feature target without running a GPU.
Then build the isolated extension:

```bash
FLASH_KDA_CUDA_ARCHS=103a NVCC_THREADS=4 \
  <REMOTE_HOME>/FlashKDA/.venv/bin/python setup_h3.py build_ext --inplace
```

Compilation alone does not validate the HMMA fragment orientation or BF16
rounding.  The first GPU gate, when separately scheduled, must compare P, W,
U, output and final state against the saved CPU probe over non-symmetric data,
tails, nonzero state and multi-chunk recurrence before any timing.

## Stop conditions

- Do not proceed to tcgen05 if matched-HMMA H3 fails the existing recurrence
  gate.
- If full-call H3 does not improve despite a faster K2, record the K1/workspace
  transfer as the cause; do not time K2 alone as a win.
- If H3 matched-HMMA wins, it becomes the correct parent for later H2/H6
  combinations; those changes require separate path identities.
