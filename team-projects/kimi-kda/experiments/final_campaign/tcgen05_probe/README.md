# FlashKDA K2 Phase-6 `tcgen05` probe

This directory is an isolated decision probe for the assignment question:

> FlashKDA currently uses the SM80 `mma.sync` atom on Blackwell. Is a B300
> execution-engine rewrite worth pursuing?

It does **not** claim to be an optimized SM100 FlashKDA kernel. It measures one
real K2 operation, with explicit lower/upper-envelope scopes, before we spend a
scarce 15-minute B300 slot on a full rewrite.

## Exact operation and instruction shapes

K2 Phase 6 computes, per head and per 16-token chunk:

```text
delta_state[128,V] = k_restored_t[128,16] @ U[16,V]
state[128,V] = BF16(FP32(state) * g_total[row] + delta_state)
```

`V` is swept over `16,32,64,128`; `V=128` is the official Kimi K3 shape. The
existing K2 path decomposes this into `m16n8k16` BF16 `mma.sync` atoms. The
candidate issues one `tcgen05.mma.cta_group::1.kind::f16` with shape
`m128nVk16`. NVIDIA's actual constraint is `M in {64,128}` and `N=8..256` in
steps of 8, so `V=16/32/64/128` are all legal; the course's `m128n64k64`
example is not the instruction's minimum N.

The candidate uses compact K=16 32-byte swizzle atoms, not a K=64 tile padded
to the course example's 128-byte swizzle. TMEM allocation is rounded only as
required by the ISA: 32 columns for `V=16/32`, 64 for `V=64`, and 128 for
`V=128`.

## What L0 and L1 mean

| Level | Included in both paths | Extra in `tcgen05` | Interpretation |
|---|---|---|---|
| L0 | global-to-shared staging, Phase-6 GEMM, result consumption/store, full kernel launch | TMEM alloc/commit/wait/load/dealloc | Optimistic core probe; operands are already in each path's preferred on-chip layout. |
| L1 | L0 plus BF16 state load/update/store and per-row FP32 gate | Every inner phase scalar-reformats logical `U[16,V]` into the compact descriptor layout | Conservative materialization envelope, not the final implementation. |

Important limitations:

- Current FlashKDA enters Phase 6 with `U` already in an SM80 register fragment.
  L0 does not charge the candidate for changing that representation.
- L1's scalar shared-to-shared copy charges both a load and a store. A real
  implementation should use transposed `stmatrix`/`tcgen05.st`, or keep `U^T`
  in TMEM across Phases 1/3/4. Therefore L1 is intentionally conservative.
- The probe does not include K1, K2's TMA producer/store warps, pipeline
  overlap, shared-memory aliasing, or whole-forward scheduling.
- `inner=1` is complete launch time but pessimistically pays TMEM allocation
  once per phase. `inner=64` allocates once and reports `kernel_us/inner`; it
  approximates persistent reuse, but repeats fixed operands rather than the
  full recurrence.
- CUDA's occupancy API reports register/shared-memory limits. It may not expose
  every TMEM scheduling constraint; confirm the winner with profiler data.

These caveats are also printed in the CSV `scope` field so the data cannot be
mistaken for an end-to-end KDA result.

## Build and smoke test

CUDA 13 and a Blackwell data-center target are required. On B300:

```bash
cd /path/to/tcgen05_probe
make CUDA_HOME=/usr/local/cuda-13.0 ARCH=103a GUARDRAILS=1
./phase6_probe --validate-only --check-inners 1,2,4
```

The guarded build uses PTXAS TMEM bounds checking. Do not benchmark it. A
successful smoke test prints exact-match PASS rows for L0 and bit-exact BF16
PASS rows for L1, at every V and at barrier rounds 1/2/4.

Release build and the complete requested sweep:

```bash
make clean
make -j CUDA_HOME=/usr/local/cuda-13.0 ARCH=103a
./phase6_probe \
  --benchmark-only \
  --values 16,32,64,128 --grids 12,148 --inners 1,64 \
  --warmup 30 --iters 200 --repeats 5 \
  --csv phase6_probe.csv
make CUDA_HOME=/usr/local/cuda-13.0 sass
```

On the current B300 login node, invoke the toolkit through
`/usr/local/cuda-13.0/bin/nvcc`; the `/usr/local/bin/nvcc` symlink does not
discover its own headers and fails with `cuda_runtime.h: No such file`.

`grid=12` represents one TP8 Kimi K3 request (96 heads / TP8), while
`grid=148` approximately fills one CTA per B300 SM. Each repeat records the
entire kernel with CUDA events; path order alternates across repeats. CSV
contains median/min kernel microseconds, normalized per-inner-phase time,
speedup, registers, static shared memory, and API-estimated blocks/SM.

## One 15-minute Slurm job

Copy this directory to the B300 campaign location, then submit:

```bash
sbatch --export=ALL,TCGEN05_PROBE_DIR=/absolute/path/to/tcgen05_probe \
  run_03_tcgen05_probe.sbatch
```

The job first builds with guardrails and runs correctness under a three-minute
timeout. Only after that passes does it rebuild without guardrails, dump SASS,
and run the timing sweep under a ten-minute timeout. The default output paths
match the existing C1 final campaign and can be overridden with
`TCGEN05_RESULT_DIR` and `TCGEN05_ARCH`.

## Decision rule

Treat the result as a gate, not a victory claim:

1. If `tcgen05` loses L0 at `V=128` for both grids after amortization, stop: the
   execution primitive itself is not promising for this phase.
2. If L0 wins but L1 loses, the next work is representation/layout design
   (`U^T` in TMEM or transposed store), not whole-kernel integration.
3. If L1 wins at `grid=12`, implement the small four-phase transposed-dataflow
   prototype while preserving every BF16 rounding point.
4. Only a full FlashKDA forward improvement above run-to-run noise can answer
   the assignment's final “worth it” question. This probe alone cannot.

Primary ISA references:

- [NVIDIA PTX ISA: `tcgen05.mma`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-mma)
- [NVIDIA PTX ISA: TMEM allocation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-alloc-tcgen05-dealloc-tcgen05-relinquish-alloc-permit)
- [NVIDIA PTX ISA: `tcgen05.ld/st/wait`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory-and-register-load-store-instructions)
