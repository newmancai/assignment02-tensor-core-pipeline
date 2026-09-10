# Stage 11: V16 HMMA vs `tcgen05` probe

This is the smallest compile/run entry for the Stage 11 question: can the
Blackwell `m128n16k16` primitive beat the favorable HMMA Phase-6 path at the
actual Kimi-K3 TP8 ValueSlice grid?

The executable reuses the reviewed implementation and oracle in
`../final_campaign/tcgen05_probe/phase6_probe.cu`, but fixes all semantic and
shape controls:

| Field | Fixed value |
|---|---:|
| KDA head dimension `D` | 128 |
| token chunk / MMA `K` | 16 |
| value slice `V` | 16 |
| TP8 local heads | 12 |
| slices per head | 8 |
| grid | 96 CTAs |
| Tensor Core candidate | `tcgen05.mma` CTA-group 1, `m128n16k16` |

The wrapper rejects options that could change `V`, grid, inner counts, or the
correctness grid.  Only timing counts, CSV path, and validate/benchmark mode are
configurable.

## Fairness and scope

Both arms receive the same BF16 `A[128,16]`, `U[16,16]`, initial BF16 state,
and FP32 row gate.  Both use FP32 MMA accumulation.  L1 applies the same update
after every inner step:

```text
state = BF16(FP32(state) * gate[row] + delta_state)
```

Correctness checks include the FP32 Phase-6 output and the bit-exact BF16 state
after 1, 2, and 4 iterations.  Those iterations exercise both mbarrier parity
phases and preserve the CHUNK=16 rounding boundary.

The HMMA arm is an intentionally favorable Phase-6 baseline: `U` is loaded
into register fragments before the timed inner loop, matching the important
operand-residency property of optimized ValueSlice.  The tcgen05 L0 arm also
receives its preferred descriptor layout.  L1 additionally charges tcgen05 a
conservative scalar `U` reformat on every iteration.

This is **not** a complete FlashKDA comparison.  In particular, the standalone
probe does not reproduce the production kernel's 96-thread load/compute/store
warp specialization, Phase-6 `StatePrefetch=4`, Phase-1 lookahead, TMA overlap,
or public forward wrapper.  Therefore:

- a loss is useful evidence against a Phase-6 instruction substitution;
- a win only authorizes a full-K2 integration candidate against the actual
  `ValueSlice + P4 + Phase1` baseline;
- this probe's microseconds must not be reported as KDA, Kimi prefill, or
  serving latency.

## Build without submitting GPU work

CUDA 13 with an SM103a-capable toolkit is required:

```bash
cd experiments/stage11_v16_tcgen_probe
make CUDA_HOME=/usr/local/cuda-13.0 ARCH=103a GUARDRAILS=1
./stage11_v16_tcgen_probe --validate-only
```

The guardrailed build is for correctness only.  Rebuild before timing:

```bash
make clean
make -j CUDA_HOME=/usr/local/cuda-13.0 ARCH=103a
./stage11_v16_tcgen_probe \
  --warmup 30 --iters 200 --repeats 5 \
  --csv stage11_v16_tcgen_probe.csv
make CUDA_HOME=/usr/local/cuda-13.0 sass
```

Expected correctness coverage is exactly 96 CTAs for both arms:

```text
VALIDATE,level=L0,path=mma_sync,V=16,grid=96,inner=1,PASS
VALIDATE,level=L0,path=tcgen05,V=16,grid=96,inner=1,PASS
VALIDATE,level=L1,...,inner=1|2|4,...,PASS
```

## Prepared Slurm entry

`run_stage11_v16_tcgen_probe.sbatch` performs a guardrailed correctness build,
then a clean release build, SASS dump, correctness rerun, and paired timing.
It is provided for review only; creating the file does not submit a job.

Primary screen criterion: the release `tcgen05` arm must beat HMMA at grid 96
for the amortized L0 scope.  L1 then shows whether per-iteration layout
materialization destroys that signal.  Any output/state mismatch is an
immediate stop.
