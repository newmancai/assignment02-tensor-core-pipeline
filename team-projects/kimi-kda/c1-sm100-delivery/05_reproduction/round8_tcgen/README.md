# Round 8 tcgen05 V16 preferred-layout bridge

This directory is a self-contained, vendored probe. It does not include or
modify the archived Stage 11 source at build time.

## Question and budget

The single structural mutation asks whether tcgen05 closes the V16 L1 gap when
its producer contract supplies the physical `B^T[V,K]` descriptor layout once,
so the timed inner lifecycle no longer performs the scalar shared-to-shared
reformat. The experiment keeps V=16, D=128, K=16, grid=96, state/gate work,
BF16 rounding, TMEM protocol, and correctness oracle unchanged.

This is one implementation candidate, one correctness build, and one release
timing build. It is the tcgen05 Round 8 equal structural-budget arm. It is not a
full K2 integration and does not charge producer-side preferred-layout
conversion inside the repeated phase.

## Arms and gates

- `mma_sync`: resident HMMA L1 baseline.
- `tcgen05_scalar`: archived conservative L1 control with a scalar reformat on
  every repetition.
- `tcgen05_preferred`: the only mutation; it stages the same logical input into
  the preferred descriptor layout once before the inner loop.

Correctness requires zero BF16-bit mismatches for all three L1 arms at
inner=1,2,4. Timing uses a three-way rotating order for every repeat and reports
inner=1 and inner=64. The practical promotion gate is preferred/HMMA >= 1.03
with a positive paired confidence interval; preferred must also recover at
least 0.328 us/phase from the archived scalar tcgen05 L1 result. A failed gate
is still useful evidence that removing the measured reformat is insufficient.

## Build and run

```bash
make clean
make -j8 CUDA_HOME=/usr/local/cuda-13.0 ARCH=103a GUARDRAILS=1
./round8_v16_preferred --validate-only

make clean
make -j8 CUDA_HOME=/usr/local/cuda-13.0 ARCH=103a
make CUDA_HOME=/usr/local/cuda-13.0 ARCH=103a sass
./round8_v16_preferred --validate-only
./round8_v16_preferred --benchmark-only --warmup 30 --iters 200 \
  --repeats 5 --csv results/round8_v16_preferred.csv
```

On Slurm, set `ROUND8_PROBE_DIR` to this directory and submit
`run_round8_v16_preferred.sbatch`. The script rejects devices other than
compute capability 10.3. It also writes a SHA-256 receipt for the vendored
sources, scripts, release binary, and SASS beside the CSV; the JSON receipt
records the source provenance, experiment identity, and structural budget.

## Interpretation limits

The preferred arm is a producer-contract mechanism probe. It does not prove
that FlashKDA Phase 3 can produce this layout for free, preserve it across
Phase 4/6, or match the production 192-thread role pipeline. Record registers,
static shared memory, active blocks per SM, the release SASS, and raw repeat
timings before drawing an integration conclusion.
