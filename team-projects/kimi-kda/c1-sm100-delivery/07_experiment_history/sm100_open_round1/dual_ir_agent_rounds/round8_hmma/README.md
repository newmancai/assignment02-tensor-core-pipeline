# Round 8 HMMA L3 build and screen

This directory contains the isolated structural-build arm for HMMA.  It changes
only the Phase-1 lookahead depth used by the existing guarded V16 path when a
nonzero BF16 initial state is present.  Phase-6 StatePrefetch remains four,
the source carrier and arithmetic order remain unchanged, and the generated
extension has its own Python/C++ module name.

The frozen L2 source is copied into a job-specific directory.  `l3.patch` then:

1. admits `Phase1Prefetch=3` in the CUDA template assertion; and
2. selects depth three instead of depth two for `HasStateIn`.

Submit after copying this directory to
`<REMOTE_HOME>/kda-dual-ir-agent-rounds-20260910/round8_hmma`:

```bash
sbatch round8_hmma_l3.sbatch
```

The job writes to
`<REMOTE_HOME>/kda-dual-ir-agent-rounds-20260910/round8-hmma-l3-job-$SLURM_JOB_ID`.
`build-artifact/build_receipt.json` records frozen and patched source hashes,
the patch and binary identities, the exact command and ptxas resource lines.
`results/summary.json` contains the isolated-process performance comparison.

The screen uses H12 fixed T8192, a nonzero BF16 initial state, V16 for both
optimized arms, two balanced blocks, 20 dry runs and 100 cold-L2 CUPTI samples
per block.  Every L2/L3 output and updated state must be finite and bitwise
equal to the same official peer.  L3 advances only at `L2/L3 >= 1.03`; this is
a two-block structural screen, not deployment qualification.
