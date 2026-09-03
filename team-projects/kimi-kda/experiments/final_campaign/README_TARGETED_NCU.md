# Targeted NCU: H12, T8192, fixed single sequence

This campaign compares the untouched official FlashKDA K2 (`V=128`) with the
validated integrated ValueSlice K2 forced to `V=16`.  It profiles exactly one
`_flash_kda_fwd_recurrence` launch per report; tensor creation, warmup, K1
prepare, copies, and allocator activity remain outside the selected kernel.

Collected sections are deliberately small and stable: SpeedOfLight,
LaunchStats, Occupancy, and SchedulerStats.  The job also queries the installed
NCU/GPU pair and adds any supported tensor-pipe-active percentage metric from a
short allowlist.  This is sufficient to compare duration, SM versus DRAM
throughput, tensor utilization, achieved occupancy, scheduler eligibility,
registers, dynamic shared memory, block size, and grid/CTA count.

Remote staging and execution:

```bash
scp profile_k2_targeted_ncu.py export_targeted_ncu.py \
  run_05_targeted_ncu.sbatch \
  b300-login:/home/lcpu/<USER_ID>/FlashKDA/profile/c1-final/
ssh b300-login \
  'sbatch /home/lcpu/<USER_ID>/FlashKDA/profile/c1-final/run_05_targeted_ncu.sbatch'
```

The expected grids are 12 CTAs for official V128 (`1 x 12 x 1`) and 96 CTAs
for ValueSlice V16 (`1 x 12 x 8`).  The exporter treats a different CTA count
as an error, which catches wrong-extension and wrong-dispatch captures.

Expected wall time is 2--6 minutes.  The Slurm limit is 15 minutes; only two
kernel reports are collected and each uses four stable sections plus at most a
few supported tensor raw metrics.

The job writes two `.ncu-rep` files, one long metric CSV, one compact summary
CSV, the NCU log, and the queried tensor-metric list beneath
`/home/lcpu/<USER_ID>/FlashKDA-c1-results/`.

For the public repository, the log and CSV exports are committed, while the
binary `.ncu-rep` files remain local because they embed host, account, and GPU
metadata.  Their filenames and SHA-256 digests are recorded in
[`artifacts/ncu/SHA256SUMS`](artifacts/ncu/SHA256SUMS).
