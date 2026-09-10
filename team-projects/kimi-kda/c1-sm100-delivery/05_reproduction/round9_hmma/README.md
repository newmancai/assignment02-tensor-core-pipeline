# Round 9 HMMA paired-warps experiment

The current ValueSlice kernel assigns one compute warp to each 16-column
block for V32 and V64.  The V128 path already proves that one warp can carry
two independent column blocks.  This experiment reuses that mapping for V32
and V64, reducing the K2 block from 128 to 96 threads at V32 and from 192 to
128 threads at V64 without changing grid geometry, BF16 rounding, public-call
scope, or recurrence order.

`paired_warps.patch` is applied to the frozen, hash-checked HMMA source in an
isolated build.  `bench_hmma_paired_warps.py` compares the candidate with the
unchanged ValueSlice parent and official FlashKDA in fresh worker processes.
Both output and updated state must be bitwise equal to the official peer before
timing begins.
