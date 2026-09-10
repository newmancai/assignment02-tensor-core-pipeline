# Round 10: minimum P3/P4 tcgen05 lifecycle

This SM103a mechanism probe executes the exact typed sequence
`P3 -> BF16 round -> P4` with identical deterministic inputs in two kernels.
The HMMA arm uses one fused CTA; the tcgen05 arm allocates TMEM once, reuses
the same accumulator columns for both MMAs, and exposes the rounded value
through one safe 32B-swizzled SMEM carrier. Both arms preserve the intermediate
and final BF16 rounding points.

The experiment validates BF16 output bits for inner counts 1/2/4 before
timing grids 12/96 at inner 1/64. It isolates lifecycle and carrier cost; it is
not a public FlashKDA result and does not include P1/P2/P5/P6, beta, TMA, or the
production warp-specialized schedule.

The accepted Job 25377 path uses the corrected `Swizzle<1,4,3>` mapping
(address bit 4 XOR address bit 7), a K-nondegenerate validation input, one TMEM
allocation across both MMAs, and only the required tcgen/thread-proxy fences.
It passes exact BF16 validation and reaches 1.4460x/1.4303x over the fused HMMA
probe for inner=64 at grids 12/96, while inner=1 remains launch/lifecycle bound
at 0.7651x/0.7089x. These are mechanism results, not public-full speedups.

Two direct D-fragment-to-A-TMEM layouts were also compiled and rejected by
exact validation (Jobs 25382 and 25385). Their source and logs are retained in
`../../04_evidence/agent_rounds/round12_tmem_carrier_rejection/`; the failing
arm is intentionally absent here so the accepted reproduction never times an
incorrect kernel.

Submit `run_round10_p34_lifecycle.sbatch` after replacing `<REMOTE_HOME>`.
