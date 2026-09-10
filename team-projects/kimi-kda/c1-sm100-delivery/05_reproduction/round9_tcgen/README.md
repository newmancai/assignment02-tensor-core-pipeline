# Round 9 tcgen05 direct physical-layout probe

The Round 8 preferred arm still reads logical `B[K,V]` with a strided access
while writing the 32-byte-swizzled `B^T[V,K]` descriptor tile.  This experiment
keeps its math, TMEM lifecycle, grid, state epilogue, and timing method fixed,
but supplies global B in producer-ready NK order.  The candidate therefore
tests whether closing the physical producer/consumer contract is better than
merely moving a logical transpose outside a synthetic inner loop.

The patch is applied to a copied, previously validated probe.  Both logical and
physical binaries run the same L0/L1 correctness oracle before their V16/grid96
timings.  This remains a Phase-6 mechanism result, not a public-full FlashKDA
speedup and not evidence that Phase 4 can emit the layout for free.

Important validation correction: the original Job 25317 input was constant
along K and could not detect a 16-byte swizzle sub-block permutation. The
accepted reproduction uses the corrected `Swizzle<1,4,3>` row bit and a
K-nondegenerate input. Job 25380 is authoritative; its files live in
`../../04_evidence/agent_rounds/round11_tcgen_revalidation/`. The original
result is retained only as superseded provenance.
