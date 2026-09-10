# Corrected tcgen05 layout revalidation

This is the authoritative replacement for the original Round 9 Job 25317.

- Job 25379 revalidates the Round 8 V16 preferred-layout path after correcting
  `Swizzle<1,4,3>` and making A0 genuinely vary along K.
- Job 25380 reruns the logical-versus-producer-ready global-B comparison using
  the same corrected validation contract.

All L0/L1 HMMA, scalar-tcgen and preferred-tcgen validation cases pass. The
corrected conclusion is unchanged in direction but not in raw timing: the
producer-ready input is 0.99994x/0.99263x versus logical at inner 1/64, and the
inner-64 physical path is 0.94606x versus HMMA. Use `summary.json` and these
raw files for every Stage 12 citation.
