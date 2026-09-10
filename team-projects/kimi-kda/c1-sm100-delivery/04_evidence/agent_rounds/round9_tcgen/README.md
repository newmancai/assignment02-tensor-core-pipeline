# Superseded Round 9 tcgen05 evidence

Job 25317 compiled and ran, but its A0 validation input used a coefficient that
made the tensor constant along K modulo 3. It therefore could not detect the
incorrect 16-byte swizzle sub-block permutation. Do not use this directory for
performance conclusions.

The corrected, K-nondegenerate revalidation is Job 25380 in
`../round11_tcgen_revalidation/`. The files here are retained only to preserve
the audit trail; no result was silently overwritten.
