# Upstream review: explicit FlashKDA evolution backend

## Recommendation

Submit the explicit `backend="evolution"` adapter as one reviewable patch. Do
not modify `backend="auto"` in the same patch. The five-shape activation policy
has scope-identical B300 evidence, but separating mechanism from default policy
keeps the first upstream change opt-in and gives maintainers a clean rollback
boundary.

## Public contract

- Accepted domain is intentionally narrow: contiguous BF16 Q/K/V/G/beta,
  float32 A_log/dt_bias, explicit contiguous BF16 initial state, fused QK norm,
  fused gate, beta logits, finite negative lower bound, SM100/SM103, and no
  checkpoints/indexed state/spec decode/CUDA graph capture.
- Output storage is checked for overlap using the existing CAKE validator.
- Generated kernels use distinct initial/final state pointers. The adapter uses
  per-stream final-state scratch and a same-stream copy-back to preserve public
  in-place state updates.
- Shape metadata and prepared routes are cached on the existing per-stream
  workspace; dynamic rotating-state pointers are rebound per call.
- Only five exact qualified shapes resolve to evolution. H96 mixed and every
  unqualified shape resolve to CAKE inside the explicit adapter.
- `backend="auto"` is unchanged.

## Evidence

- Six-shape public output/final-state correctness: PASS.
- Paired public CAKE/evolution CUPTI protocol: two 20+100 blocks per backend,
  cold L2, rotating identical initial states.
- Five activated shapes: 1.0319x to 1.2614x; every bootstrap 95% lower bound is
  above one.
- H96 mixed fallback: 1.0001x with interval crossing one.
- Guarded six-shape geomean: 1.1420x.
- Existing B300 evolution GPU tests: 6/6 PASS.
- Non-GPU evolution tests: 189 PASS.
- Typed-IR tests: 20/20 PASS.

The complete proof chain and implementation hashes are in
`evidence/b300_stage3_activation_certificate.json`.

## Review risks

1. The adapter currently attaches private caches to FlashInfer's existing
   stream workspace. Upstream may prefer named fields or a dedicated workspace
   type; this is an internal-structure cleanup, not a semantic change.
2. The 32-entry prepared-route cache is bounded but not LRU. Exact serving
   workloads are expected to reuse a small tensor set; a maintainer may prefer
   a formal cache helper before default routing.
3. CUDA graph capture is rejected. Supporting it requires caller-owned stable
   scratch and descriptor lifetime, and should be a separate change.
4. Packed sequence values are read once per offsets tensor version during eager
   warmup. This matches the existing eager CAKE planning model but should remain
   explicit in API documentation.
5. `auto` activation should be a follow-up policy patch after the mechanism is
   accepted, with the same five exact shape guards and CAKE fallback.
