# FlashKDA Runtime Profile Evolution - Stage 5 Handoff

## Outcome

Stage 5 tested the Stage 4 H12 prepare-grid policy on six shapes fixed before
the first held-out GPU run and disjoint from the development preset. All four
shapes where the policy applies have bootstrap 95% lower bounds above one. The
two boundary controls at 64 and 128 chunks remain unchanged, as required.

The certificate is `evidence/b300_stage5_h12_heldout_certificate.json`.

## Held-out results

| Shape | Chunks | Baseline | cpc=9 | Speedup | Bootstrap 95% | Result |
|---|---:|---:|---:|---:|---:|---|
| fixed 1024 | 64 | 83.073 us | 83.120 us | 0.9994x | [0.9969, 1.0027] | unaffected control |
| fixed 2048 | 128 | 146.657 us | 146.833 us | 0.9988x | [0.9969, 1.0007] | boundary control |
| fixed 4096 | 256 | 274.097 us | 273.011 us | 1.0040x | [1.0026, 1.0058] | supports policy |
| fixed 16384 | 1024 | 997.205 us | 985.141 us | 1.0122x | [1.0113, 1.0138] | supports policy |
| packed skew | 512 | 507.747 us | 489.570 us | 1.0371x | [1.0347, 1.0397] | supports policy |
| packed irregular | 418 | 282.786 us | 276.176 us | 1.0239x | [1.0205, 1.0268] | supports policy |

- All output/final-state checks: PASS.
- Affected-shape support: 4/4.
- Affected-shape geomean: 1.019245x.
- Guarded held-out six-shape geomean: 1.012789x.
- B300 allocation: 5.133528 GPU-s, estimated 1.141105 kJ.

## Interpretation

The result supports the total-chunk threshold but does not yet prove that 9 is
the architecture-normalized optimum. The effect is not monotonic in sequence
length alone: packed skew benefits more than fixed workloads with the same
order of total chunks. The next model should therefore include sequence skew,
prepare CTA wave count, and tail utilization in addition to total chunks.

## Next falsifiable step

Freeze a larger stratified H12 matrix before measurement, fit the policy only
on a training split, and evaluate an untouched test split on both B200 and
B300. Use prepare-only and chain-only counter attribution to determine whether
the improvement comes from fewer CTA scheduling waves, less fixed CTA overhead,
or a different tail-efficiency regime. Then validate the selected policy in an
end-to-end Kimi-K3 SGLang run.
