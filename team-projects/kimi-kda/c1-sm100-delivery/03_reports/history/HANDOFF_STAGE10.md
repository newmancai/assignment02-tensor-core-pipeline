# FlashKDA MARPE - Stage 10 Handoff

## Outcome

Stage 10 turns the Stage 8 capacity formula into a sealed prospective test. It
passes on all four untouched interpolation profiles, across eight fresh Python
processes on one B300 GPU UUID, without screening candidate cpc values first.

The precise rule is:

```text
capacity_ctas = SM_count * resident_prepare_ctas_per_SM
groups_per_capacity_fill = floor(capacity_ctas / local_heads)
cpc* = ceil(total_chunks / groups_per_capacity_fill)
```

For this exact H12 prepare image, NCU reports five resident CTAs/SM, so the
capacity is `148 * 5 = 740` CTAs and there are 61 per-head chunk groups. This is
a **resident-grid-capacity fill**, not the ordinary one-CTA-per-SM meaning of a
CUDA wave.

## Sealed prospective qualification

Before job `23248` was submitted, the following were frozen and SHA-256 sealed:

- profiles: W384 fixed and packed balanced-4; W768 fixed and packed balanced-4;
- predictions: `W384 -> cpc7`, `W768 -> cpc13`;
- baseline: current `cpc9`;
- exact `sm_103a` logical and physical route identifiers;
- eight process-isolated epochs, serialized as a Slurm array;
- alternating ABBA/BAAB timing order;
- public output/final-state correctness at `atol=rtol=1e-2`;
- CUPTI cold-L2 timing with 256 rotating state slots;
- Bonferroni one-sided 98.75% lower bounds over process epochs;
- conjunctive mechanism gate `lower > 1.0` and shadow-deployment gate
  `lower >= 1.005` for every profile;
- failure, route-drift, and no-post-result-neighbor-search rules.

The test is direction-reversing. At W384, cpc7 increases the prepare grid from
516 to 660 CTAs. At W768, cpc13 reduces it from 1032 to 720 CTAs. In both cases
the next smaller cpc gives 768 CTAs, just above the 740-CTA capacity. Joint
success therefore rejects the simpler hypothesis that fewer CTAs are always
better.

| Profile | Prediction | Point speedup | Adjusted one-sided 98.75% lower | Epoch range |
|---|---:|---:|---:|---:|
| W384 fixed | cpc7 | 1.013345x | 1.012949x | 1.011463--1.014041x |
| W384 balanced-4 | cpc7 | 1.033139x | 1.032156x | 1.031209--1.035661x |
| W768 fixed | cpc13 | 1.049362x | 1.048779x | 1.048592--1.049678x |
| W768 balanced-4 | cpc13 | 1.135382x | 1.135244x | 1.134914--1.136020x |

All 32 profile-epoch output and final-state comparisons have zero maximum
absolute difference. All route, scope, accounting, mechanism, and 0.5% gates
pass. `evidence/b300_stage10_certificate.json` has status
`prospective_capacity_rule_confirmed_and_shadow_activation_qualified`.

The first certifier pass is intentionally retained. It failed only because the
code compared the driver product name with `NVIDIA B300`; `nvidia-smi` reports
the exact, already-known Stage 8 string `NVIDIA B300 SXM6 AC`. The non-statistical
amendment, first-pass certificate, and prior accounting hash are archived. No
prediction, sample, statistic, threshold, or claim boundary changed.

## Post-qualification mechanism attribution

Job `23260` ran only after the prospective decision was fixed, so it cannot
change qualification. CUPTI activity separates prepare and chain:

- full-span saving: `3.9685--31.7130 us`;
- prepare saving: `3.8885--31.5675 us`;
- prepare fraction: `97.984--101.155%`;
- largest absolute chain 95% bound: `0.303 us`;
- every chain interval lies inside the pre-frozen +/-1% equivalence band.

`evidence/b300_stage10_mechanism_certificate.json` passes all nine checks. The
combined interpretation is now stronger than a post-hoc correlation: total
chunks, local heads, SM count, and compiled-kernel resident capacity predict
the prepare work assignment; maximum per-sequence chunks controls the unchanged
recurrent chain. The coefficients and evidence domain remain hardware and
kernel specific.

## Proof-carrying shadow resolver

`kda_ir/runtime_policy.py` adds:

- `PrepareKernelReceipt`, with exact target, route/image identity, resource
  footprint, five independent occupancy limits, and toolchain hash;
- `RuntimeCallContract`, for the Kimi-K3 TP8 H12 ABI and state semantics;
- `ResidentGridPolicyV1`, with `SHADOW_ONLY`, `QUALIFIED`, and `ACTIVE` states;
- `PolicyResolution`, which records recommendation, actual selection, all guard
  results, hashes, policy IDs, and fallback reason;
- ten `RPEPxxx` guard families.

The current status is deliberately `SHADOW_ONLY`. It may recommend cpc7/13 but
still selects the existing cpc9 fallback. Only an exact, hash-bound `ACTIVE`
bundle can mutate a schedule. A naked `resident_ctas_per_sm=5` is not accepted
as production authority. The local suite is **37/37 PASS**.

## Accounting

- prospective eight-epoch qualification: `38.009433` GPU-s, `9.735025 kJ`;
- post-qualification phase attribution: `7.759441` GPU-s, `1.637695 kJ`;
- Stage 10 total: `45.768874` GPU-s, `11.372720 kJ`, nine attempts;
- Stage 4--10 total: 28 attempts, `153.526` GPU-s, `35.303 kJ` rounded.

These are allocation-level values including setup and idle gaps, not
kernel-only energy.

## Paper release

- PDF: `output/pdf/runtime-profile-evolution-stage10.pdf`;
- CAKE-template source archive:
  `paper/runtime_profile_evolution/releases/runtime-profile-evolution-stage10-cake-template-source.tar.gz`;
- 16 letter-size pages, 30 references, fresh-archive compile verified;
- visual QA inspected all pages; no clipping or nearly blank trailing page;
- only benign underfull bibliography warnings remain.

## Claim boundary and next experiment

Established: within the exact B300/H12/BT16 resource receipt, the capacity rule
prospectively improves the current cpc9 policy at W384 and W768, with process-
level family-wise confidence and prepare-phase attribution.

Not established: unique global cpc optimum, unmeasured W, H other than 12,
changed kernel/toolchain resource footprints, another B300, B200, end-to-end
Kimi-K3 serving, or multi-agent superiority/emergence.

The best next step is not another local cpc sweep. It is either (a) acquire a
second device/receipt and test transfer, or (b) execute the already frozen Stage
9 equal-budget four-arm multi-agent ablation. Until then the capacity policy
should remain shadow-only and the production dispatcher should remain unchanged.
