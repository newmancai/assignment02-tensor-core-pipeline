# FlashKDA MARPE - Stage 8 Handoff

## Outcome

Stage 8 converts the Stage 7 scheduling-wave interpretation into a falsifiable,
hardware-normalized policy and tests it at two new total-work levels on the same
NVIDIA B300. The formula predicts every screen winner:

```text
cpc*(W,H,S,R) = ceil(W / floor((S * R) / H))
```

Here `W` is total 16-token chunks, `H` is local heads, `S` is SM count, and `R`
is resident prepare CTAs per SM under the compiled resource footprint. Stage 7
NCU establishes `H=12`, `S=148`, and `R=5` for this kernel. The resulting
predictions and measured winners are:

| Total chunks | Predicted cpc | Screen winner | Profiles |
|---:|---:|---:|---:|
| 256 | 5 | 5 | 2/2 |
| 512 | 9 | 9 | 4/4 (Stage 6) |
| 1024 | 17 | 17 | 2/2 |

This is the first result in the project where a physical model predicts an
unmeasured optimum region before the GPU screen rather than explaining a winner
afterward.

## Preregistered B300 experiment

Stage 8 froze four profiles before job `23211`:

- W256 fixed 4096;
- W256 balanced four-way `(976,1008,1040,1072)`;
- W1024 fixed 16384;
- W1024 balanced four-way `(4000,4064,4128,4192)`.

Each profile searched the same cpc set
`{3,4,5,6,8,9,12,16,17,18,20,24}` through the public CAKE route with cold-L2
CUPTI timing, output/final-state correctness, 256 state rotations, 20 dry runs,
and 100 measured iterations. All 48 candidates passed correctness and retained
the same physical prepare/chain route.

## Same-object confirmation

Job `23217` compared the predicted cpc with the current cpc-9 rule using
baseline/candidate/candidate/baseline order and two 100-sample blocks per arm.

| Profile | Candidate | Baseline (us) | Candidate (us) | Speedup | Bootstrap 95% |
|---|---:|---:|---:|---:|---:|
| W256 fixed | 5 | 156.578 | 152.018 | 1.0300x | [1.0286,1.0314] |
| W256 balanced-4 | 5 | 68.769 | 63.329 | 1.0859x | [1.0833,1.0885] |
| W1024 fixed | 17 | 567.320 | 555.719 | 1.0209x | [1.0195,1.0214] |
| W1024 balanced-4 | 17 | 209.475 | 198.723 | 1.0541x | [1.0522,1.0560] |

All four lower bounds exceed one; geometric-mean speedup is `1.04742x`.

## Phase attribution

The second command in job `23217` independently captured public-call CUPTI
activities. Prepare saves `5.09-5.20 us` at W256 and `11.02-11.23 us` at W1024.
It explains at least `96.67%` of each full-span improvement. Chain differences
remain below `0.61 us`. This reproduces the Stage 7 separation:

```text
total parallel chunks + resident grid capacity -> prepare cpc
maximum sequence chunks                    -> chain critical path
```

The absolute prepare saving changes with total work, so Stage 8 refines the old
"constant 18 us" observation. What transfers is the decomposition and wave
boundary, not the coefficient.

## Code and certificate

- `kda_ir/runtime_profile.py` now exposes
  `recommend_one_resident_wave_cpc()` and a typed
  `ResidentWaveRecommendation` receipt.
- `certify_stage8.py` validates prediction order, correctness, physical scope,
  paired intervals, phase attribution, Stage 6 middle-level evidence, and all
  accounting.
- `evidence/b300_stage8_certificate.json`: all 11 certification checks pass.
- Local typed-IR suite: **32/32 PASS** after adding resident-wave and fair
  multi-agent-ablation tests.

## Accounting

Stage 8 retains three successful attempts:

- screen: `10.978168` GPU-s, `2.225370 kJ`;
- paired confirmation: `5.634634` GPU-s, `1.160490 kJ`;
- phase attribution: `9.920421` GPU-s, `2.539090 kJ`.

Stage 8 total: `26.533223` GPU-s and `5.924950 kJ`. Stage 4-8 total: 19
attempts, `107.757` GPU-s, and `23.930 kJ` (rounded). These are allocation-level,
not kernel-only, energy numbers.

## Multi-agent next stage is now executable

`kda_ir/ablation.py` implements a deterministic fairness auditor for four arms:

1. single generalist;
2. one Agent sequentially role-playing the same roles;
3. homogeneous independent Agent pool;
4. role-specialized Agents with conclusions-only memory.

The auditor rejects a claim when model/profile/prompt/candidate space differ,
candidate or GPU budgets are not matched, held-out is unsealed, or the
multi-role arm does not beat both roleplay and homogeneous controls. It also
requires a cross-lineage compound candidate before calling the result
collaboration rather than best-of-N.

`evidence/stage9_multiagent_ablation_preregistration.json` freezes three
trajectories per arm, 150k proposal tokens, 24 proposal/screen slots, four
qualifications, and approximately 60 B300 GPU-seconds per arm. It is explicitly
marked `preregistered_not_executed` and supports no emergence claim yet.

## Claim boundary

Established on one B300 for this H12 kernel/resource footprint:

- the one-resident-wave formula predicts the best cpc at W=256/512/1024;
- paired public-path gains and phase attribution support every new prediction;
- the physical variables are total chunks, local heads, SM count, resident CTAs
  per SM, per-CTA resources, and maximum dependency length.

Not established:

- deployment safety of the new piecewise rule;
- other head counts, kernel resource footprints, B300 replicas, or B200;
- end-to-end Kimi-K3 throughput;
- multi-agent superiority or emergence.

The production dispatcher therefore remains unchanged until an untouched
qualification is designed and executed.
