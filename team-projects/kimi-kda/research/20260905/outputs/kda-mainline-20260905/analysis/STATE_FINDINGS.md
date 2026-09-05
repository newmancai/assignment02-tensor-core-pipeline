# Job 19903: state-contract and tail supplement

## Strict conclusion

This independent run confirms that the clean, guarded release improves all 27 tested shapes in all three measurement scopes. The useful refinement is that **initial-state presence divides the gain into two regimes**: roughly 17–20% with initial state (`both`/`in`), versus roughly 7–9% without it (`out`/`none`). The same pattern holds for one packed sequence. Tail cases 2049/4095/8191 remain positive. These are reductions relative to the old automatic dispatcher, not additional percentage points to add to the earlier slice-selection gain.

No timings are combined with Job 19901. Source: [state_matrix_19903.log](../state_matrix_19903.log#L2). Candidate SHA256 is `34e2c68cf80de0bd24278afb035fdcd44e4a6205241dfcb9aab4ee95891bc486` on the 148-SM B300. The scope is H12/B1, D128, BF16 state, and either fixed length or exactly one packed sequence; no multi-sequence claim follows.

## Completeness and numerical contract

The independent parser reports `PASS`, with no missing or duplicated cases, variants, rounds, or markers:

- 27 exact shapes: 3 lengths × 4 state contracts × fixed/packed = 24, plus 3 fixed-length `both` tail cases. This shape construction is explicit in [state_matrix_probe.py](../state_matrix_probe.py#L26).
- 81 emitted correctness rows, all `PASS`, all expected outputs bitwise-equal and finite. Importantly, 27 rows compare `release_v128` with itself; **54 rows are non-reference comparisons** (`baseline_auto` and `release_auto` against `release_v128`). The self-comparison is visible at [state_matrix_probe.py](../state_matrix_probe.py#L45). Thus “81 independent cross-implementation checks” would be inaccurate.
- 243 timing rows = 27 × 3 variants × 3 rounds. Each contains eager/graph/cache-perturbed statistics with 60/60/30 event samples respectively. Exactly 27 shape-completion markers and one final `state_matrix_complete` match the counts at [state_matrix_19903.log](../state_matrix_19903.log#L354).
- Correctness is checked after graph capture/replay and before timing. There is no separate post-timing correctness marker, sanitizer run, or chained-state test in this supplement. Those are separate Job 19901 evidence, not implied by this run's `PASS`.

All reported latencies are medians of the three round medians, not a pool of samples from different jobs. p10/p90 are descriptive within-round quantiles, not confidence intervals. The cache experiment clears 256 MiB **before the call**; K1 may subsequently warm K2 workspace. [Measurement loop](../state_matrix_probe.py#L48).

## Gains by state, length, and packing

Values below are latency reductions versus `baseline_auto`, in percent. Each triple is **eager / graph / pre-call cache perturbation**. Packed means `lengths=[T]`, not many sequences.

| T | State contract | Fixed: E / G / P | Packed single: E / G / P |
|---:|---|---:|---:|
| 2048 | both | 17.28 / 17.88 / 17.01 | 16.84 / 18.72 / 16.46 |
| 2048 | in | 17.42 / 17.86 / 17.02 | 16.95 / 17.63 / 16.46 |
| 2048 | out | 8.78 / 8.92 / 7.54 | 7.43 / 8.89 / 7.00 |
| 2048 | none | 8.83 / 7.82 / 8.69 | 8.41 / 7.74 / 7.80 |
| 4096 | both | 18.71 / 18.42 / 18.82 | 18.45 / 18.92 / 18.49 |
| 4096 | in | 18.15 / 18.35 / 18.22 | 17.97 / 18.93 / 18.22 |
| 4096 | out | 9.13 / 8.70 / 8.60 | 7.87 / 8.62 / 8.12 |
| 4096 | none | 8.59 / 8.68 / 8.27 | 7.86 / 8.62 / 7.84 |
| 8192 | both | 19.42 / 19.52 / 19.41 | 19.26 / 19.44 / 19.24 |
| 8192 | in | 19.30 / 19.58 / 19.10 | 19.57 / 19.66 / 19.35 |
| 8192 | out | 8.84 / 8.92 / 8.84 | 8.75 / 8.85 / 8.61 |
| 8192 | none | 8.83 / 8.73 / 8.81 | 8.54 / 8.56 / 8.53 |

Across core cases with initial state, the aggregate shape/channel gains range from **16.456% to 19.655%**. Without initial state the corresponding range is **6.997% to 9.129%**. These are shape/channel extrema, not uncertainty bounds. The worst individual paired-round gain in the entire matrix is still positive: **6.676%**, packed T2048 `out`, cache perturbation. None of the 243 shape/channel/repeat comparisons against old auto loses in this observed run; this is not a universal no-regression guarantee.

Representative raw latencies show why the reference must be named:

| Shape and timing | Old auto ms | Release auto ms | Release 128 ms | Reduction vs old | Reduction vs 128 |
|---|---:|---:|---:|---:|---:|
| Fixed T8192 both, eager | 0.569840 | 0.459168 | 0.784784 | 19.422% | 41.491% |
| Fixed T8192 both, graph | 0.566768 | 0.456128 | 0.779808 | 19.521% | 41.508% |
| Fixed T8192 both, perturbation | 0.570416 | 0.459696 | 0.784448 | 19.410% | 41.399% |
| Fixed T8192 out, eager | 0.602480 | 0.549216 | 0.788928 | 8.841% | 30.385% |
| Fixed T8192 out, graph | 0.597536 | 0.544256 | 0.785904 | 8.917% | 30.748% |
| Fixed T8192 out, perturbation | 0.603120 | 0.549824 | 0.789440 | 8.837% | 30.353% |

The ~41.5% and ~30.4% columns include the older slice-selection advantage; the ~19.4% and ~8.8% columns isolate the clean-release increment relative to old auto in this job.

## Tail cases

These are fixed H12/B1, BF16, `both`; they do not test tails for all four state modes or packed tails.

| T | Eager old → release ms | Eager gain | Graph gain | Perturbation gain |
|---:|---:|---:|---:|---:|
| 2049 | 0.155088 → 0.127456 | 17.817% | 17.635% | 18.044% |
| 4095 | 0.295360 → 0.242000 | 18.066% | 18.557% | 18.652% |
| 8191 | 0.571792 → 0.460688 | 19.431% | 19.361% | 19.408% |

The direct evidence is not restricted to exact multiples of the 16-token tile, while still staying inside the production candidate's guarded length range. Tail data begin at [state_matrix_19903.log](../state_matrix_19903.log#L315).

## Most useful insight, with causal limits

The smaller no-initial-state gain is not explained convincingly by a fixed initialization overhead alone. For the release's fixed `out` versus `both`, eager latency differences at T2048/4096/8192 are **20.576 / 43.152 / 90.048 µs**; graph differences are **22.560 / 43.008 / 88.128 µs**. The gap grows roughly with sequence length. Meanwhile, `both` tracks `in`, and `out` tracks `none`, so final-state output is not the dominant observed separator.

The reviewed kernel explicitly distinguishes `HasStateIn` in the pre-loop TMA-load versus shared-zero-initialization path: fwd_kernel2.cuh（历史临时路径，参见本目录实验补丁）, zero-initialization branch（历史临时路径，参见本目录实验补丁）. **Hypothesis, not a demonstrated cause:** the template specialization changes steady-state generated code, register allocation, or scheduling; alternatively, data-dependent effects or other specialization interactions could contribute. The next high-information read-only comparison would be `HasStateIn=true/false` SASS and per-specialization resource counts, followed by a targeted profiler control if authorized. Optimizing only the one-time zero fill is not yet supported as the right fix.

This supplement strengthens the case for the narrowly guarded prefetch change, including absent-state and nonmultiple tails. It does not erase the separately observed multi-request boundary, prove cold-K2 behavior, establish model/training throughput, or justify widening to arbitrary heads/batches/packed multi-sequence inputs.

## Reproduction of this audit

The summarizer is pure standard library, with generic helpers imported from the adjacent `summarize_release.py`. It does not load CUDA or write results. Missing/duplicated markers, shapes, variants, rounds, or expected output fields produce `UNVERIFIED`; explicit failed equality/finite checks produce `FAIL`.

```sh
python3 outputs/kda-mainline-20260905/analysis/summarize_state_matrix.py \
  outputs/kda-mainline-20260905/state_matrix_19903.log --format markdown
python3 -m unittest discover \
  -s outputs/kda-mainline-20260905/analysis \
  -p 'test_summarize_state_matrix.py' -v
```

Omit `--format markdown` for JSON containing every round, both baseline comparisons, per-shape worst paired regression, and separate initial-state/packing gain ranges. Ten in-memory fixtures passed. The CLI's process completion is not itself an experiment verdict; inspect the report's `status`.
