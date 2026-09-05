# Matched-input zero-state experiment: protocol and falsification

Status: protocol only; no GPU execution or production change is implied. The experiment asks whether initial-free first prefill can recover the faster state-present specialization **without changing D128/C16, state rounding, or public semantics**. “First prefill” here means no mathematical initial state, not a cold process, cold allocator, or cold GPU.

## 1. Question and four minimum arms

Build each shape once. All arms share the exact same q/k/v/g/beta/A_log/dt_bias tensors and scalars, with identical device, dtype, strides, packing, and scale. Do not call `make_case` separately for each arm: it generates different random inputs. Existing helper behavior is visible at [release_probe.py](../kda-mainline-20260905/release_probe.py#L61).

Each arm has its own preallocated `out` and BF16 `final_state`. No output/state aliases any input or another arm's outputs. Every invocation starts from the same initial condition; do not feed its final state back into the next invocation.

| Arm | `initial_state` passed to the real wrapper | Timed per-call work | Mathematical reference |
|---|---|---|---|
| `none` | `None` | Unchanged wrapper call | Absent/zero initial state |
| `zero_reuse` | Immutable, preallocated BF16 tensor containing positive zero | Wrapper call; zero creation excluded and explicitly recorded | Same as `none` |
| `zero_alloc_each_call` | `torch.zeros(state_shape, device=…, dtype=torch.bfloat16)` constructed **inside every invocation** | Allocation request + zero fill + wrapper call | Same as `none` |
| `nonzero_reuse` | Immutable, preallocated ordinary finite BF16 state | Wrapper call; creation excluded | Its own nonzero initial state |

The public BF16 state shape is `(N,H,128,128)`, where N is the batch count for fixed data or the number of packed sequences. The first target N1/H12 state occupies 393,216 bytes (384 KiB). Even the zero-reuse arm introduces a real state input/TMA read that the `None` arm does not have.

`zero_reuse` is a mechanism control and possible upper-bound opportunity, **not a demonstrated production optimization**. The experiment does not authorize a global cached zero tensor, a change in stream/lifetime ownership, hidden state allocation, or new public semantics. A warmed `torch.zeros` allocation request also does not imply a fresh `cudaMalloc` every call.

### What each contrast can and cannot identify

- `none` versus `zero_reuse`: equal mathematical initial state, different state-present specialization and initialization mechanism. A persistent difference rules out nonzero mathematical state as the sole explanation. It does **not** isolate compiler scheduling alone: the initial TMA read versus generic shared-zero path, address/cache behavior, and resource allocation also change.
- `zero_reuse` versus `nonzero_reuse`: same state-present branch, different state values. A difference suggests value-dependent effects or another uncontrolled factor; verify the actual selected kernel is identical. Distinct state-buffer addresses are a residual confound, not a mathematical proof.
- `zero_reuse` versus `zero_alloc_each_call`: practical cost of constructing the substitute state in this execution scope, including fill/allocator/cache/lifetime effects. Do not label the entire difference “allocation overhead” or subtract a separately measured zero-fill cost as if timings were necessarily additive.
- `none` versus `zero_alloc_each_call`: the relevant first test of whether explicit-zero substitution still helps after its required per-call work. Even a win is a candidate implementation result, not evidence that caching a zero tensor is safe or necessary.

For a small fifth control, use `zero_refill_each_call`: keep storage preallocated, execute `zero_buffer.zero_()` inside the timed invocation, then call the wrapper. It separates “fill plus its cache effects” from the additional allocator/Python/lifetime path more cleanly than the four arms alone. A standalone `zeros` microbenchmark is useful context, not a substitute for the composed call.

## 2. Minimal bounded matrix

Primary performance matrix: B1/H12, D128/C16, BF16 q/k/v/g/beta and final state, fixed T2048/4096/8192, always request both `out` and `final_state`; four arms × three scopes × three randomized-order rounds. Use 60 event samples per eager/graph round and 30 per perturbed round, preserving the earlier release measurement convention. The three scopes must be summarized separately.

Add only high-information checks before widening the experiment:

- One packed single sequence at T8192, to verify that the result survives the varlen specialization.
- Correctness at T1, T17, T2049, T4095, T8191, T8193; the last is a guard-boundary negative control. Performance on all these tails is optional initially.
- A second random seed at T8192 and one zero-valued q/k/v input sanity case. They help expose accidental state/output reuse; they are not broad numerical coverage.
- If a kernel candidate is proposed, verify state-output absent as a separate API contract before generalizing beyond this output-present experiment.

Use the same binary for all four arms in the mechanism experiment. Record the loaded extension path/SHA256, source/patch identity, compiler flags, device SM/L2 resources, PyTorch/CUDA/driver versions, seed, and stream. Existing `explain_k2_dispatch` reports ValueSlice, not Prefetch4 or `HasStateIn` identity.

The primary result should retain actual automatic dispatch. Require its recorded ValueSlice to match across arms before describing the result as a matched kernel-specialization comparison. If state metadata changes the selection, report that fact and add a separately labeled forced-V16 control; do not silently replace auto with forced dispatch. Identify the actual K2 specialization in a representative profile/SASS record before claiming a compiler mechanism.

## 3. Correctness gates

Before timing, establish two references using the established baseline wrapper with explicit V128 dispatch and the **same input objects**:

1. `reference_zero`: no initial state (optionally cross-check against an explicitly zero BF16 state under that same V128 implementation).
2. `reference_nonzero`: the exact BF16 nonzero state used by `nonzero_reuse`.

Then require both `out` and BF16 `final_state` to be finite and bitwise equal:

- `none`, `zero_reuse`, `zero_alloc_each_call`, and optional `zero_refill_each_call` against `reference_zero`, and against each other.
- `nonzero_reuse` against `reference_nonzero` only. **Do not require it to equal `reference_zero`: it is a different mathematical input.** Conversely, nonzero versus zero output equality is not itself a test failure; some inputs can erase initial-state influence.

Run these checks after ordinary eager execution, after graph capture/replay, and after the timing loops. Assert before emitting a `PASS` row. For the immutable reuse controls, verify state contents remain unchanged before/after the experiment. Maintain disjoint initial and final state buffers and explicitly confirm the initial-state reference remains alive until GPU consumption completes.

A `torch.zeros` tensor must reach the wrapper in that invocation. Beware closures around `buffers`: [prepare](../kda-mainline-20260905/release_probe.py#L88) constructs its own dictionary. Allocating a local zero tensor, or modifying a different dictionary than the lambda closes over, can accidentally leave the original initial state in use. Log the semantic arm and passed-state presence/dtype/shape outside the timed hot path; use a small instrumented call to verify object flow.

Any mismatch blocks performance acceptance. Preserve the failing case/seed, output-specific result, selected kernel, and arm; do not loosen tolerance to rescue a bitwise claim. This is parity with the existing implementation, not a new high-precision oracle for the entire algorithm.

## 4. Timing scopes and allocation accounting

### Eager CUDA events

Record the start event, execute the whole arm invocation, and record the stop event on the same stream. For `zero_alloc_each_call`, the `torch.zeros` call belongs after the start-event enqueue and before the wrapper call. Avoid constructing the state once in `prepare` and accidentally timing a reuse arm under the allocation label.

This is a GPU-timeline interval. Host-induced gaps may appear in it, but it does not generally account for all Python or allocator CPU time. Therefore **“event time includes a zero-allocation request and fill” is not equivalent to “full CPU allocation cost measured.”**

For any claim about practical per-call allocation cost, add a separately labeled host-wall-clock end-to-end measurement (e.g. `perf_counter_ns`, synchronize before each start and after the invocation). Apply the same synchronization protocol to every arm and report its scope; the added synchronization changes the workload. A many-call batch wall-time with one final synchronization is another useful throughput-oriented scope, but must not be relabeled single-request latency.

Warm all arms consistently. Record allocator warmup and do not empty the allocator cache between only one arm's samples. Cold-process/first-ever allocation cost, if desired, needs its own isolated protocol and must not be averaged into steady-state samples.

### CUDA Graph replay

Capture the **whole arm invocation** separately for each arm. If `torch.zeros` is inside the invocation, verify that the captured graph actually retains the required fill operation and the wrapper consumes that state; graph construction and its memory-pool setup are not part of replay timing. Do not build a zero tensor before capture and claim graph replay measured per-call zero construction.

Graph replay normally reuses captured addresses/resources and omits per-replay Python allocation/dispatch. Thus label this arm “captured zero fill + forward,” not “same eager dynamic allocator cost.” Keep graph-owned state and buffers alive, avoid unintended graph-pool sharing between comparisons, and verify outputs after repeated replay. Record capture failures as `FAIL`/`UNVERIFIED`, not a silent eager fallback.

### Pre-call cache perturbation

Zero a separate 256 MiB eviction tensor before the start event, on the same stream. Then time the entire eager arm, including state creation/fill where specified. Eviction cost is excluded for every arm. Use the label `cache_perturbed`, not guaranteed “cold.”

The dynamic/refill zero operation itself changes state-cache residency; that is part of its real composed cost. K1 can still warm K2 workspace after the eviction. This protocol does not isolate cold K2, prove all caches were evicted, or equalize every arm's state-buffer warmth.

## 5. Minimal log and analysis contract

Use a new job/schema; do not mix this experiment's medians with Job 19901/19903. Suggested records:

- `zero_environment`: binary/source/device identity, shape list, names, seed, expected repeats/counts, stream and timing-scope definitions.
- `zero_correctness`: shape/case/arm/reference/stage (`eager`, `graph`, `post_timing`), output-specific `bitwise` and `finite`, status. Reference self-checks must be counted separately.
- `zero_performance`: shape/case/arm/repeat, selected dispatch, `eager`, `graph`, `cache_perturbed` and optional separately named `wall_sync`; each contains count/median/p10/p90 with units. Record graph allocation-scope distinction in metadata.
- `zero_shape_complete`: exact case/shape and emitted row counts.
- `zero_complete`: exact shape/correctness/performance counts. A partial run, capture failure, or missing terminal marker cannot be reported `PASS`.

For every shape and scope, retain all three per-round medians and the paired-round ratios; report median-of-round-medians and the worst observed paired regression. The contrast baseline is named explicitly (`none` or `zero_reuse`). Do not treat p10/p90, the three rounds, or a shape-to-shape range as a confidence interval. Do not divide joined multi-request time by two or extrapolate single-request savings to serving throughput.

## 6. Acceptance and high-information falsifiers

### Mechanism experiment acceptance

The experiment is informative even if all substitution arms lose. Minimum acceptance means complete logging, exact matched inputs, correct dispatch identity, bitwise/finite gates satisfied, and all four arms measured under the same declared scope. It does not require a performance win.

The strongest outcomes and their counterclaims are:

| Observation | Supported interpretation | Counterclaim not established |
|---|---|---|
| `zero_reuse ≈ nonzero_reuse < none` | Zero mathematical values are not required to incur the slower path; state-presence/initialization specialization is a strong lead | Compiler scheduling alone is proven causal |
| `zero_reuse ≈ none < nonzero_reuse`, or zero/nonzero differ materially | Values or other controls matter; revisit address/cache/input matching and actual kernel identity | The previous state-contract gap was solely `HasStateIn` code generation |
| `zero_reuse` wins but `zero_alloc_each_call` does not | A mechanism opportunity exists but this substitution does not recover practical per-call cost | A reusable global zero tensor is now a production optimization |
| Graph wins but eager event/host-wall does not | Replay-specific scheduling or Python/allocation scope matters | A general first-prefill latency gain |
| Hot wins but perturbed loses | Benefit depends on the tested residency/initialization path | Cache-independent or cold-K2 speedup |
| NoState candidate wins only by changing slice or arithmetic/rounding | Experiment is confounded or violates the intended contract | A localized no-state code-generation improvement |

Two inexpensive falsification controls are especially valuable:

1. **Address crossover:** reuse one preallocated state storage for an auxiliary `zero` versus `nonzero` trial, writing its contents outside the timed interval and synchronizing before measurement. This reduces state-address/layout confounding; revalidate expected outputs for each content. Do not mutate storage belonging to another live graph or concurrent request.
2. **Length scaling:** inspect raw `none − zero_reuse` differences at T2048/4096/8192, separately for each scope. A roughly length-proportional gap motivates inspecting steady-state SASS/resources; a constant gap points more toward initialization/fixed work. Neither pattern alone proves causality. Do not subtract large noisy timings and report a false-precision crossover.

### Candidate implementation acceptance

A production-oriented kernel change must keep the **real no-initial-state call** as its input contract and compare against the unchanged release `none` arm, including required allocation/initialization work. If the candidate instead constructs an explicit zero state, count that work and qualify graph/eager behavior separately. Do not substitute the excluded-cost `zero_reuse` control as the production baseline or result.

Before considering promotion: unchanged bitwise/finite results on the bounded matrix and tails; complete three-round eager/graph/perturbed results; no concealed regression in control shapes; source/build identity; representative actual-kernel and register/spill evidence; and targeted synchronization/memory checks for any changed shared/TMA initialization. Preserve D128/C16 and the original rounding points. Small gains comparable to paired control variation remain an unresolved measurement result, not an automatic promotion. Concurrency, other H/N, FP32 state, packed multi-sequence, aliasing, and multi-GPU behavior remain unverified until explicitly tested.
