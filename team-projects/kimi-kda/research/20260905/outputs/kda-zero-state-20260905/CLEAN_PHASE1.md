# Clean Phase1 prefetch candidate (prospective 0005)

Status: the isolated clean source was freshly built through its original `setup.py`; **Job19934 completed the recorded single-GPU correctness, graph, hardening, targeted sanitizer, and timing checks**. This remains a guarded, compile-time opt-in candidate, not a claim of complete production qualification. Multi-GPU and separate alias-extension GPU checks are SKIP. Job19935 profile collection and the separate performance/SASS reviews are complete; see `CLEAN_FINDINGS.md` and `CLEAN_SASS.md`. Earlier experiment-binary results are not substituted for the clean-binary evidence. No original source tree or dirty repository was modified.

Base: four-patch source `/private/tmp/kda-release-prefetch4.DStLoA`.
Candidate source: `/private/tmp/kda-clean-phase1.KHJ8t3`.
Patch: `clean-phase1.patch`, applied after 0001 + 0002 + hardening 0003 + guarded Phase6 Prefetch4 0004.

## Exact retained change

The K2 template gains `Phase1Prefetch=1`. Value 1 executes the original Phase1 body; values 2/4 use the k/q/state triple ring from the measured experiment. Initial-state handling remains the original compile-time HasStateIn branch and scalar zero initialization. There is no `InitStrategy`, runtime `load_initial_state` argument, experimental selector ID, vector-zero branch, or canonicalized HasStateIn in this candidate.

When enabled, Phase1 prefetch selection is nested **inside** the existing Phase6 Prefetch4 selected branch:

- D128 / C16, ValueSlice already selected as 16, public state mode not FP32.
- N=1, H=12, and inclusive `2048 <= T_total <= 8192`.
- HasStateIn=false selects Phase1Prefetch=4; HasStateIn=true selects 2.
- HasStateOut does not add another tuning dimension. A legal packed single sequence uses the same HasStateIn rule; no IsVarlen-specific tuning is added.

Other shapes/slices retain Phase1Prefetch=1 and their prior Phase6 selection. Python policy is unchanged. The build target restriction is explicit SM103a, not a new runtime B300 product check; raw forced V16 still bypasses the pre-existing Python device/resource policy. Nothing broadens D/C, dtype, device, or input legality guarantees.

The ring primes L triples, consumes k in ascending order, performs the k-GEMM followed by the q-GEMM, then refills the consumed slot with k+L only when in range. There are still eight k blocks and sixteen GEMM calls per tile. Both consumers finish before overwrite. Phase2 onward—including state FMA expressions, BF16 conversions, shared-memory writes, pipeline releases, barriers, and Phase6 StatePrefetch=4—is unchanged. See `PHASE1_DRAFT.md` for the consume/overwrite invariant; the clean extraction only replaces the experimental strategy-derived depth with a dedicated template parameter.

## Template / call-chain review

| Boundary | Template order / change | Review result |
|---|---|---|
| Public `launch_fwd` / `csrc/fwd.h` | `<D, HasStateIn, HasStateOut, StateFP32, IsVarlen>` | Unchanged declaration and call interface |
| `launch_fwd_impl` | `<D, K2Value, HasStateIn, HasStateOut, StateFP32, IsVarlen, StatePrefetch=1, Phase1Prefetch=1>` | New parameter is appended, not inserted among booleans |
| K2 template after its 11 TMA descriptor types | `<CHUNK,D,V,InputStages,OutputStages,NumThreads,HasStateIn,HasStateOut,StateFP32,IsVarlen,StatePrefetch,Phase1Prefetch>` | Both prefetch depths forwarded in order; all state booleans preserved |
| K2 runtime argument list | Existing descriptors, output pointer, sizes, cu_seqlens, total_tiles | Byte-identical signature to four-patch base; no added runtime bool |
| Guarded fast launch | Same runtime guard; StatePrefetch=4; Phase1Prefetch=HasStateIn?2:4 if enabled, otherwise 1 | Both depths are compile-time constants |
| All ordinary slice launches | Omit both trailing template arguments | Defaults remain StatePrefetch=1 / Phase1Prefetch=1 |

The kernel statically limits Phase1Prefetch to 1/2/4; the ring branch additionally requires V16, one value block per compute warp, and depth no greater than K_BLOCKS. The clean binding only accepts its original 16/32/64/128 selector values.

## Build configuration and rollback

Enable the new optimization only at build time:

```sh
FLASH_KDA_CUDA_ARCHS=103a FLASH_KDA_ENABLE_V16_PREFETCH4=1 FLASH_KDA_ENABLE_V16_PHASE1_PREFETCH=1 FLASH_KDA_EXTENSION_NAME=flash_kda_phase1_C python setup.py build_ext --build-lib /absolute/new-build/lib --build-temp /absolute/new-build/temp --force
```

Use the established CUDA/Python development-header environment and fixed CUTLASS dependency revision. The isolated source copy does not bundle CUTLASS; provide it at the expected `cutlass` path without changing dependency versions.

The actual recipe is `build_clean.sh`, with compiler output and source hashes in `build_clean.log`. It builds `/home/lcpu/YOUR_USER_ID/kda-zero-state-20260905/clean_source` into separate `clean_build/{lib,temp}` directories using the established Python 3.12 venv, CUDA 13.0, `MAX_JOBS=2`, `NVCC_THREADS=8`, and the existing user-local Python development headers supplied through CPATH. Its original `setup.py build_ext ... --force` command does not install over the existing package. The nvcc command records both prefetch feature macros and `-gencode arch=compute_103a,code=sm_103a`; the source hashes match the Identity table. Keep `build_clean.sh`, `build_clean.log`, `run_clean.sbatch`, `clean_probe.py`, the shared `release_probe.py` helper, entry-hardening test, and Job19934 logs together for reproduction.

Job19934 loaded `/home/lcpu/YOUR_USER_ID/kda-zero-state-20260905/clean_build/lib/flash_kda_phase1_C.cpython-312-x86_64-linux-gnu.so`, SHA-256 `f6f80fa402cc1dc00b09a8082b10806bbe17c0e533d067e931dd774d270b9270`. Its external four-patch reference was `/home/lcpu/YOUR_USER_ID/kda-mainline-20260905/release_build/lib/flash_kda_release_C.cpython-312-x86_64-linux-gnu.so`, SHA-256 `34e2c68cf80de0bd24278afb035fdcd44e4a6205241dfcb9aab4ee95891bc486`. The unchanged wrapper hash was `c638962a3d333680e923884ba47ffcd1cc4f26b1db4b2097af9bfa01b0b4f50f`. The run used NVIDIA B300 SXM6 AC, driver 580.126.09, PyTorch 2.10.0+cu130, SM10.3 / 148 SMs / 132644864 bytes L2. These identify one tested hardware/software configuration, not general SM103-product coverage.

`get_release_flags()` raises a clear RuntimeError if Phase1 opt-in is enabled without Phase6 Prefetch4 enabled, or if the enabled prefetch build does not explicitly target `FLASH_KDA_CUDA_ARCHS=103a`. Unset/auto/all/other/mixed architecture settings do not satisfy that requirement. The Phase1 and Phase6 feature macros are consumed only by the CUDA launch translation unit, so they are passed to nvcc, not C++; the existing `K2_VALUE_SLICE` binding/alias define remains passed to both where applicable. Both production and optional alias go through the same checked extension factory.

Defaults remain opt-in off. A build with only Phase6 Prefetch4 enabled retains the previous Phase1 body. Unsetting the Phase1 environment variable at runtime does not change an already compiled binary: rebuild without that flag to restore the old Phase6-only V16 path. The existing runtime `FLASH_KDA_K2_VALUE_SLICE=128` is a compatibility rollback to V128, not an A/B selector between the two V16 prefetch implementations. `explain_k2_dispatch()` still explains ValueSlice only; binary build flags/hash and selected-kernel evidence identify Phase1 variants.

## CPU checks performed

```sh
python3 test_clean_phase1_contract.py /private/tmp/kda-release-prefetch4.DStLoA /private/tmp/kda-clean-phase1.KHJ8t3
```

- Forward and reverse `git apply --check`: PASS.
- Source contract: PASS. Removing only the new template parameter/assert and Phase1 branch/wrapper restores the entire old K2 file byte-for-byte. K2 runtime arguments, binding, public fwd declaration, and Python package compare unchanged. Experiment symbols/IDs are absent, and the full kernel template forwarding order is checked.
- Real setup-function contract: **20 checks PASS**, including default/off/Phase6-only/both feature flags, production/alias macro placement, missing Phase6 dependency, and invalid architecture combinations. This AST-isolated test imports neither PyTorch nor setup side effects.
- Real C++ launch selector: **234 checks PASS** (39 cases under 6 preprocessor configurations), compiled on CPU against a recording `launch_fwd_impl` stub. Covers 2047/2048/8192/8193, three central tail boundaries, 16384, both HasStateIn values, all state-presence combinations, FP32 state, N2, H11/H13, V32/64/128, and packed single/multiple sequences. It also verifies all state booleans reach the stub unchanged.
- One CPU preprocessor-only case manually defines Phase1 without Phase6 and verifies the enclosing Phase6 guard keeps the old route; this is **not** a supported setup configuration—the build factory separately rejects it.

These checks do not compile CUDA, instantiate CuTE register fragments, verify SASS/register usage, establish bitwise GPU results, or measure speed. Recompilation can alter generated code even for the same old branch; retain the external four-patch binary and, if needed, a separately built Phase6-only binary from this source when interpreting GPU results. Unlike the experiment binary, this clean candidate has no same-binary runtime selector to disable Phase1 within its enabled envelope.

## Recorded clean-binary acceptance — Job19934

Counts below are kept separate by purpose; in particular, a reference self-comparison is not counted as an independent candidate correctness comparison. The present tensors in the listed checks are finite and bitwise equal to their designated comparison path. This is regression against the existing implementation, not a new high-precision numerical oracle.

| Check | Evidence | Recorded result / scope |
|---|---|---|
| Fresh setup build and loaded identity | `build_clean.sh`, `build_clean.log`, `clean_19934.log` | PASS; matching clean source hashes, expected flags, and loaded `.so` identity |
| Core cross-path regression | `clean_19934.log`, `release_probe.py` | 120 comparisons PASS: 40 configurations × auto/force16/off, against four-patch V128 |
| Extra tail/state/packed regression | `clean_19934.log`, `clean_probe.py` | 14 comparisons PASS: 7 additional configurations × auto/force16, against four-patch V128 |
| State chains | `clean_19934.log` | 6 steps PASS across two 3-step chains; includes first prefill without initial state followed by continued state-carrying calls |
| Checks after graph capture/replay | `clean_19934.log` | 120 logged PASS rows: **80 cross-path comparisons plus 40 V128 self-comparisons**, across 40 shapes |
| Checks after the timed workload | `clean_19934.log` | A separate 80 `post_correctness` comparisons PASS; not merged with the graph rows |
| Two-stream correctness | `clean_19934.log` | 4 comparisons PASS: two requests each for `both` and `out` state modes |
| Entry hardening | `clean_19934.log` | 5 cases PASS: binding, alignment, parity, non-default stream/graph, and CPU-device rejection. Multi-GPU SKIP: only one CUDA device visible; separate alias-default GPU check SKIP: no alias extension supplied |
| Targeted memcheck | `clean_19934_memcheck.log` | 20 comparisons PASS; process exit 0 and `ERROR SUMMARY: 0 errors` |
| Targeted synccheck | `clean_19934_synccheck.log` | 20 comparisons PASS; process exit 0 and `ERROR SUMMARY: 0 errors` |
| Matched profiling | `clean_profile_19935.log`, CSV/logs, `CLEAN_SASS.md` | Four profile processes exited 0 (out/both × four-patch/Phase1); exact P4/L4 and P4/L2 identity, preserved matrix structure, improved issue/short-scoreboard, and real no-state control spill independently reviewed; 16-pass single K2 profiles are not repeated forward timings |

Each sanitizer's 20 comparisons consist of the 6-check reduced core matrix plus the 14 extra tail/state/packed comparisons; they do not represent sanitizer execution of the full main matrix. The focused matrix and graph checks cover the relevant presence/absence of initial/final state and representative packed and tail paths without changing the release guard. Multi-GPU remains untested in this allocation. The candidate deliberately uses ring2 for HasStateIn=true; no result here is a claim that ring4 benefits that branch.

### Performance scope and remaining interpretation

Against the four-patch Phase6-only binary, the **34 tested in-envelope single-request configurations show lower three-round median-of-medians** under each separately recorded eager CUDA-event, graph replay, cache-perturbed CUDA-event, and synchronized host-wall measure. The **6 out-of-envelope controls remain near parity** and are not pooled into the optimized-domain gain. Detailed numbers and their interpretation are in `CLEAN_FINDINGS.md`; matched instruction/code-generation detail is in `CLEAN_SASS.md`. Those reports, not merely successful profile collection, govern mechanism/resource claims. All 408 optimized-domain paired-round gains are positive, with a minimum 3.72%; fallback median drift stays within ±0.9%, but the worst fallback paired round regresses 1.32%.

The two-stream `out` pair regressed by approximately **1.52%** relative to the four-patch reference in this run. It times a joined interval for two full requests; the candidate is **not presented as a concurrent-throughput optimization**, and single-request latency improvements must not be extrapolated to serving throughput. The `both` pair is close to parity.

Eager CUDA events measure a GPU-timeline interval around the real unchanged wrapper, not CPU wall time or a server request. Graph replay excludes per-call Python dispatch and graph construction. Cache perturbation writes a 256 MiB buffer before the start event and excludes that write's cost; it does not prove every cache is cold. `wall_sync` is separately measured with a host clock around one wrapper call and its completion synchronization (with a pre-synchronization before timing); it still excludes serving, tokenization, model-wide work, and request queuing. Profile timings are separate instrumented measurements, not replacements for these workload timings.

The remaining boundary is deliberate: these are focused single-B300 candidate checks, not complete production workload, concurrency, multi-GPU, or hardware-product qualification. Profile/SASS analysis should be read from the dedicated reports as finalized; the source and patch are not promoted to broader defaults or support solely because Job19934 passed. Do not infer acceptance from Job19924 or the rejected initialization/canonical experiments.

## Identity

| Artifact | SHA-256 |
|---|---|
| `clean-phase1.patch` | `fae72eccda8eea94d5609fd30df75f9855dc9c9c00231300b2af00f89da910d1` |
| `csrc/smxx/fwd_kernel2.cuh` | `950684947df54a3432732468c138a571bca08b44ab96b91c8540d9f0fd97db31` |
| `csrc/smxx/fwd_launch.cu` | `ffe7a15ad1196d1b3d771a55b3f9bdc3900e9ddac5bd51b20f0f2cec27c9643c` |
| `setup.py` | `487b01f3bdc8f9f232fc547b8f6519215899193e4ee076c5eae734c6858fbda8` |
| Job19934 loaded `flash_kda_phase1_C` binary | `f6f80fa402cc1dc00b09a8082b10806bbe17c0e533d067e931dd774d270b9270` |
