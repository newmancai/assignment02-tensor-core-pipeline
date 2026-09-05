# Guarded V16 Prefetch4 release candidate

Status: the clean release candidate was freshly built through its original `setup.py` and exercised on B300 in **Job19901**. CPU contracts, the recorded single-GPU correctness/hardening checks, and targeted sanitizer checks passed. Multi-GPU and the separate alias-extension GPU check remain **SKIP**; complete NCU kernel-identity/resource interpretation is still **PENDING**. Prior Job19896 results came from the experiment binary and are not substituted for this clean release build.

Apply `0004-guarded-v16-prefetch4.patch` after 0001 + 0002 + `0003-release-entry-hardening.patch`. The candidate changes only `csrc/smxx/fwd_kernel2.cuh`, `csrc/smxx/fwd_launch.cu`, and `setup.py`. No experiment selector, OutputSchedule, early-output, or delayed-state-store implementation is included.

## Exact scope

The model stays **D128 / C16**. The existing state-update ring gains a compile-time `StatePrefetch` parameter, default 1. Prefetch4 loads four independent state-row fragments ahead instead of one; state layout, arithmetic, output scheduling, barrier placement, and rounding expressions are unchanged.

Prefetch4 is selected only when all of the following hold:

- The binary was built with `FLASH_KDA_ENABLE_V16_PREFETCH4=1`.
- The already-selected ValueSlice is 16.
- D is 128, state mode is not FP32, N is 1, H is 12, and `2048 <= T_total <= 8192`.

BF16 initial/final state and absent state qualify for the non-FP32 branch. A legal packed single sequence can qualify; multiple sequences, FP32 public state, other H, short/long lengths outside the interval, and V32/V64/V128 keep Prefetch1. No runtime selector values are added to the public C++ API.

This is a **B300-oriented compile-time opt-in that refines the existing guarded ValueSlice path**. `setup.py` requires an explicit `FLASH_KDA_CUDA_ARCHS=103a`; `auto`, `all`, multi-architecture lists, and other targets are rejected when this opt-in is enabled. The feature macro is passed only to nvcc. It is disabled by default.

The build constraint is an SM103 target check, **not** a runtime check that every device is the calibrated B300 configuration. The unchanged Python auto policy still checks B300 SM/L2 resources and its calibrated shape domain before choosing V16. Raw calls or the existing forced-V16 override bypass that Python policy and may run on another compatible SM103 product; this candidate does not claim hardware-wide coverage. `explain_k2_dispatch()` still explains the ValueSlice policy, not the new compiled prefetch subvariant; identify Prefetch4 with the binary build manifest and kernel/profile evidence.

## Build and rollback

Use the fixed upstream/CUTLASS versions and an independent source/build directory. The source copy prepared by this subtask is `/private/tmp/kda-release-prefetch4.DStLoA`; it includes the C++/CUDA source, Python package, and setup script. Provide the established CUTLASS tree at its expected `cutlass` path rather than introducing a new dependency revision.

Example opt-in build, using the established Python/CUDA/include environment:

```sh
FLASH_KDA_CUDA_ARCHS=103a FLASH_KDA_ENABLE_V16_PREFETCH4=1 FLASH_KDA_EXTENSION_NAME=flash_kda_release_C python setup.py build_ext --build-lib /absolute/new-build/lib --build-temp /absolute/new-build/temp --force
```

Set the environment at **build time**. Setting or unsetting `FLASH_KDA_ENABLE_V16_PREFETCH4` only at runtime cannot change an already built binary. Build without the opt-in to retain Prefetch1 everywhere; compare against that separate binary when measuring old V16. In an enabled binary, forcing V16 inside the envelope selects Prefetch4. For a runtime compatibility path, the existing `FLASH_KDA_K2_VALUE_SLICE=128` selects V128/Prefetch1; Python policy and override priority are unchanged.

The custom module name above is for isolated validation. The unchanged public
wrapper imports `flash_kda_C`; merely placing `flash_kda_release_C` on PYTHONPATH
does **not** redirect that wrapper. `../release_probe.py` explicitly binds two
independent copies of the real wrapper to the old and new extension objects, and
prints the loaded binary hash. For a normal package, build the standard
`flash_kda_C` name in a separate environment; do not rename the `.so` or overwrite
the existing installation and assume its extension initialization name changes.

For the known server venv, the previously verified build include environment is:

```sh
export CPATH="/usr/local/cuda-13.0/targets/x86_64-linux/include:/home/lcpu/YOUR_USER_ID/FlashKDA/.deps/python312-dev/usr/include:/home/lcpu/YOUR_USER_ID/FlashKDA/.deps/python312-dev/usr/include/python3.12:/home/lcpu/YOUR_USER_ID/FlashKDA/.deps/python312-dev/usr/include/x86_64-linux-gnu/python3.12"
```

Do not infer that a build is enabled merely from the module name. Record setup environment, compiler flags, loaded `.so` path/hash, and representative selected-kernel evidence.

The actual fresh-build recipe is `../build_release.sh`, with output in `../build_release.log`. It uses `/home/lcpu/YOUR_USER_ID/kda-mainline-20260905/release_source` and separate `release_build/{lib,temp}` directories, enables the opt-in and explicit `103a`, and invokes the source's original `setup.py build_ext ... --force`. The recorded source hashes match the candidate below; the nvcc command contains both `-DFLASH_KDA_ENABLE_V16_PREFETCH4=1` and `-gencode arch=compute_103a,code=sm_103a`. This workflow changes neither the original FlashKDA source tree nor the established baseline binary. The module name and build output paths isolate the release candidate; the build does not install it over the user's existing package.

For reproduction, retain the four-patch chain, the fixed dependency tree, `build_release.sh`/`build_release.log`, `../run_release.sbatch`, `../release_probe.py`, `../hardening/test_entry_hardening.py`, and the Job19901 logs. The run script supplies both the isolated release and existing baseline library paths; inspect the emitted loaded-binary path/hash before interpreting a rerun. The raw extension defaults to V128; Prefetch4 remains build-time **off by default**, independently of the unchanged Python auto-dispatch default. To roll back an enabled binary at runtime, set `FLASH_KDA_K2_VALUE_SLICE=128`; to restore old V16 itself, rebuild without the Prefetch4 opt-in.

## CPU validation completed

```sh
python test_build_contract.py /absolute/candidate-source
python test_guard_contract.py /absolute/candidate-source
```

- Forward and reverse `git apply --check`: PASS.
- Build-function tests: 13 checks PASS. Default-off, explicit-off, opt-in SM103, production/alias macro consistency, and rejection of unset/auto/all/other/multi-architecture opt-in targets were tested without importing PyTorch or executing setup.
- Guard tests: 51 checks PASS. The test extracts the **actual `launch_fwd` selector** from the candidate, compiles it unchanged against a recording CPU stub, and executes 17 cases under no macro, macro 0, and macro 1. It covers T2047/2048/4096/8192/8193, N2, H11/H13, FP32 state, V32/V64/V128, packed N1/N2, and state input/output/None.
- C++ binding and Python package comparisons against the hardened base: unchanged.

The CPU guard test verifies branch selection, not CUDA code generation, kernel execution, numerical correctness, register pressure, or speed. The old hardening build-factory-only test predates `get_release_flags`; use this directory's `test_build_contract.py` for the final setup script. The hardening GPU tests remain applicable to the new extension.

## Clean release GPU acceptance — Job19901

| Item | Job / artifact | Result |
|---|---|---|
| Fresh build from original setup with opt-in + explicit SM103 | `../build_release.sh`, `../build_release.log` | PASS; independent source/build directories and expected nvcc flags |
| Loaded release `.so` path/hash | `../release_19901.log` | PASS; identity recorded below |
| Core bitwise regression against old V128 | `../release_19901.log`, `../release_probe.py` | 120 comparisons PASS: 40 input configurations × auto/force16/off; output and present final state are finite and bitwise equal |
| Recurrent state-chain and concurrent correctness | `../release_19901.log` | 3 state-chain steps and 2 concurrent-request comparisons PASS against the old implementation |
| Hardening alignment/stream/device tests | `../release_19901.log` | 5 cases PASS: binding, alignment, parity, stream/graph, CPU-device rejection. Multi-GPU SKIP: only one CUDA device visible. Separate alias-extension default check SKIP: no alias extension supplied |
| Prefetch4/Prefetch1 guard envelope | CPU selector test plus `../release_19901.log` | 51 CPU selector checks PASS; GPU correctness executed in-domain, immediate boundary, FP32, multi-sequence, other-H, and V128 paths. `explain_k2_dispatch` reports ValueSlice only; per-case GPU prefetch identity is not asserted by this log |
| Targeted memory/synchronization checks | `../release_19901_memcheck.log`, `../release_19901_synccheck.log` | Both exit 0 and report `ERROR SUMMARY: 0 errors`; each covers dense T2049 and packed-single T2048 with output-only state under auto/force16/off, not the entire regression matrix |
| Paired forward timing against old auto V16 and release V128 | `../release_19901.log` | Completed over 3 randomized-order rounds; representative measurements below |
| NCU/resource checks for the clean release kernel | `../release_19901_{baseline,release}_ncu.log` and corresponding CSV; `../analysis/NCU_FINDINGS.md` | Independently verified P4 kernel suffix, unchanged 96 CTA / 96 threads / shared memory; target registers 54 to 70, no target spills, issue/eligible improved. One 16-pass K2 profile per variant, not a timing distribution or all-shape qualification |

The loaded binary was `/home/lcpu/YOUR_USER_ID/kda-mainline-20260905/release_build/lib/flash_kda_release_C.cpython-312-x86_64-linux-gnu.so`, SHA-256 `34e2c68cf80de0bd24278afb035fdcd44e4a6205241dfcb9aab4ee95891bc486`. The run recorded NVIDIA B300 SXM6 AC, driver 580.126.09, PyTorch 2.10.0+cu130, SM10.3 / 148 SMs / 132644864 bytes L2. This is one tested hardware configuration, not multi-product or multi-GPU qualification.

The clean-binary matrix includes T2047/2048/8192/8193, intermediate/tail lengths, BF16/FP32 public state, missing state, legal packed single and multiple sequences (including an empty segment), batches 2/4, H24/48/96, and two gate extremes. CUDA Graph execution is represented by the hardening stream/graph case and the wrapper timing runs. These are regression checks against the existing implementation, not a new high-precision oracle or proof that upstream FP32-state semantics have changed.

### Timing interpretation

For B1 / H12 / T8192 / D128 / C16 with BF16 input and BF16 initial/final state, the following values are the **median of the three per-round medians**, in milliseconds. Each round samples 60 eager and 60 graph intervals, or 30 cache-perturbed intervals; variant order is shuffled in each round. The separate two-stream test samples 30 joined request pairs per round.

| Measurement | Old auto (V16) | Release auto (V16/Prefetch4 envelope) | Latency reduction |
|---|---:|---:|---:|
| Eager wrapper, CUDA events | 0.569712 | 0.459184 | 19.40% |
| CUDA Graph replay | 0.566816 | 0.454304 | 19.85% |
| Cache-perturbed wrapper, CUDA events | 0.571376 | 0.459808 | 19.53% |
| Two-stream pair of full requests | 0.672208 | 0.669040 | 0.47% |

The eager interval uses CUDA events around an actual wrapper call, including its dispatch/workspace-allocation path. It measures a GPU-timeline interval, which can include host-induced gaps, **not CPU wall-clock latency or server end-to-end latency**. Graph replay excludes per-call Python dispatch; graph construction is outside the timed interval. Cache perturbation writes a 256 MiB buffer before the start event, so its cost is excluded; this is a cache-perturbed measurement, not a guarantee that every cache is cold. The two-stream result times a joined interval for both requests, not one request or serving throughput. Its approximately 0.47% difference does not support extrapolating the single-request gain to concurrent serving. NCU was collected separately with uncontrolled caches/clocks; its instrumented durations are not mixed into these timing numbers.

### State-contract supplement — Job19903

The separate state matrix passed 27 shapes and 243 timing rows, covering
T2048/4096/8192, fixed/packed single sequence and all four state input/output
combinations, plus T2049/4095/8191 tails. Of 81 correctness rows, 54 are non-self
cross-path comparisons and 27 are reference self-sanity checks. Each mode has
three rounds with 60 eager / 60 graph / 30 cache-perturbed intervals.

Across the 24 core shapes and three timing scopes, having an initial BF16 state
gave 16.46–19.66% additional reduction; absent initial state gave 7.00–9.13%.
Do not advertise the ~19% continuation-state result as the initial-free first
prefill result. All observed paired-round reductions remained positive, but the
finite samples do not establish every integer length in the build guard or
every serving environment. See
[state findings](../analysis/STATE_FINDINGS.md)
and [full mainline result](../MAINLINE_RESULT.md).

## Source identity

The hardened base `/private/tmp/kda-hardening.khB2d9` was not edited. The candidate contains no broad experiment patch.

| Candidate artifact | SHA-256 |
|---|---|
| `0004-guarded-v16-prefetch4.patch` | `246a6aa347a1779215d5cdb72d84f2be18357ea0b77be8df987c8c46118b9a96` |
| `csrc/smxx/fwd_kernel2.cuh` | `78e21ed05cfeada41d04b0018232ea98a3478aa26b7bc7f3a38a4d5b29877546` |
| `csrc/smxx/fwd_launch.cu` | `0c03beaa93c072418bd89646a245b7b686f77200e89460af6e997e3f87171051` |
| `setup.py` | `49360624f93a74b8a3433dc0a12a02f72e62d83b27951e7f36e27533bfbb7307` |
