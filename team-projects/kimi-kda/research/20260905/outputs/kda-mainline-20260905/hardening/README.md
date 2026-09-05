# FlashKDA release-entry hardening candidate

This directory is an isolated candidate, not a production-source edit or a released build. Apply `0003-release-entry-hardening.patch` after the existing `0001-k2-value-slice-and-dispatch.patch` and `0002-dispatch-packed-single-sequence.patch` on FlashKDA `1ce47ea`.

## Changes

Only `csrc/flash_kda.cpp` and `setup.py` change:

1. Validate that k/v/g/beta/out/workspace/A_log/dt_bias and optional initial_state/final_state/cu_seqlens are on q's device. Hold a `c10::cuda::CUDAGuard` through ATen copies and CUDA launch so the correct device's current stream is used and the caller's device is restored on return or exception.
2. After the existing beta transpose/contiguous step, clone only if the actual beta base address is not 16-byte aligned. This repairs legal contiguous offset views when T_total=1 or H=1 without adding another copy to the already aligned path. It does not synchronize the host or change beta values.
3. Forward the `K2_VALUE_SLICE` macro to the C++ binding compiler as well as nvcc. CUDA-only isolated-kernel flags remain nvcc-only. Production builds without a macro still default to V128 at the raw binding; the existing Python wrapper continues explicitly passing the auto/forced selection.

There are no kernel, D128/C16, arithmetic, rounding, Python default-policy, or dispatcher edits. This patch does not repair FP32 empty-state quantization, add arbitrary-stride support, add alignment copies for other tensor arguments, or change workspace ownership. Other TMA bases still require aligned storage under the candidate's stated input domain; mixed-device pointers are now rejected rather than dispatched.

## Reproduction

Use a fresh independent checkout/build directory, not a dirty working tree. From a tree with 0001 and 0002 applied:

```sh
git apply --check /absolute/path/hardening/0003-release-entry-hardening.patch
git apply /absolute/path/hardening/0003-release-entry-hardening.patch
python /absolute/path/hardening/test_build_flags.py .
```

Build with the established PyTorch/CUDA13/CUTLASS environment and SM103 target. Do not reuse an old `.so` or infer binary identity from its module name. The tests print each loaded extension's absolute path and SHA-256. A dedicated extension name may be supplied to avoid colliding with an existing installation:

```sh
python /absolute/path/hardening/test_entry_hardening.py --extension flash_kda_C
```

If the optional V16 alias was freshly built, also run:

```sh
python /absolute/path/hardening/test_entry_hardening.py --extension flash_kda_C --alias-extension flash_kda_vsplit16_C
```

On an allocation with two matching B300 devices, add `--require-two-gpus`. With only one visible GPU the device-1/caller-device-0 and mixed-GPU cases are explicitly SKIP, not proof of multi-GPU correctness. Each case runs in its own subprocess with a 180-second timeout. No job is submitted by this script. It may take several minutes including separate PyTorch initializations.

The test accepts `--cases binding,alignment,parity,stream_graph,cpu_rejection,multi_gpu` to select subsets and `--expected-default` for a deliberately nondefault macro build. The default expected production raw-binding value is 128; `--alias-extension` always checks for 16. An optional alias that was not supplied is explicitly reported as SKIP.

## Validation scope

Main-agent update, 2026-09-05: the historical subtask handoff table below is now
superseded for GPU acceptance by Job19896 (experiment build) and Job19901 (clean
release build). Both rebuilt binaries passed the five single-GPU entry cases;
multi-GPU remained SKIP because the allocation exposes one GPU, and the optional
compiled alias remained SKIP because no alias binary was built. See
[the release acceptance record](../release/README.md)
and [raw Job19901 log](../release_19901.log).
The updated final setup adds release flags; run `release/test_build_contract.py`
against the four-patch source, not this earlier factory-only CPU test.

| Check | Purpose | Status when handed to the main agent |
|---|---|---|
| Patch forward and reverse `git apply --check` | Confirm exact compatibility with the two-patch source baseline and edited copy | PASS locally |
| `test_build_flags.py` production/alias/isolated-kernel cases | Confirm host/device macro agreement and exclusion of CUDA-only flags from C++ | PASS locally, no CUDA imported |
| Python syntax compilation | Confirm both test scripts compile | PASS locally; cache redirected to the isolated temporary directory |
| Kernel tree and Python package comparison | Confirm only the binding and setup changed | PASS locally |
| Fresh C++/CUDA extension build | Verify headers, API compatibility and binary generation | PENDING; main agent owns build and scheduling |
| Offset beta vs aligned clone | T1/H2, T1/H12, T17/H1 and copying-transpose control; all four slices, BF16/FP32 public state | GPU test prepared, NOT RUN by this subtask |
| Same-binary V128 parity | T1, fixed B2/T17, packed lengths 1/16/16; state input/output/None | GPU test prepared, NOT RUN by this subtask |
| Non-default stream and CUDA Graph | Verify the conditional beta clone remains ordered and capture-compatible | GPU test prepared, NOT RUN by this subtask |
| CPU and mixed-GPU rejection | Verify all relevant pointer-device validation reaches a catchable exception before descriptor construction | GPU test prepared, NOT RUN by this subtask |
| Input device 1 with caller device 0 | Verify correct launch, caller-device restoration and restoration on a later validation exception | Two-GPU test prepared, NOT RUN by this subtask |
| Compiled production/alias signature | Verify actual binding defaults after rebuilding, not merely build-factory intent | Test prepared, NOT RUN by this subtask |

The same-binary parity cases are targeted regression checks, not a replacement for the main release campaign's untouched-upstream vs patched-V128 reference or high-value-shape performance measurement. The guard and common aligned beta path introduce no new CUDA kernels. The historical table states what the subtask itself ran; completed main-agent GPU results are linked above. Job19901 measures real-wrapper CUDA-event latency, not isolated CPU wall overhead.

## Source identity

The implementation was edited only in `/private/tmp/kda-hardening.khB2d9`, copied from the already two-patch source `/private/tmp/kda-review-code.PDkQDX`. The original directory was not edited.

| File | Before SHA-256 | After SHA-256 |
|---|---|---|
| csrc/flash_kda.cpp | `6ea353ff4af3a1fad0f6f3c376c82fe58cceae23eba7c67f21fe0589883a5863` | `2c07c7ef52007bc7d8f09bfca3d2ef870ad3f111ddea54d1474f65ffa4311b2d` |
| setup.py | `2412204d9c44a63cd482abfc57ef6ffa762d9f8ba15b293125bf4e898d00e709` | `3e53aa0849192322aaca0b8830ab61f332a76790f4183e1998ede752cd54191d` |

Patch SHA-256: `0f888bf678f0111b5ad394bd4beda4641d472dbca9fc5e02c2e7988fc5bbaefe`.
