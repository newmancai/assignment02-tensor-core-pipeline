# Phase1 triple-ring lookahead2/4 — experimental draft

Base: `/private/tmp/kda-zero-experiment.rZovhK` (existing strategies 0/1/2/3).
Candidate: `/private/tmp/kda-phase1-draft.7weRM6`.
Incremental patch: `phase1-draft.patch`, SHA-256 `e10834179290cac2a833ee2642c11045ee4ee4bb07a54ffc41fa5696e0429179`.

Only three files change: `csrc/smxx/fwd_kernel2.cuh`, `csrc/smxx/fwd_launch.cu`, `csrc/flash_kda.cpp`. No setup, Python policy, original source, existing selectors, or GPU job was changed by this subtask.

## Selection and scope

- Experimental raw selector `40016` uses `InitStrategy=4`, Phase1 lookahead 2.
- Experimental raw selector `50016` uses `InitStrategy=5`, Phase1 lookahead 4.
- Both IDs are exposed only under the existing `KDA_ZERO_ABLATION` macro. They use the existing experiment launch macro: D128/BF16 public state routes to V16 with Phase6 `StatePrefetch=4`; its pre-existing FP32 behavior falls back to ordinary V16/Prefetch1.
- The new Phase1 branch statically requires `V==16 && kBlocksPerWarp==1`. HasStateIn is **not** canonicalized for strategies 4/5. Their no-state branch retains original scalar zero initialization; neither vector/onewarp nor runtime-init strategy is combined with them.
- The experiment macro intentionally has a wider experimental shape envelope than the released auto policy. These IDs are measurement controls, not public release dispatch additions.
- Strategies 0/1/2/3 retain the literal old Phase1 body in the new `else`; initialization and Phase2 onward are otherwise unchanged. Recompilation may still change generated code, so retain the external four-patch binary and the same-binary strategy 0 control.

## Consume / overwrite invariant

Let `L=2` or `4`, `K_BLOCKS=D/16=8`. Each ring slot holds three owning raw register fragments `(k_decayed[k], q_decayed[k], state[k])`; the state fragment is shared by the two GEMMs at that k.

1. Prologue loads exactly blocks `0..L-1` into their matching slots for all three operands.
2. At iteration `k`, slot `k % L` contains block `k` in all three rings. The implementation applies the same `cute::transform(..., identity{})` into the original MMA operand fragments.
3. It executes the original k-GEMM into `u_acc[0]`, then the original q-GEMM into `out_acc[0]`. Accumulator order is still strictly `k=0,1,...,7`; no reassociation, skipped zero product, changed cast, or extra MMA is introduced.
4. **Only after both consumers** does it refill the slot with block `k+L`, and only if `k+L<8`. That slot is not consumed again until iteration `k+L`. Tail iterations issue no out-of-range loads.

Each operand therefore loads all eight blocks exactly once, and there are still sixteen GEMM calls per recurrence tile (eight per accumulator). Lookahead 2 allows one intervening k iteration before a refilled slot is used; lookahead 4 allows three. This is source-level scheduling intent, not a claim that ptxas preserves the exact load-to-HMMA distance.

The prefetched shared inputs remain stable throughout Phase1: the input stage is not released until the existing end-of-tile pipeline release, and `state_acc` is not updated until the later state-update phase. No new shared stores, barriers, TMA transactions, or host synchronization were added. Ring storage is register-fragment storage; increased register pressure/spills or compiler rescheduling may erase the intended benefit and must be measured.

## Validation performed here

```sh
python3 test_phase1_contract.py /private/tmp/kda-zero-experiment.rZovhK /private/tmp/kda-phase1-draft.7weRM6
```

The CPU contract checks that removing only the new Phase1 branch/wrapper restores the entire original kernel file byte-for-byte, that launch/binding differ only by the two added experimental IDs, and that an independent abstract ring model consumes k in order and loads each block once for both lookaheads. The model does **not** execute CUDA source.

Forward and reverse `git apply --check` passed. CUDA compilation, register/spill inspection, actual SASS lookahead, same-input bitwise parity, sanitizer checks, and GPU timings remain for the main agent. In particular, this draft does not infer success from the earlier initialization experiments or from source-level prefetch distance alone.
