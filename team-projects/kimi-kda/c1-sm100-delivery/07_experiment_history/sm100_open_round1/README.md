# SM100 open research, round 1

Question: **FlashKDA 官方 kernel 当前使用 SM80 MMA，分析迁移到 SM100 是否值得**.

This directory resumes from `HANDOFF_SM100_OPEN_RESEARCH.md`.  It keeps
proposal identity, Typed IR/representation status, and evidence status
separate.  Historical measurements are indexed but are not relabeled as new
results.

## Selected first falsifier: H3 P/W precomputation

The first prototype tests the algebraic rewrite

```text
official: U = INV * diag(beta) * (V - Kd * S)
H3:       P = INV * diag(beta) * V
          W = INV * diag(beta) * Kd
          U = P - W * S
```

It is selected because it changes the serial K2 boundary and is technically
different from another isolated opcode swap.  The real-valued identity alone
does not preserve FlashKDA's BF16 rounding DAG, so the candidate is explicitly
`algorithm_equivalent_not_official_bitwise`.

Before results are inspected, the single-chunk entry gate is fixed as follows:

- compare official ordering and H3 against one FP64 reference;
- check `U`, output, and updated final state;
- H3 worst relative-L2 must be no more than `max(2 * official, 0.01)`;
- H3 worst normalized-Linf must be no more than `max(4 * official, 0.02)`;
- cover 20 seeds at scales 0.02, 0.1, 0.5, and 1.0, shape C16/D128/V128.

This gate only decides whether to build a complete multi-chunk oracle.  It is
not a production tolerance and cannot establish official bitwise compatibility.

The next recurrence gate is also fixed before execution.  It covers 2, 4, 16,
and 64 chunks; lengths 17, 33, and 65 (tails); nonzero initial state; strong,
mixed, and weak decay; a first-chunk cancellation construction; and a two-call
state handoff.  Against the same FP64 chunk recurrence, H3 must satisfy, for
every scenario:

- output/final-state relative-L2 `<= max(2 * official, 0.02)`;
- output/final-state normalized-Linf `<= max(4 * official, 0.05)`;
- finite values and exact equivalence between one-call and two-call execution
  when both consume the same precomputed chunk list.

As above, this is an early algorithm-equivalence screen, not the final accuracy
contract against the FLA/official input-level references.

Run with the bundled NumPy environment:

```bash
<LOCAL_HOME>/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  experiments/sm100_open_round1/prototype_h3_rounding.py \
  --output experiments/sm100_open_round1/evidence/h3_rounding_probe.json

<LOCAL_HOME>/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  experiments/sm100_open_round1/prototype_h3_recurrence.py \
  --output experiments/sm100_open_round1/evidence/h3_recurrence_probe.json

<LOCAL_HOME>/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  experiments/sm100_open_round1/prototype_h3_input_oracle.py \
  --output experiments/sm100_open_round1/evidence/h3_input_oracle.json
```

## Artifact map

- `paths/index.json`: cross-path identity, representation and evidence scope.
- `prototype_h3_rounding.py`: executable CPU numerical falsifier.
- `prototype_h3_recurrence.py`: multi-chunk/tail/state-handoff falsifier.
- `prototype_h3_input_oracle.py`: shared q/k/v/g/beta CPU oracle against an
  FP64 FLA-style token recurrence.
- `evidence/h3_rounding_probe.json`: generated result, once run.
- `evidence/h3_recurrence_probe.json`: generated recurrence result, once run.
- `evidence/h3_input_oracle.json`: generated input-level CPU result.
- `evidence/h3_build_attempt2.log`: successful SM103a build receipt.
- `evidence/h3_gpu_correctness.json`: passing B300 GPU correctness gate.
- `evidence/h3_b300_pilot.json`: H12/T8192 matched-HMMA pilot.
- `evidence/h3_b300_full_matrix.json`: historical same-process
  six-H12/three-H96 diagnostic, superseded for formal magnitude.
- `evidence/h3_b300_isolated_process_cupti.json`: authoritative H3
  isolated-process CUPTI matrix; older same-process timing is diagnostic.
- `evidence/h3_source_manifest.json`: H3 source and B300 binary identities.
- `ir/cake_sm100_full_stack.json`: source-grounded partial IR for the actual
  CAKE SM103a routes and their explicit representation gaps.
- `evidence/b300_stage12_cake_official_isolated_pair.json`: authoritative
  isolated-process CAKE/official paired timing over six H12 and three H96
  profiles.
- `evidence/cake_source_binary_manifest.json`: exact CAKE dispatch/JIT source,
  cached SM103a binary, measurement and log hashes for job 24380.
- `DEFENSE_SELF_AUDIT.md`: sixty-four-question claim, method, path, causality and
  external-validity audit for the mainline defense.
- `TCGEN_OPTION_VALUE.md`: formal estimand and matched H0/H1/T0/T1/T2 matrix
  for separating common optimization, direct ISA/protocol, and tcgen-enabled
  optimization value.
- `H3_MATCHED_HMMA_PATCH_DRAFT.md`: source-audited implementation draft.
- `H3_B300_MEASUREMENT.md`, correctness/pilot sbatch files, benchmark runner,
  and `schema/`: executable B300 gates and evidence contract.
- `ROUND1_REVIEW.md`: proposals, cross-criticism, selection and next code task.

The archived JSON files copied back from `<B300_LOGIN_HOST>` are B300 evidence; no
benchmark executed on this macOS host is a B300 result.  New GPU work must use
`<B300_LOGIN_HOST>`, record the actual SM103a environment, and preserve the same
public-full scope and implementation isolation.
