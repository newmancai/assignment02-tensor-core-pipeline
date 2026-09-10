# Open-source multi-agent stack for MARPE

## Decision

Use a three-plane composition rather than pretending one upstream repository
already implements the paper's system:

```text
PIKE-B search plane
  independent scouts -> typed proposals -> branch allocation -> repair/pivot
                              |
                              v
MARPE proof-carrying control plane (existing code)
  canonical ID -> KIR verifier -> dedup -> equal budget -> sealed held-out gate
                              |
                              v
Atrex execution/evidence plane
  sandbox -> correctness -> profile -> ABBA -> Git episode -> durable memory
```

The smallest credible Stage 9 configuration is four proposal lanes, not ten:

1. `wave/capacity scout` explores resident-grid boundaries;
2. `tail/skew scout` explores workload imbalance and longest-chain effects;
3. `resource scout` explores register, shared-memory, occupancy, and launch
   geometry counterfactuals;
4. `critic/synthesizer` attacks assumptions and may compose typed deltas from
   different lineages.

Each lane starts from a private context. Communication happens only after a
round through `Conclusion` records and immutable measurement receipts. Agents
never share hidden reasoning, never edit the incumbent directly, and never
decide activation.

## What is reused

From PIKE:

- the phased pattern in `scripts/parallel_tree_search.py`;
- parent allocation strategies in `src/query_strategies.py`;
- separate initial brainstorming and error-repair roles;
- per-agent/per-step artifacts and budget accounting conventions.

From Atrex:

- `orchestrator/agent_runtime/` for Codex/Claude/Qoder/Pi process adapters;
- `long_horizon/` for isolated branches, resumable journals, structured
  handoffs, and promotion records;
- `tools/sandbox.py` plus SSH/Bubblewrap execution for a low-privilege remote
  GPU account;
- `tools/profile_nvidia.sh` and profiler helpers;
- same-allocation incumbent/candidate ABBA verification and canonical memory.

From the current FlashKDA prototype (kept authoritative):

- `kda_ir.multi_agent.coordinate_contributions` for typed verification before
  semantic merge;
- `kda_ir.autonomous_agent.canonical_candidate_id` and conclusions-only memory;
- `kda_ir.ablation.EqualBudgetContract` and the frozen Stage 9 arms;
- output/final-state correctness, public-scope parity, B300 run accounting,
  confidence-bound activation, fallback, and sealed held-out evaluation.

## Deliberately not reused

- PIKE's KernelBench evaluator and `kernel.py` parser: FlashKDA is multi-file,
  stateful, dispatcher-inclusive, and has final-state semantics.
- PIKE's raw latency-only ranking: development score must include correctness,
  scope parity, uncertainty, duplicate cost, and profile coverage.
- Atrex's assumption that an episode candidate changes only `kernel.py`: MARPE
  candidates are typed schedule/policy deltas with content-addressed receipts.
- Atrex's optional reference-project submodules: they add size and supply-chain
  surface without helping the frozen FlashKDA experiment.
- OpenEvolve as a third runtime dependency: MAP-Elites/islands are useful later,
  but PIKE-B already supplies the Stage 9 competition required to distinguish
  role collaboration from best-of-N.

## Adapter boundary to implement

Only three thin adapters are needed:

1. `ProposalBackend`: run one isolated role prompt and return an
   `AgentContribution`; no source code is accepted from the model.
2. `BranchAllocator`: translate PIKE parent selection into canonical KIR
   candidate IDs while preserving lineage and charging duplicate proposals.
3. `AtrexMeasurementOracle`: package a verified typed candidate for the
   existing B300 scripts, then translate correctness/profile/ABBA artifacts into
   `ScreenResult` and `QualificationResult`.

The first executable experiment should be the already preregistered four-arm
Stage 9 study. All arms must use the same base model, token cap, 24 proposal
slots, 24 screen slots, four qualification slots, GPU-second budget, workload
split, and candidate space. The GPU remains a single-writer FIFO resource;
parallelism is in reasoning and proposal generation, not concurrent timing.

## Honest claim boundary

This import does not make the current replay a real multi-agent result. The
system becomes a competitive multi-agent optimizer only after independent
processes generate separate proposals, branches survive or die by deterministic
receipts, and the equal-budget Stage 9 run is completed. Until then, the paper
should describe the implementation as a proof-carrying multi-agent protocol and
the performance evidence as a deterministic replay, not as emergent
collaboration.
