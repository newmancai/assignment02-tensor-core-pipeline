# Stage 9 best-swarm pilot handoff

Status: engineering pilot complete; production activation and multi-agent
superiority claims are not authorized.

## Implemented loop

Four roles (capacity, tail/skew, resource, critic/synthesizer) run in isolated
repository-external workspaces. Round 0 proposals are typed, verified,
content-addressed, deduplicated, and measured by one B300 FIFO executor. Only
compact verified conclusions cross the barrier. Round 1 creates single-parent
mutations, followed by a verifier-enforced two-parent compound. Memory is
append-only and hash-chained; any tool event rejects a complete Agent request.

## Final v5 pilot

- Model: `gpt-5.6-sol`, high reasoning.
- Agent calls: 8 successful; 137,317 total recorded tokens; zero tool events.
- Proposal slots: 8; unique semantic policies: 4.
- Development qualification: compound 1.0238245x; original parent 1.0236658x.
- Frozen winner: `kir-policy:ad2eaac3aa3ba8654fd5f9a3acf49267e7c4ff776ec956c5854c558b02e85f7c`.
- Randomized remote-only held-out: 8/8 correctness; 1.0063543x geometric mean.
- Parent held-out geometric means: 1.0060692x and 1.0059206x.
- Compound advantage over best parent: 0.02834%; not statistically or
  practically sufficient to claim emergent collaboration.
- Some per-profile 95% lower bounds are below one; activation remains denied.

The earlier v1--v4 attempts are retained as failure evidence. They exposed and
fixed: raw-latency comparison across allocations, single-parent mutations being
misclassified as compounds, parallel synthesis before parent verification, a
route-out-of-scope randomized profile, and held-out errors returning exit zero.

## Primary artifacts

- `evidence/stage9_best_swarm_pilot_certificate.json`
- `evidence/stage9_best_swarm_pilot_protocol.json`
- `evidence/stage9_best_swarm_pilot_runs_sol/proposal_manifest_v5.json`
- `evidence/b300_stage9_pilot_v5_screen.jsonl`
- `evidence/b300_stage9_pilot_v5_qualification.jsonl`
- `evidence/b300_stage9_pilot_v5_heldout.jsonl`
- `evidence/stage9_pilot_v5_winner_freeze.json`
- `evidence/stage9_pilot_v5_heldout_reveal.json`

## Next scientific step

A no-model vNext audit subsequently enforced three genuinely distinct semantic
niches and screened 17 typed policies (16 development behaviors) in B300 job
23823. All were correct. The best capacity policy reproduced the parent's exact
behavior and measured 1.000346x; the best tail/skew and mixed policies measured
0.978989x and 0.988022x. The run used 30.206 allocated GPU-seconds. Artifacts:
`evidence/b300_stage9_vnext_niche_report.json` and
`evidence/b300_stage9_vnext_niche_accounting.json`.

Do not spend more model tokens on synthesis in the same tree grammar. The next
scientific step is a new typed action knob or another hardware receipt. Only if
a multi-agent superiority claim becomes necessary, run a matched comparison
against sequential single-agent and homogeneous best-of-N baselines with
multiple independent trajectories. Do not infer that claim from this pilot.
