# Audited upstream agent stack

This directory contains source snapshots, not locally invented replacements.
The exact revisions and licenses are recorded in `UPSTREAM_LOCK.json`.

## Selected packages

- `atrex-kernel-agent/` supplies the trustworthy execution plane: NVIDIA
  profiling, isolated local/remote GPU execution, Git-worktree episodes,
  resumable journals, conclusions memory, and incumbent/candidate ABBA
  verification. Its optional reference-project submodules are deliberately not
  imported; the FlashKDA campaign already owns its implementation references.
- `pike/` supplies the actual population-search precedent: independent initial
  ideas, parallel branches, error-fixing agents, top-branch selection, and
  experiment/result tooling.

Neither repository is treated as the paper's contribution. Local adapters must
preserve upstream notices and keep the FlashKDA KIR verifier, correctness
oracle, candidate identifiers, B300 accounting, held-out split, and activation
gate authoritative.

## Important boundary

Atrex alone is not a competitive population. Its normal optimization path is
one coding-agent episode at a time, supplemented by independent reviewers and
mechanical promotion. PIKE-B is a real branching search, but its stock
evaluator assumes a KernelBench-style single Python program and ranks mainly by
runtime. The intended integration therefore uses PIKE's search plane and
Atrex's execution/evidence plane without replacing the existing proof-carrying
MARPE control plane.

Do not initialize Atrex's full `reference-projects/` submodule collection by
default. Several entries are large and some use SSH-only upstream URLs.
