# Stage 9 bounded proposal protocol

> **Historical narrow protocol.** This grammar is authoritative only for the
> sealed Stage 9 H12/BT16/cpc ablation. It is not the MARPE-wide proposal plane
> and must not constrain HMMA/tcgen05 migration research. See
> `FRAMEWORK_SCOPE_CORRECTION_20260910.md` and `SM100_OPEN_AGENT_PROTOCOL.md`.

This file is the common public envelope shown to every Stage 9 proposal
backend. It deliberately contains no held-out profile, Stage 10 result, paper
text, source tree, or executable candidate code.

The task is to propose a total runtime policy for the existing H12/BF16
`bt16_prepare_chain_m64` route on one fixed B300 resource receipt. A policy maps
four integer profile features to `chunks_per_prepare_cta`:

- `total_chunks`;
- `max_chunks`;
- `num_sequences`;
- `resident_grid_capacity_ctas` (740 in the development receipt).

The output language is a depth-at-most-two decision tree, serialized as a flat
node array with integer child indices. A leaf uses `kind=leaf`, an allowed
`cpc`, `feature=cmp=none`, `threshold=0`, and child indices -1. A branch uses
`kind=branch`, `cpc=0`, one feature, `le` or `gt`, a positive threshold, and
valid `then_index`/`else_index`. Raw CUDA, Python,
shell commands, imports, prose outside the requested JSON, and file access are
forbidden.

All policies must be total. Prefer falsifiable physical hypotheses over curve
fitting. The evaluator, not the model, owns correctness, performance ranking,
candidate identity, deduplication, budget accounting, and held-out access.

Development evidence is supplied inside each request. Latencies are public-path
CUPTI measurements in microseconds; lower is better. Every arm receives the
same evidence and candidate grammar. Proposal failures and duplicates consume
proposal opportunities.
