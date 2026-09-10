from __future__ import annotations

import json

from kda_ir import (
    B300,
    FailureLedger,
    Proposal,
    break_token_order,
    cake_bt16_prepare_chain,
    evolve,
    lowering_plan,
    value_sliced_v16,
)


for candidate in (value_sliced_v16(), cake_bt16_prepare_chain()):
    print(json.dumps(lowering_plan(candidate, B300), indent=2))

ledger = FailureLedger()
unsafe = break_token_order(value_sliced_v16())
result = evolve(
    (
        Proposal(value_sliced_v16(), "increase residency"),
        Proposal(cake_bt16_prepare_chain(), "split recurrence"),
        Proposal(unsafe, "speculative token reorder"),
        Proposal(unsafe, "second observation of the same failure"),
    ),
    B300,
    ledger,
)
print(
    json.dumps(
        {
            "accepted": [item.proposal.schedule.name for item in result.accepted],
            "rejected": [item.proposal.schedule.name for item in result.rejected],
            "pareto_frontier": [
                item.proposal.schedule.name for item in result.pareto_frontier
            ],
            "rule_promotion_candidates": ledger.promotion_candidates(),
        },
        indent=2,
    )
)
