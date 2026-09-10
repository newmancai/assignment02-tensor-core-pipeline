from __future__ import annotations

import unittest

from kda_ir import (
    B300,
    AgentContribution,
    AgentRole,
    AutonomousRuntimeProfileAgent,
    Proposal,
    QualificationResult,
    ScreenResult,
    break_token_order,
    cake_bt16_prepare_chain,
    canonical_candidate_id,
    set_prepare_chunks_per_cta,
)


class FakeOracle:
    def __init__(self, latencies, qualifications):
        self.latencies = latencies
        self.qualifications = qualifications
        self.screened = []
        self.qualified = []

    def screen(self, candidate):
        candidate_id = canonical_candidate_id(candidate.schedule)
        self.screened.append(candidate_id)
        return ScreenResult(candidate_id, self.latencies[candidate_id], True, True)

    def qualify(self, candidate):
        candidate_id = canonical_candidate_id(candidate.schedule)
        self.qualified.append(candidate_id)
        speedup, interval = self.qualifications[candidate_id]
        return QualificationResult(
            candidate_id,
            speedup,
            interval,
            True,
            True,
            True,
            ("paired.json",),
        )


class AutonomousAgentTest(unittest.TestCase):
    def test_closed_loop_deduplicates_verifies_measures_and_activates(self):
        cpc4 = set_prepare_chunks_per_cta(cake_bt16_prepare_chain(), 4)
        cpc9 = set_prepare_chunks_per_cta(cake_bt16_prepare_chain(), 9)
        ids = {cpc: canonical_candidate_id(schedule) for cpc, schedule in ((4, cpc4), (9, cpc9))}
        oracle = FakeOracle(
            {ids[4]: 519.0, ids[9]: 501.0},
            {ids[9]: (1.035, (1.033, 1.037))},
        )
        agent = AutonomousRuntimeProfileAgent(
            B300, oracle, top_k=1, max_rounds=2, plateau_rounds=1
        )

        def proposals(round_index, memory):
            if round_index > 1:
                return ()
            return (
                AgentContribution("a", AgentRole.PROFILE_ANALYST, Proposal(cpc4, "base"), "cpc4"),
                AgentContribution("b", AgentRole.SCHEDULE_EXPLORER, Proposal(cpc9, "candidate"), "cpc9"),
                AgentContribution("c", AgentRole.META_REVIEWER, Proposal(cpc9, "duplicate"), "same typed delta"),
                AgentContribution("d", AgentRole.SAFETY_CRITIC, Proposal(break_token_order(cpc9), "invalid"), "must fail"),
            )

        result = agent.run(proposals)
        self.assertEqual(result.incumbent_candidate_id, ids[9])
        self.assertEqual(oracle.qualified, [ids[9]])
        self.assertEqual(len(oracle.screened), 2)
        self.assertEqual(result.rounds[0].rejected_count, 1)
        self.assertTrue(any(item.state == "verifier_rejected" for item in agent.memory.entries))

    def test_confidence_crossing_one_cannot_activate(self):
        schedule = set_prepare_chunks_per_cta(cake_bt16_prepare_chain(), 9)
        candidate_id = canonical_candidate_id(schedule)
        oracle = FakeOracle(
            {candidate_id: 500.0},
            {candidate_id: (1.001, (0.999, 1.004))},
        )
        agent = AutonomousRuntimeProfileAgent(
            B300, oracle, max_rounds=1, plateau_rounds=1
        )
        result = agent.run(
            lambda _round, _memory: (
                AgentContribution("a", AgentRole.SCHEDULE_EXPLORER, Proposal(schedule, "try"), "candidate"),
            )
        )
        self.assertIsNone(result.incumbent_candidate_id)
        self.assertEqual(agent.memory.entries[-1].state, "measure")


if __name__ == "__main__":
    unittest.main()
