#!/usr/bin/env python3
"""Replay the autonomous proof-carrying loop over Stage 6 B300 evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from kda_ir import (
    B300,
    AgentContribution,
    AgentRole,
    AutonomousRuntimeProfileAgent,
    ConclusionsMemory,
    Proposal,
    QualificationResult,
    ScreenResult,
    cake_bt16_prepare_chain,
    canonical_candidate_id,
    set_prepare_chunks_per_cta,
)


class EvidenceOracle:
    def __init__(self, screen_row: dict, qualification_row: dict) -> None:
        self.screen_row = screen_row
        self.qualification_row = qualification_row

    @staticmethod
    def _cpc(candidate) -> int:
        return candidate.schedule.recurrence.prepare_chunks_per_cta

    def screen(self, candidate) -> ScreenResult:
        cpc = self._cpc(candidate)
        row = next(
            item for item in self.screen_row["candidates"] if item["chunks_per_cta"] == cpc
        )
        return ScreenResult(
            canonical_candidate_id(candidate.schedule),
            row["median_ms"] * 1000.0,
            row["correctness"]["passed"],
            self.screen_row["route_invariant"],
            ("evidence/b300_stage6_h12_profile_matrix.json",),
        )

    def qualify(self, candidate) -> QualificationResult:
        if self._cpc(candidate) != 9:
            raise RuntimeError("Stage 6 paired qualification exists only for cpc=9")
        row = self.qualification_row
        return QualificationResult(
            canonical_candidate_id(candidate.schedule),
            row["speedup"],
            tuple(row["bootstrap_95"]),
            row["correct"],
            True,
            True,
            ("evidence/b300_stage6_h12_profile_pair.json",),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screen", type=Path, default=Path("evidence/b300_stage6_h12_profile_matrix.json")
    )
    parser.add_argument(
        "--qualification", type=Path, default=Path("evidence/b300_stage6_h12_profile_pair.json")
    )
    parser.add_argument(
        "--memory", type=Path, default=Path("evidence/stage7_conclusions_memory.json")
    )
    parser.add_argument(
        "--json", type=Path, default=Path("evidence/stage7_autonomous_agent_replay.json")
    )
    args = parser.parse_args()
    screen = json.loads(args.screen.read_text())
    qualification = json.loads(args.qualification.read_text())
    paired_by_name = {row["name"]: row for row in qualification["rows"]}
    memory = ConclusionsMemory.empty()
    runs = []

    for screen_row in screen["rows"]:
        profile_id = screen_row["name"]
        oracle = EvidenceOracle(screen_row, paired_by_name[profile_id])
        agent = AutonomousRuntimeProfileAgent(
            B300,
            oracle,
            profile_id=profile_id,
            top_k=1,
            max_rounds=2,
            plateau_rounds=1,
            max_screened_candidates=len(screen["cpc_candidates"]),
            memory=memory,
        )

        def proposals(round_index, _memory):
            if round_index > 1:
                return ()
            items = []
            for cpc in screen["cpc_candidates"]:
                schedule = set_prepare_chunks_per_cta(cake_bt16_prepare_chain(), cpc)
                items.append(
                    AgentContribution(
                        f"{profile_id}-cpc-{cpc}",
                        AgentRole.SCHEDULE_EXPLORER,
                        Proposal(schedule, f"screen prepare work assignment cpc={cpc}"),
                        f"candidate prepare chunks per CTA is {cpc}",
                        ("evidence/b300_stage6_h12_profile_matrix.json",),
                    )
                )
            return tuple(items)

        result = agent.run(proposals)
        runs.append(
            {
                "profile_id": profile_id,
                "incumbent_candidate_id": result.incumbent_candidate_id,
                "stopped_reason": result.stopped_reason,
                "rounds": [asdict(item) for item in result.rounds],
                "activated_cpc": 9 if result.incumbent_candidate_id else None,
            }
        )

    memory.write(args.memory)
    report = {
        "schema_version": 1,
        "analysis": "stage7-autonomous-runtime-profile-agent-evidence-replay",
        "authority_boundary": (
            "Agents propose; typed verification, B300 receipts, confidence gates, "
            "fallback, memory, and stopping are deterministic."
        ),
        "runs": runs,
        "all_profiles_activated_cpc9": all(run["activated_cpc"] == 9 for run in runs),
        "memory": str(args.memory),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
