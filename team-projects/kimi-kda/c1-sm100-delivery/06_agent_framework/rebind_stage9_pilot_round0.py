#!/usr/bin/env python3
"""Bind semantically unchanged pilot round-0 proposals to a runner-only amendment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path


def digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_bytes, amendment_bytes = args.manifest.read_bytes(), args.amendment.read_bytes()
    old, amendment = json.loads(old_bytes), json.loads(amendment_bytes)
    if (old.get("round"), old.get("status")) not in {
        (0, "round_0_proposal_frozen"), (None, "proposal_frozen")
    }:
        raise ValueError("only a frozen pilot manifest can be rebound")
    if not amendment.get("pilot") or amendment.get("status") != "amended_ready_for_execution":
        raise ValueError("target must be a sealed pilot amendment")
    if amendment.get("supersedes_protocol_digest") != old.get("protocol_digest"):
        raise ValueError("amendment does not explicitly supersede source protocol")
    protocol = amendment["protocol"]
    if tuple(old["arms"]) != tuple(protocol["arms"]):
        raise ValueError("arm semantics changed")
    backend = protocol["proposal_backend"]
    if old["model"] != backend["model"] or old["reasoning_effort"] != backend["reasoning_effort"]:
        raise ValueError("proposal backend changed")
    rebound = dict(old)
    rebound.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "amendment_sha256": sha256(amendment_bytes).hexdigest(),
        "amendment_protocol_digest": amendment["protocol_digest"],
        "protocol_digest": amendment["protocol_digest"],
        "candidate_space_digest": digest(protocol["candidate_space"]),
        "arm_prompt_digests": backend["arm_protocol_digests"],
        "arm_protocol_digests": backend["arm_protocol_digests"],
        "supersedes_manifest_sha256": sha256(old_bytes).hexdigest(),
        "rebind_scope": "round1_adapter_and_synthesizer_order_only_round0_prompt_and_proposals_unchanged",
    })
    rebound["proposal_manifest_sha256"] = digest(rebound["proposals"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rebound, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
