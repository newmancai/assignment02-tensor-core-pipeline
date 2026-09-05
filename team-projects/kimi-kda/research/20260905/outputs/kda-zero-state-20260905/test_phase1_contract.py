#!/usr/bin/env python3
"""CPU-only source preservation and abstract ring-order checks; no CUDA claim."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    rel = Path("csrc/smxx/fwd_kernel2.cuh")
    base = (args.base / rel).read_text()
    candidate = (args.candidate / rel).read_text()
    opening = "            if constexpr (InitStrategy == 4 || InitStrategy == 5) {\n"
    otherwise = "            } else {\n"
    closing = "            }\n\n            // ======== Phase 2:"
    begin = candidate.index(opening)
    old_begin = candidate.index(otherwise, begin) + len(otherwise)
    old_end = candidate.index(closing, old_begin)
    restored = candidate[:begin] + candidate[old_begin:old_end] + candidate[old_end + len("            }\n"):]
    assert restored == base, "Changes escaped the Phase1 wrapper/new branch"
    print(json.dumps(dict(check="original_kernel_body_preserved", status="PASS")))

    for rel, anchor, addition in (
        ("csrc/smxx/fwd_launch.cu",
         "        case 30016: LAUNCH_ZERO_STRATEGY(3); break;\n",
         "        case 40016: LAUNCH_ZERO_STRATEGY(4); break;\n"
         "        case 50016: LAUNCH_ZERO_STRATEGY(5); break;\n"),
        ("csrc/flash_kda.cpp",
         "        k2_value_slice == 10016 || k2_value_slice == 20016 || k2_value_slice == 30016 ||\n",
         "        k2_value_slice == 40016 || k2_value_slice == 50016 ||\n"),
    ):
        old = (args.base / rel).read_text()
        new = (args.candidate / rel).read_text()
        assert old.count(anchor) == 1
        assert new == old.replace(anchor, anchor + addition), rel
        print(json.dumps(dict(check="only_new_experiment_ids", file=rel, status="PASS")))

    # This is an independent abstract model of the documented schedule, not
    # execution of the CUDA source or a substitute for matched GPU tests.
    for lookahead in (2, 4):
        k_blocks = 8
        rings = {operand: list(range(lookahead)) for operand in ("k", "q", "state")}
        loads = [(operand, i) for i in range(lookahead) for operand in rings]
        consumed = []
        for k in range(k_blocks):
            slot = k % lookahead
            assert all(ring[slot] == k for ring in rings.values())
            consumed.extend((("k", k), ("q", k)))
            if k + lookahead < k_blocks:
                for operand, ring in rings.items():
                    ring[slot] = k + lookahead
                    loads.append((operand, k + lookahead))
        assert consumed == [(operand, k) for k in range(k_blocks) for operand in ("k", "q")]
        assert sorted(loads) == sorted((operand, k) for operand in rings for k in range(k_blocks))
        print(json.dumps(dict(check="abstract_ring_order", lookahead=lookahead,
                              loads=len(loads), gemm_calls=len(consumed), status="PASS")))


if __name__ == "__main__":
    main()
