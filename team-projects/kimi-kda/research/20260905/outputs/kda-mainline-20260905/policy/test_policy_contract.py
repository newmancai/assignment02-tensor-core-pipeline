#!/usr/bin/env python3
"""CPU-only regression of the checked-in FlashKDA policy and rollback contract.

No torch import, CUDA context, extension load, network access, or repo edits.
dispatch.py is imported directly. Actual wrapper function ASTs run with explicit
metadata/extension stubs; this verifies host decisions, not GPU execution.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


POLICY_DIR = Path(__file__).resolve().parent
WORKSPACE = POLICY_DIR.parents[2]
SOURCE_DIR = WORKSPACE / "assignment02-github/team-projects/kimi-kda/experiments/final_campaign/implementation/current/flash_kda"
B300 = {
    "major": 10, "minor": 3, "sm_count": 148,
    "l2_bytes": 132_644_864, "shared_memory_per_sm": 233_472,
    "registers_per_sm": 65_536, "max_threads_per_sm": 2048,
    "max_blocks_per_sm": 32,
}
ENV_KEYS = ("FLASH_KDA_K2_VALUE_SLICE", "FLASH_KDA_K2_DISPATCH")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def load_dispatch(path):
    sys.dont_write_bytecode = True  # keep the inspected source tree untouched
    name = "flash_kda_policy_contract_subject"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves its defining module
    spec.loader.exec_module(module)
    return module


class MetadataOffsets:
    """Only numel is available; data access, conversion and copies are forbidden."""

    def __init__(self, numel):
        self._numel = numel
        self.numel_calls = 0

    def numel(self):
        self.numel_calls += 1
        return self._numel

    def __getattr__(self, name):
        raise AssertionError(f"cu_seqlens data access is forbidden: {name}")

    def __getitem__(self, index):
        raise AssertionError("cu_seqlens indexing is forbidden")

    def __iter__(self):
        raise AssertionError("cu_seqlens iteration is forbidden")


class NoMetadata:
    def __getattr__(self, name):
        raise AssertionError(f"override/off unexpectedly read metadata: {name}")


def tensor(batch=1, tokens=8192, heads=12, dtype="bfloat16", device=0):
    return SimpleNamespace(shape=(batch, tokens, heads, 128), dtype=dtype,
                           device=SimpleNamespace(index=device))


def make_wrapper(path, dispatch):
    calls = {"raw": [], "workspace": [], "empty": [], "device": []}

    def raw(*args, **kwargs):
        calls["raw"].append((args, kwargs))

    def get_workspace_size(*args):
        calls["workspace"].append(args)
        return 256

    def empty(*args, **kwargs):
        calls["empty"].append((args, kwargs))
        return "workspace_stub"

    def device(index):
        calls["device"].append(index)
        return nullcontext()

    namespace = {
        "os": os, "lru_cache": lru_cache,
        "torch": SimpleNamespace(float32="float32", uint8="uint8", empty=empty,
                                 cuda=SimpleNamespace(device=device)),
        "_fwd_raw": raw, "get_workspace_size": get_workspace_size,
        "get_device_characteristics": lambda: dict(B300),
        "select_k2_value_slice": dispatch.select_k2_value_slice,
    }
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    check({"_dispatch_decision", "explain_k2_dispatch", "fwd"}.issubset(
        node.name for node in functions), "expected wrapper functions are missing")
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--json-out", type=Path,
                        help="Optional generated evidence file; must stay in this policy directory")
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    if args.json_out:
        output_path = args.json_out.resolve()
        check(output_path.parent == POLICY_DIR, "evidence output must remain in the owned policy directory")
    dispatch_path = source_dir / "dispatch.py"
    wrapper_path = source_dir / "__init__.py"
    dispatch = load_dispatch(dispatch_path)
    wrapper, calls = make_wrapper(wrapper_path, dispatch)
    results = []

    def run(name, group, action):
        try:
            details = action()
            results.append({"name": name, "group": group, "status": "PASS", "details": details})
        except Exception as exc:
            results.append({"name": name, "group": group, "status": "FAIL",
                            "error": f"{type(exc).__name__}: {exc}"})

    def select(**changes):
        params = dict(batch=1, tokens_per_sequence=8192, heads=12,
                      state_fp32=False, is_varlen=False, device=dict(B300))
        params.update(changes)
        return dispatch.select_k2_value_slice(**params)

    def fallback(expected_reason, **changes):
        decision = select(**changes)
        check((decision.value_slice, decision.reason) == (128, expected_reason),
              f"expected V128/{expected_reason}, got {decision}")
        check(not decision.predicted_ms, "early fallback must not imply a calibrated score")
        return {"value_slice": decision.value_slice, "reason": decision.reason}

    def calibrated(tokens, fp32=False):
        decision = select(tokens_per_sequence=tokens, state_fp32=fp32)
        check(decision.value_slice == 16, f"H12 expected V16 at T{tokens}, got {decision}")
        check(decision.reason == "calibrated_score_with_guard_band", "missing calibrated selection")
        check(set(decision.predicted_ms) == {16, 32, 64, 128}, "expected all four candidates at H12")
        check(all(math.isfinite(x) and x > 0 for x in decision.predicted_ms.values()), "invalid prediction")
        return {"tokens": tokens, "state_fp32": fp32,
                "evidence_class": "calibration_anchor" if tokens in (2048, 4096, 8192) else "interpolated_held_out",
                "value_slice": decision.value_slice, "reason": decision.reason,
                "predicted_ms": dict(decision.predicted_ms)}

    for tokens in (2048, 4096, 8192, 3072, 6144):
        run(f"bf16_T{tokens}", "length", lambda t=tokens: calibrated(t))
    run("fp32_T4096", "length", lambda: calibrated(4096, True))
    for tokens in (0, 1, 2047, 8193, 16384):
        run(f"bf16_outside_T{tokens}", "length", lambda t=tokens: fallback(
            "recurrence_length_not_calibrated", tokens_per_sequence=t))
    for tokens in (2048, 3072, 4095, 4097, 6144, 8192):
        run(f"fp32_outside_T{tokens}", "length", lambda t=tokens: fallback(
            "recurrence_length_not_calibrated", tokens_per_sequence=t, state_fp32=True))

    def interpolation(midpoint, left, right):
        mid = select(tokens_per_sequence=midpoint).predicted_ms
        lo = select(tokens_per_sequence=left).predicted_ms
        hi = select(tokens_per_sequence=right).predicted_ms
        check(all(math.isclose(mid[v], (lo[v] + hi[v]) / 2, rel_tol=1e-13) for v in mid),
              "midpoint prediction is not the anchor midpoint")
        return {"midpoint": midpoint, "anchors": [left, right], "gpu_validated": False}

    run("interpolation_3072", "length", lambda: interpolation(3072, 2048, 4096))
    run("interpolation_6144", "length", lambda: interpolation(6144, 4096, 8192))
    for major, minor in ((10, 0), (10, 1), (9, 0), (12, 0)):
        run(f"cc_{major}_{minor}_fallback", "device", lambda a=major, b=minor: fallback(
            "architecture_not_calibrated", device={**B300, "major": a, "minor": b}))
    for sms in (147, 149):
        run(f"sm_{sms}_fallback", "device", lambda n=sms: fallback(
            "sm_topology_not_calibrated", device={**B300, "sm_count": n}))
    lower_l2 = math.ceil(B300["l2_bytes"] * 0.95)
    upper_l2 = math.floor(B300["l2_bytes"] * 1.05)

    def accepted_l2(size):
        decision = select(device={**B300, "l2_bytes": size})
        check(decision.value_slice == 16 and bool(decision.predicted_ms), "L2 boundary must be accepted")
        return {"l2_bytes": size, "value_slice": decision.value_slice}

    for size in (lower_l2, upper_l2):
        run(f"l2_inside_{size}", "device", lambda n=size: accepted_l2(n))
    for size in (0, lower_l2 - 1, upper_l2 + 1):
        run(f"l2_outside_{size}", "device", lambda n=size: fallback(
            "l2_capacity_not_calibrated", device={**B300, "l2_bytes": n}))
    run("baseline_resource_infeasible", "device", lambda: fallback(
        "official_variant_not_feasible", device={**B300, "shared_memory_per_sm": 100_000}))
    for batch, heads in ((0, 12), (1, 0), (1, 97), (9, 12), (-1, 12)):
        run(f"bh_outside_B{batch}_H{heads}", "domain", lambda b=batch, h=heads: fallback(
            "sequence_head_domain_exceeded", batch=b, heads=h))

    def inside_bh(batch, heads, expected_slice):
        decision = select(batch=batch, heads=heads)
        check(bool(decision.predicted_ms), "valid B*H was rejected before scoring")
        check(decision.value_slice == expected_slice, f"unexpected selected slice {decision}")
        return {"batch": batch, "heads": heads, "value_slice": decision.value_slice,
                "reason": decision.reason, "candidate_slices": sorted(decision.predicted_ms)}

    for batch, heads, value in ((1, 1, 16), (1, 12, 16), (2, 12, 32), (4, 12, 64), (8, 12, 128)):
        run(f"bh_inside_B{batch}_H{heads}", "domain", lambda b=batch, h=heads, v=value: inside_bh(b, h, v))

    def cta_limit(heads, candidate, expected_presence):
        decision = select(heads=heads)
        check((candidate in decision.predicted_ms) == expected_presence, "two CTA-layer bound changed")
        return {"sequence_heads": heads, "candidate": candidate, "candidate_scored": expected_presence}

    for heads, candidate, presence in ((37, 16, True), (38, 16, False), (74, 32, True), (75, 32, False)):
        run(f"cta_layers_H{heads}_V{candidate}", "domain", lambda h=heads, v=candidate, p=presence: cta_limit(h, v, p))
    run("varlen_direct_fallback", "domain", lambda: fallback("varlen_not_calibrated", is_varlen=True))
    run("varlen_precedes_architecture_guard", "domain", lambda: fallback(
        "varlen_not_calibrated", is_varlen=True, device={**B300, "major": 9, "minor": 0}))

    def wrapper_decision(env=None, q=None, initial=None, final=None, offsets=None):
        with patch.dict(os.environ, {k: v for k, v in os.environ.items() if k not in ENV_KEYS}, clear=True):
            os.environ.update(env or {})
            return wrapper["explain_k2_dispatch"](q or tensor(), initial, final, offsets)

    def override(value, mode="off"):
        decision = wrapper_decision({ENV_KEYS[0]: str(value), ENV_KEYS[1]: mode},
                                    q=NoMetadata(), initial=NoMetadata(), offsets=NoMetadata())
        check(decision == {"value_slice": value, "reason": "environment_override"}, "forced slice did not win")
        return decision

    for value in (16, 32, 64, 128):
        run(f"force_V{value}_wins_over_off", "wrapper_override", lambda v=value: override(v))
    for mode in ("off", "OFF", "0", "false", "FaLsE"):
        def disabled(m=mode):
            decision = wrapper_decision({ENV_KEYS[1]: m}, q=NoMetadata(), offsets=NoMetadata())
            check(decision == {"value_slice": 128, "reason": "dispatch_disabled"}, "off contract changed")
            return decision
        run(f"disabled_{mode}", "wrapper_override", disabled)
    for value in ("", "abc", "0", "15", "256"):
        def invalid(v=value):
            try:
                wrapper_decision({ENV_KEYS[0]: v, ENV_KEYS[1]: "off"}, q=NoMetadata())
            except ValueError:
                return {"forced_value": v, "raises": "ValueError", "off_does_not_suppress_invalid_override": True}
            raise AssertionError("invalid force should fail even when dispatch is off")
        run(f"invalid_override_{value!r}", "wrapper_override", invalid)

    def default_auto():
        decision = wrapper_decision()
        check(decision["value_slice"] == 16 and decision["reason"] == "calibrated_score_with_guard_band",
              "unset environment must use auto")
        return {"value_slice": decision["value_slice"], "reason": decision["reason"]}
    run("unset_environment_is_auto", "wrapper_override", default_auto)

    def packed(count):
        offsets = MetadataOffsets(count + 1)
        decision = wrapper_decision(offsets=offsets)
        check(offsets.numel_calls == 1, "dispatch must only require one numel metadata read")
        if count == 1:
            check(decision == wrapper_decision(), "packed single sequence differs from fixed B1")
        else:
            check((decision["value_slice"], decision["reason"]) == (128, "varlen_not_calibrated"),
                  "multi-sequence packed must fall back before scoring")
        return {"sequence_count": count, "numel_calls": offsets.numel_calls,
                "value_slice": decision["value_slice"], "reason": decision["reason"],
                "offset_values_checked": False}
    for count in (1, 2, 6, 32):
        run(f"packed_{count}_metadata_only", "wrapper_packed", lambda n=count: packed(n))

    def state_dtype(initial, final, tokens, expected_slice, expected_reason):
        decision = wrapper_decision(q=tensor(tokens=tokens), initial=initial, final=final)
        check((decision["value_slice"], decision["reason"]) == (expected_slice, expected_reason),
              "public state dtype dispatch changed")
        return {"tokens": tokens, "initial_dtype": getattr(initial, "dtype", None),
                "final_dtype": getattr(final, "dtype", None), "value_slice": expected_slice, "reason": expected_reason}
    bf16 = SimpleNamespace(dtype="bfloat16")
    fp32 = SimpleNamespace(dtype="float32")
    for name, initial, final, tokens, value, reason in (
        ("bf16_buffers", bf16, bf16, 8192, 16, "calibrated_score_with_guard_band"),
        ("fp32_initial_4096", fp32, None, 4096, 16, "calibrated_score_with_guard_band"),
        ("fp32_final_4096", None, fp32, 4096, 16, "calibrated_score_with_guard_band"),
        ("fp32_initial_8192", fp32, bf16, 8192, 128, "recurrence_length_not_calibrated"),
        ("fp32_final_8192", bf16, fp32, 8192, 128, "recurrence_length_not_calibrated"),
    ):
        run(name, "wrapper_dtype", lambda i=initial, f=final, t=tokens, v=value, r=reason: state_dtype(i, f, t, v, r))

    def forward_contract(env, expected_slice, offsets=None):
        q = tensor()
        initial = SimpleNamespace(dtype="bfloat16")
        final = SimpleNamespace(dtype="bfloat16")
        previous_calls = len(calls["raw"])
        with patch.dict(os.environ, {k: v for k, v in os.environ.items() if k not in ENV_KEYS}, clear=True):
            os.environ.update(env)
            wrapper["fwd"](q, "k", "v", "g", "beta", 0.125, "out", "A_log", "dt_bias", -5.0,
                           initial_state=initial, final_state=final, cu_seqlens=offsets)
        check(len(calls["raw"]) == previous_calls + 1, "forward should make exactly one extension call")
        raw_args, raw_kwargs = calls["raw"][-1]
        check(raw_kwargs["k2_value_slice"] == expected_slice, "forward did not transmit selected slice")
        check(raw_args[7] == "workspace_stub", "allocated workspace not passed through")
        check(raw_kwargs["initial_state"] is initial and raw_kwargs["final_state"] is final, "state buffers altered")
        check(raw_kwargs["cu_seqlens"] is offsets, "packed offsets identity altered")
        expected_n = offsets._numel - 1 if offsets else 1
        check(calls["workspace"][-1] == (8192, 12, expected_n), "workspace dimensions changed")
        return {"k2_value_slice": expected_slice, "extension_call_count": 1,
                "same_raw_entry": True, "gpu_execution": False}
    run("forward_force_V128_rollback", "forward_stub", lambda: forward_contract({ENV_KEYS[0]: "128"}, 128))
    run("forward_off_rollback", "forward_stub", lambda: forward_contract({ENV_KEYS[1]: "off"}, 128))
    run("forward_force_V16_over_off", "forward_stub", lambda: forward_contract({ENV_KEYS[0]: "16", ENV_KEYS[1]: "off"}, 16))
    run("forward_packed_offsets_preserved", "forward_stub", lambda: forward_contract(
        {ENV_KEYS[0]: "128"}, 128, MetadataOffsets(3)))
    run("forward_default_auto_V16", "forward_stub", lambda: forward_contract({}, 16))
    run("forward_packed_multi_auto_V128", "forward_stub", lambda: forward_contract({}, 128, MetadataOffsets(7)))
    run("torch_never_imported", "isolation", lambda: check("torch" not in sys.modules, "torch was imported"))

    failed = [case for case in results if case["status"] != "PASS"]
    report = {
        "status": "FAIL" if failed else "PASS",
        "tests": len(results), "passed": len(results) - len(failed), "failed": len(failed),
        "source": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (dispatch_path, wrapper_path)},
        "device_fixture": B300,
        "scope": "CPU policy + real wrapper AST under metadata/extension stubs; no torch, CUDA, .so or network",
        "not_validated": ["GPU tensor correctness", "GPU performance including held-out lengths",
                          "real device-property query", "CUDA stream/device handling", "TMA alignment",
                          "packed offset data validity", "loaded binary identity", "fresh release build"],
        "results": results,
    }
    if args.json_out:
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "tests", "passed", "failed", "source", "scope")},
                     ensure_ascii=False, indent=2))
    for failure in failed:
        print(json.dumps(failure, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
