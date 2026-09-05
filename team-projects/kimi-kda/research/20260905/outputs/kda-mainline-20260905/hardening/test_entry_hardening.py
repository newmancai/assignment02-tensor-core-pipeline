"""Targeted GPU entry tests; run only against a freshly built hardened extension.

Every case is isolated in a child process so an old TMA assertion is a FAIL,
not a lost test run. This script does not submit GPU jobs or build extensions.
"""

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys


CASES = ("binding", "alignment", "parity", "stream_graph", "cpu_rejection", "multi_gpu")


def emit(**fields):
    print(json.dumps(fields, sort_keys=True), flush=True)


def inputs(torch, *, tokens=17, heads=2, batch=1, device="cuda:0", offset_beta=False):
    shape = (batch, tokens, heads, 128)
    result = {name: torch.randn(shape, dtype=torch.bfloat16, device=device)
              for name in ("q", "k", "v", "g")}
    count = batch * tokens * heads
    if offset_beta:
        backing = torch.randn(count + 1, dtype=torch.bfloat16, device=device)
        result["beta"] = backing[1:].reshape(batch, tokens, heads)
        assert result["beta"].is_contiguous()
        assert result["beta"].data_ptr() % 16 == 2
    else:
        result["beta"] = torch.randn(shape[:-1], dtype=torch.bfloat16, device=device)
    result["A_log"] = torch.zeros(heads, dtype=torch.float32, device=device)
    result["dt_bias"] = torch.zeros(heads, 128, dtype=torch.float32, device=device)
    return result


def call_args(torch, extension, data, *, value_slice=128, state_dtype=None, cu=None,
              initial=True, final=True):
    q = data["q"]
    batch, tokens, heads, _ = q.shape
    sequences = cu.numel() - 1 if cu is not None else batch
    state_dtype = torch.bfloat16 if state_dtype is None else state_dtype
    state = torch.full((sequences, heads, 128, 128), 0.125,
                       dtype=state_dtype, device=q.device)
    return dict(
        **data, scale=1 / math.sqrt(128), lower_bound=-5.0,
        out=torch.empty_like(data["v"]),
        workspace=torch.empty(extension.get_workspace_size(batch * tokens, heads, sequences),
                              dtype=torch.uint8, device=q.device),
        initial_state=state if initial else None,
        final_state=torch.empty_like(state) if final else None,
        cu_seqlens=cu, k2_value_slice=value_slice,
    )


def execute(extension, call):
    extension.fwd(**call)
    return call["out"], call["final_state"]


def equal(torch, actual, expected):
    for name, lhs, rhs in zip(("out", "final_state"), actual, expected):
        if rhs is None:
            assert lhs is None, name
        else:
            assert torch.isfinite(lhs).all().item(), name + " not finite"
            assert torch.equal(lhs, rhs), name + " is not bitwise equal"


def binding_case(torch, extension, args):
    doc = extension.fwd.__doc__ or ""
    match = re.search(r"k2_value_slice[^=\n]*=\s*(\d+)", doc)
    assert match, "extension binding does not expose its slice default"
    assert int(match.group(1)) == args.expected_default, doc
    emit(check="binding_default", extension=args.extension, default=int(match.group(1)))
    if args.alias_extension:
        alias = importlib.import_module(args.alias_extension)
        alias_match = re.search(r"k2_value_slice[^=\n]*=\s*(\d+)", alias.fwd.__doc__ or "")
        assert alias_match and int(alias_match.group(1)) == 16, alias.fwd.__doc__
        emit(check="alias_default", extension=args.alias_extension, default=16,
             path=alias.__file__)
    else:
        emit(check="alias_default", status="SKIP", reason="no --alias-extension supplied")


def alignment_case(torch, extension, args):
    # T_total=1 and H=1 preserve the beta view through transpose.contiguous().
    # T=17,H=2 is a control where the transpose already copies the input.
    for tokens, heads in ((1, 2), (1, 12), (17, 1), (17, 2)):
        data = inputs(torch, tokens=tokens, heads=heads, offset_beta=True)
        aligned = {**data, "beta": data["beta"].clone()}
        for dtype in (torch.bfloat16, torch.float32):
            for value_slice in (128, 16, 32, 64):
                expected = execute(extension, call_args(torch, extension, aligned,
                                   value_slice=value_slice, state_dtype=dtype))
                actual = execute(extension, call_args(torch, extension, data,
                                 value_slice=value_slice, state_dtype=dtype))
                torch.cuda.synchronize(0)
                equal(torch, actual, expected)
                emit(check="offset_beta", tokens=tokens, heads=heads,
                     state_dtype=str(dtype), value_slice=value_slice, status="PASS")


def parity_case(torch, extension, args):
    for tokens, batch, packed in ((1, 1, False), (17, 2, False), (33, 1, True)):
        data = inputs(torch, tokens=tokens, batch=batch)
        cu = torch.tensor([0, 1, 17, 33], dtype=torch.int64, device="cuda:0") if packed else None
        for dtype in (torch.bfloat16, torch.float32):
            for initial, final in ((True, True), (False, True), (True, False), (False, False)):
                expected = execute(extension, call_args(torch, extension, data,
                                   state_dtype=dtype, cu=cu, initial=initial, final=final))
                for value_slice in (16, 32, 64):
                    actual = execute(extension, call_args(torch, extension, data,
                                     value_slice=value_slice, state_dtype=dtype, cu=cu,
                                     initial=initial, final=final))
                    torch.cuda.synchronize(0)
                    equal(torch, actual, expected)
                emit(check="v128_parity", tokens=tokens, batch=batch, packed=packed,
                     dtype=str(dtype), initial=initial, final=final, status="PASS")
    # This compares variants in the same fresh binary, not untouched upstream.


def stream_graph_case(torch, extension, args):
    data = inputs(torch, tokens=1, heads=2, offset_beta=True)
    aligned = {**data, "beta": data["beta"].clone()}
    expected = execute(extension, call_args(torch, extension, aligned, value_slice=16))
    default = torch.cuda.current_stream(0)
    side = torch.cuda.Stream(device=0)
    side.wait_stream(default)
    with torch.cuda.stream(side):
        actual = execute(extension, call_args(torch, extension, data, value_slice=16))
    default.wait_stream(side)
    equal(torch, actual, expected)
    # Keep all capture arguments/output buffers alive through replay.
    capture_call = call_args(torch, extension, data, value_slice=16)
    side.wait_stream(default)
    with torch.cuda.stream(side):
        for _ in range(3):
            execute(extension, capture_call)
    side.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        captured = execute(extension, capture_call)
    graph.replay()
    torch.cuda.synchronize(0)
    equal(torch, captured, expected)
    emit(check="offset_beta_nondefault_stream_and_graph", status="PASS")


def cpu_rejection_case(torch, extension, args):
    data = inputs(torch)
    cu = torch.tensor([0, 17], dtype=torch.int64, device="cuda:0")
    for field in ("A_log", "dt_bias", "initial_state", "final_state", "cu_seqlens"):
        call = call_args(torch, extension, data, cu=cu)
        call[field] = call[field].cpu()
        try:
            execute(extension, call)
        except RuntimeError as error:
            assert field + " must be on cuda:0" in str(error), str(error)
        else:
            raise AssertionError("CPU " + field + " was accepted")
        assert torch.cuda.current_device() == 0
        emit(check="same_device_rejection", field=field, status="PASS")


def multi_gpu_case(torch, extension, args):
    if torch.cuda.device_count() < 2:
        if args.require_two_gpus:
            raise AssertionError("two CUDA devices were required")
        emit(check="multi_gpu", status="SKIP", reason="only one CUDA device visible")
        return "SKIP"
    if torch.cuda.get_device_capability(0) != torch.cuda.get_device_capability(1):
        raise RuntimeError("use two matching B300 devices for this binary-specific test")
    with torch.cuda.device(1):
        data = inputs(torch, tokens=1, heads=2, device="cuda:1", offset_beta=True)
        reference_call = call_args(torch, extension, data, value_slice=16)
        expected = execute(extension, reference_call)
        torch.cuda.synchronize(1)
        call = call_args(torch, extension, data, value_slice=16)
    torch.cuda.set_device(0)
    actual = execute(extension, call)
    assert torch.cuda.current_device() == 0, "guard did not restore caller device"
    torch.cuda.synchronize(1)
    equal(torch, actual, expected)
    emit(check="q_device1_caller_device0", status="PASS")
    # Check every raw pointer: the error must be raised before descriptor setup.
    cu = torch.tensor([0, 1], dtype=torch.int64, device="cuda:1")
    for field in ("k", "v", "g", "beta", "out", "workspace", "A_log", "dt_bias",
                  "initial_state", "final_state", "cu_seqlens"):
        mixed = call_args(torch, extension, data, cu=cu)
        mixed[field] = mixed[field].to("cuda:0")
        try:
            execute(extension, mixed)
        except RuntimeError as error:
            assert field + " must be on cuda:1" in str(error), str(error)
        else:
            raise AssertionError("mixed-device " + field + " was accepted")
        assert torch.cuda.current_device() == 0
        emit(check="mixed_gpu_rejection", field=field, status="PASS")
    # This exception occurs after the guard has been constructed.
    invalid = call_args(torch, extension, data)
    invalid["k"] = torch.empty((1, 2, 2, 128), dtype=torch.bfloat16, device="cuda:1")
    try:
        execute(extension, invalid)
    except RuntimeError as error:
        assert "k must match q shape" in str(error), str(error)
    else:
        raise AssertionError("invalid k shape accepted")
    assert torch.cuda.current_device() == 0, "validation exception leaked the guard device"
    emit(check="device_restored_on_exception", status="PASS")


def child(args):
    import torch
    extension = importlib.import_module(args.extension)
    path = Path(extension.__file__).resolve()
    emit(check="loaded_binary", extension=args.extension, path=str(path),
         sha256=hashlib.sha256(path.read_bytes()).hexdigest(), torch=torch.__version__)
    if args.case != "binding":
        assert torch.cuda.is_available(), "CUDA device required"
        torch.cuda.set_device(0)
        torch.manual_seed(20260905)
    handlers = dict(binding=binding_case, alignment=alignment_case, parity=parity_case,
                    stream_graph=stream_graph_case, cpu_rejection=cpu_rejection_case,
                    multi_gpu=multi_gpu_case)
    with torch.inference_mode():
        outcome = handlers[args.case](torch, extension, args)
    emit(case=args.case, status="SKIP" if outcome == "SKIP" else "PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", default="flash_kda_C")
    parser.add_argument("--alias-extension")
    parser.add_argument("--expected-default", type=int, default=128)
    parser.add_argument("--require-two-gpus", action="store_true")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case:
        child(args)
        return
    failures = []
    skipped = []
    for case in args.cases.split(","):
        if case not in CASES:
            raise ValueError("unknown case " + case)
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--extension", args.extension, "--expected-default", str(args.expected_default),
                   "--case", case]
        if args.alias_extension:
            command += ["--alias-extension", args.alias_extension]
        if args.require_two_gpus:
            command += ["--require-two-gpus"]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            failures.append(case)
            emit(case=case, status="FAIL", reason="child exceeded 180 seconds")
            continue
        print(result.stdout, end="", flush=True)
        if result.returncode:
            failures.append(case)
            emit(case=case, status="FAIL", returncode=result.returncode, stderr=result.stderr)
        else:
            for line in result.stdout.splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("case") == case and row.get("status") == "SKIP":
                    skipped.append(case)
    emit(suite="entry_hardening", status="FAIL" if failures else "PASS",
         failures=failures, skipped=skipped)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
