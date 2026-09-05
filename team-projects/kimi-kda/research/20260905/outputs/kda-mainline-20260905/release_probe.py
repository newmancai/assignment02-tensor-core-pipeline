"""Clean release validation: unchanged real wrapper, guarded auto, no ablation IDs.

Eager CUDA-event measurements include wrapper dispatch/workspace allocation calls.
Graph replay measures captured GPU work, not Python overhead. Cache perturbation
zeros a separate 256 MiB tensor before each timed event (excluded from latency).
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

import torch
import flash_kda_C as baseline_raw
import flash_kda_release_C as release_raw


def emit(kind, **data):
    print(json.dumps(dict(kind=kind, **data), sort_keys=True), flush=True)


def load_wrapper(name, raw, source):
    spec = importlib.util.spec_from_file_location(
        name, source / 'flash_kda' / '__init__.py',
        submodule_search_locations=[str(source / 'flash_kda')])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.modules['flash_kda_C']
    try:
        sys.modules['flash_kda_C'] = raw
        spec.loader.exec_module(module)
    finally:
        sys.modules['flash_kda_C'] = previous
    return module


@contextmanager
def dispatch(force=None, off=False):
    names = ('FLASH_KDA_K2_VALUE_SLICE', 'FLASH_KDA_K2_DISPATCH')
    previous = {key: os.environ.get(key) for key in names}
    try:
        for key in names:
            os.environ.pop(key, None)
        if force is not None:
            os.environ[names[0]] = str(force)
        if off:
            os.environ[names[1]] = 'off'
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def make_case(tokens, heads=12, batch=1, lengths=None, fp32=False,
              state_mode='both', gate=None):
    shape = (batch, tokens, heads, 128)
    q, k, v, g = [torch.randn(shape, dtype=torch.bfloat16, device='cuda') for _ in range(4)]
    if gate is not None:
        g.fill_(gate)
    beta = torch.randn(shape[:-1], dtype=torch.bfloat16, device='cuda')
    cu = None
    if lengths is not None:
        assert batch == 1 and sum(lengths) == tokens
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu = torch.tensor(offsets, dtype=torch.int64, device='cuda')
    n = len(lengths) if lengths is not None else batch
    initial = torch.randn((n, heads, 128, 128), device='cuda',
                          dtype=torch.float32 if fp32 else torch.bfloat16) * .125
    inputs = dict(q=q, k=k, v=v, g=g, beta=beta, scale=1/math.sqrt(128),
                  A_log=torch.zeros(heads, device='cuda'),
                  dt_bias=torch.zeros(heads, 128, device='cuda'), lower_bound=-5.,
                  initial_state=initial if state_mode in ('both', 'in') else None,
                  cu_seqlens=cu)
    meta = dict(tokens=tokens, heads=heads, batch=batch, lengths=lengths,
                fp32=fp32, state_mode=state_mode, gate=gate)
    return meta, inputs, initial


def prepare(data, wrapper):
    meta, inputs, initial = data
    buffers = dict(inputs, out=torch.empty_like(inputs['v']),
                   final_state=torch.empty_like(initial) if meta['state_mode'] in ('both', 'out') else None)
    return lambda: wrapper.fwd(**buffers), buffers


def compare(actual, expected):
    checks = {}
    for field in ('out', 'final_state'):
        x, y = actual[field], expected[field]
        if y is None:
            assert x is None
            continue
        checks[field] = dict(bitwise=bool(torch.equal(x, y)), finite=bool(torch.isfinite(x).all()))
        assert all(checks[field].values()), checks
    return checks


def correctness(old, new, sanitizer=False):
    if sanitizer:
        descriptions = [dict(tokens=2049), dict(tokens=2048, state_mode='out', lengths=[2048])]
    else:
        descriptions = [dict(tokens=t) for t in (1, 17, 2047, 2048, 2049, 3072, 4095, 4096, 4097, 6144, 8191, 8192, 8193, 16384)]
        descriptions += [dict(tokens=t, lengths=[t] if packed else None, state_mode=s)
                         for t in (2048, 8192) for packed in (False, True)
                         for s in ('in', 'out', 'none')]
        descriptions += [dict(tokens=4096, fp32=True, state_mode=s) for s in ('both', 'in', 'out')]
        descriptions += [dict(tokens=8192, heads=h) for h in (24, 48, 96)]
        descriptions += [dict(tokens=8192, batch=b) for b in (2, 4)]
        descriptions += [dict(tokens=8192, lengths=lengths) for lengths in
                         ([8192], [1024]*8, [16, 32, 512, 1024, 2512, 4096], [0,8192])]
        descriptions += [dict(tokens=8192, gate=g) for g in (-8., 12.)]
    rows = 0
    for index, description in enumerate(descriptions):
        data = make_case(**description)
        ref_run, ref = prepare(data, old)
        with dispatch(force=128):
            ref_run()
        run, actual = prepare(data, new)
        # Auto is the production entry. Force16 verifies Prefetch1 fallback and
        # every guarded specialization even where the cost model picks another slice.
        for label, force, off in (('auto',None,False), ('force16',16,False), ('off',None,True)):
            with dispatch(force, off):
                decision = new.explain_k2_dispatch(actual['q'], actual['initial_state'], actual['final_state'], actual['cu_seqlens'])
                run()
                torch.cuda.synchronize()
                checks = compare(actual, ref)
            emit('correctness', case=index, shape=data[0], mode=label, decision=decision, tensors=checks, status='PASS')
            rows += 1
    if not sanitizer:
        # Carry recurrent state through actual wrapper calls, without aliasing
        # state input/output. Both implementations see identical fresh inputs.
        old_state = new_state = None
        for step in range(3):
            data = make_case(2048)
            old_run, ref = prepare(data, old)
            new_run, actual = prepare(data, new)
            if step:
                ref['initial_state'], actual['initial_state'] = old_state, new_state
            with dispatch():
                old_run()
                new_run()
                checks = compare(actual, ref)
            old_state, new_state = ref['final_state'], actual['final_state']
            emit('state_chain', step=step, tensors=checks, status='PASS')
    emit('correctness_complete', comparison_rows=rows, sanitizer=sanitizer)


def timed(run, count, evict=None):
    for _ in range(8):
        run()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    for start, stop in zip(starts, stops):
        if evict is not None:
            evict.zero_()
        start.record()
        run()
        stop.record()
    torch.cuda.synchronize()
    values = sorted(start.elapsed_time(stop) for start, stop in zip(starts, stops))
    return dict(median_ms=statistics.median(values), p10_ms=values[count//10],
                p90_ms=values[min(count-1, count*9//10)], count=count)


def performance(old, new, count, repeats):
    descriptions = [dict(tokens=t) for t in (2048,3072,4096,6144,8192,16384)]
    descriptions += [dict(tokens=8192, lengths=[8192]), dict(tokens=8192, batch=2),
                     dict(tokens=8192, lengths=[1024]*8), dict(tokens=4096, fp32=True),
                     dict(tokens=8192, state_mode='out')]
    for index, description in enumerate(descriptions):
        data = make_case(**description)
        runs = {}
        for name, wrapper, force in (('baseline_auto',old,None), ('release_auto',new,None), ('release_v128',new,128)):
            run, buffers = prepare(data, wrapper)
            with dispatch(force):
                decision = wrapper.explain_k2_dispatch(buffers['q'], buffers['initial_state'], buffers['final_state'], buffers['cu_seqlens'])
                for _ in range(3):
                    run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                graph.replay()
            runs[name] = (run, buffers, force, graph, decision)
        for name, value in runs.items():
            compare(value[1], runs['baseline_auto'][1])
        evict = torch.empty(256*1024*1024, dtype=torch.uint8, device='cuda')
        for repeat in range(repeats):
            names = list(runs)
            random.Random(20260905+index*31+repeat).shuffle(names)
            for name in names:
                run, buffers, force, graph, decision = runs[name]
                with dispatch(force):
                    eager = timed(run, count)
                    captured = timed(graph.replay, count)
                    cold = timed(run, max(20,count//2), evict)
                emit('performance', case=index, shape=data[0], repeat=repeat, name=name,
                     decision=decision, eager=eager, graph=captured, cache_perturbed=cold)
        emit('shape_complete', case=index, shape=data[0])
        del evict


def concurrent(old, new, count):
    # A two-stream stress envelope, not a serving throughput claim. A single
    # joined interval times both full requests; no mixing with single-call data.
    data = [make_case(8192), make_case(8192)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    prepared = {}
    for name, wrapper in (('baseline_auto',old), ('release_auto',new)):
        work = [prepare(item, wrapper) for item in data]
        prepared[name] = work
    torch.cuda.synchronize()
    for repeat in range(3):
        names = list(prepared)
        random.Random(780+repeat).shuffle(names)
        for name in names:
            work = prepared[name]
            def run_pair():
                parent = torch.cuda.current_stream()
                for stream, (run, _) in zip(streams, work):
                    stream.wait_stream(parent)
                    with torch.cuda.stream(stream):
                        run()
                for stream in streams:
                    parent.wait_stream(stream)
            with dispatch():
                result = timed(run_pair, count)
            emit('concurrent', name=name, repeat=repeat, requests=2, shape=data[0][0], pair=result)
    for i in range(2):
        emit('concurrent_correctness', case=i, tensors=compare(prepared['release_auto'][i][1], prepared['baseline_auto'][i][1]), status='PASS')


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--count', type=int, default=60)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--sanitizer', action='store_true')
    parser.add_argument('--profile', choices=('baseline','release'))
    args = parser.parse_args()
    torch.manual_seed(20260905)
    old = load_wrapper('kda_baseline_wrapper', baseline_raw, args.source)
    new = load_wrapper('kda_release_wrapper', release_raw, args.source)
    emit('environment', torch=torch.__version__, device=release_raw.get_device_characteristics(),
         extension=release_raw.__file__, baseline=baseline_raw.__file__,
         candidate_sha256=hashlib.sha256(Path(release_raw.__file__).read_bytes()).hexdigest(),
         scope='actual Python wrapper; eager includes dispatch and workspace allocation; graph replay excludes Python; cache perturbation before start event')
    if args.profile:
        data = make_case(8192)
        wrapper = old if args.profile == 'baseline' else new
        run, buffers = prepare(data, wrapper)
        with dispatch():
            run()
            torch.cuda.synchronize()
        emit('profile_complete', mode=args.profile)
        return
    correctness(old, new, args.sanitizer)
    if not args.sanitizer:
        performance(old, new, args.count, args.repeats)
        concurrent(old, new, max(20,args.count//2))
    emit('complete', sanitizer=args.sanitizer)


if __name__ == '__main__':
    main()
