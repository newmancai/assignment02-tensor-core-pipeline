"""True K1+K2 forward ablations; D128/C16 unchanged, workspace preallocated.

Experiment-only raw selector codes are NOT public model/value dimensions:
116 delayed output acquire; 216 state prefetch2; 316 both;
416 state prefetch4; 516 early non-final output publication.
"""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch
import flash_kda_C as legacy
import flash_kda_mainline_C as candidate

MODES = {128: 'v128', 64: 'v64', 32: 'v32', 16: 'v16',
         116: 'v16_delayed_acquire', 216: 'v16_prefetch2',
         316: 'v16_prefetch2_delayed', 416: 'v16_prefetch4',
         516: 'v16_early_publish'}


def emit(kind, **data):
    print(json.dumps(dict(kind=kind, **data), sort_keys=True), flush=True)


def case(tokens, heads=12, batch=1, lengths=None, fp32=False,
         state_mode='both', gate=None):
    shape = (batch, tokens, heads, 128)
    q, k, v, g = [torch.randn(shape, device='cuda', dtype=torch.bfloat16)
                  for _ in range(4)]
    if gate is not None:
        g.fill_(gate)
    beta = torch.randn(shape[:-1], device='cuda', dtype=torch.bfloat16)
    cu = None
    if lengths is not None:
        assert batch == 1 and sum(lengths) == tokens
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu = torch.tensor(offsets, device='cuda', dtype=torch.int64)
    nseq = len(lengths) if lengths is not None else batch
    dtype = torch.float32 if fp32 else torch.bfloat16
    initial = torch.randn((nseq, heads, 128, 128), device='cuda', dtype=dtype) * 0.125
    meta = dict(tokens=tokens, heads=heads, batch=batch, lengths=lengths,
                fp32=fp32, state_mode=state_mode, gate=gate)
    args = dict(q=q, k=k, v=v, g=g, beta=beta, scale=1/math.sqrt(128),
                A_log=torch.zeros(heads, device='cuda'),
                dt_bias=torch.zeros(heads, 128, device='cuda'), lower_bound=-5.,
                initial_state=initial if state_mode in ('both', 'in') else None,
                cu_seqlens=cu)
    return meta, args, initial


def prepare(data, module, mode):
    meta, inputs, initial = data
    q = inputs['q']
    nseq = initial.shape[0]
    args = dict(inputs, out=torch.empty_like(inputs['v']),
                final_state=torch.empty_like(initial) if meta['state_mode'] in ('both', 'out') else None,
                workspace=torch.empty(module.get_workspace_size(q.shape[0]*q.shape[1], q.shape[2], nseq),
                                      device='cuda', dtype=torch.uint8),
                k2_value_slice=mode)
    return lambda: module.fwd(**args), args


def compare(actual, expected):
    result = {}
    for name in ('out', 'final_state'):
        x, ref = actual[name], expected[name]
        if ref is None:
            assert x is None
            continue
        same = bool(torch.equal(x, ref))
        finite = bool(torch.isfinite(x).all())
        result[name] = dict(bitwise=same, finite=finite)
        if not same:
            xf, rf = x.float(), ref.float()
            result[name]['rel_rmse'] = float((xf-rf).square().mean().sqrt()/rf.square().mean().sqrt().clamp_min(1e-20))
        assert same and finite, result
    return result


def correctness():
    descriptions = [dict(tokens=t, heads=12) for t in (1, 15, 16, 17, 31, 32, 33, 127)]
    descriptions += [dict(tokens=33, heads=12, batch=2, fp32=fp, state_mode=s)
                     for fp in (False, True) for s in ('both', 'in', 'out', 'none')]
    descriptions += [dict(tokens=256, heads=12, lengths=[0, 1, 16, 16, 223], fp32=fp)
                     for fp in (False, True)]
    descriptions += [dict(tokens=t, heads=h, gate=g) for t,h,g in
                     ((3072,12,None), (6144,12,None), (8192,12,None), (8192,12,-8.), (8192,96,None))]
    descriptions += [dict(tokens=8192, lengths=[8192]),
                     dict(tokens=8192, lengths=[1024]*8)]
    count = 0
    for index, description in enumerate(descriptions):
        data = case(**description)
        reference_run, reference = prepare(data, legacy, 128)
        reference_run()
        for mode in MODES:
            run, actual = prepare(data, candidate, mode)
            run()
            torch.cuda.synchronize()
            details = compare(actual, reference)
            emit('correctness', case=index, shape=data[0], mode=mode,
                 name=MODES[mode], status='PASS', tensors=details)
            count += 1
    emit('correctness_complete', comparison_rows=count)


def timed(run, count):
    for _ in range(10):
        run()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    for start, stop in zip(starts, stops):
        start.record()
        run()
        stop.record()
    torch.cuda.synchronize()
    values = sorted(start.elapsed_time(stop) for start,stop in zip(starts,stops))
    return dict(median_ms=statistics.median(values), min_ms=values[0],
                p10_ms=values[count//10], p90_ms=values[min(count-1,count*9//10)],
                count=count)


def performance(args):
    descriptions = [dict(tokens=t) for t in (2048, 3072, 4096, 6144, 8192, 16384)]
    descriptions += [dict(tokens=8192, heads=h) for h in (24,48,96)]
    descriptions += [dict(tokens=8192, batch=b) for b in (2,4)]
    descriptions += [dict(tokens=8192, lengths=[8192]),
                     dict(tokens=8192, lengths=[16,32,512,1024,2512,4096]),
                     dict(tokens=8192, lengths=[1024]*8),
                     dict(tokens=8192, lengths=[256]*32)]
    if args.quick:
        descriptions = [dict(tokens=8192), dict(tokens=3072), dict(tokens=6144)]
    for index, description in enumerate(descriptions):
        data = case(**description)
        ref_run, reference = prepare(data, legacy, 128)
        ref_run()
        runs = {}
        for mode in MODES:
            run, buffers = prepare(data, candidate, mode)
            run()
            compare(buffers, reference)
            runs[MODES[mode]] = (run, buffers)
        for mode in (128,16):
            runs['legacy'+str(mode)] = prepare(data, legacy, mode)
        graphs = {}
        for name, (run, buffers) in runs.items():
            for _ in range(3):
                run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            graph.replay()
            compare(buffers, reference)
            graphs[name] = graph
        for repeat in range(args.repeats):
            names = list(runs)
            random.Random(20260905 + index*31 + repeat).shuffle(names)
            for name in names:
                eager = timed(runs[name][0], args.count)
                graph = timed(graphs[name].replay, args.count)
                emit('performance', case=index, shape=data[0], repeat=repeat, name=name,
                     eager=eager, graph=graph)
        emit('shape_complete', case=index, shape=data[0])


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=60)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--skip-correctness', action='store_true')
    args=parser.parse_args()
    torch.manual_seed(20260905)
    emit('environment', torch=torch.__version__, device=candidate.get_device_characteristics(),
         extension=candidate.__file__, legacy=legacy.__file__, modes=MODES,
         candidate_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
         scope='complete raw forward K1+K2 including beta transpose; reusable workspace, no Python dispatch')
    if not args.skip_correctness:
        correctness()
    performance(args)
    emit('complete')


if __name__ == '__main__':
    main()
