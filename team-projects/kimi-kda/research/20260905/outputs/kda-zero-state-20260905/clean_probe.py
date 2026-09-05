"""Five-patch candidate: real wrappers, no experimental selector/proxy.

Reference is the separate four-patch P4 binary; V128 is a same-job fallback
control. Event, graph and synchronized host-wall scopes are kept separate.
"""
import argparse
import hashlib
from pathlib import Path
import random
import statistics
import time

import torch
import flash_kda_phase1_C as candidate
from release_probe import (release_raw, load_wrapper, make_case, prepare,
                           dispatch, compare, timed, emit, correctness)


def wall_sync(run, count=20):
    values = []
    for _ in range(count):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        run()
        torch.cuda.synchronize()
        values.append((time.perf_counter_ns() - start) / 1e6)
    values.sort()
    return dict(median_ms=statistics.median(values), p10_ms=values[count//10],
                p90_ms=values[count*9//10], count=count)


def extra_correctness(old, new, sanitizer=False):
    descriptions = [dict(tokens=t, state_mode=s, lengths=[t] if packed else None)
                    for t, s, packed in ((2049, 'out', False), (4095, 'out', False),
                        (8191, 'out', False), (2049, 'none', False),
                        (2049, 'out', True), (4095, 'both', True),
                        (8191, 'in', False))]
    for index, description in enumerate(descriptions):
        data = make_case(**description)
        ref_run, ref = prepare(data, old)
        with dispatch(force=128):
            ref_run()
        run, actual = prepare(data, new)
        for label, force in (('auto', None), ('force16', 16)):
            with dispatch(force=force):
                run()
                torch.cuda.synchronize()
                checks = compare(actual, ref)
            emit('extra_correctness', case=index, shape=data[0], mode=label,
                 tensors=checks, status='PASS', sanitizer=sanitizer)


def first_prefill_chain(old, new):
    old_state = new_state = None
    for step in range(3):
        data = make_case(2048, state_mode='out')
        old_run, ref = prepare(data, old)
        new_run, actual = prepare(data, new)
        ref['initial_state'], actual['initial_state'] = old_state, new_state
        with dispatch():
            old_run()
            new_run()
            torch.cuda.synchronize()
            checks = compare(actual, ref)
        emit('first_prefill_chain', step=step, initial_present=step > 0,
             tensors=checks, status='PASS')
        old_state, new_state = ref['final_state'], actual['final_state']


def performance(old, new):
    descriptions = [dict(tokens=t, state_mode=state, lengths=[t] if packed else None)
                    for t in (2048, 4096, 8192)
                    for state in ('both', 'in', 'out', 'none')
                    for packed in (False, True)]
    descriptions += [dict(tokens=t, state_mode=s)
                     for t in (2049, 4095, 8191) for s in ('both', 'out')]
    descriptions += [dict(tokens=t, state_mode=s)
                     for t in (3072, 6144) for s in ('both', 'out')]
    # Out-of-envelope controls must stay in the acceptance data, not be pooled
    # into the optimized-domain gain.
    descriptions += [dict(tokens=2047, state_mode='out'),
                     dict(tokens=8193, state_mode='out'),
                     dict(tokens=4096, fp32=True),
                     dict(tokens=8192, batch=2),
                     dict(tokens=8192, heads=24),
                     dict(tokens=8192, lengths=[1024]*8)]
    for case_id, description in enumerate(descriptions):
        data = make_case(**description)
        runs = {}
        for name, wrapper, force in (('p4_auto', old, None), ('phase1_auto', new, None),
                                     ('v128', new, 128)):
            run, buffers = prepare(data, wrapper)
            with dispatch(force=force):
                decision = wrapper.explain_k2_dispatch(buffers['q'], buffers['initial_state'],
                                                       buffers['final_state'], buffers['cu_seqlens'])
                for _ in range(3):
                    run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                graph.replay()
            runs[name] = (run, buffers, force, graph, decision)
        for name, item in runs.items():
            emit('graph_correctness', case=case_id, shape=data[0], name=name,
                 tensors=compare(item[1], runs['v128'][1]), status='PASS')
        evict = torch.empty(256*1024*1024, dtype=torch.uint8, device='cuda')
        for repeat in range(3):
            names = list(runs)
            random.Random(19925+case_id*31+repeat).shuffle(names)
            for name in names:
                run, buffers, force, graph, decision = runs[name]
                with dispatch(force=force):
                    eager = timed(run, 60)
                    replay = timed(graph.replay, 60)
                    cold = timed(run, 30, evict)
                    wall = wall_sync(run)
                emit('performance', case=case_id, shape=data[0], name=name, repeat=repeat,
                     decision=decision, eager=eager, graph=replay,
                     cache_perturbed=cold, wall_sync=wall)
        for name in ('p4_auto', 'phase1_auto'):
            emit('post_correctness', case=case_id, shape=data[0], name=name,
                 tensors=compare(runs[name][1], runs['v128'][1]), status='PASS')
        emit('shape_complete', case=case_id, shape=data[0])
        del evict
    emit('performance_complete', shapes=len(descriptions), rows=9*len(descriptions))


def concurrent(old, new):
    for state_mode in ('both', 'out'):
        data = [make_case(8192, state_mode=state_mode) for _ in range(2)]
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        prepared = {name: [prepare(item, wrapper) for item in data]
                    for name, wrapper in (('p4_auto', old), ('phase1_auto', new))}
        torch.cuda.synchronize()
        for repeat in range(3):
            names = list(prepared)
            random.Random(19926+repeat).shuffle(names)
            for name in names:
                def pair():
                    parent = torch.cuda.current_stream()
                    for stream, (run, _) in zip(streams, prepared[name]):
                        stream.wait_stream(parent)
                        with torch.cuda.stream(stream):
                            run()
                    for stream in streams:
                        parent.wait_stream(stream)
                with dispatch():
                    result = timed(pair, 30)
                emit('concurrent', state_mode=state_mode, name=name, repeat=repeat,
                     requests=2, pair=result)
        for index in range(2):
            emit('concurrent_correctness', state_mode=state_mode, case=index,
                 tensors=compare(prepared['phase1_auto'][index][1],
                                 prepared['p4_auto'][index][1]), status='PASS')


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--sanitizer', action='store_true')
    parser.add_argument('--profile', choices=('p4', 'phase1'))
    parser.add_argument('--state-mode', choices=('out', 'both'), default='out')
    args = parser.parse_args()
    torch.manual_seed(20260908)
    old = load_wrapper('phase1_clean_old', release_raw, args.source)
    new = load_wrapper('phase1_clean_new', candidate, args.source)
    emit('environment', experiment='clean-phase1', torch=torch.__version__,
         device=candidate.get_device_characteristics(), candidate=candidate.__file__,
         candidate_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
         baseline=release_raw.__file__,
         baseline_sha256=hashlib.sha256(Path(release_raw.__file__).read_bytes()).hexdigest(),
         wrapper_sha256=hashlib.sha256((args.source/'flash_kda/__init__.py').read_bytes()).hexdigest(),
         scope='actual unchanged wrappers; no raw-ID proxies; event/graph/cache-perturbed/host-wall separate')
    if args.profile:
        data = make_case(8192, state_mode=args.state_mode)
        run, buffers = prepare(data, old if args.profile == 'p4' else new)
        with dispatch():
            run()
            torch.cuda.synchronize()
        emit('profile_complete', variant=args.profile, state_mode=args.state_mode)
        return
    correctness(old, new, args.sanitizer)
    extra_correctness(old, new, args.sanitizer)
    if not args.sanitizer:
        first_prefill_chain(old, new)
        performance(old, new)
        concurrent(old, new)
    emit('clean_complete', sanitizer=args.sanitizer)


if __name__ == '__main__':
    main()
