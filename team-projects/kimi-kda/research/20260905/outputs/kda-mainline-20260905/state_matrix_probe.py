"""Supplement: separate state-in/out contracts and token-tail performance.

Uses the clean release wrapper helpers. Results are not pooled with Job19901.
"""
import argparse
import hashlib
from pathlib import Path
import random

import torch
from release_probe import (baseline_raw, release_raw, load_wrapper, make_case,
                           prepare, dispatch, compare, timed, emit)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20260905)
    old = load_wrapper('kda_state_old', baseline_raw, args.source)
    new = load_wrapper('kda_state_new', release_raw, args.source)
    emit('state_environment', device=release_raw.get_device_characteristics(),
         candidate_sha256=hashlib.sha256(Path(release_raw.__file__).read_bytes()).hexdigest(),
         scope='separate state-contract supplement; true wrapper; CUDA events; no pooled jobs')
    descriptions = [dict(tokens=t, state_mode=state, lengths=[t] if packed else None)
                    for t in (2048,4096,8192) for state in ('both','in','out','none')
                    for packed in (False,True)]
    descriptions += [dict(tokens=t) for t in (2049,4095,8191)]
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
            checks = compare(value[1], runs['release_v128'][1])
            emit('state_correctness', case=index, shape=data[0], name=name, tensors=checks, status='PASS')
        evict = torch.empty(256*1024*1024, dtype=torch.uint8, device='cuda')
        for repeat in range(3):
            names = list(runs)
            random.Random(19902+31*index+repeat).shuffle(names)
            for name in names:
                run, buffers, force, graph, decision = runs[name]
                with dispatch(force):
                    eager = timed(run, 60)
                    captured = timed(graph.replay, 60)
                    cold = timed(run, 30, evict)
                emit('state_performance', case=index, shape=data[0], repeat=repeat, name=name,
                     decision=decision, eager=eager, graph=captured, cache_perturbed=cold)
        emit('state_shape_complete', case=index, shape=data[0])
        del evict
    emit('state_matrix_complete', shapes=len(descriptions), correctness_rows=3*len(descriptions), performance_rows=9*len(descriptions))


if __name__ == '__main__':
    main()
