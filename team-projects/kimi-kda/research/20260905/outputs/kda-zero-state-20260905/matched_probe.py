"""Same-input no-state vs explicit-zero controls, using existing binaries.

No-state/zero-reused/zero-created-per-call are mathematically equivalent and
must match out plus final_state bitwise. Nonzero state is timing-only control.
"""
import argparse
import hashlib
from pathlib import Path
import random

import torch
from release_probe import (baseline_raw, release_raw, load_wrapper, make_case,
                           dispatch, compare, timed, emit)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--profile', choices=('release_none','release_zero','legacy_none','legacy_zero'))
    args = parser.parse_args()
    torch.manual_seed(20260906)
    old = load_wrapper('matched_old', baseline_raw, args.source)
    new = load_wrapper('matched_new', release_raw, args.source)
    emit('environment', experiment='matched-zero-state', device=release_raw.get_device_characteristics(),
         binary_sha256=hashlib.sha256(Path(release_raw.__file__).read_bytes()).hexdigest(),
         scope='same input tensors within each case; actual wrapper; CUDA event intervals; per-call zero creation included')
    descriptions = [dict(tokens=t, gate=g) for t in (2048,4096,8192) for g in (None,-8.)]
    descriptions += [dict(tokens=8192,lengths=[8192]),dict(tokens=2049),dict(tokens=8191),dict(tokens=8192,state_mode='none')]
    if args.profile:
        descriptions = [dict(tokens=8192)]
    for case_id, description in enumerate(descriptions):
        meta, inputs, state = make_case(**description)
        zero = torch.zeros_like(state)
        torch.cuda.synchronize()
        runs = {}
        for name, wrapper, initial in (('release_none',new,None), ('release_zero',new,zero),
                                      ('release_zero_each',new,'create'), ('release_nonzero',new,state),
                                      ('legacy_none',old,None), ('legacy_zero',old,zero)):
            if args.profile and name != args.profile:
                continue
            buffers = dict(inputs, out=torch.empty_like(inputs['v']),
                           initial_state=zero if isinstance(initial,str) else initial,
                           final_state=None if meta['state_mode']=='none' else torch.empty_like(state))
            if isinstance(initial,str):
                def run(w=wrapper, b=buffers, prototype=state):
                    b['initial_state'] = torch.zeros_like(prototype)
                    w.fwd(**b)
            else:
                def run(w=wrapper, b=buffers):
                    w.fwd(**b)
            with dispatch():
                run()
                torch.cuda.synchronize()
            if args.profile:
                emit('profile_complete',mode=name)
                return
            with dispatch():
                for _ in range(3):
                    run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                graph.replay()
            runs[name] = (run,buffers,graph)
        for name, (run,buffers,graph) in runs.items():
            if name != 'release_nonzero':
                emit('correctness',case=case_id,shape=meta,name=name,status='PASS',
                     reference_sanity=name=='legacy_none',tensors=compare(buffers,runs['legacy_none'][1]))
        nonzero_reference = dict(runs['release_nonzero'][1],
                                 out=torch.empty_like(inputs['v']),
                                 final_state=None if meta['state_mode']=='none' else torch.empty_like(state))
        with dispatch(force=128):
            old.fwd(**nonzero_reference)
        emit('nonzero_correctness',case=case_id,status='PASS',
             tensors=compare(runs['release_nonzero'][1],nonzero_reference))
        # A future optimization cannot rely on the reused zero tensor remaining
        # hot: the same perturbation precedes each timed variant's interval.
        eviction = torch.empty(256*1024*1024, dtype=torch.uint8,device='cuda')
        for repeat in range(3):
            names = list(runs)
            random.Random(19905+31*case_id+repeat).shuffle(names)
            for name in names:
                run,buffers,graph = runs[name]
                with dispatch():
                    eager = timed(run,60)
                    replay = timed(graph.replay,60)
                    cold = timed(run,30,eviction)
                emit('performance',case=case_id,shape=meta,name=name,repeat=repeat,
                     eager=eager,graph=replay,cache_perturbed=cold)
        # Graph capture may replace the per-call zero buffer. Check the final
        # eager result too, so capture-only correctness cannot conceal a bug.
        for name, (run,buffers,graph) in runs.items():
            if name != 'release_nonzero':
                emit('post_timing_correctness',case=case_id,name=name,status='PASS',
                     tensors=compare(buffers,runs['legacy_none'][1]))
        emit('shape_complete',case=case_id,shape=meta)
    emit('matched_complete',shapes=len(descriptions))


if __name__=='__main__':
    main()
