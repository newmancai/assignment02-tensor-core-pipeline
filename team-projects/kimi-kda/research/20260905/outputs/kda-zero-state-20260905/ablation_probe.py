"""Same-binary initializer ablations through the actual wrapper.

Only experiment proxies map an already selected/forced V16 to diagnostic raw
IDs. IDs are NOT dimensions or a public policy change. Each case shares inputs.
"""
import argparse
import hashlib
from pathlib import Path
import random
import statistics
import time
from types import SimpleNamespace

import torch
import flash_kda_zero_C as candidate
from release_probe import (release_raw, load_wrapper, make_case, prepare,
                           dispatch, compare, timed, emit)

STRATEGIES = {'base':16,'vector':10016,'onewarp':20016,'unified':30016}


def proxy(selector):
    def fwd(*args,**kwargs):
        if kwargs.get('k2_value_slice',128)==16:
            kwargs['k2_value_slice']=selector
        candidate.fwd(*args,**kwargs)
    return SimpleNamespace(fwd=fwd,get_device_characteristics=candidate.get_device_characteristics,
                           get_workspace_size=candidate.get_workspace_size)


def synchronized_wall(run,count=20):
    values=[]
    for _ in range(count):
        torch.cuda.synchronize()
        start=time.perf_counter_ns()
        run()
        torch.cuda.synchronize()
        values.append((time.perf_counter_ns()-start)/1e6)
    values.sort()
    return dict(median_ms=statistics.median(values),p10_ms=values[count//10],
                p90_ms=values[count*9//10],count=count)


def correctness(old,wrappers):
    descriptions=[dict(tokens=t,state_mode='out') for t in
                  (1,17,33,2047,2048,2049,4095,4096,4097,8191,8192,8193)]
    descriptions += [dict(tokens=t,state_mode=s) for t in (2048,8192) for s in ('both','in','none')]
    descriptions += [dict(tokens=8192,state_mode='out',gate=g) for g in (-8.,12.)]
    descriptions += [dict(tokens=t,lengths=[t],state_mode=s) for t in (2048,8192) for s in ('both','out','none')]
    descriptions += [dict(tokens=33,batch=2,state_mode='out'),
                     dict(tokens=2048,lengths=[0,1,16,2031],state_mode='out'),
                     dict(tokens=4096,fp32=True,state_mode='out'),
                     dict(tokens=4096,fp32=True,state_mode='both'),
                     dict(tokens=8192,heads=24,state_mode='out')]
    for case_id, description in enumerate(descriptions):
        data=make_case(**description)
        ref_run,ref=prepare(data,old)
        with dispatch(force=128):
            ref_run()
        for name,wrapper in wrappers.items():
            run,buffers=prepare(data,wrapper)
            with dispatch(force=16):
                run()
                torch.cuda.synchronize()
                checks=compare(buffers,ref)
            emit('correctness',case=case_id,shape=data[0],name=name,status='PASS',tensors=checks)
    emit('correctness_complete',shapes=len(descriptions),rows=len(descriptions)*len(wrappers))


def performance(old,wrappers,count=60):
    descriptions=[dict(tokens=t,state_mode=s) for t in (2048,4096,8192) for s in ('out','both')]
    descriptions += [dict(tokens=8192,state_mode='none'),dict(tokens=8192,lengths=[8192],state_mode='out'),
                     dict(tokens=2049,state_mode='out'),dict(tokens=4095,state_mode='out'),dict(tokens=8191,state_mode='out')]
    for case_id,description in enumerate(descriptions):
        data=make_case(**description)
        runs={}
        for name,wrapper in dict(legacy=old,**wrappers).items():
            run,buffers=prepare(data,wrapper)
            with dispatch():
                decision=wrapper.explain_k2_dispatch(buffers['q'],buffers['initial_state'],buffers['final_state'],buffers['cu_seqlens'])
                for _ in range(3):
                    run()
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                graph.replay()
            runs[name]=(run,buffers,graph,decision)
        for name,(_,buffers,_,_) in runs.items():
            emit('graph_correctness',case=case_id,name=name,status='PASS',tensors=compare(buffers,runs['legacy'][1]))
        evict=torch.empty(256*1024*1024,dtype=torch.uint8,device='cuda')
        for repeat in range(3):
            names=list(runs)
            random.Random(19920+case_id*31+repeat).shuffle(names)
            for name in names:
                run,buffers,graph,decision=runs[name]
                with dispatch():
                    eager=timed(run,count)
                    replay=timed(graph.replay,count)
                    cold=timed(run,count//2,evict)
                    wall=synchronized_wall(run)
                emit('performance',case=case_id,shape=data[0],name=name,repeat=repeat,decision=decision,
                     eager=eager,graph=replay,cache_perturbed=cold,wall_sync=wall)
        emit('shape_complete',case=case_id,shape=data[0])
    emit('performance_complete',shapes=len(descriptions),rows=3*(len(wrappers)+1)*len(descriptions))


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--phase1',action='store_true',help='Evaluate Phase1 ring2/ring4 instead of initializer variants')
    args=parser.parse_args()
    torch.manual_seed(20260907)
    old=load_wrapper('ablation_legacy',release_raw,args.source)
    strategies={'base':16,'phase1_ring2':40016,'phase1_ring4':50016} if args.phase1 else STRATEGIES
    wrappers={name:load_wrapper('ablation_'+name,proxy(selector),args.source)
              for name,selector in strategies.items()}
    emit('environment',experiment='zero-init-ablation',device=candidate.get_device_characteristics(),
         candidate=candidate.__file__,candidate_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
         legacy=release_raw.__file__,legacy_sha256=hashlib.sha256(Path(release_raw.__file__).read_bytes()).hexdigest(),
         strategies=strategies,scope='actual wrapper via experiment-only raw adapter; GPU events and separate per-call synchronized host wall')
    correctness(old,wrappers)
    performance(old,wrappers)
    emit('ablation_complete')


if __name__=='__main__':
    main()
