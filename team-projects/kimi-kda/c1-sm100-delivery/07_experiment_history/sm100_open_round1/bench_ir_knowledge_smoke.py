#!/usr/bin/env python3
"""Small, isolated-process check of archived HMMA advice; not equal-budget search."""
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path('<REMOTE_HOME>')
CASES = [('h12_fixed8192',12,[8192]), ('h12_packed8x1024',12,[1024]*8),
         ('h96_fixed8192',96,[8192])]
VARIANTS = ['official', 'hmma_auto', 'hmma_force16', 'cake']

def worker(a):
    import torch
    from flashinfer.testing import bench_gpu_time
    name, heads, lengths = CASES[a.case]
    total, sequences = sum(lengths), len(lengths)
    gen = torch.Generator(device='cuda').manual_seed(31000+a.case)
    def rand(shape,dtype=torch.bfloat16):
        return torch.randn(shape,generator=gen,device='cuda').to(dtype)
    shape = (1,total,heads,128)
    q,k,v,g = [rand(shape) for _ in range(4)]
    beta = rand((1,total,heads))
    alog = torch.rand(heads,generator=gen,device='cuda')
    bias = torch.rand((heads,128),generator=gen,device='cuda')
    initial = (rand((sequences,heads,128,128),torch.float32)*.25).to(torch.bfloat16)
    states = initial.unsqueeze(0).expand(64,*initial.shape).clone()
    cu = None
    if sequences > 1:
        offsets=[0]
        for length in lengths: offsets.append(offsets[-1]+length)
        cu=torch.tensor(offsets,dtype=torch.int64,device='cuda')
    out,final = torch.empty_like(q),torch.empty_like(initial)
    decision={}
    if a.variant == 'cake':
        from flashinfer.kda import recurrent_kda
        module_path=importlib.import_module('flashinfer.kda').__file__
        def invoke(state):
            recurrent_kda(q=q,k=k,v=v,g=g,beta=beta,A_log=alog,dt_bias=bias,
                scale=1/math.sqrt(128),initial_state=state,output=out,
                output_final_state=False,use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,lower_bound=-5.,cu_seqlens=cu,
                beta_is_logit=True,backend='cake')
    else:
        if a.variant == 'official':
            sys.path.insert(0,str(ROOT/'FlashKDA-c1-official'))
            module=importlib.import_module('flash_kda_C')
            extra={}
            decision={'value_slice':128,'reason':'official'}
        else:
            sys.path.insert(0,str(ROOT/'kda-zero-state-20260905/clean_build/lib'))
            module=importlib.import_module('flash_kda_phase1_C')
            # Only this extension is loaded; the archived Python wrapper imports its API.
            sys.modules['flash_kda_C']=module
            sys.path.insert(0,str(ROOT/'kda-zero-state-20260905/clean_source'))
            wrapper=importlib.import_module('flash_kda')
            os.environ.pop('FLASH_KDA_K2_VALUE_SLICE',None)
            os.environ.pop('FLASH_KDA_K2_DISPATCH',None)
            if a.variant=='hmma_force16': os.environ['FLASH_KDA_K2_VALUE_SLICE']='16'
            decision=wrapper.explain_k2_dispatch(q,initial,final,cu)
            extra={'k2_value_slice':decision['value_slice']}
            decision['phase_prefetch_guard_from_source']=bool(
                decision['value_slice']==16 and sequences==1 and heads==12 and 2048<=total<=8192)
        module_path=module.__file__
        workspace=torch.empty(module.get_workspace_size(total,heads,sequences),
                              dtype=torch.uint8,device='cuda')
        def invoke(state):
            module.fwd(q,k,v,g,beta,1/math.sqrt(128),out,workspace,alog,bias,-5.,
                       initial_state=state,final_state=final,cu_seqlens=cu,**extra)
            state.copy_(final)
    # Common peer test, before timing. HMMA must be bitwise; CAKE uses a declared
    # 1% relative-L2 and normalized-Linf screen, not a new independent oracle.
    probe_state=initial.clone()
    invoke(probe_state)
    torch.cuda.synchronize()
    actual={'output':out.cpu(),'state':probe_state.cpu()}
    refpath=a.output/f'{name}_official.pt'
    checks={}
    if a.variant=='official' and a.block==0:
        torch.save(actual,refpath)
    reference=torch.load(refpath,weights_only=True)
    for key,x in actual.items():
        ref=reference[key]
        delta=x.float()-ref.float()
        rel=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(ref.float()).clamp_min(1e-30))
        linf=float(delta.abs().max()/ref.float().abs().max().clamp_min(1e-30))
        finite=bool(torch.isfinite(x).all())
        exact=torch.equal(x,ref)
        strict=finite and rel<=.01 and linf<=.01
        historical=finite and bool(torch.isclose(x,ref,atol=.01,rtol=.01).all())
        passed=finite and (exact if a.variant!='cake' else
                          (historical if a.contract=='historical_peer' else strict))
        checks[key]={'finite':finite,'bitwise':exact,'relative_l2':rel,
                     'normalized_linf':linf,'max_abs':float(delta.abs().max()),
                     'strict_1pct_passed':strict,'historical_atol_rtol_1e2_passed':historical,
                     'passed':passed}
    if not all(x['passed'] for x in checks.values()):
        # Preserve the prespecified threshold. A failed candidate receives no
        # timing, but does not prevent independent cases from being evaluated.
        result={'case':name,'variant':a.variant,'block':a.block,'decision':decision,
            'module_path':module_path,'correctness':checks,'samples_ms':[],
            'median_ms':None,'status':'correctness_failed_not_timed'}
        (a.output/f'{name}_{a.variant}_{a.block}.json').write_text(json.dumps(result,indent=2))
        return
    del actual, reference
    cursor=0
    def run():
        nonlocal cursor
        if cursor>=len(states): raise RuntimeError('state rotation exhausted')
        state=states[cursor];cursor+=1
        invoke(state)
    samples=[float(x) for x in bench_gpu_time(run,enable_cupti=True,
        cold_l2_cache=True,use_cuda_graph=False,dry_run_iters=10,repeat_iters=40)]
    result={'case':name,'variant':a.variant,'block':a.block,'decision':decision,
            'module_path':module_path,'correctness':checks,'samples_ms':samples,
            'median_ms':statistics.median(samples),'calls':cursor,
            'device':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability()}
    (a.output/f'{name}_{a.variant}_{a.block}.json').write_text(json.dumps(result,indent=2))

def controller(a):
    a.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for case,(name,_,_) in enumerate(CASES):
        for block in range(4):
            order=VARIANTS[block:]+VARIANTS[:block]
            for variant in order:
                stem=f'{name}_{variant}_{block}'
                command=[sys.executable,__file__,'--worker','--contract',a.contract,'--case',str(case),
                         '--variant',variant,'--block',str(block),'--output',str(a.output)]
                with (a.output/f'{stem}.log').open('w') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                result=json.loads((a.output/f'{stem}.json').read_text())
                results.append(result)
                print(stem,result['median_ms'],result['decision'],
                      result.get('status','measured'),flush=True)
    summary=[]
    for name,_,_ in CASES:
        medians={}
        for v in VARIANTS:
            values=[r['median_ms'] for r in results if r['case']==name and r['variant']==v]
            medians[v]=statistics.median(values) if all(x is not None for x in values) else None
        def ratio(left,right):
            return medians[left]/medians[right] if medians[left] is not None and medians[right] is not None else None
        summary.append({'case':name,'median_ms':medians,
            'official_over_auto':ratio('official','hmma_auto'),
            'auto_over_cake':ratio('hmma_auto','cake'),
            'auto_over_force16':ratio('hmma_auto','hmma_force16')})
    payload={'scope':'exploratory public-full preallocated; per-call transforms and state copy included',
        'correctness_contract':a.contract,
        'job_id':os.getenv('SLURM_JOB_ID'),'blocks':4,'warmups':10,'samples_per_block':40,
        'policy':'one implementation per worker; cold-L2 CUPTI; balanced cyclic order',
        'limitations':['not equal-budget optimization','three previously known profiles',
                      'nonzero BF16 initial state only','peer oracle, not FP64 accuracy study'],
        'summary':summary,'results':results}
    (a.output/'summary.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--worker',action='store_true')
    p.add_argument('--contract',choices=['strict','historical_peer'],default='strict')
    p.add_argument('--case',type=int)
    p.add_argument('--block',type=int)
    p.add_argument('--variant',choices=VARIANTS)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    worker(a) if a.worker else controller(a)
