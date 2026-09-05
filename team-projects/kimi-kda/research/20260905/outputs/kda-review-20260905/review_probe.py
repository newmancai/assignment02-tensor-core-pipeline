"""Isolated B300 review probes; does not modify the FlashKDA implementation."""
import json
import math
import os
import statistics
import subprocess
import sys

import torch
import flash_kda
import flash_kda_C


def emit(kind, **fields):
    print(json.dumps(dict(kind=kind, **fields), sort_keys=True), flush=True)


def metrics(x, ref):
    x, ref = x.float(), ref.float()
    err = x - ref
    return dict(finite=bool(x.isfinite().all()), bitwise=bool(torch.equal(x, ref)),
                rel_rmse=float(err.square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(1e-20)),
                max_abs=float(err.abs().max()))


def make_inputs(t, h, gate=None):
    shape = (1, t, h, 128)
    q, k, v, g = [torch.randn(shape, device='cuda', dtype=torch.bfloat16) for _ in range(4)]
    if gate is not None:
        g.fill_(gate)
    beta = torch.randn(shape[:-1], device='cuda', dtype=torch.bfloat16)
    a = torch.zeros(h, device='cuda', dtype=torch.float32)
    dt = torch.zeros(h, 128, device='cuda', dtype=torch.float32)
    return q, k, v, g, beta, a, dt


def run_raw(inp, value_slice=128, state=None, packed=None, state_dtype=torch.bfloat16, alias=False):
    q, k, v, g, beta, a, dt = inp
    n = packed.numel() - 1 if packed is not None else q.shape[0]
    out = torch.empty_like(v)
    final = state if alias else torch.empty((n, q.shape[2], 128, 128), device='cuda', dtype=state_dtype)
    ws = torch.empty(flash_kda_C.get_workspace_size(q.shape[0]*q.shape[1], q.shape[2], n), device='cuda', dtype=torch.uint8)
    flash_kda_C.fwd(q,k,v,g,beta,1/math.sqrt(128),out,ws,a,dt,-5.,state,final,packed,k2_value_slice=value_slice)
    return out, final


def numerical_probes():
    for dtype in (torch.bfloat16, torch.float32):
        inp = make_inputs(17, 2)
        initial = torch.full((2,2,128,128), 1.001, device='cuda', dtype=dtype)
        cu = torch.tensor([0,0,17], device='cuda', dtype=torch.int64)
        for vs in (128,16):
            out, final = run_raw(inp,vs,initial,cu,dtype)
            emit('empty_sequence_identity', dtype=str(dtype), value_slice=vs,
                 initial=float(initial[0,0,0,0]), final=float(final[0,0,0,0]),
                 **metrics(final[0],initial[0]))
    for gate in (None,-8.):
        inp = make_inputs(256, 2, gate)
        initial = torch.randn((1,2,128,128),device='cuda',dtype=torch.bfloat16)
        whole_out,whole_final = run_raw(inp,128,initial)
        for lengths in ([16]*16,[17,239],[1]*256):
            for vs in (128,16):
                state=initial.clone()
                parts=[]
                offset=0
                for n in lengths:
                    # clone, not just contiguous: a contiguous view may retain
                    # a beta storage offset that violates TMA 16-byte alignment.
                    part=tuple(x[:,offset:offset+n].clone() if i<5 else x for i,x in enumerate(inp))
                    out,state=run_raw(part,vs,state)
                    parts.append(out)
                    offset+=n
                emit('segmentation',gate=gate,first_segment=lengths[0],segments=len(lengths),value_slice=vs,
                     output=metrics(torch.cat(parts,dim=1),whole_out),state=metrics(state,whole_final))
        expected_out,expected_state=run_raw(inp,16,initial.clone())
        alias_out,alias_state=run_raw(inp,16,initial.clone(),alias=True)
        emit('state_alias',gate=gate,output=metrics(alias_out,expected_out),state=metrics(alias_state,expected_state))


def timed(run, count=80, evict=None):
    for _ in range(12): run()
    torch.cuda.synchronize()
    starts=[torch.cuda.Event(enable_timing=True) for _ in range(count)]
    ends=[torch.cuda.Event(enable_timing=True) for _ in range(count)]
    for i in range(count):
        if evict is not None: evict.zero_()
        starts[i].record()
        run()
        ends[i].record()
    torch.cuda.synchronize()
    vals=[a.elapsed_time(b) for a,b in zip(starts,ends)]
    return statistics.median(vals)


def perf_probes():
    inp=make_inputs(8192,12)
    q,k,v,g,beta,a,dt=inp
    state=torch.randn((1,12,128,128),device='cuda',dtype=torch.bfloat16)
    out=torch.empty_like(v)
    final=torch.empty_like(state)
    def run(): flash_kda.fwd(q,k,v,g,beta,1/math.sqrt(128),out,a,dt,-5.,state,final)
    os.environ.pop('FLASH_KDA_K2_VALUE_SLICE',None)
    os.environ.pop('FLASH_KDA_K2_DISPATCH',None)
    emit('dispatch',decision=flash_kda.explain_k2_dispatch(q,state,final,None))
    run()
    auto_out,auto_state=out.clone(),final.clone()
    os.environ['FLASH_KDA_K2_VALUE_SLICE']='128'
    run()
    emit('auto_wrapper',output=metrics(auto_out,out),state=metrics(auto_state,final))
    eviction=torch.empty(256*1024*1024,device='cuda',dtype=torch.uint8)
    for repeat in range(3):
        modes=(128,16) if repeat%2==0 else (16,128)
        for mode in modes:
            os.environ['FLASH_KDA_K2_VALUE_SLICE']=str(mode)
            hot=timed(run)
            disturbed=timed(run,evict=eviction)
            for _ in range(3): run()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): run()
            graph_hot=timed(graph.replay)
            graph_disturbed=timed(graph.replay,evict=eviction)
            emit('perf',repeat=repeat,value_slice=mode,hot_ms=hot,pre_call_eviction_ms=disturbed,
                 graph_hot_ms=graph_hot,graph_pre_call_eviction_ms=graph_disturbed)
    # Strong invariant: prior-output allocation remains fixed through graph replay.
    # Eviction is before the complete operator; K1 may warm workspace for K2 again.


def alignment_probe(clone):
    inp=list(make_inputs(1,2))
    backing=torch.randn(4,device='cuda',dtype=torch.bfloat16)
    inp[4]=backing[1:3].reshape(1,1,2)
    if clone: inp[4]=inp[4].clone()
    emit('alignment_input',cloned=clone,contiguous=inp[4].is_contiguous(),
         beta_pointer_mod16=inp[4].data_ptr()%16)
    out,state=run_raw(tuple(inp))
    torch.cuda.synchronize()
    emit('alignment_completed',cloned=clone,finite=bool(out.isfinite().all()))


@torch.inference_mode()
def main():
    torch.manual_seed(20260905)
    emit('environment',torch=torch.__version__,extension=flash_kda_C.__file__,
         wrapper=flash_kda.__file__,device=flash_kda_C.get_device_characteristics())
    numerical_probes()
    perf_probes()
    for mode in ('aligned','view'):
        child=subprocess.run([sys.executable,__file__,'--alignment',mode],capture_output=True,text=True)
        emit('alignment_child',mode=mode,returncode=child.returncode,stdout=child.stdout,stderr=child.stderr)
    emit('complete')


if __name__=='__main__':
    if len(sys.argv)>1:
        with torch.inference_mode(): alignment_probe(sys.argv[2]=='aligned')
    else: main()
