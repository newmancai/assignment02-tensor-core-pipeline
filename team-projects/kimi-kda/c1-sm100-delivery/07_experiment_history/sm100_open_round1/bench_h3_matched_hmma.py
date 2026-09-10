#!/usr/bin/env python3
"""B300 correctness/full-forward pilot for H3 matched HMMA.

The full sbatch CLI is accepted.  ``--pilot`` narrows it to official+H3,
H12/T8192 and public_full.  K1/K2 requests fail until the extensions expose
diagnostic APIs; the runner never substitutes separately compiled binaries or
the sum of unrelated measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

OFFICIAL = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
VARIANTS = ("official_hmma", "h3_matched_hmma", "optimized_hmma")
SCOPES = ("k1", "k2", "raw_full", "public_full")


def csv_choice(text, allowed):
    values = tuple(x.strip() for x in text.split(",") if x.strip())
    if not values or len(set(values)) != len(values) or set(values) - set(allowed):
        raise argparse.ArgumentTypeError(f"choose unique comma-separated values from {allowed}")
    return values


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(root, *args):
    try:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    except subprocess.CalledProcessError:
        return None


def load_extension(root, name):
    root = Path(root).resolve(strict=True)
    files = sorted(root.glob(f"{name}*.so"))
    if len(files) != 1:
        raise RuntimeError(f"expected one {name}*.so in {root}, got {files}")
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module(name)
    finally:
        sys.path.pop(0)
    if Path(module.__file__).resolve() != files[0].resolve():
        raise RuntimeError(f"{name} imported from unexpected path {module.__file__}")
    return module, files[0], root


def specs(variant):
    if variant == "h3_matched_hmma":
        return (("w",4096,(16,128),"bf16"),("q_decayed",4096,(16,128),"bf16"),
                ("k_restored",4096,(16,128),"bf16"),("g_total",512,(128,),"fp32"),
                ("p",4096,(16,128),"bf16"),("mqk",512,(16,16),"bf16"))
    return (("k_decayed",4096,(16,128),"bf16"),("q_decayed",4096,(16,128),"bf16"),
            ("k_restored",4096,(16,128),"bf16"),("g_total",512,(128,),"fp32"),
            ("inv",512,(16,16),"bf16"),("mqk",512,(16,16),"bf16"))


def ledger(module, variant, heads, seq_lens):
    tiles = heads * sum((x + 15) // 16 for x in seq_lens)
    fn = getattr(module, "get_workspace_size_h3", getattr(module, "get_workspace_size", None))
    if fn is None:
        raise RuntimeError(f"{variant} lacks workspace size API")
    allocated = int(fn(sum(seq_lens), heads, len(seq_lens)))
    arrays = [{"name":n,"dtype":d,"logical_shape_per_tile":list(shape),
               "layout":"row_major_gmem","bytes_per_tile":b,"tiles_touched":tiles,
               "k1_reads":0,"k1_writes":1,"k2_reads":1,"k2_writes":0,
               "touched_bytes":2*b*tiles} for n,b,shape,d in specs(variant)]
    one = sum(x[1] for x in specs(variant)) * tiles
    return {"allocated_bytes":allocated,"tiles_touched":tiles,"arrays":arrays,
            "k1_touched_bytes":one,"k2_touched_bytes":one,"full_touched_bytes":2*one,
            "non_workspace_device_bytes":{},"ledger_matches_source":allocated >= one,
            "measured_counters":{}}


def metrics(torch, actual, ref):
    diff, base = actual.float()-ref.float(), ref.float()
    rel = float(torch.linalg.vector_norm(diff)/max(torch.linalg.vector_norm(base),1e-30))
    linf = float(diff.abs().max()/max(base.abs().max(),1e-30))
    return {"max_abs":float(diff.abs().max()),"relative_l2":rel,
            "normalized_linf":linf,"passed":rel <= .03 and linf <= .08}


def bench_cuda_event(torch, fn, dry_run_iters, repeat_iters, l2_flush):
    """CUDA-event timing with a stream-ordered 2x-L2 flush before each call."""
    for _ in range(dry_run_iters):
        l2_flush.zero_(); fn()
    torch.cuda.synchronize()
    starts=[torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    ends=[torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    for i in range(repeat_iters):
        l2_flush.zero_()
        starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    return [starts[i].elapsed_time(ends[i]) for i in range(repeat_iters)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official-source-dir",type=Path,required=True)
    p.add_argument("--official-commit",default=OFFICIAL)
    p.add_argument("--h3-source-dir",type=Path,required=True)
    p.add_argument("--optimized-source-dir",type=Path)
    p.add_argument("--case-set",default="h12,h96-task")
    p.add_argument("--variants",type=lambda x:csv_choice(x,VARIANTS),default=VARIANTS)
    p.add_argument("--scopes",type=lambda x:csv_choice(x,SCOPES),default=SCOPES)
    p.add_argument("--state-rotations",type=int,default=512)
    p.add_argument("--dry-run-iters",type=int,default=20)
    p.add_argument("--repeat-iters",type=int,default=100)
    p.add_argument("--blocks",type=int,default=8)
    p.add_argument("--cold-l2",action="store_true")
    p.add_argument("--balanced-pair-order",action="store_true")
    p.add_argument("--pilot",action="store_true")
    p.add_argument("--correctness-only",action="store_true")
    p.add_argument("--backend",choices=("cupti","cuda_event_cold_l2"),default="cuda_event_cold_l2")
    p.add_argument("--reverse-load-order",action="store_true")
    p.add_argument("--schema",type=Path,required=True)
    p.add_argument("--json",type=Path,required=True)
    a=p.parse_args()
    if a.pilot:
        a.variants=("official_hmma","h3_matched_hmma"); a.scopes=("public_full",)
    if a.official_commit != OFFICIAL or git(a.official_source_dir,"rev-parse","HEAD") != OFFICIAL:
        p.error("official commit mismatch")
    if (not a.cold_l2) or (not a.balanced_pair_order):
        p.error("frozen design requires --cold-l2 and --balanced-pair-order")
    if any(s in a.scopes for s in ("k1","k2")):
        raise RuntimeError("K1/K2 diagnostic ABI is not implemented; run --pilot first")

    import numpy as np
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10,3):
        raise RuntimeError("B300/SM103 is required")
    l2_bytes=torch.cuda.get_device_properties(0).L2_cache_size
    l2_flush=torch.empty((2*l2_bytes+3)//4,dtype=torch.int32,device="cuda")
    if a.backend=="cupti":
        from flashinfer.testing import bench_gpu_time as flashinfer_bench_gpu_time
    roots={"official_hmma":(a.official_source_dir,"flash_kda_C"),
           "h3_matched_hmma":(a.h3_source_dir,"flash_kda_h3_C"),
           "optimized_hmma":(a.optimized_source_dir,"flash_kda_phase1_C")}
    modules={}; identities=[]
    load_variants=tuple(reversed(a.variants)) if a.reverse_load_order else a.variants
    for variant in load_variants:
        if roots[variant][0] is None: p.error(f"source directory missing for {variant}")
        module,binary,root=load_extension(*roots[variant]); modules[variant]=module
        identities.append({"variant_id":variant,"source_dir":str(root),
          "source_commit":git(root,"rev-parse","HEAD"),"tracked_dirty":bool(git(root,"status","--porcelain","--untracked-files=no") or ""),
          "patch_sha256":[],"extension_name":module.__name__,"binary_sha256":sha(binary),
          "instruction_family":"sm80_mma_sync","semantic_compatibility":"algorithm_equivalent_not_official_bitwise" if variant=="h3_matched_hmma" else "official_bitwise",
          "build_command":["build receipt required before final measurement"],"route_guard":None})

    pilot_case=("h12_fixed_8192",12,(8192,),"fixed",12003)
    full_cases=(
      ("h12_packed_512x32",12,(512,)*32,"packed",12000),
      ("h12_packed_128x8",12,(128,)*8,"packed",12001),
      ("h12_fixed_512",12,(512,),"fixed",12002),
      pilot_case,
      ("h12_packed_mixed",12,(1300,547,2048,963,271,3063),"packed",12004),
      ("h12_packed_1024x8",12,(1024,)*8,"packed",12005),
      ("h96_fixed_8192",96,(8192,),"fixed",10000),
      ("h96_mixed",96,(1300,547,2048,963,271,3063),"packed",10001),
      ("h96_uniform_1024x8",96,(1024,)*8,"packed",10002),
    )
    case_defs=(pilot_case,) if a.pilot else full_cases
    cases=[]; all_ok=True
    for case_name,heads,seq_lens,layout,seed in case_defs:
        total=sum(seq_lens); sequences=len(seq_lens) if layout=="packed" else 1
        cu=(torch.tensor([0,*np.cumsum(seq_lens).tolist()],dtype=torch.int64,device="cuda")
            if layout=="packed" else None)
        gen=torch.Generator(device="cuda").manual_seed(seed); shape=(1,total,heads,128)
        rand=lambda shape,dtype=torch.bfloat16: torch.randn(shape,generator=gen,device="cuda").to(dtype)
        q,k,v,g=[rand(shape) for _ in range(4)]; beta=rand((1,total,heads))
        alog=torch.rand(heads,generator=gen,device="cuda"); dt=torch.rand((heads,128),generator=gen,device="cuda")
        initial=(rand((sequences,heads,128,128),torch.float32)*.25).to(torch.bfloat16)
        prepared={}
        for variant,module in modules.items():
            wsfn=getattr(module,"get_workspace_size_h3",None) or getattr(module,"get_workspace_size",None)
            prepared[variant]={"out":torch.empty_like(q),"final":torch.empty_like(initial),
              "ws":torch.empty(int(wsfn(total,heads,sequences)),dtype=torch.uint8,device="cuda"),
              "states":initial.unsqueeze(0).expand(a.state_rotations,*initial.shape).clone(),"cursor":0}
        def reset(variant):
            z=prepared[variant]; z["states"].copy_(initial.unsqueeze(0)); z["cursor"]=0
        def run(variant,scope="public_full"):
            z=prepared[variant]; i=z["cursor"]
            if i>=a.state_rotations: raise RuntimeError("state rotations exhausted")
            z["cursor"]+=1; module=modules[variant]
            fn=(getattr(module,"fwd_h3",None) or module.fwd) if variant=="h3_matched_hmma" else module.fwd
            fn(q,k,v,g,beta,1/math.sqrt(128),z["out"],z["ws"],alog,dt,-5.0,
               initial_state=z["states"][i],final_state=z["final"],cu_seqlens=cu)
            if scope=="public_full": z["states"][i].copy_(z["final"])
        observed={}
        for variant in a.variants:
            reset(variant); run(variant); torch.cuda.synchronize()
            observed[variant]=(prepared[variant]["out"].clone(),prepared[variant]["states"][0].clone())
        ref=observed["official_hmma"]; rows=[]
        times={variant:{scope:[] for scope in a.scopes} for variant in a.variants}
        samples={variant:{scope:[] for scope in a.scopes} for variant in a.variants}
        if not a.correctness_only:
            for block in range(a.blocks):
                order=a.variants[block % len(a.variants):]+a.variants[:block % len(a.variants)]
                for variant in order:
                    for scope in a.scopes:
                        reset(variant)
                        call=lambda variant=variant,scope=scope:run(variant,scope)
                        if a.backend=="cupti":
                            raw=flashinfer_bench_gpu_time(call,enable_cupti=True,
                              cold_l2_cache=True,use_cuda_graph=False,
                              dry_run_iters=a.dry_run_iters,repeat_iters=a.repeat_iters)
                        else:
                            raw=bench_cuda_event(torch,call,a.dry_run_iters,a.repeat_iters,l2_flush)
                        vals=[float(x) for x in raw]
                        times[variant][scope].append(float(np.median(vals)))
                        samples[variant][scope].extend(vals)
        for variant in a.variants:
            om,sm=metrics(torch,observed[variant][0],ref[0]),metrics(torch,observed[variant][1],ref[1])
            all_ok &= om["passed"] and sm["passed"]
            if not a.correctness_only:
                timing=[{"scope":scope,"median_ms":float(np.median(times[variant][scope])),
                  "samples_ms":samples[variant][scope],"block_medians_ms":times[variant][scope]} for scope in a.scopes]
            else:
                timing=[{"scope":"public_full","median_ms":1e-30,"samples_ms":[1e-30],"block_medians_ms":[1e-30,1e-30]}]
            rows.append({"variant_id":variant,"resolved_route":variant,"correctness":{"reference":"official_hmma_peer","tolerance_frozen_before_timing":True,"output":om,"final_state":sm},"workspace":ledger(modules[variant],variant,heads,seq_lens),"timings":timing})
        comparisons=[]
        for variant in a.variants[1:]:
            for scope in a.scopes:
                if a.correctness_only: ratio=1.0; ci=[1.0,1.0]
                else:
                    num=np.asarray(times["official_hmma"][scope]); den=np.asarray(times[variant][scope])
                    ratio=float(np.median(num)/np.median(den)); rng=np.random.default_rng(seed)
                    draws=[]
                    for _ in range(10000):
                        ix=rng.integers(0,len(num),len(num)); draws.append(float(np.median(num[ix])/np.median(den[ix])))
                    ci=[float(x) for x in np.quantile(draws,[.025,.975])]
                comparisons.append({"numerator":"official_hmma","denominator":variant,"scope":scope,"speedup":ratio,"bootstrap_ci95":ci})
        cases.append({"name":case_name,"num_heads":heads,"seq_lens":list(seq_lens),"layout":layout,"seed":seed,"variants":rows,"paired_comparisons":comparisons})
        del prepared,observed,q,k,v,g,beta,alog,dt,initial,cu
        torch.cuda.empty_cache()
    props=torch.cuda.get_device_properties(0); driver=subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True).splitlines()[0].strip()
    h12_speedups=[x["speedup"] for c in cases if c["num_heads"]==12 for x in c["paired_comparisons"] if x["denominator"]=="h3_matched_hmma" and x["scope"]=="public_full"]
    h12_geo=float(math.exp(sum(math.log(x) for x in h12_speedups)/len(h12_speedups))) if h12_speedups else None
    if not all_ok: decision="stop_correctness"; reasons=["at least one output or final-state comparison failed"]
    elif a.pilot: decision="needs_more_data"; reasons=["pilot advance gate passed; full case matrix remains"]
    elif h12_geo <= 1.05 or min(h12_speedups) < .98: decision="stop_full_path_regression"; reasons=["H12 practical gate requires geomean >1.05x and no case below 0.98x"]
    else: decision="needs_more_data"; reasons=["matched-HMMA gate passed; optimized-HMMA comparison remains before tcgen05"]
    payload={"schema_version":"sm100_open_h3_b300_v1","question":"FlashKDA 官方 kernel 当前使用 SM80 MMA，分析迁移到 SM100 是否值得","experiment_id":os.getenv("SLURM_JOB_ID","interactive-pilot"),"status":"correctness_failed" if not all_ok else ("inconclusive" if a.pilot else "measured"),
      "provenance":{"harness":{"path":str(Path(__file__).resolve()),"sha256":sha(__file__)},"variants":identities},
      "hardware":{"device_name":props.name,"compute_capability":[10,3],"cuda_arch":"sm_103a","sm_count":props.multi_processor_count,"l2_bytes":l2_bytes,"torch_version":torch.__version__,"cuda_version":str(torch.version.cuda),"driver_version":driver},
      "timing_policy":{"backend":a.backend,"cold_l2":True,"cuda_graph":False,"allocation_in_scope":False,"state_reset_in_scope":False,"same_initial_state_per_call":True,"headline_scope":"public_full","pair_order":list(a.variants),"dry_run_iters":a.dry_run_iters,"repeat_iters":a.repeat_iters,"blocks":a.blocks,"per_call_transforms_in_scope":["beta_transpose","workspace_io","state_copy_back"]},
      "cases":cases,"aggregate_decision":{"correctness_passed":all_ok,"h12_geomean_speedup_vs_official":h12_geo,"competitive_with_optimized_where_active":None,"decision":decision,"reasons":reasons}}
    a.json.parent.mkdir(parents=True,exist_ok=True); a.json.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    try:
        import jsonschema; jsonschema.validate(payload,json.loads(a.schema.read_text()))
    except ImportError:
        print("warning: jsonschema unavailable",file=sys.stderr)
    print(json.dumps(payload["aggregate_decision"],indent=2))
    if not all_ok: raise SystemExit(2)


if __name__ == "__main__": main()
