#!/usr/bin/env python3
"""CPU-only contracts for the clean Phase1 candidate; does not execute CUDA."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from unittest.mock import patch


def source_contract(base, candidate):
    rel = "csrc/smxx/fwd_kernel2.cuh"
    old = (base / rel).read_text()
    new = (candidate / rel).read_text()
    opening = "            if constexpr (Phase1Prefetch > 1) {\n"
    otherwise = "            } else {\n"
    closing = "            }\n\n            // ======== Phase 2:"
    begin = new.index(opening)
    old_begin = new.index(otherwise, begin) + len(otherwise)
    old_end = new.index(closing, old_begin)
    restored = new[:begin] + new[old_begin:old_end] + new[old_end + len("            }\n"):]
    restored = restored.replace("    int StatePrefetch = 1,\n    int Phase1Prefetch = 1\n",
                                "    int StatePrefetch = 1\n")
    restored = restored.replace("    static_assert(Phase1Prefetch == 1 || Phase1Prefetch == 2 || Phase1Prefetch == 4);\n", "")
    assert restored == old, "Kernel changes escaped the template/assert/Phase1 wrapper"
    launch = (candidate / "csrc/smxx/fwd_launch.cu").read_text()
    assert "HasStateIn, HasStateOut, StateFP32, IsVarlen, StatePrefetch, Phase1Prefetch" in launch
    assert "int StatePrefetch = 1, int Phase1Prefetch = 1>" in launch
    signature = r"_flash_kda_fwd_recurrence\((.*?)\n\) \{"
    assert re.search(signature, new, re.S).group(1) == re.search(signature, old, re.S).group(1)
    for text in (new, launch, (candidate / "csrc/flash_kda.cpp").read_text()):
        assert not re.search(r"InitStrategy|\bload_initial_state\b|KDA_ZERO_ABLATION|[12345]0016", text)
    for rel in ("csrc/flash_kda.cpp", "csrc/fwd.h"):
        assert (base / rel).read_bytes() == (candidate / rel).read_bytes(), rel
    originals = {p.relative_to(base / "flash_kda") for p in (base / "flash_kda").rglob("*") if p.is_file()}
    copies = {p.relative_to(candidate / "flash_kda") for p in (candidate / "flash_kda").rglob("*") if p.is_file()}
    assert originals == copies
    assert all((base / "flash_kda" / p).read_bytes() == (candidate / "flash_kda" / p).read_bytes() for p in originals)
    print(json.dumps(dict(check="source_preservation_and_template_forwarding", status="PASS")))


def build_contract(candidate):
    tree = ast.parse((candidate / "setup.py").read_text())
    names = {"is_flag_set", "get_release_flags", "make_cuda_extension"}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == names
    scope = dict(os=os, this_dir=str(candidate), CUDAExtension=lambda **kw: kw,
                 get_nvcc_thread_args=lambda: [], get_arch_flags=lambda: [])
    exec(compile(ast.Module(body=functions, type_ignores=[]), "clean_phase1_build", "exec"), scope)
    phase1 = "FLASH_KDA_ENABLE_V16_PHASE1_PREFETCH"
    phase6 = "FLASH_KDA_ENABLE_V16_PREFETCH4"
    arch = "FLASH_KDA_CUDA_ARCHS"
    checks = 0
    for env, expected in (
        ({}, (False, False)),
        ({phase1: "0", arch: "90a"}, (False, False)),
        ({phase6: "1", arch: "103a"}, (True, False)),
        ({phase6: "1", phase1: "0", arch: "103a"}, (True, False)),
        ({phase6: "1", phase1: "1", arch: "103a"}, (True, True)),
    ):
        with patch.dict(os.environ, env, clear=True):
            for alias in (False, True):
                result = scope["make_cuda_extension"]("test", ["-DK2_VALUE_SLICE=16"] if alias else [])
                cxx, nvcc = (result["extra_compile_args"][k] for k in ("cxx", "nvcc"))
                for flag, enabled in zip((phase6, phase1), expected):
                    macro = "-D" + flag + "=1"
                    assert (macro in nvcc) == enabled and macro not in cxx
                assert ("-DK2_VALUE_SLICE=16" in cxx) == alias
                assert ("-DK2_VALUE_SLICE=16" in nvcc) == alias
                checks += 1
    invalid = [({phase1: "1", arch: "103a"}, phase6 + "=1"),
               ({phase1: "1", phase6: "0", arch: "103a"}, phase6 + "=1"),
               ({phase1: "1", phase6: "false", arch: "103a"}, phase6 + "=1")]
    for target in (None, "auto", "all", "90a", "100a", "120a", "103a,90a"):
        env = {phase1: "1", phase6: "1"}
        if target is not None:
            env[arch] = target
        invalid.append((env, "requires explicit " + arch + "=103a"))
    for env, message in invalid:
        with patch.dict(os.environ, env, clear=True):
            try:
                scope["make_cuda_extension"]("test", [])
            except RuntimeError as error:
                assert message in str(error), str(error)
            else:
                raise AssertionError("invalid build configuration accepted: " + str(env))
            checks += 1
    print(json.dumps(dict(check="build_contract", checks=checks, status="PASS")))


PRELUDE = r"""
#include <cstdint>
#include <cstdlib>
#include <iostream>
namespace cutlass { struct bfloat16_t {}; }
using cudaStream_t = void*;
int value=-1, phase6=-1, phase1=-1, checks=0;
bool hi=false, ho=false, fp32=false, vl=false;
template<int D, int V, bool HI, bool HO, bool FP32, bool VL,
         int P6=1, int P1=1, class... Args>
void launch_fwd_impl(Args...) {
    value=V; phase6=P6; phase1=P1; hi=HI; ho=HO; fp32=FP32; vl=VL;
}
"""

CHECKS = r"""
template<bool FP32=false, bool VL=false, bool HI=true, bool HO=true>
void check(int t, int h, int n, int v, int p6, int p1) {
    value=phase6=phase1=-1;
    launch_fwd<128,HI,HO,FP32,VL>(
        nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,1.0f,
        nullptr,nullptr,nullptr,1,t,h,n,nullptr,nullptr,nullptr,-5.0f,v,nullptr);
    if (value!=v || phase6!=p6 || phase1!=p1 || hi!=HI || ho!=HO || fp32!=FP32 || vl!=VL) {
        std::cerr << "T=" << t << " HI=" << HI << " got " << value << '/' << phase6 << '/' << phase1
                  << " expected " << v << '/' << p6 << '/' << p1;
        std::exit(1);
    }
    ++checks;
}
int main() {
#if defined(FLASH_KDA_ENABLE_V16_PREFETCH4) && FLASH_KDA_ENABLE_V16_PREFETCH4
    constexpr int fast6=4;
#if defined(FLASH_KDA_ENABLE_V16_PHASE1_PREFETCH) && FLASH_KDA_ENABLE_V16_PHASE1_PREFETCH
    constexpr int fast_true=2, fast_false=4;
#else
    constexpr int fast_true=1, fast_false=1;
#endif
#else
    constexpr int fast6=1, fast_true=1, fast_false=1;
#endif
    struct Case {int t; bool fast;};
    for (auto c : {Case{1,false},Case{2047,false},Case{2048,true},Case{2049,true},
                   Case{4095,true},Case{4096,true},Case{4097,true},Case{8191,true},
                   Case{8192,true},Case{8193,false},Case{16384,false}}) {
        check(c.t,12,1,16,c.fast?fast6:1,c.fast?fast_true:1);
        check<false,false,false,true>(c.t,12,1,16,c.fast?fast6:1,c.fast?fast_false:1);
    }
    check(4096,12,2,16,1,1);
    check<false,false,false,true>(4096,12,2,16,1,1);
    for (int h : {11,13}) {
        check(4096,h,1,16,1,1);
        check<false,false,false,true>(4096,h,1,16,1,1);
    }
    check<true>(4096,12,1,16,1,1);
    check<true,false,false,true>(4096,12,1,16,1,1);
    for (int v : {32,64,128}) check(4096,12,1,v,1,1);
    check<false,true>(4096,12,1,16,fast6,fast_true);
    check<false,true,false,true>(4096,12,1,16,fast6,fast_false);
    check<false,true>(4096,12,2,16,1,1);
    check<false,true,false,true>(4096,12,2,16,1,1);
    check<false,false,true,false>(4096,12,1,16,fast6,fast_true);
    check<false,false,false,false>(4096,12,1,16,fast6,fast_false);
    std::cout << checks;
}
"""


def guard_contract(candidate):
    text = (candidate / "csrc/smxx/fwd_launch.cu").read_text()
    start = text.index("template <int D, bool HasStateIn, bool HasStateOut, bool StateFP32, bool IsVarlen>")
    selector = text[start:text.index("// Explicit instantiations", start)]
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    p6, p1 = "-DFLASH_KDA_ENABLE_V16_PREFETCH4=", "-DFLASH_KDA_ENABLE_V16_PHASE1_PREFETCH="
    with tempfile.TemporaryDirectory(prefix="kda-clean-phase1-cpu-") as temporary:
        for name, defines in (("default", []), ("phase6_only", [p6+"1"]),
                              ("both", [p6+"1", p1+"1"]), ("phase1_off", [p6+"1", p1+"0"]),
                              ("both_off", [p6+"0", p1+"0"]), ("manual_phase1_without_phase6", [p1+"1"])):
            executable = str(Path(temporary) / name)
            subprocess.run(compiler + ["-std=c++17", "-O0", *defines, "-x", "c++", "-", "-o", executable],
                           input=PRELUDE+selector+CHECKS, text=True, capture_output=True, check=True, timeout=45)
            result = subprocess.run([executable], text=True, capture_output=True, check=True, timeout=5)
            print(json.dumps(dict(check="actual_launch_selector", configuration=name,
                                  checks=int(result.stdout), status="PASS")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    source_contract(args.base, args.candidate)
    build_contract(args.candidate)
    guard_contract(args.candidate)


if __name__ == "__main__":
    main()
