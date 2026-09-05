"""CPU-only audit of the candidate's real build functions via isolated AST.

Does not import PyTorch, execute setup.py, initialize CUTLASS, or build CUDA.
"""

import argparse
import ast
import json
import os
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    tree = ast.parse((args.source / "setup.py").read_text())
    names = {"is_flag_set", "get_release_flags", "make_cuda_extension"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in names]
    assert {node.name for node in functions} == names
    scope = {
        "os": os, "this_dir": str(args.source),
        "CUDAExtension": lambda **kwargs: kwargs,
        "get_nvcc_thread_args": lambda: [], "get_arch_flags": lambda: [],
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "release_build", "exec"), scope)
    macro = "-DFLASH_KDA_ENABLE_V16_PREFETCH4=1"
    for label, env, enabled in (
        ("default_off", {}, False),
        ("explicit_off_other_arch", {"FLASH_KDA_ENABLE_V16_PREFETCH4": "0",
                                     "FLASH_KDA_CUDA_ARCHS": "90a"}, False),
        ("enabled_sm103", {"FLASH_KDA_ENABLE_V16_PREFETCH4": "1",
                           "FLASH_KDA_CUDA_ARCHS": "103a"}, True),
    ):
        with patch.dict(os.environ, env, clear=True):
            for alias in (False, True):
                flags = ["-DK2_VALUE_SLICE=16"] if alias else []
                extension = scope["make_cuda_extension"]("alias" if alias else "production", flags)
                cxx = extension["extra_compile_args"]["cxx"]
                nvcc = extension["extra_compile_args"]["nvcc"]
                assert (macro in nvcc) == enabled
                assert macro not in cxx
                assert ("-DK2_VALUE_SLICE=16" in cxx) == alias
                assert ("-DK2_VALUE_SLICE=16" in nvcc) == alias
                print(json.dumps({"case": label, "alias": alias, "status": "PASS",
                                  "release_macro": macro in nvcc}))
    for arch in (None, "auto", "all", "90a", "100a", "120a", "103a,90a"):
        env = {"FLASH_KDA_ENABLE_V16_PREFETCH4": "1"}
        if arch is not None:
            env["FLASH_KDA_CUDA_ARCHS"] = arch
        with patch.dict(os.environ, env, clear=True):
            try:
                scope["make_cuda_extension"]("production", [])
            except RuntimeError as error:
                assert "requires explicit FLASH_KDA_CUDA_ARCHS=103a" in str(error)
            else:
                raise AssertionError("unsupported opt-in architecture accepted: " + str(arch))
        print(json.dumps({"case": "reject_optin_arch", "arch": arch, "status": "PASS"}))


if __name__ == "__main__":
    main()
