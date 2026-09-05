"""CPU-only AST audit of setup.py's extension factory, without running setup."""

import argparse
import ast
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="FlashKDA source after 0001 + 0002 + 0003")
    args = parser.parse_args()
    tree = ast.parse((args.source / "setup.py").read_text())
    factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "make_cuda_extension")
    scope = {
        "os": os, "this_dir": str(args.source),
        "CUDAExtension": lambda **kwargs: kwargs,
        "get_nvcc_thread_args": lambda: [], "get_arch_flags": lambda: [],
    }
    exec(compile(ast.Module(body=[factory], type_ignores=[]), "setup_factory", "exec"), scope)
    cases = (
        ("production", [], None),
        ("alias_v16", ["-DK2_VALUE_SLICE=16"], "-DK2_VALUE_SLICE=16"),
        ("isolated_k2", ["-DBLOCK_LEVEL_K1=-1", "-DK2_VALUE_SLICE=32"], "-DK2_VALUE_SLICE=32"),
    )
    for label, flags, expected in cases:
        extension = scope["make_cuda_extension"](label, flags)
        cxx = extension["extra_compile_args"]["cxx"]
        nvcc = extension["extra_compile_args"]["nvcc"]
        expected_defines = [] if expected is None else [expected]
        assert [x for x in cxx if x.startswith("-DK2_VALUE_SLICE=")] == expected_defines
        assert [x for x in nvcc if x.startswith("-DK2_VALUE_SLICE=")] == expected_defines
        assert not any(x.startswith("-DBLOCK_LEVEL_") for x in cxx)
        assert all(x in nvcc for x in flags)
        print(json.dumps({"case": label, "status": "PASS", "cxx_flags": cxx}))


if __name__ == "__main__":
    main()
