"""Read-only setup flag audit. Takes an already patched FlashKDA source directory.

Runs only the AST-extracted extension factory against a harmless dictionary
stub. Does not import PyTorch, invoke setup, build CUDA, or access the network.
"""

import ast
import json
import os
import sys
from pathlib import Path


source_root = Path(sys.argv[1])
setup_source = (source_root / "setup.py").read_text()
setup_tree = ast.parse(setup_source)
factory = next(
    node for node in setup_tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "make_cuda_extension"
)
namespace = {
    "os": os,
    "this_dir": str(source_root),
    "CUDAExtension": lambda **kwargs: kwargs,
    "get_nvcc_thread_args": lambda: [],
    "get_arch_flags": lambda: [],
}
exec(compile(ast.Module(body=[factory], type_ignores=[]), "setup_factory", "exec"), namespace)
extension = namespace["make_cuda_extension"]("flash_kda_vsplit16_C", ["-DK2_VALUE_SLICE=16"])
flags = extension["extra_compile_args"]
binding = (source_root / "csrc/flash_kda.cpp").read_text()
result = {
    "source_root": str(source_root),
    "extension": extension["name"],
    "cxx_flags": flags["cxx"],
    "nvcc_value_slice_flags": [flag for flag in flags["nvcc"] if "K2_VALUE_SLICE" in flag],
    "binding_default_uses_macro": "constexpr int64_t kDefaultK2ValueSlice = K2_VALUE_SLICE;" in binding,
    "binding_without_macro_defaults_128": "constexpr int64_t kDefaultK2ValueSlice = 128;" in binding,
    "finding_reproduced": not any("K2_VALUE_SLICE" in flag for flag in flags["cxx"]),
}
print(json.dumps(result, indent=2))
assert result["finding_reproduced"]
