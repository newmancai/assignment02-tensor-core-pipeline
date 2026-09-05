"""Execute the actual launch-selection code on CPU with a recording stub.

Extracts launch_fwd from the candidate source. CUDA/template implementation
calls are replaced by a stub that records Value and StatePrefetch; the original
guard and switch are compiled unchanged. This is not a GPU correctness test.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


PRELUDE = r"""
#include <cstdint>
#include <cstdlib>
#include <iostream>
namespace cutlass { struct bfloat16_t {}; }
using cudaStream_t = void*;
int selected_value = -1;
int selected_prefetch = -1;
int checks = 0;
template<int D, int K2Value, bool HasStateIn, bool HasStateOut,
         bool StateFP32, bool IsVarlen, int StatePrefetch = 1, class... Args>
void launch_fwd_impl(Args...) {
    selected_value = K2Value;
    selected_prefetch = StatePrefetch;
}
"""

CHECKS = r"""
template<bool FP32 = false, bool Varlen = false, bool StateIn = true, bool StateOut = true>
void check(const char* name, int tokens, int heads, int sequences, int value, int expected) {
    selected_value = selected_prefetch = -1;
    launch_fwd<128, StateIn, StateOut, FP32, Varlen>(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1.0f,
        nullptr, nullptr, nullptr, 1, tokens, heads, sequences,
        nullptr, nullptr, nullptr, -5.0f, value, nullptr);
    if (selected_value != value || selected_prefetch != expected) {
        std::cerr << name << " got V" << selected_value << "/P" << selected_prefetch
                  << " expected V" << value << "/P" << expected << '\n';
        std::exit(1);
    }
    ++checks;
    std::cout << "PASS " << name << " V=" << selected_value
              << " Prefetch=" << selected_prefetch << '\n';
}

int main() {
#if defined(FLASH_KDA_ENABLE_V16_PREFETCH4) && FLASH_KDA_ENABLE_V16_PREFETCH4
    constexpr int fast = 4;
#else
    constexpr int fast = 1;
#endif
    check("T2047", 2047, 12, 1, 16, 1);
    check("T2048", 2048, 12, 1, 16, fast);
    check("T4096", 4096, 12, 1, 16, fast);
    check("T8192", 8192, 12, 1, 16, fast);
    check("T8193", 8193, 12, 1, 16, 1);
    check("N2", 4096, 12, 2, 16, 1);
    check("H11", 4096, 11, 1, 16, 1);
    check("H13", 4096, 13, 1, 16, 1);
    check<true>("FP32", 4096, 12, 1, 16, 1);
    check("V32", 4096, 12, 1, 32, 1);
    check("V64", 4096, 12, 1, 64, 1);
    check("V128", 4096, 12, 1, 128, 1);
    check<false, true>("packed_N1", 4096, 12, 1, 16, fast);
    check<false, true>("packed_N2", 4096, 12, 2, 16, 1);
    check<false, false, false, false>("state_none", 4096, 12, 1, 16, fast);
    check<false, false, true, false>("state_input_only", 4096, 12, 1, 16, fast);
    check<false, false, false, true>("state_output_only", 4096, 12, 1, 16, fast);
    std::cout << "checks=" << checks << '\n';
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = (args.source / "csrc/smxx/fwd_launch.cu").read_text()
    match = re.search(
        r"template <int D, bool HasStateIn, bool HasStateOut, bool StateFP32, bool IsVarlen>"
        r"\s*void launch_fwd\(", source)
    assert match, "could not identify the actual production launch selector"
    end = source.index("// Explicit instantiations", match.start())
    selector = source[match.start():end]
    program = PRELUDE + selector + CHECKS
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    with tempfile.TemporaryDirectory(prefix="kda-prefetch-guard-") as temporary:
        for label, defines in (
            ("default_no_macro", []),
            ("explicit_macro_zero", ["-DFLASH_KDA_ENABLE_V16_PREFETCH4=0"]),
            ("optin_macro_one", ["-DFLASH_KDA_ENABLE_V16_PREFETCH4=1"]),
        ):
            executable = str(Path(temporary) / label)
            subprocess.run(compiler + ["-std=c++17", "-O0", *defines, "-x", "c++", "-", "-o", executable],
                           input=program, text=True, capture_output=True, check=True, timeout=45)
            result = subprocess.run([executable], text=True, capture_output=True, check=True, timeout=5)
            print(json.dumps({"configuration": label, "status": "PASS", "checks": 17,
                              "output": result.stdout}), flush=True)


if __name__ == "__main__":
    main()
