// Round 8 fixed-shape entry for the FlashKDA Phase-6 preferred-layout bridge.
//
// Keep the already reviewed HMMA/tcgen05 implementations and correctness
// oracle in one place.  This wrapper fixes the experiment to the Kimi-K3 TP8
// ValueSlice shape: D=128, CHUNK/K=16, V=16, and 12 heads * 8 slices = 96
// CTAs.  Timing controls and the output CSV path remain configurable.

#define main round8_phase6_probe_main
#include "phase6_probe_preferred.cu"
#undef main

#include <cstdio>
#include <string>
#include <vector>

namespace {

bool takes_value(const std::string& argument) {
    return argument == "--warmup" || argument == "--iters" ||
           argument == "--repeats" || argument == "--csv";
}

bool is_flag(const std::string& argument) {
    return argument == "--validate-only" ||
           argument == "--benchmark-only";
}

void print_round8_usage(const char* program) {
    std::printf(
        "usage: %s [--warmup N] [--iters N] [--repeats N] [--csv PATH] "
        "[--validate-only|--benchmark-only]\n",
        program);
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> arguments = {
        argv[0],
        "--values", "16",
        "--grids", "96",
        "--inners", "1,64",
        "--check-grid", "96",
        "--check-inners", "1,2,4",
    };

    bool validate_only = false;
    bool benchmark_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help" || argument == "-h") {
            print_round8_usage(argv[0]);
            return 0;
        }
        if (takes_value(argument)) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after %s\n", argument.c_str());
                return 2;
            }
            arguments.push_back(argument);
            arguments.emplace_back(argv[++i]);
            continue;
        }
        if (is_flag(argument)) {
            validate_only |= argument == "--validate-only";
            benchmark_only |= argument == "--benchmark-only";
            arguments.push_back(argument);
            continue;
        }
        std::fprintf(
            stderr,
            "Round 8 fixes V=16 and grid/check-grid=96; unsupported option: %s\n",
            argument.c_str());
        print_round8_usage(argv[0]);
        return 2;
    }
    if (validate_only && benchmark_only) {
        std::fprintf(stderr,
                     "--validate-only and --benchmark-only are mutually exclusive\n");
        return 2;
    }

    std::printf(
        "ROUND8,profile=kimi_k3_tp8_h12_v16,chunk=16,D=128,V=16,grid=96,"
        "arms=hmma_resident|tcgen_scalar|tcgen_preferred,"
        "candidate=tcgen05_m128n16k16_preferred_layout_contract,"
        "producer_conversion_outside_inner=1,full_flashkda=0\n");

    std::vector<char*> forwarded;
    forwarded.reserve(arguments.size());
    for (std::string& argument : arguments) {
        forwarded.push_back(argument.data());
    }
    return round8_phase6_probe_main(
        static_cast<int>(forwarded.size()), forwarded.data());
}
