// FlashKDA K2 Phase-6 probe for Blackwell/B300.
//
// The mathematical operation is
//   delta_state[D,V] = k_restored_t[D,16] @ U[16,V]
// with D=128 and V in {16,32,64,128}.  Both paths use BF16 inputs and FP32
// accumulation.  L1 additionally evaluates the real K2 epilogue
//   state = BF16(FP32(state) * gate[row] + delta_state).
//
// This is deliberately a probe, not an SM100 FlashKDA implementation:
//   L0: operands are staged once.  It measures the two MMA execution paths.
//   L1: the tcgen05 path also reformats U from a logical shared-memory tile to
//       compact 32B-swizzled shared memory on every inner iteration.  That
//       scalar shared->shared copy is a conservative proxy for the missing
//       Phase-4-register-fragment -> tcgen05-operand materialization.  It is
//       NOT the final STSM/TMEM-resident implementation.
// Full-kernel CUDA-event time is reported.  inner=1 includes allocation and
// staging; inner>1 amortizes one-time costs and reports kernel_us/inner too.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error_ = (call);                                            \
        if (error_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",              \
                         cudaGetErrorName(error_), __FILE__, __LINE__,           \
                         cudaGetErrorString(error_));                            \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

using BF16 = __nv_bfloat16;

constexpr int kM = 128;
constexpr int kK = 16;
constexpr int kThreads = 128;

__host__ __device__ inline int swizzle_32b_offset(int row, int col_byte) {
    // One compact K=16 BF16 row is 32 bytes.  The 32B swizzle atom is
    // 8 rows x 32 bytes and repeats on a 256-byte boundary.
    const int atom = row >> 3;
    const int row_in_atom = row & 7;
    const int chunk16 = col_byte >> 4;
    const int byte_in_chunk = col_byte & 15;
    return atom * 256 + row_in_atom * 32 +
           ((chunk16 ^ (row_in_atom & 1)) << 4) + byte_in_chunk;
}

__device__ inline uint64_t make_desc_sm100(uint32_t shared_address,
                                            uint32_t leading_byte_offset,
                                            uint32_t stride_byte_offset,
                                            uint32_t layout_type) {
    uint64_t desc = 0;
    desc |= uint64_t((shared_address >> 4) & 0x3fff);
    desc |= uint64_t((leading_byte_offset >> 4) & 0x3fff) << 16;
    desc |= uint64_t((stride_byte_offset >> 4) & 0x3fff) << 32;
    desc |= uint64_t(1) << 46;  // SM100 matrix descriptor version.
    desc |= uint64_t(layout_type & 7) << 61;
    return desc;
}

__device__ inline void mbarrier_wait_parity(uint32_t mbar, uint32_t parity) {
    uint32_t done = 0;
    while (!done) {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.b32 %0, 1, 0, p;\n"
            "}"
            : "=r"(done)
            : "r"(mbar), "r"(parity));
    }
}

__device__ inline uint32_t pack_bf16(BF16 lo, BF16 hi) {
    const __nv_bfloat162 packed = __halves2bfloat162(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&packed);
}

__device__ inline void mma_m16n8k16_bf16(const uint32_t (&a)[4],
                                          const uint32_t (&b)[2],
                                          float (&d)[4]) {
    const float zero0 = 0.0f;
    const float zero1 = 0.0f;
    const float zero2 = 0.0f;
    const float zero3 = 0.0f;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
          "r"(b[1]), "f"(zero0), "f"(zero1), "f"(zero2), "f"(zero3));
}

template <int V, bool L1>
__global__ void __launch_bounds__(kThreads) phase6_mma_sync(
    const BF16* __restrict__ global_a,
    const BF16* __restrict__ global_b,
    const BF16* __restrict__ global_state,
    const float* __restrict__ global_gate,
    float* __restrict__ global_delta,
    BF16* __restrict__ global_state_out,
    int inner) {
    static_assert(V == 16 || V == 32 || V == 64 || V == 128,
                  "supported V values are 16, 32, 64, 128");
    constexpr int kNTiles = V / 8;
    constexpr int kMaxTilesPerWarp = (kNTiles + 3) / 4;

    __shared__ BF16 shared_a[kM * kK];
    __shared__ BF16 shared_b[kK * V];
    __shared__ BF16 shared_state[L1 ? kM * V : 1];
    __shared__ float shared_gate[L1 ? kM : 1];

    const int tid = int(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    const size_t block_a = size_t(blockIdx.x) * kM * kK;
    const size_t block_b = size_t(blockIdx.x) * kK * V;
    const size_t block_state = size_t(blockIdx.x) * kM * V;

    for (int i = tid; i < kM * kK; i += blockDim.x) {
        shared_a[i] = global_a[block_a + i];
    }
    for (int i = tid; i < kK * V; i += blockDim.x) {
        shared_b[i] = global_b[block_b + i];
    }
    if constexpr (L1) {
        for (int i = tid; i < kM * V; i += blockDim.x) {
            shared_state[i] = global_state[block_state + i];
        }
        for (int i = tid; i < kM; i += blockDim.x) {
            shared_gate[i] = global_gate[size_t(blockIdx.x) * kM + i];
        }
    }
    __syncthreads();

    // K2 enters Phase 6 with U already in register fragments.  Preloading the
    // B fragments here gives mma.sync the same favorable operand residency.
    uint32_t b_frag[kMaxTilesPerWarp][2];
#pragma unroll
    for (int slot = 0; slot < kMaxTilesPerWarp; ++slot) {
        const int n_tile = warp + slot * 4;
        if (n_tile < kNTiles) {
            const int n = n_tile * 8 + group;
            b_frag[slot][0] = pack_bf16(
                shared_b[(thread_in_group * 2) * V + n],
                shared_b[(thread_in_group * 2 + 1) * V + n]);
            b_frag[slot][1] = pack_bf16(
                shared_b[(thread_in_group * 2 + 8) * V + n],
                shared_b[(thread_in_group * 2 + 9) * V + n]);
        }
    }

    for (int repetition = 0; repetition < inner; ++repetition) {
#pragma unroll
        for (int m_tile = 0; m_tile < kM / 16; ++m_tile) {
            const int row0 = m_tile * 16 + group;
            const int row1 = row0 + 8;
            const int k0 = thread_in_group * 2;

            uint32_t a_frag[4];
            a_frag[0] = pack_bf16(shared_a[row0 * kK + k0],
                                  shared_a[row0 * kK + k0 + 1]);
            a_frag[1] = pack_bf16(shared_a[row1 * kK + k0],
                                  shared_a[row1 * kK + k0 + 1]);
            a_frag[2] = pack_bf16(shared_a[row0 * kK + k0 + 8],
                                  shared_a[row0 * kK + k0 + 9]);
            a_frag[3] = pack_bf16(shared_a[row1 * kK + k0 + 8],
                                  shared_a[row1 * kK + k0 + 9]);

#pragma unroll
            for (int slot = 0; slot < kMaxTilesPerWarp; ++slot) {
                const int n_tile = warp + slot * 4;
                if (n_tile < kNTiles) {
                    float d[4];
                    mma_m16n8k16_bf16(a_frag, b_frag[slot], d);
                    const int col0 = n_tile * 8 + thread_in_group * 2;
                    const int col1 = col0 + 1;

                    if constexpr (L1) {
                        shared_state[row0 * V + col0] = __float2bfloat16(
                            __bfloat162float(shared_state[row0 * V + col0]) *
                                shared_gate[row0] +
                            d[0]);
                        shared_state[row0 * V + col1] = __float2bfloat16(
                            __bfloat162float(shared_state[row0 * V + col1]) *
                                shared_gate[row0] +
                            d[1]);
                        shared_state[row1 * V + col0] = __float2bfloat16(
                            __bfloat162float(shared_state[row1 * V + col0]) *
                                shared_gate[row1] +
                            d[2]);
                        shared_state[row1 * V + col1] = __float2bfloat16(
                            __bfloat162float(shared_state[row1 * V + col1]) *
                                shared_gate[row1] +
                            d[3]);
                    } else if (repetition + 1 == inner) {
                        const size_t out = block_state;
                        global_delta[out + row0 * V + col0] = d[0];
                        global_delta[out + row0 * V + col1] = d[1];
                        global_delta[out + row1 * V + col0] = d[2];
                        global_delta[out + row1 * V + col1] = d[3];
                    }
                }
            }
        }
    }

    if constexpr (L1) {
        __syncthreads();
        for (int i = tid; i < kM * V; i += blockDim.x) {
            global_state_out[block_state + i] = shared_state[i];
        }
    }
}

template <int V>
struct TmemColumns {
    static constexpr int value = V <= 32 ? 32 : V;
};

template <int V, bool L1>
__global__ void __launch_bounds__(kThreads) phase6_tcgen05(
    const BF16* __restrict__ global_a,
    const BF16* __restrict__ global_b,
    const BF16* __restrict__ global_state,
    const float* __restrict__ global_gate,
    float* __restrict__ global_delta,
    BF16* __restrict__ global_state_out,
    int inner) {
    static_assert(V == 16 || V == 32 || V == 64 || V == 128,
                  "supported V values are 16, 32, 64, 128");
    constexpr int kAllocatedColumns = TmemColumns<V>::value;

    __shared__ __align__(256) uint8_t shared_a_bytes[kM * kK * sizeof(BF16)];
    __shared__ __align__(256) uint8_t shared_bt_bytes[V * kK * sizeof(BF16)];
    // L1 keeps a logical KxV copy and pays a conservative scalar reformat to
    // the N x K descriptor layout on every inner iteration.
    __shared__ BF16 shared_b_logical[L1 ? kK * V : 1];
    __shared__ BF16 shared_state[L1 ? kM * V : 1];
    __shared__ float shared_gate[L1 ? kM : 1];
    __shared__ __align__(8) uint64_t completion_barrier;
    __shared__ uint32_t shared_tmem_address[1];

    const int tid = int(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const size_t block_a = size_t(blockIdx.x) * kM * kK;
    const size_t block_b = size_t(blockIdx.x) * kK * V;
    const size_t block_state = size_t(blockIdx.x) * kM * V;
    const uint32_t mbar =
        uint32_t(__cvta_generic_to_shared(&completion_barrier));

    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::
                         "r"(mbar), "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    if (warp == 0) {
        const uint32_t dst =
            uint32_t(__cvta_generic_to_shared(shared_tmem_address));
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
            "[%0], %1;" ::
                "r"(dst), "r"(kAllocatedColumns));
        asm volatile(
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    // Compact K=16, 32B-swizzled A staging.
    for (int i = tid; i < kM * kK; i += blockDim.x) {
        const int row = i / kK;
        const int k = i % kK;
        *reinterpret_cast<BF16*>(
            &shared_a_bytes[swizzle_32b_offset(row, k * int(sizeof(BF16)))]) =
            global_a[block_a + i];
    }

    if constexpr (L1) {
        for (int i = tid; i < kK * V; i += blockDim.x) {
            shared_b_logical[i] = global_b[block_b + i];
        }
        for (int i = tid; i < kM * V; i += blockDim.x) {
            shared_state[i] = global_state[block_state + i];
        }
        for (int i = tid; i < kM; i += blockDim.x) {
            shared_gate[i] = global_gate[size_t(blockIdx.x) * kM + i];
        }
    } else {
        // L0 receives logical B[K,V] but directly stages the tcgen05 physical
        // B descriptor, whose K-major storage is B^T[V,K].
        for (int i = tid; i < V * kK; i += blockDim.x) {
            const int n = i / kK;
            const int k = i % kK;
            *reinterpret_cast<BF16*>(
                &shared_bt_bytes[swizzle_32b_offset(n, k * int(sizeof(BF16)))]) =
                global_b[block_b + k * V + n];
        }
    }

    // Generic shared writes become visible to the tcgen05 asynchronous proxy.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    const uint32_t tmem = shared_tmem_address[0];
    const uint32_t a_base =
        uint32_t(__cvta_generic_to_shared(shared_a_bytes));
    const uint32_t b_base =
        uint32_t(__cvta_generic_to_shared(shared_bt_bytes));
    constexpr uint32_t kInstructionDescriptor =
        (1u << 4) |             // D is F32.
        (1u << 7) |             // A is BF16.
        (1u << 10) |            // B is BF16.
        (uint32_t(V / 8) << 17) |  // N.
        (uint32_t(kM / 16) << 24); // M.

    // 32B swizzle: LBO is ignored, SBO is one 8-row x 32B = 256B atom,
    // layout_type=6.  This is the compact layout for BF16 K=16.
    const uint64_t a_desc = make_desc_sm100(a_base, 0, 256, 6);
    const uint64_t b_desc = make_desc_sm100(b_base, 0, 256, 6);

    for (int repetition = 0; repetition < inner; ++repetition) {
        if constexpr (L1) {
            // Conservative materialization envelope.  The intended integrated
            // path would use a transposed STSM or keep U^T in TMEM; this scalar
            // shared load+store intentionally does not claim that optimization.
            for (int i = tid; i < V * kK; i += blockDim.x) {
                const int n = i / kK;
                const int k = i % kK;
                *reinterpret_cast<BF16*>(
                    &shared_bt_bytes[swizzle_32b_offset(
                        n, k * int(sizeof(BF16)))]) =
                    shared_b_logical[k * V + n];
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
        }

        if (tid == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile(
                "{\n"
                ".reg .pred p;\n"
                "setp.ne.b32 p, %4, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 "
                "[%0], %1, %2, %3, p;\n"
                "}\n" ::
                    "r"(tmem), "l"(a_desc), "l"(b_desc),
                "r"(kInstructionDescriptor), "r"(0));
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                ".shared::cluster.b64 [%0];" ::
                    "r"(mbar)
                : "memory");
        }

        // A new barrier phase is completed by each commit.  Fixed parity=0 is
        // the classic two-iteration deadlock/race bug; parity must alternate.
        mbarrier_wait_parity(mbar, uint32_t(repetition & 1));
        asm volatile("tcgen05.fence::after_thread_sync;");

#pragma unroll
        for (int col = 0; col < V; col += 8) {
            const uint32_t src =
                tmem + (uint32_t(warp * 32) << 16) + uint32_t(col);
            float result[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=f"(result[0]), "=f"(result[1]), "=f"(result[2]),
                  "=f"(result[3]), "=f"(result[4]), "=f"(result[5]),
                  "=f"(result[6]), "=f"(result[7])
                : "r"(src));
            asm volatile("tcgen05.wait::ld.sync.aligned;");

            const int row = warp * 32 + lane;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if constexpr (L1) {
                    shared_state[row * V + col + j] = __float2bfloat16(
                        __bfloat162float(shared_state[row * V + col + j]) *
                                shared_gate[row] +
                            result[j]);
                } else if (repetition + 1 == inner) {
                    global_delta[block_state + row * V + col + j] = result[j];
                }
            }
        }

        // All four warps must finish their TMEM lanes before the issuing thread
        // is allowed to overwrite the same TMEM accumulator in the next phase.
        asm volatile("tcgen05.fence::before_thread_sync;");
        __syncthreads();
    }

    if constexpr (L1) {
        for (int i = tid; i < kM * V; i += blockDim.x) {
            global_state_out[block_state + i] = shared_state[i];
        }
    }

    __syncthreads();
    if (warp == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::
                "r"(tmem), "r"(kAllocatedColumns));
    }
}

struct Options {
    int warmup = 30;
    int iterations = 200;
    int repeats = 5;
    int check_grid = 12;
    std::vector<int> values = {16, 32, 64, 128};
    std::vector<int> grids = {12, 148};
    std::vector<int> inners = {1, 64};
    std::vector<int> check_inners = {1, 2, 4};
    bool validate = true;
    bool benchmark = true;
    std::string csv_path;
};

static std::vector<int> parse_int_list(const std::string& text) {
    std::vector<int> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) values.push_back(std::stoi(item));
    }
    if (values.empty()) throw std::runtime_error("empty integer list");
    return values;
}

static void print_usage(const char* argv0) {
    std::printf(
        "usage: %s [options]\n"
        "  --warmup N          warm-up launches per repeat (default 30)\n"
        "  --iters N           timed launches per repeat (default 200)\n"
        "  --repeats N         timing repeats (default 5)\n"
        "  --values LIST       V list (default 16,32,64,128)\n"
        "  --grids LIST        CTA grid list (default 12,148)\n"
        "  --inners LIST       inner phase repeats (default 1,64)\n"
        "  --check-inners LIST correctness barrier phases (default 1,2,4)\n"
        "  --check-grid N      correctness grid (default 12)\n"
        "  --csv PATH          also write benchmark rows to PATH\n"
        "  --validate-only     skip timing\n"
        "  --benchmark-only    skip correctness (not recommended)\n",
        argv0);
}

static Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value after " + arg);
            }
            return argv[++i];
        };
        if (arg == "--warmup") options.warmup = std::stoi(next());
        else if (arg == "--iters") options.iterations = std::stoi(next());
        else if (arg == "--repeats") options.repeats = std::stoi(next());
        else if (arg == "--values") options.values = parse_int_list(next());
        else if (arg == "--grids") options.grids = parse_int_list(next());
        else if (arg == "--inners") options.inners = parse_int_list(next());
        else if (arg == "--check-inners")
            options.check_inners = parse_int_list(next());
        else if (arg == "--check-grid") options.check_grid = std::stoi(next());
        else if (arg == "--csv") options.csv_path = next();
        else if (arg == "--validate-only") options.benchmark = false;
        else if (arg == "--benchmark-only") options.validate = false;
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.warmup < 0 || options.iterations <= 0 ||
        options.repeats <= 0 || options.check_grid <= 0) {
        throw std::runtime_error("counts must be positive (warmup may be zero)");
    }
    return options;
}

struct Buffers {
    BF16* a = nullptr;
    BF16* b = nullptr;
    BF16* state = nullptr;
    float* gate = nullptr;
    float* delta = nullptr;
    BF16* state_out = nullptr;

    Buffers() = default;
    Buffers(const Buffers&) = delete;
    Buffers& operator=(const Buffers&) = delete;
    Buffers(Buffers&& other) noexcept
        : a(other.a),
          b(other.b),
          state(other.state),
          gate(other.gate),
          delta(other.delta),
          state_out(other.state_out) {
        other.a = nullptr;
        other.b = nullptr;
        other.state = nullptr;
        other.gate = nullptr;
        other.delta = nullptr;
        other.state_out = nullptr;
    }
    Buffers& operator=(Buffers&& other) noexcept {
        if (this != &other) {
            cudaFree(a);
            cudaFree(b);
            cudaFree(state);
            cudaFree(gate);
            cudaFree(delta);
            cudaFree(state_out);
            a = other.a;
            b = other.b;
            state = other.state;
            gate = other.gate;
            delta = other.delta;
            state_out = other.state_out;
            other.a = nullptr;
            other.b = nullptr;
            other.state = nullptr;
            other.gate = nullptr;
            other.delta = nullptr;
            other.state_out = nullptr;
        }
        return *this;
    }

    ~Buffers() {
        cudaFree(a);
        cudaFree(b);
        cudaFree(state);
        cudaFree(gate);
        cudaFree(delta);
        cudaFree(state_out);
    }
};

template <int V>
static Buffers allocate_and_initialize(int grid,
                                       std::vector<BF16>* host_a = nullptr,
                                       std::vector<BF16>* host_b = nullptr,
                                       std::vector<BF16>* host_state = nullptr,
                                       std::vector<float>* host_gate = nullptr) {
    const size_t a_count = size_t(grid) * kM * kK;
    const size_t b_count = size_t(grid) * kK * V;
    const size_t state_count = size_t(grid) * kM * V;
    const size_t gate_count = size_t(grid) * kM;
    std::vector<BF16> a(a_count), b(b_count), state(state_count);
    std::vector<float> gate(gate_count);

    for (int block = 0; block < grid; ++block) {
        for (int m = 0; m < kM; ++m) {
            for (int k = 0; k < kK; ++k) {
                const int value = ((m * 5 + k * 3 + block) % 3) - 1;
                a[(size_t(block) * kM + m) * kK + k] =
                    __float2bfloat16(float(value));
            }
        }
        for (int k = 0; k < kK; ++k) {
            for (int n = 0; n < V; ++n) {
                const int value = ((k * 7 + n * 2 + block * 2) % 3) - 1;
                b[(size_t(block) * kK + k) * V + n] =
                    __float2bfloat16(float(value));
            }
        }
        for (int m = 0; m < kM; ++m) {
            gate[size_t(block) * kM + m] = (m & 1) ? 0.5f : 1.0f;
            for (int n = 0; n < V; ++n) {
                const int value = ((m + n + block) % 3) - 1;
                state[(size_t(block) * kM + m) * V + n] =
                    __float2bfloat16(float(value));
            }
        }
    }

    Buffers device;
    CUDA_CHECK(cudaMalloc(&device.a, a_count * sizeof(BF16)));
    CUDA_CHECK(cudaMalloc(&device.b, b_count * sizeof(BF16)));
    CUDA_CHECK(cudaMalloc(&device.state, state_count * sizeof(BF16)));
    CUDA_CHECK(cudaMalloc(&device.gate, gate_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device.delta, state_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device.state_out, state_count * sizeof(BF16)));
    CUDA_CHECK(cudaMemcpy(device.a, a.data(), a_count * sizeof(BF16),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.b, b.data(), b_count * sizeof(BF16),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.state, state.data(),
                          state_count * sizeof(BF16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.gate, gate.data(), gate_count * sizeof(float),
                          cudaMemcpyHostToDevice));

    if (host_a) *host_a = std::move(a);
    if (host_b) *host_b = std::move(b);
    if (host_state) *host_state = std::move(state);
    if (host_gate) *host_gate = std::move(gate);
    return device;
}

template <int V>
static std::vector<float> reference_delta(const std::vector<BF16>& a,
                                          const std::vector<BF16>& b,
                                          int grid) {
    std::vector<float> reference(size_t(grid) * kM * V, 0.0f);
    for (int block = 0; block < grid; ++block) {
        for (int m = 0; m < kM; ++m) {
            for (int n = 0; n < V; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < kK; ++k) {
                    sum += __bfloat162float(
                               a[(size_t(block) * kM + m) * kK + k]) *
                           __bfloat162float(
                               b[(size_t(block) * kK + k) * V + n]);
                }
                reference[(size_t(block) * kM + m) * V + n] = sum;
            }
        }
    }
    return reference;
}

template <int V>
static std::vector<BF16> reference_state(const std::vector<float>& delta,
                                         const std::vector<BF16>& initial,
                                         const std::vector<float>& gate,
                                         int grid, int inner) {
    std::vector<BF16> reference = initial;
    for (int repetition = 0; repetition < inner; ++repetition) {
        for (int block = 0; block < grid; ++block) {
            for (int m = 0; m < kM; ++m) {
                for (int n = 0; n < V; ++n) {
                    const size_t index =
                        (size_t(block) * kM + m) * V + n;
                    reference[index] = __float2bfloat16(
                        __bfloat162float(reference[index]) *
                                gate[size_t(block) * kM + m] +
                            delta[index]);
                }
            }
        }
    }
    return reference;
}

static uint16_t bf16_bits(BF16 value) {
    uint16_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

template <int V>
static bool validate_v(int grid, const std::vector<int>& check_inners) {
    std::vector<BF16> host_a, host_b, host_state;
    std::vector<float> host_gate;
    Buffers device = allocate_and_initialize<V>(
        grid, &host_a, &host_b, &host_state, &host_gate);
    const size_t output_count = size_t(grid) * kM * V;
    const std::vector<float> delta_ref =
        reference_delta<V>(host_a, host_b, grid);
    bool ok = true;

    for (const char* path : {"mma_sync", "tcgen05"}) {
        CUDA_CHECK(cudaMemset(device.delta, 0xff,
                              output_count * sizeof(float)));
        if (std::strcmp(path, "mma_sync") == 0) {
            phase6_mma_sync<V, false><<<grid, kThreads>>>(
                device.a, device.b, device.state, device.gate, device.delta,
                device.state_out, 1);
        } else {
            phase6_tcgen05<V, false><<<grid, kThreads>>>(
                device.a, device.b, device.state, device.gate, device.delta,
                device.state_out, 1);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> got(output_count);
        CUDA_CHECK(cudaMemcpy(got.data(), device.delta,
                              output_count * sizeof(float),
                              cudaMemcpyDeviceToHost));
        size_t bad = 0;
        for (size_t i = 0; i < output_count; ++i) {
            if (got[i] != delta_ref[i]) {
                if (bad < 3) {
                    std::fprintf(stderr,
                                 "L0 mismatch path=%s V=%d index=%zu got=%g "
                                 "want=%g\n",
                                 path, V, i, got[i], delta_ref[i]);
                }
                ++bad;
            }
        }
        std::printf("VALIDATE,level=L0,path=%s,V=%d,grid=%d,inner=1,%s,bad=%zu\n",
                    path, V, grid, bad ? "FAIL" : "PASS", bad);
        ok &= bad == 0;
    }

    for (int inner : check_inners) {
        const std::vector<BF16> state_ref = reference_state<V>(
            delta_ref, host_state, host_gate, grid, inner);
        for (const char* path : {"mma_sync", "tcgen05"}) {
            CUDA_CHECK(cudaMemset(device.state_out, 0xff,
                                  output_count * sizeof(BF16)));
            if (std::strcmp(path, "mma_sync") == 0) {
                phase6_mma_sync<V, true><<<grid, kThreads>>>(
                    device.a, device.b, device.state, device.gate,
                    device.delta, device.state_out, inner);
            } else {
                phase6_tcgen05<V, true><<<grid, kThreads>>>(
                    device.a, device.b, device.state, device.gate,
                    device.delta, device.state_out, inner);
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<BF16> got(output_count);
            CUDA_CHECK(cudaMemcpy(got.data(), device.state_out,
                                  output_count * sizeof(BF16),
                                  cudaMemcpyDeviceToHost));
            size_t bad = 0;
            for (size_t i = 0; i < output_count; ++i) {
                if (bf16_bits(got[i]) != bf16_bits(state_ref[i])) {
                    if (bad < 3) {
                        std::fprintf(
                            stderr,
                            "L1 mismatch path=%s V=%d inner=%d index=%zu "
                            "got=%g want=%g\n",
                            path, V, inner, i, __bfloat162float(got[i]),
                            __bfloat162float(state_ref[i]));
                    }
                    ++bad;
                }
            }
            std::printf(
                "VALIDATE,level=L1,path=%s,V=%d,grid=%d,inner=%d,%s,bad=%zu\n",
                path, V, grid, inner, bad ? "FAIL" : "PASS", bad);
            ok &= bad == 0;
        }
    }
    return ok;
}

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() & 1) return values[middle];
    return 0.5 * (values[middle - 1] + values[middle]);
}

static double time_batch_us(const std::function<void()>& launch, int warmup,
                            int iterations) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return double(elapsed_ms) * 1000.0 / double(iterations);
}

struct TimingPair {
    double mma_median_us;
    double mma_min_us;
    double tcgen_median_us;
    double tcgen_min_us;
};

static TimingPair time_pair(const std::function<void()>& mma_launch,
                            const std::function<void()>& tcgen_launch,
                            int warmup, int iterations, int repeats) {
    std::vector<double> mma, tcgen;
    mma.reserve(repeats);
    tcgen.reserve(repeats);
    for (int repeat = 0; repeat < repeats; ++repeat) {
        if ((repeat & 1) == 0) {
            mma.push_back(time_batch_us(mma_launch, warmup, iterations));
            tcgen.push_back(time_batch_us(tcgen_launch, warmup, iterations));
        } else {
            tcgen.push_back(time_batch_us(tcgen_launch, warmup, iterations));
            mma.push_back(time_batch_us(mma_launch, warmup, iterations));
        }
    }
    return {median(mma), *std::min_element(mma.begin(), mma.end()),
            median(tcgen), *std::min_element(tcgen.begin(), tcgen.end())};
}

struct KernelResources {
    int registers;
    size_t static_shared_bytes;
    int active_blocks_per_sm;
};

template <int V, bool L1, bool Tcgen>
static KernelResources kernel_resources() {
    cudaFuncAttributes attributes{};
    int active = 0;
    if constexpr (Tcgen) {
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, phase6_tcgen05<V, L1>));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, phase6_tcgen05<V, L1>, kThreads, 0));
    } else {
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, phase6_mma_sync<V, L1>));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, phase6_mma_sync<V, L1>, kThreads, 0));
    }
    return {attributes.numRegs, attributes.sharedSizeBytes, active};
}

static const char* csv_header =
    "level,V,grid,inner,warmup,iters,repeats,"
    "mma_median_kernel_us,mma_min_kernel_us,"
    "tcgen_median_kernel_us,tcgen_min_kernel_us,"
    "mma_median_phase_us,tcgen_median_phase_us,speedup_median,"
    "mma_regs,tcgen_regs,mma_static_smem_B,tcgen_static_smem_B,"
    "mma_active_blocks_per_sm,tcgen_active_blocks_per_sm,scope";

template <int V, bool L1>
static void benchmark_case(int grid, int inner, const Options& options,
                           std::ostream* csv) {
    Buffers device = allocate_and_initialize<V>(grid);
    auto mma_launch = [&] {
        phase6_mma_sync<V, L1><<<grid, kThreads>>>(
            device.a, device.b, device.state, device.gate, device.delta,
            device.state_out, inner);
    };
    auto tcgen_launch = [&] {
        phase6_tcgen05<V, L1><<<grid, kThreads>>>(
            device.a, device.b, device.state, device.gate, device.delta,
            device.state_out, inner);
    };
    const TimingPair timing =
        time_pair(mma_launch, tcgen_launch, options.warmup,
                  options.iterations, options.repeats);
    const KernelResources mma = kernel_resources<V, L1, false>();
    const KernelResources tcgen = kernel_resources<V, L1, true>();
    const double speedup = timing.mma_median_us / timing.tcgen_median_us;
    const char* level = L1 ? "L1" : "L0";
    const char* scope =
        L1 ? "phase6_plus_state_gate_plus_scalar_U_reformat_upper_envelope"
           : "phase6_core_operands_prestaged_optimistic_tcgen";

    std::ostringstream row;
    row.setf(std::ios::fixed);
    row.precision(6);
    row << level << ',' << V << ',' << grid << ',' << inner << ','
        << options.warmup << ',' << options.iterations << ','
        << options.repeats << ',' << timing.mma_median_us << ','
        << timing.mma_min_us << ',' << timing.tcgen_median_us << ','
        << timing.tcgen_min_us << ','
        << timing.mma_median_us / double(inner) << ','
        << timing.tcgen_median_us / double(inner) << ',' << speedup << ','
        << mma.registers << ',' << tcgen.registers << ','
        << mma.static_shared_bytes << ',' << tcgen.static_shared_bytes << ','
        << mma.active_blocks_per_sm << ',' << tcgen.active_blocks_per_sm << ','
        << scope;
    std::printf("RESULT,%s\n", row.str().c_str());
    if (csv) {
        *csv << row.str() << '\n';
        csv->flush();
    }
}

template <int V>
static bool run_v(const Options& options, std::ostream* csv) {
    bool ok = true;
    if (options.validate) {
        ok = validate_v<V>(options.check_grid, options.check_inners);
        if (!ok) return false;
    }
    if (options.benchmark) {
        for (int grid : options.grids) {
            for (int inner : options.inners) {
                benchmark_case<V, false>(grid, inner, options, csv);
                benchmark_case<V, true>(grid, inner, options, csv);
            }
        }
    }
    return ok;
}

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse_options(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "argument error: %s\n", error.what());
        print_usage(argv[0]);
        return 2;
    }

    int device_index = 0;
    CUDA_CHECK(cudaGetDevice(&device_index));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device_index));
    int driver_version = 0, runtime_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    std::printf(
        "META,device=%s,cc=%d.%d,sm_count=%d,driver=%d,runtime=%d,"
        "D=%d,K=%d,threads=%d\n",
        properties.name, properties.major, properties.minor,
        properties.multiProcessorCount, driver_version, runtime_version, kM,
        kK, kThreads);
    std::printf(
        "META,L0=prestaged_core_probe,L1=state_gate_plus_conservative_"
        "scalar_U_reformat,not_full_K2=1\n");

    if (properties.major < 10) {
        std::fprintf(stderr,
                     "tcgen05 requires Blackwell data-center architecture; "
                     "found compute capability %d.%d\n",
                     properties.major, properties.minor);
        return 3;
    }

    std::ofstream csv_file;
    std::ostream* csv = nullptr;
    if (!options.csv_path.empty()) {
        csv_file.open(options.csv_path);
        if (!csv_file) {
            std::fprintf(stderr, "cannot open CSV path: %s\n",
                         options.csv_path.c_str());
            return 2;
        }
        csv_file << "# device=" << properties.name << ",cc="
                 << properties.major << '.' << properties.minor
                 << ",driver=" << driver_version
                 << ",runtime=" << runtime_version << '\n';
        csv_file << csv_header << '\n';
        csv = &csv_file;
    }
    if (options.benchmark) std::printf("CSV_HEADER,%s\n", csv_header);

    bool all_ok = true;
    for (int value : options.values) {
        switch (value) {
            case 16: all_ok &= run_v<16>(options, csv); break;
            case 32: all_ok &= run_v<32>(options, csv); break;
            case 64: all_ok &= run_v<64>(options, csv); break;
            case 128: all_ok &= run_v<128>(options, csv); break;
            default:
                std::fprintf(stderr, "unsupported V=%d\n", value);
                return 2;
        }
        if (!all_ok) {
            std::fprintf(stderr,
                         "correctness failed; timing is intentionally stopped\n");
            return 1;
        }
    }

    std::printf("SUMMARY,%s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
