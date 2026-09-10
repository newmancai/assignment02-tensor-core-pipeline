// Minimum P3 -> BF16 round -> P4 lifecycle probe for SM103a.
// Reuses the verified descriptor, barrier, timing, and HMMA helpers from the
// Stage-11/12 Phase-6 probe. This is a mechanism probe, not public FlashKDA.
#define main phase6_probe_original_main
#include "phase6_probe_preferred.cu"
#undef main

constexpr int kP34N = 16;

struct P34Buffers {
    BF16* a0 = nullptr;
    BF16* inv = nullptr;
    BF16* mqk = nullptr;
    BF16* output = nullptr;

    P34Buffers() = default;
    P34Buffers(const P34Buffers&) = delete;
    P34Buffers& operator=(const P34Buffers&) = delete;
    P34Buffers(P34Buffers&& other) noexcept
        : a0(other.a0), inv(other.inv), mqk(other.mqk),
          output(other.output) {
        other.a0 = nullptr;
        other.inv = nullptr;
        other.mqk = nullptr;
        other.output = nullptr;
    }
    P34Buffers& operator=(P34Buffers&& other) noexcept {
        if (this != &other) {
            cudaFree(a0);
            cudaFree(inv);
            cudaFree(mqk);
            cudaFree(output);
            a0 = other.a0;
            inv = other.inv;
            mqk = other.mqk;
            output = other.output;
            other.a0 = nullptr;
            other.inv = nullptr;
            other.mqk = nullptr;
            other.output = nullptr;
        }
        return *this;
    }
    ~P34Buffers() {
        cudaFree(a0);
        cudaFree(inv);
        cudaFree(mqk);
        cudaFree(output);
    }
};

__device__ inline void load_hmma_b16(const BF16* shared_b,
                                     int warp, int group,
                                     int thread_in_group,
                                     uint32_t (&fragment)[2]) {
    const int n = warp * 8 + group;
    fragment[0] = pack_bf16(
        shared_b[(thread_in_group * 2) * kP34N + n],
        shared_b[(thread_in_group * 2 + 1) * kP34N + n]);
    fragment[1] = pack_bf16(
        shared_b[(thread_in_group * 2 + 8) * kP34N + n],
        shared_b[(thread_in_group * 2 + 9) * kP34N + n]);
}

__device__ inline void load_hmma_a16(const BF16* shared_a,
                                     int row0, int row1,
                                     int thread_in_group,
                                     uint32_t (&fragment)[4]) {
    const int k0 = thread_in_group * 2;
    fragment[0] = pack_bf16(shared_a[row0 * kK + k0],
                            shared_a[row0 * kK + k0 + 1]);
    fragment[1] = pack_bf16(shared_a[row1 * kK + k0],
                            shared_a[row1 * kK + k0 + 1]);
    fragment[2] = pack_bf16(shared_a[row0 * kK + k0 + 8],
                            shared_a[row0 * kK + k0 + 9]);
    fragment[3] = pack_bf16(shared_a[row1 * kK + k0 + 8],
                            shared_a[row1 * kK + k0 + 9]);
}

__global__ void __launch_bounds__(kThreads) p34_hmma(
    const BF16* __restrict__ global_a0,
    const BF16* __restrict__ global_inv,
    const BF16* __restrict__ global_mqk,
    BF16* __restrict__ global_output,
    int inner) {
    __shared__ BF16 shared_a0[kM * kK];
    __shared__ BF16 shared_inv[kK * kP34N];
    __shared__ BF16 shared_mqk[kK * kP34N];
    __shared__ BF16 shared_u[kM * kP34N];

    const int tid = int(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    const size_t a_offset = size_t(blockIdx.x) * kM * kK;
    const size_t b_offset = size_t(blockIdx.x) * kK * kP34N;
    const size_t out_offset = size_t(blockIdx.x) * kM * kP34N;

    for (int i = tid; i < kM * kK; i += blockDim.x)
        shared_a0[i] = global_a0[a_offset + i];
    for (int i = tid; i < kK * kP34N; i += blockDim.x) {
        shared_inv[i] = global_inv[b_offset + i];
        shared_mqk[i] = global_mqk[b_offset + i];
    }
    __syncthreads();

    uint32_t inv_fragment[2] = {0, 0};
    uint32_t mqk_fragment[2] = {0, 0};
    if (warp < 2) {
        load_hmma_b16(shared_inv, warp, group, thread_in_group,
                      inv_fragment);
        load_hmma_b16(shared_mqk, warp, group, thread_in_group,
                      mqk_fragment);
    }

    for (int repetition = 0; repetition < inner; ++repetition) {
        if (warp < 2) {
#pragma unroll
            for (int m_tile = 0; m_tile < kM / 16; ++m_tile) {
                const int row0 = m_tile * 16 + group;
                const int row1 = row0 + 8;
                uint32_t a_fragment[4];
                float result[4];
                load_hmma_a16(shared_a0, row0, row1, thread_in_group,
                              a_fragment);
                mma_m16n8k16_bf16(a_fragment, inv_fragment, result);
                const int col0 = warp * 8 + thread_in_group * 2;
                const int col1 = col0 + 1;
                shared_u[row0 * kP34N + col0] =
                    __float2bfloat16(result[0]);
                shared_u[row0 * kP34N + col1] =
                    __float2bfloat16(result[1]);
                shared_u[row1 * kP34N + col0] =
                    __float2bfloat16(result[2]);
                shared_u[row1 * kP34N + col1] =
                    __float2bfloat16(result[3]);
            }
        }
        __syncthreads();

        if (warp < 2) {
#pragma unroll
            for (int m_tile = 0; m_tile < kM / 16; ++m_tile) {
                const int row0 = m_tile * 16 + group;
                const int row1 = row0 + 8;
                uint32_t u_fragment[4];
                float result[4];
                load_hmma_a16(shared_u, row0, row1, thread_in_group,
                              u_fragment);
                mma_m16n8k16_bf16(u_fragment, mqk_fragment, result);
                if (repetition + 1 == inner) {
                    const int col0 = warp * 8 + thread_in_group * 2;
                    const int col1 = col0 + 1;
                    global_output[out_offset + row0 * kP34N + col0] =
                        __float2bfloat16(result[0]);
                    global_output[out_offset + row0 * kP34N + col1] =
                        __float2bfloat16(result[1]);
                    global_output[out_offset + row1 * kP34N + col0] =
                        __float2bfloat16(result[2]);
                    global_output[out_offset + row1 * kP34N + col1] =
                        __float2bfloat16(result[3]);
                }
            }
        }
        __syncthreads();
    }
}

__device__ inline uint64_t p34_desc(uint8_t* pointer) {
    return make_desc_sm100(uint32_t(__cvta_generic_to_shared(pointer)),
                           0, 256, 6);
}

template <bool TmemCarrier>
__global__ void __launch_bounds__(kThreads) p34_tcgen_carrier(
    const BF16* __restrict__ global_a0,
    const BF16* __restrict__ global_inv,
    const BF16* __restrict__ global_mqk,
    BF16* __restrict__ global_output,
    int inner,
    int diagnostic_mode) {
    __shared__ __align__(256) uint8_t shared_a0[kM * kK * sizeof(BF16)];
    __shared__ __align__(256) uint8_t shared_u[kM * kK * sizeof(BF16)];
    __shared__ __align__(256) uint8_t shared_inv[kP34N * kK * sizeof(BF16)];
    __shared__ __align__(256) uint8_t shared_mqk[kP34N * kK * sizeof(BF16)];
    __shared__ __align__(8) uint64_t completion_barrier;
    __shared__ uint32_t shared_tmem_address[1];

    const int tid = int(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const size_t a_offset = size_t(blockIdx.x) * kM * kK;
    const size_t b_offset = size_t(blockIdx.x) * kK * kP34N;
    const size_t out_offset = size_t(blockIdx.x) * kM * kP34N;
    const uint32_t mbar =
        uint32_t(__cvta_generic_to_shared(&completion_barrier));

    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::
                         "r"(mbar), "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    if (warp == 0) {
        const uint32_t destination =
            uint32_t(__cvta_generic_to_shared(shared_tmem_address));
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
            "[%0], %1;" ::
                "r"(destination), "r"(32));
        asm volatile(
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    for (int i = tid; i < kM * kK; i += blockDim.x) {
        const int row = i / kK;
        const int k = i % kK;
        *reinterpret_cast<BF16*>(
            &shared_a0[swizzle_32b_offset(row, k * int(sizeof(BF16)))]) =
            global_a0[a_offset + i];
    }
    for (int i = tid; i < kP34N * kK; i += blockDim.x) {
        const int n = i / kK;
        const int k = i % kK;
        const size_t source = b_offset + k * kP34N + n;
        *reinterpret_cast<BF16*>(
            &shared_inv[swizzle_32b_offset(n, k * int(sizeof(BF16)))]) =
            global_inv[source];
        *reinterpret_cast<BF16*>(
            &shared_mqk[swizzle_32b_offset(n, k * int(sizeof(BF16)))]) =
            global_mqk[source];
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    const uint32_t tmem = shared_tmem_address[0];
    const uint64_t a0_desc = p34_desc(shared_a0);
    const uint64_t inv_desc = p34_desc(shared_inv);
    const uint64_t mqk_desc = p34_desc(shared_mqk);
    constexpr uint32_t instruction_descriptor =
        (1u << 4) | (1u << 7) | (1u << 10) |
        (uint32_t(kP34N / 8) << 17) | (uint32_t(kM / 16) << 24);

    for (int repetition = 0; repetition < inner; ++repetition) {
        if (tid == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile(
                "{\n"
                ".reg .pred p;\n"
                "setp.ne.b32 p, %4, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 "
                "[%0], %1, %2, %3, p;\n"
                "}\n" ::
                    "r"(tmem), "l"(a0_desc), "l"(inv_desc),
                    "r"(instruction_descriptor), "r"(0));
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                ".shared::cluster.b64 [%0];" ::
                    "r"(mbar)
                : "memory");
        }
        mbarrier_wait_parity(mbar, uint32_t((2 * repetition) & 1));
        asm volatile("tcgen05.fence::after_thread_sync;");

#pragma unroll
        for (int col = 0; col < kP34N; col += 8) {
            const uint32_t source =
                tmem + (uint32_t(warp * 32) << 16) + uint32_t(col);
            float result[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=f"(result[0]), "=f"(result[1]), "=f"(result[2]),
                  "=f"(result[3]), "=f"(result[4]), "=f"(result[5]),
                  "=f"(result[6]), "=f"(result[7])
                : "r"(source));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            const int row = warp * 32 + lane;
            if constexpr (TmemCarrier) {
                uint32_t packed[4];
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    packed[j] = pack_bf16(__float2bfloat16(result[2 * j]),
                                          __float2bfloat16(result[2 * j + 1]));
                const uint32_t destination =
                    tmem + uint32_t(16 + col) +
                    (uint32_t(warp * 32) << 16);
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x4.unpack::16b.b32 "
                    "[%0], {%1,%2,%3,%4};"
                    :: "r"(destination), "r"(packed[0]), "r"(packed[1]),
                       "r"(packed[2]), "r"(packed[3]));
                asm volatile("tcgen05.wait::st.sync.aligned;");
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    *reinterpret_cast<BF16*>(
                        &shared_u[swizzle_32b_offset(
                            row, (col + j) * int(sizeof(BF16)))]) =
                        __float2bfloat16(result[j]);
                }
            }
        }
        asm volatile("tcgen05.fence::before_thread_sync;");
        if constexpr (!TmemCarrier)
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();

        if constexpr (!TmemCarrier) {
            if (diagnostic_mode == 1) {
                for (int i = tid; i < kM * kP34N; i += blockDim.x) {
                    const int row = i / kP34N;
                    const int col = i % kP34N;
                    global_output[out_offset + i] = *reinterpret_cast<BF16*>(
                        &shared_u[swizzle_32b_offset(
                            row, col * int(sizeof(BF16)))]);
                }
                __syncthreads();
                if (warp == 0) {
                    asm volatile(
                        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::
                            "r"(tmem), "r"(32));
                }
                return;
            }
        }

        const uint64_t u_desc = p34_desc(shared_u);
        const uint64_t p4_a_desc =
            diagnostic_mode == 2 ? a0_desc : u_desc;
        if (tid == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            if constexpr (TmemCarrier) {
                const uint32_t a_tmem = tmem + 16;
                asm volatile(
                    "{\n"
                    ".reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 "
                    "[%0], [%1], %2, %3, p;\n"
                    "}\n" ::
                        "r"(tmem), "r"(a_tmem), "l"(mqk_desc),
                        "r"(instruction_descriptor), "r"(0));
            } else {
                asm volatile(
                    "{\n"
                    ".reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 "
                    "[%0], %1, %2, %3, p;\n"
                    "}\n" ::
                        "r"(tmem), "l"(p4_a_desc), "l"(mqk_desc),
                        "r"(instruction_descriptor), "r"(0));
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                ".shared::cluster.b64 [%0];" ::
                    "r"(mbar)
                : "memory");
        }
        mbarrier_wait_parity(mbar, uint32_t((2 * repetition + 1) & 1));
        asm volatile("tcgen05.fence::after_thread_sync;");

#pragma unroll
        for (int col = 0; col < kP34N; col += 8) {
            const uint32_t source =
                tmem + (uint32_t(warp * 32) << 16) + uint32_t(col);
            float result[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=f"(result[0]), "=f"(result[1]), "=f"(result[2]),
                  "=f"(result[3]), "=f"(result[4]), "=f"(result[5]),
                  "=f"(result[6]), "=f"(result[7])
                : "r"(source));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            if (repetition + 1 == inner) {
                const int row = warp * 32 + lane;
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    global_output[out_offset + row * kP34N + col + j] =
                        __float2bfloat16(result[j]);
            }
        }
        asm volatile("tcgen05.fence::before_thread_sync;");
        __syncthreads();
    }

    if (warp == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::
                "r"(tmem), "r"(32));
    }
}

static P34Buffers allocate_p34(int grid, std::vector<BF16>* host_a0,
                               std::vector<BF16>* host_inv,
                               std::vector<BF16>* host_mqk) {
    const size_t a_count = size_t(grid) * kM * kK;
    const size_t b_count = size_t(grid) * kK * kP34N;
    const size_t output_count = size_t(grid) * kM * kP34N;
    std::vector<BF16> a0(a_count), inv(b_count), mqk(b_count);
    for (int block = 0; block < grid; ++block) {
        for (int m = 0; m < kM; ++m)
            for (int k = 0; k < kK; ++k)
                // Keep K genuinely non-constant so validation detects a
                // misplaced 16-byte swizzle sub-block.
                a0[(size_t(block) * kM + m) * kK + k] =
                    __float2bfloat16(float(((m * 5 + k * 2 + block) % 3) - 1));
        for (int k = 0; k < kK; ++k) {
            for (int n = 0; n < kP34N; ++n) {
                inv[(size_t(block) * kK + k) * kP34N + n] =
                    __float2bfloat16(float(((k * 7 + n * 2 + block) % 3) - 1));
                mqk[(size_t(block) * kK + k) * kP34N + n] =
                    __float2bfloat16(float(((k * 2 + n * 5 + block) % 3) - 1));
            }
        }
    }
    P34Buffers device;
    CUDA_CHECK(cudaMalloc(&device.a0, a_count * sizeof(BF16)));
    CUDA_CHECK(cudaMalloc(&device.inv, b_count * sizeof(BF16)));
    CUDA_CHECK(cudaMalloc(&device.mqk, b_count * sizeof(BF16)));
    CUDA_CHECK(cudaMalloc(&device.output, output_count * sizeof(BF16)));
    CUDA_CHECK(cudaMemcpy(device.a0, a0.data(), a_count * sizeof(BF16),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.inv, inv.data(), b_count * sizeof(BF16),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.mqk, mqk.data(), b_count * sizeof(BF16),
                          cudaMemcpyHostToDevice));
    if (host_a0) *host_a0 = std::move(a0);
    if (host_inv) *host_inv = std::move(inv);
    if (host_mqk) *host_mqk = std::move(mqk);
    return device;
}

static std::vector<BF16> reference_p34(const std::vector<BF16>& a0,
                                       const std::vector<BF16>& inv,
                                       const std::vector<BF16>& mqk,
                                       int grid) {
    std::vector<BF16> u(size_t(grid) * kM * kP34N);
    std::vector<BF16> output(size_t(grid) * kM * kP34N);
    for (int block = 0; block < grid; ++block) {
        for (int m = 0; m < kM; ++m) {
            for (int n = 0; n < kP34N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < kK; ++k)
                    sum += __bfloat162float(
                               a0[(size_t(block) * kM + m) * kK + k]) *
                           __bfloat162float(
                               inv[(size_t(block) * kK + k) * kP34N + n]);
                u[(size_t(block) * kM + m) * kP34N + n] =
                    __float2bfloat16(sum);
            }
        }
        for (int m = 0; m < kM; ++m) {
            for (int n = 0; n < kP34N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < kK; ++k)
                    sum += __bfloat162float(
                               u[(size_t(block) * kM + m) * kP34N + k]) *
                           __bfloat162float(
                               mqk[(size_t(block) * kK + k) * kP34N + n]);
                output[(size_t(block) * kM + m) * kP34N + n] =
                    __float2bfloat16(sum);
            }
        }
    }
    return output;
}

static std::vector<BF16> reference_p3(const std::vector<BF16>& a0,
                                      const std::vector<BF16>& inv,
                                      int grid) {
    std::vector<BF16> u(size_t(grid) * kM * kP34N);
    for (int block = 0; block < grid; ++block)
        for (int m = 0; m < kM; ++m)
            for (int n = 0; n < kP34N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < kK; ++k)
                    sum += __bfloat162float(
                               a0[(size_t(block) * kM + m) * kK + k]) *
                           __bfloat162float(
                               inv[(size_t(block) * kK + k) * kP34N + n]);
                u[(size_t(block) * kM + m) * kP34N + n] =
                    __float2bfloat16(sum);
            }
    return u;
}

static std::vector<BF16> reference_direct_p4(
    const std::vector<BF16>& a0, const std::vector<BF16>& mqk, int grid) {
    std::vector<BF16> output(size_t(grid) * kM * kP34N);
    for (int block = 0; block < grid; ++block)
        for (int m = 0; m < kM; ++m)
            for (int n = 0; n < kP34N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < kK; ++k)
                    sum += __bfloat162float(
                               a0[(size_t(block) * kM + m) * kK + k]) *
                           __bfloat162float(
                               mqk[(size_t(block) * kK + k) * kP34N + n]);
                output[(size_t(block) * kM + m) * kP34N + n] =
                    __float2bfloat16(sum);
            }
    return output;
}

static bool validate_p34(int grid, const std::vector<int>& inners) {
    std::vector<BF16> a0, inv, mqk;
    P34Buffers device = allocate_p34(grid, &a0, &inv, &mqk);
    const auto reference = reference_p34(a0, inv, mqk, grid);
    const auto phase3_reference = reference_p3(a0, inv, grid);
    const auto direct_p4_reference = reference_direct_p4(a0, mqk, grid);
    const size_t count = size_t(grid) * kM * kP34N;
    bool ok = true;
    p34_tcgen_carrier<false><<<grid, kThreads>>>(
        device.a0, device.inv, device.mqk, device.output, 1, true);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<BF16> phase3_got(count);
    CUDA_CHECK(cudaMemcpy(phase3_got.data(), device.output,
                          count * sizeof(BF16), cudaMemcpyDeviceToHost));
    size_t phase3_bad = 0;
    for (size_t i = 0; i < count; ++i)
        if (bf16_bits(phase3_got[i]) != bf16_bits(phase3_reference[i]))
            ++phase3_bad;
    std::printf("VALIDATE,path=tcgen_phase3_carrier,grid=%d,inner=1,%s,bad=%zu\n",
                grid, phase3_bad ? "FAIL" : "PASS", phase3_bad);
    ok &= phase3_bad == 0;
    CUDA_CHECK(cudaMemset(device.output, 0xff, count * sizeof(BF16)));
    p34_tcgen_carrier<false><<<grid, kThreads>>>(
        device.a0, device.inv, device.mqk, device.output, 1, 2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<BF16> direct_p4_got(count);
    CUDA_CHECK(cudaMemcpy(direct_p4_got.data(), device.output,
                          count * sizeof(BF16), cudaMemcpyDeviceToHost));
    size_t direct_p4_bad = 0;
    for (size_t i = 0; i < count; ++i)
        if (bf16_bits(direct_p4_got[i]) !=
            bf16_bits(direct_p4_reference[i]))
            ++direct_p4_bad;
    std::printf("VALIDATE,path=tcgen_second_issue_a0,grid=%d,inner=1,%s,bad=%zu\n",
                grid, direct_p4_bad ? "FAIL" : "PASS", direct_p4_bad);
    ok &= direct_p4_bad == 0;
    for (int inner : inners) {
        for (const char* path :
             {"hmma", "tcgen_shared_carrier", "tcgen_tmem_carrier"}) {
            CUDA_CHECK(cudaMemset(device.output, 0xff, count * sizeof(BF16)));
            if (std::strcmp(path, "hmma") == 0)
                p34_hmma<<<grid, kThreads>>>(device.a0, device.inv,
                                             device.mqk, device.output, inner);
            else if (std::strcmp(path, "tcgen_shared_carrier") == 0)
                p34_tcgen_carrier<false><<<grid, kThreads>>>(
                    device.a0, device.inv, device.mqk, device.output, inner,
                    false);
            else
                p34_tcgen_carrier<true><<<grid, kThreads>>>(
                    device.a0, device.inv, device.mqk, device.output, inner,
                    false);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<BF16> got(count);
            CUDA_CHECK(cudaMemcpy(got.data(), device.output,
                                  count * sizeof(BF16),
                                  cudaMemcpyDeviceToHost));
            size_t bad = 0;
            for (size_t i = 0; i < count; ++i) {
                if (bf16_bits(got[i]) != bf16_bits(reference[i])) {
                    if (bad < 20)
                        std::fprintf(stderr,
                                     "mismatch path=%s inner=%d index=%zu "
                                     "got=%g want=%g\n",
                                     path, inner, i, __bfloat162float(got[i]),
                                     __bfloat162float(reference[i]));
                    ++bad;
                }
            }
            std::printf("VALIDATE,path=%s,grid=%d,inner=%d,%s,bad=%zu\n",
                        path, grid, inner, bad ? "FAIL" : "PASS", bad);
            ok &= bad == 0;
        }
    }
    return ok;
}

static void benchmark_p34(int grid, int inner, const Options& options,
                          std::ostream* csv) {
    P34Buffers device = allocate_p34(grid, nullptr, nullptr, nullptr);
    auto hmma_launch = [&] {
        p34_hmma<<<grid, kThreads>>>(device.a0, device.inv, device.mqk,
                                     device.output, inner);
    };
    auto shared_launch = [&] {
        p34_tcgen_carrier<false><<<grid, kThreads>>>(
            device.a0, device.inv, device.mqk, device.output, inner, false);
    };
    auto tmem_launch = [&] {
        p34_tcgen_carrier<true><<<grid, kThreads>>>(
            device.a0, device.inv, device.mqk, device.output, inner, false);
    };
    std::vector<double> hmma, shared, tmem;
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        if (repeat % 3 == 0) {
            hmma.push_back(time_batch_us(hmma_launch, options.warmup,
                                         options.iterations));
            shared.push_back(time_batch_us(shared_launch, options.warmup,
                                           options.iterations));
            tmem.push_back(time_batch_us(tmem_launch, options.warmup,
                                         options.iterations));
        } else if (repeat % 3 == 1) {
            shared.push_back(time_batch_us(shared_launch, options.warmup,
                                           options.iterations));
            tmem.push_back(time_batch_us(tmem_launch, options.warmup,
                                         options.iterations));
            hmma.push_back(time_batch_us(hmma_launch, options.warmup,
                                         options.iterations));
        } else {
            tmem.push_back(time_batch_us(tmem_launch, options.warmup,
                                         options.iterations));
            hmma.push_back(time_batch_us(hmma_launch, options.warmup,
                                         options.iterations));
            shared.push_back(time_batch_us(shared_launch, options.warmup,
                                           options.iterations));
        }
    }
    cudaFuncAttributes hmma_attributes{}, shared_attributes{}, tmem_attributes{};
    int hmma_active = 0, shared_active = 0, tmem_active = 0;
    CUDA_CHECK(cudaFuncGetAttributes(&hmma_attributes, p34_hmma));
    CUDA_CHECK(cudaFuncGetAttributes(&shared_attributes,
                                     p34_tcgen_carrier<false>));
    CUDA_CHECK(cudaFuncGetAttributes(&tmem_attributes,
                                     p34_tcgen_carrier<true>));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &hmma_active, p34_hmma, kThreads, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &shared_active, p34_tcgen_carrier<false>, kThreads, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &tmem_active, p34_tcgen_carrier<true>, kThreads, 0));
    const double hmma_us = median(hmma);
    const double shared_us = median(shared);
    const double tmem_us = median(tmem);
    std::ostringstream row;
    row.setf(std::ios::fixed);
    row.precision(6);
    row << grid << ',' << inner << ',' << options.warmup << ','
        << options.iterations << ',' << options.repeats << ',' << hmma_us
        << ',' << shared_us << ',' << tmem_us << ','
        << hmma_us / shared_us << ',' << hmma_us / tmem_us << ','
        << shared_us / tmem_us << ',' << hmma_us / double(inner) << ','
        << shared_us / double(inner) << ',' << tmem_us / double(inner) << ','
        << hmma_attributes.numRegs << ',' << shared_attributes.numRegs << ','
        << tmem_attributes.numRegs << ','
        << hmma_attributes.sharedSizeBytes << ','
        << shared_attributes.sharedSizeBytes << ','
        << tmem_attributes.sharedSizeBytes << ',' << hmma_active << ','
        << shared_active << ',' << tmem_active
        << ",p3_bf16_round_p4_single_tmem_allocation_carrier_ablation";
    std::printf("RESULT,%s\n", row.str().c_str());
    if (csv) {
        *csv << row.str() << '\n';
        csv->flush();
    }
}

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse_options(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "argument error: %s\n", error.what());
        return 2;
    }
    int device_index = 0;
    CUDA_CHECK(cudaGetDevice(&device_index));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device_index));
    int driver = 0, runtime = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    std::printf("META,device=%s,cc=%d.%d,sm_count=%d,driver=%d,runtime=%d,"
                "D=%d,N=%d,K=%d,threads=%d,not_full_K2=1\n",
                properties.name, properties.major, properties.minor,
                properties.multiProcessorCount, driver, runtime,
                kM, kP34N, kK, kThreads);
    if (properties.major != 10 || properties.minor != 3) return 3;

    if (options.validate &&
        !validate_p34(options.check_grid, options.check_inners))
        return 4;

    std::ofstream csv_file;
    std::ostream* csv = nullptr;
    static const char* header =
        "grid,inner,warmup,iters,repeats,hmma_median_us,"
        "tcgen_shared_median_us,tcgen_tmem_median_us,"
        "shared_speedup,tmem_speedup,tmem_over_shared_speedup,"
        "hmma_phase_us,tcgen_shared_phase_us,tcgen_tmem_phase_us,"
        "hmma_regs,tcgen_shared_regs,tcgen_tmem_regs,hmma_smem_B,"
        "tcgen_shared_smem_B,tcgen_tmem_smem_B,"
        "hmma_active_blocks_per_sm,tcgen_shared_active_blocks_per_sm,"
        "tcgen_tmem_active_blocks_per_sm,scope";
    if (!options.csv_path.empty()) {
        csv_file.open(options.csv_path);
        csv_file << header << '\n';
        csv = &csv_file;
    }
    if (options.benchmark) {
        std::printf("CSV_HEADER,%s\n", header);
        for (int grid : options.grids)
            for (int inner : options.inners)
                benchmark_p34(grid, inner, options, csv);
    }
    return 0;
}
