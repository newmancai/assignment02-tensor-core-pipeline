// ============================================================================
//

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#define BM 128
#define BN 64
#define BK 64
#define THREADS 128

#ifndef STAGES
#define STAGES 3
#endif

#define CU_CHECK(x) do { CUresult r = (x);                          \
    if (r != CUDA_SUCCESS) { const char* s = nullptr;               \
        cuGetErrorName(r, &s);                                      \
        printf("CUDA driver error %s at %s:%d\n", s, __FILE__, __LINE__); \
        exit(1); } } while (0)

// ---- descriptor / idesc / elect / mbarrier helpers -------------------------
__device__ __forceinline__ uint64_t make_smem_desc(const void* smem_addr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_addr));
    uint64_t desc = 0;
    desc |= (uint64_t)((addr >> 4) & 0x3FFF);
    desc |= (uint64_t)0 << 16;
    desc |= (uint64_t)((1024u >> 4) & 0x3FFF) << 32;
    desc |= (uint64_t)1 << 46;
    desc |= (uint64_t)2 << 61;
    return desc;
}

__device__ __forceinline__ uint32_t make_idesc() {
    uint32_t id = 0;
    id |= 1u << 4;
    id |= 1u << 7;
    id |= 1u << 10;
    id |= (uint32_t)(BN >> 3) << 17;
    id |= (uint32_t)(BM >> 4) << 24;
    return id;
}

__device__ __forceinline__ bool elect_one() {
    uint32_t pred = 0;
    asm volatile(
        "{\n .reg .pred p;\n elect.sync _|p, 0xFFFFFFFF;\n"
        " selp.b32 %0, 1, 0, p;\n}\n" : "+r"(pred));
    return pred != 0;
}

__device__ __forceinline__ void mbar_init(uint64_t* mbar, uint32_t count) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count));
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* mbar, uint32_t bytes) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"(addr), "r"(bytes));
}

__device__ __forceinline__ void mbar_wait(uint64_t* mbar, uint32_t phase) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "{\n .reg .pred p;\nWAIT:\n"
        " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        " @!p bra WAIT;\n}\n" :: "r"(addr), "r"(phase));
}

__device__ __forceinline__ void tma_load_2d(const CUtensorMap* tmap,
                                            void* smem_dst,
                                            uint64_t* mbar,
                                            int c0, int c1) {
    uint32_t dst = (uint32_t)__cvta_generic_to_shared(smem_dst);
    uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
        "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(reinterpret_cast<uint64_t>(tmap)),
           "r"(c0), "r"(c1), "r"(mbar_addr)
        : "memory");
}

// ---------------------------------------------------------------------------
// pipelined kernel
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(THREADS)
gemm_pipeline_kernel(const __grid_constant__ CUtensorMap tmapA,
                     const __grid_constant__ CUtensorMap tmapB,
                     __nv_bfloat16* __restrict__ C,
                     int M, int N, int K) {
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp = tid / 32;

    constexpr uint32_t TMEM_COLS = BN;
    constexpr uint32_t TILE_BYTES = (BM * BK + BK * BN) * sizeof(__nv_bfloat16);

    // STAGES sets of A/B ring buffers. They must use opt-in dynamic shared
    // memory because S=3 already exceeds the 48 KiB static limit.
    extern __shared__ __align__(1024) __nv_bfloat16 shared_tiles[];
    auto sA = reinterpret_cast<__nv_bfloat16 (*)[BM * BK]>(shared_tiles);
    auto sB = reinterpret_cast<__nv_bfloat16 (*)[BK * BN]>(
        shared_tiles + STAGES * BM * BK);
    __shared__ __align__(8) uint64_t full[STAGES];   // TMA -> mma
    __shared__ __align__(8) uint64_t empty[STAGES];  // mma -> TMA
    __shared__ uint32_t tmem_base;

    if (warp == 0) {
        uint32_t dst = (uint32_t)__cvta_generic_to_shared(&tmem_base);
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(dst), "r"(TMEM_COLS));
        asm volatile(
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full[s], 1);
            mbar_init(&empty[s], 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    const uint32_t tmem = tmem_base;

    const int n_k = K / BK;

    // A single elected lane drives asynchronous TMA and MMA issue.
    if (warp == 0 && elect_one()) {
        uint32_t idesc = make_idesc();
        uint32_t full_phase[STAGES], empty_phase[STAGES];
        #pragma unroll
        for (int s = 0; s < STAGES; ++s) { full_phase[s] = 0; empty_phase[s] = 0; }

        // Fill every available stage before entering steady state.
        int preload = n_k < STAGES ? n_k : STAGES;
        for (int kt = 0; kt < preload; ++kt) {
            mbar_expect_tx(&full[kt], TILE_BYTES);
            tma_load_2d(&tmapA, sA[kt], &full[kt], kt * BK, bm);
            tma_load_2d(&tmapB, sB[kt], &full[kt], kt * BK, bn);
        }

        for (int kt = 0; kt < n_k; ++kt) {
            int s = kt % STAGES;

            // Consumer must not issue MMA until both TMA transfers complete.
            mbar_wait(&full[s], full_phase[s]);
            full_phase[s] ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");

            #pragma unroll
            for (int kk = 0; kk < BK / 16; ++kk) {
                uint64_t adesc = make_smem_desc(sA[s] + kk * 16);
                uint64_t bdesc = make_smem_desc(sB[s] + kk * 16);
                uint32_t scale_c = (kt > 0 || kk > 0) ? 1u : 0u;
                asm volatile(
                    "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
                    " tcgen05.mma.cta_group::1.kind::f16"
                    "   [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(tmem), "l"(adesc), "l"(bdesc), "r"(idesc),
                       "r"(scale_c));
            }
            uint32_t ea = (uint32_t)__cvta_generic_to_shared(&empty[s]);
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                ".shared::cluster.b64 [%0];"
                :: "r"(ea));

            // Refill the stage just consumed. Waiting for empty[s] prevents
            // TMA from overwriting operands still used by an in-flight MMA.
            int knext = kt + STAGES;
            if (knext < n_k) {
                mbar_wait(&empty[s], empty_phase[s]);
                empty_phase[s] ^= 1;
                mbar_expect_tx(&full[s], TILE_BYTES);
                tma_load_2d(&tmapA, sA[s], &full[s], knext * BK, bm);
                tma_load_2d(&tmapB, sB[s], &full[s], knext * BK, bn);
            }
        }

        // Drain the final MMA before the epilogue reads TMEM.
        int last = (n_k - 1) % STAGES;
        mbar_wait(&empty[last], empty_phase[last]);
    }
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    // ---- epilogue闂佹寧绋掗鐢en05.ld -> bf16 -> global闂佹寧绋戦悧鍛箔?4.1/4.2 闂佺儵鏅濋…鍫ュ箖閹剧粯鏅?--------
    float acc[64];
    {
        uint32_t* r = reinterpret_cast<uint32_t*>(acc);
        uint32_t addr0 = tmem + ((uint32_t)(warp * 32) << 16);
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
            " [%32];"
            : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),
              "=r"(r[5]),"=r"(r[6]),"=r"(r[7]),"=r"(r[8]),"=r"(r[9]),
              "=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),
              "=r"(r[15]),"=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),
              "=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),"=r"(r[24]),
              "=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),
              "=r"(r[30]),"=r"(r[31]) : "r"(addr0));
        uint32_t* r2 = reinterpret_cast<uint32_t*>(acc + 32);
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
            " [%32];"
            : "=r"(r2[0]),"=r"(r2[1]),"=r"(r2[2]),"=r"(r2[3]),"=r"(r2[4]),
              "=r"(r2[5]),"=r"(r2[6]),"=r"(r2[7]),"=r"(r2[8]),"=r"(r2[9]),
              "=r"(r2[10]),"=r"(r2[11]),"=r"(r2[12]),"=r"(r2[13]),"=r"(r2[14]),
              "=r"(r2[15]),"=r"(r2[16]),"=r"(r2[17]),"=r"(r2[18]),"=r"(r2[19]),
              "=r"(r2[20]),"=r"(r2[21]),"=r"(r2[22]),"=r"(r2[23]),"=r"(r2[24]),
              "=r"(r2[25]),"=r"(r2[26]),"=r"(r2[27]),"=r"(r2[28]),"=r"(r2[29]),
              "=r"(r2[30]),"=r"(r2[31]) : "r"(addr0 + 32));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
    }

    const int row = bm + warp * 32 + (tid % 32);
    if (row < M) {
        #pragma unroll
        for (int j = 0; j < BN; ++j) {
            int col = bn + j;
            if (col < N) C[(size_t)row * N + col] = __float2bfloat16(acc[j]);
        }
    }

    __syncthreads();
    if (warp == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :: "r"(tmem), "r"(TMEM_COLS));
    }
}

// ---------------------------------------------------------------------------
static void make_tmap_2d(CUtensorMap* tmap, const __nv_bfloat16* base,
                         uint64_t rows, uint64_t cols,
                         uint32_t box_rows, uint32_t box_cols) {
    cuuint64_t globalDim[2]   = {cols, rows};
    cuuint64_t globalStrid[1] = {cols * sizeof(__nv_bfloat16)};
    cuuint32_t boxDim[2]      = {box_cols, box_rows};
    cuuint32_t elemStrid[2]   = {1, 1};
    CU_CHECK(cuTensorMapEncodeTiled(
        tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        const_cast<__nv_bfloat16*>(base),
        globalDim, globalStrid, boxDim, elemStrid,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

static void fill_random(__nv_bfloat16* p, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; ++i)
        p[i] = __float2bfloat16((float)rand() / RAND_MAX * 2.f - 1.f);
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 4096;
    int N = (argc > 2) ? atoi(argv[2]) : 4096;
    int K = (argc > 3) ? atoi(argv[3]) : 4096;
    printf("[03_pipeline] M=%d N=%d K=%d  tile=%dx%dx%d  STAGES=%d\n",
           M, N, K, BM, BN, BK, STAGES);
    if (M % BM || N % BN || K % BK) {
        printf("ERROR: shape must be divisible by tile %dx%dx%d\n", BM, BN, BK);
        return 1;
    }

    constexpr size_t smem_bytes =
        (size_t)STAGES * (BM * BK + BK * BN) * sizeof(__nv_bfloat16);
    cudaFuncSetAttribute(gemm_pipeline_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);
    int active_blocks_per_sm = 0;
    cudaError_t occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm, gemm_pipeline_kernel, THREADS, smem_bytes);
    if (occ_err != cudaSuccess) {
        printf("occupancy query failed: %s\n", cudaGetErrorString(occ_err));
        return 1;
    }
    cudaFuncAttributes func_attr{};
    cudaDeviceProp device_prop{};
    cudaFuncGetAttributes(&func_attr, gemm_pipeline_kernel);
    cudaGetDeviceProperties(&device_prop, 0);
    printf("STAGES=%d  dynamic_smem=%zu B  max_resident_blocks_per_sm=%d\n",
           STAGES, smem_bytes, active_blocks_per_sm);
    printf("  num_regs/thread=%d  static_smem=%zu B  smem_per_sm=%zu B"
           "  regs_per_sm=%d  hw_max_blocks_per_sm=%d\n",
           func_attr.numRegs, func_attr.sharedSizeBytes,
           device_prop.sharedMemPerMultiprocessor,
           device_prop.regsPerMultiprocessor,
           device_prop.maxBlocksPerMultiProcessor);
#ifdef OCCUPANCY_ONLY
    return 0;
#endif

    __nv_bfloat16 *hA = (__nv_bfloat16*)malloc((size_t)M * K * 2);
    __nv_bfloat16 *hB = (__nv_bfloat16*)malloc((size_t)N * K * 2);
    __nv_bfloat16 *hC = (__nv_bfloat16*)malloc((size_t)M * N * 2);
    __nv_bfloat16 *hRef = (__nv_bfloat16*)malloc((size_t)M * N * 2);
    fill_random(hA, (size_t)M * K, 1);
    fill_random(hB, (size_t)N * K, 2);

    __nv_bfloat16 *dA, *dB, *dC, *dRef;
    cudaMalloc(&dA, (size_t)M * K * 2);
    cudaMalloc(&dB, (size_t)N * K * 2);
    cudaMalloc(&dC, (size_t)M * N * 2);
    cudaMalloc(&dRef, (size_t)M * N * 2);
    cudaMemcpy(dA, hA, (size_t)M * K * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, (size_t)N * K * 2, cudaMemcpyHostToDevice);

    CUtensorMap tmapA, tmapB;
    make_tmap_2d(&tmapA, dA, M, K, BM, BK);
    make_tmap_2d(&tmapB, dB, N, K, BN, BK);

    dim3 grid(N / BN, M / BM);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    gemm_pipeline_kernel<<<grid, THREADS, smem_bytes>>>(tmapA, tmapB, dC, M, N, K);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
    cudaEventRecord(t0);
    for (int it = 0; it < 10; ++it)
        gemm_pipeline_kernel<<<grid, THREADS, smem_bytes>>>(tmapA, tmapB, dC, M, N, K);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1); ms /= 10;
    double tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
    printf("STAGES=%d  %7d %7d %7d  time=%.3f ms  %.1f TFLOPS\n",
           STAGES, M, N, K, ms, tflops);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.f, beta = 0.f;
    auto cublas_run = [&] {
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                     &alpha, dB, CUDA_R_16BF, K, dA, CUDA_R_16BF, K,
                     &beta, dRef, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);
    };
    cublas_run();
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int it = 0; it < 10; ++it) cublas_run();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float cublas_ms = 0;
    cudaEventElapsedTime(&cublas_ms, t0, t1);
    cublas_ms /= 10;
    double cublas_tflops = 2.0 * M * N * K / (cublas_ms * 1e-3) / 1e12;
    cudaMemcpy(hC, dC, (size_t)M * N * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hRef, dRef, (size_t)M * N * 2, cudaMemcpyDeviceToHost);
    size_t bad = 0;
    for (size_t i = 0; i < (size_t)M * N; ++i)
        bad += reinterpret_cast<uint16_t*>(hC)[i] !=
               reinterpret_cast<uint16_t*>(hRef)[i];
    printf("cuBLAS %.3f ms %.1f TFLOPS; exact %s (bad=%zu); attainment %.1f%%\n",
           cublas_ms, cublas_tflops, bad ? "FAIL" : "PASS", bad,
           100.0 * tflops / cublas_tflops);
    cublasDestroy(handle);

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
    free(hA); free(hB); free(hC); free(hRef);
    return bad != 0;
}
