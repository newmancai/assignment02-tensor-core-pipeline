// ============================================================================
// prob 4.1 (FROM-SCRATCH): tiled GEMM, BM=128 x BN=64 x BK=64, bf16, f32 accum
//
//   C(M,N) = A(M,K) * B(K,N),  A/B row-major (K-major), C row-major
//
//
//   ./bin/m4_gemm/01_tiled [M N K]
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#define BM 128
#define BN 64
#define BK 64
#define THREADS 128

// ---------------------------------------------------------------------------
__device__ __forceinline__ int swizzle_128B(int row, int chunk) {
    return chunk ^ (row & 7);
}

// ---------------------------------------------------------------------------
// 64-bit shared memory matrix descriptor闂佹寧绋戦悧鎰帮綖?2.2 / F27闂佹寧绋戦顨篨 ISA "Shared
//   bits [16:30)  leading  dim byte offset (LBO) >> 4
//   bits [32:46)  stride   dim byte offset (SBO) >> 4
//   bits [61:64)  swizzle mode (0=none, 1=128B, 2=64B, 3=32B)
__device__ __forceinline__ uint64_t make_smem_desc(const void* smem_addr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_addr));
    uint64_t desc = 0;
    desc |= (uint64_t)((addr >> 4) & 0x3FFF);           // start address
    desc |= (uint64_t)0 << 16;                          // swizzled: LBO ignored
    desc |= (uint64_t)((1024u >> 4) & 0x3FFF) << 32;    // SBO = 1024 B
    desc |= (uint64_t)1 << 46;                          // SM100 descriptor version
    desc |= (uint64_t)2 << 61;                          // SM100 128B swizzle
    return desc;
}

// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t make_idesc() {
    uint32_t id = 0;
    id |= 1u << 4;              // d dtype: f32
    id |= 1u << 7;              // a dtype: bf16 (kind::f16 缂傚倸鍊归悧婊堟偉?
    id |= 1u << 10;             // b dtype: bf16
    id |= (uint32_t)(BN >> 3) << 17;   // N
    id |= (uint32_t)(BM >> 4) << 24;   // M
    return id;
}

// Elect one converged lane to issue tcgen05 operations.
__device__ __forceinline__ bool elect_one() {
    uint32_t pred = 0;
    asm volatile(
        "{\n .reg .pred p;\n"
        " elect.sync _|p, 0xFFFFFFFF;\n"
        " selp.b32 %0, 1, 0, p;\n}\n"
        : "+r"(pred));
    return pred != 0;
}

__device__ __forceinline__ void mbar_init(uint64_t* mbar, uint32_t count) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count));
}

__device__ __forceinline__ void mbar_wait(uint64_t* mbar, uint32_t phase) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "{\n .reg .pred p;\n"
        "WAIT:\n"
        " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        " @!p bra WAIT;\n}\n"
        :: "r"(addr), "r"(phase));
}

// ---------------------------------------------------------------------------
// GEMM kernel
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(THREADS)
gemm_tiled_kernel(const __nv_bfloat16* __restrict__ A,
                  const __nv_bfloat16* __restrict__ B,
                  __nv_bfloat16* __restrict__ C,
                  int M, int N, int K) {
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp = tid / 32;

    // m128n64 f32 accumulator occupies 64 TMEM columns.
    constexpr uint32_t TMEM_COLS = BN;

    // shared memory闂佹寧绋掗?tile (128x64 bf16, K-major+128B swizzle)
    //                B tile ( 64x64 bf16, 闂?layout)
    // Shared tiles, completion barrier, and TMEM base.
    __shared__ __align__(1024) __nv_bfloat16 sA[BM * BK];
    __shared__ __align__(1024) __nv_bfloat16 sB[BK * BN];
    __shared__ __align__(8) uint64_t mbar;
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
        mbar_init(&mbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    const uint32_t tmem = tmem_base;            // bits[31:16]=lane, [15:0]=col

    const int n_k = K / BK;                     // 闂佺顑呭ú銈夋偩?BK | K
    uint32_t phase = 0;

    for (int kt = 0; kt < n_k; ++kt) {
        const int k0 = kt * BK;

        // ---- (1) staging: global -> smem闂佹寧绋戝鍍?shared, 128B swizzle闂?-------
        // Ordinary loads/stores intentionally expose staging overhead.
        for (int i = tid; i < BM * BK; i += THREADS) {
            int row = i / BK;
            int k = i % BK;
            int physical = row * BK + swizzle_128B(row, k >> 3) * 8 + (k & 7);
            sA[physical] = A[(size_t)(bm + row) * K + k0 + k];
        }
        // B is stored as N x K so each row is contiguous in K.
        for (int i = tid; i < BN * BK; i += THREADS) {
            int row = i / BK;
            int k = i % BK;
            int physical = row * BK + swizzle_128B(row, k >> 3) * 8 + (k & 7);
            sB[physical] = B[(size_t)(bn + row) * K + k0 + k];
        }

        // Publish st.shared data from the generic proxy to the async proxy.
        asm volatile("fence.proxy.async.shared::cta;");
        __syncthreads();

        // Four k16 instructions cover BK=64; one elected lane issues them.
        if (warp == 0 && elect_one()) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            uint32_t idesc = make_idesc();
            #pragma unroll
            for (int kk = 0; kk < BK / 16; ++kk) {
                uint64_t adesc = make_smem_desc(sA + kk * 16);
                uint64_t bdesc = make_smem_desc(sB + kk * 16);
                uint32_t scale_c = (kt > 0 || kk > 0) ? 1u : 0u;
                asm volatile(
                    "{\n .reg .pred p;\n"
                    " setp.ne.b32 p, %4, 0;\n"
                    " tcgen05.mma.cta_group::1.kind::f16"
                    "   [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(tmem), "l"(adesc), "l"(bdesc), "r"(idesc),
                       "r"(scale_c));
            }
            uint32_t ma = (uint32_t)__cvta_generic_to_shared(&mbar);
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                ".shared::cluster.b64 [%0];"
                :: "r"(ma));
        }
        mbar_wait(&mbar, phase);
        phase ^= 1;
        __syncthreads();
    }

    // Two x32 loads fetch all BN=64 accumulator columns.
    asm volatile("tcgen05.fence::after_thread_sync;");
    float acc[64];
    {
        uint32_t* r = reinterpret_cast<uint32_t*>(acc);
        uint32_t addr0 = tmem + ((uint32_t)(warp * 32) << 16);
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
            " [%32];"
            : "=r"(r[ 0]),"=r"(r[ 1]),"=r"(r[ 2]),"=r"(r[ 3]),
              "=r"(r[ 4]),"=r"(r[ 5]),"=r"(r[ 6]),"=r"(r[ 7]),
              "=r"(r[ 8]),"=r"(r[ 9]),"=r"(r[10]),"=r"(r[11]),
              "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
              "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),
              "=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
              "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),
              "=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
            : "r"(addr0));
        uint32_t* r2 = reinterpret_cast<uint32_t*>(acc + 32);
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
            " [%32];"
            : "=r"(r2[ 0]),"=r"(r2[ 1]),"=r"(r2[ 2]),"=r"(r2[ 3]),
              "=r"(r2[ 4]),"=r"(r2[ 5]),"=r"(r2[ 6]),"=r"(r2[ 7]),
              "=r"(r2[ 8]),"=r"(r2[ 9]),"=r"(r2[10]),"=r"(r2[11]),
              "=r"(r2[12]),"=r"(r2[13]),"=r"(r2[14]),"=r"(r2[15]),
              "=r"(r2[16]),"=r"(r2[17]),"=r"(r2[18]),"=r"(r2[19]),
              "=r"(r2[20]),"=r"(r2[21]),"=r"(r2[22]),"=r"(r2[23]),
              "=r"(r2[24]),"=r"(r2[25]),"=r"(r2[26]),"=r"(r2[27]),
              "=r"(r2[28]),"=r"(r2[29]),"=r"(r2[30]),"=r"(r2[31])
            : "r"(addr0 + 32));
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

    // ---- TMEM 闂備焦褰冮敃銉╁棘?--------------------------------------------------------
    __syncthreads();
    if (warp == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :: "r"(tmem), "r"(TMEM_COLS));
    }
}

// ---------------------------------------------------------------------------
static void fill_random(__nv_bfloat16* p, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; ++i) {
        float v = (float)rand() / RAND_MAX * 2.f - 1.f;
        p[i] = __float2bfloat16(v);
    }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 4096;
    int N = (argc > 2) ? atoi(argv[2]) : 4096;
    int K = (argc > 3) ? atoi(argv[3]) : 4096;
    printf("[01_tiled] M=%d N=%d K=%d  tile=%dx%dx%d bf16/f32\n",
           M, N, K, BM, BN, BK);
    if (M % BM || N % BN || K % BK) {
        printf("ERROR: shape must be divisible by tile %dx%dx%d\n", BM, BN, BK);
        return 1;
    }

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

    dim3 grid(N / BN, M / BM);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    gemm_tiled_kernel<<<grid, THREADS>>>(dA, dB, dC, M, N, K);   // warmup
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int it = 0; it < 10; ++it)
        gemm_tiled_kernel<<<grid, THREADS>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1); ms /= 10;
    double tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
    printf("time = %.3f ms   %.1f TFLOPS\n", ms, tflops);

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
