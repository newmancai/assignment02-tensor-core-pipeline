// 本目录所有练习共用的小工具,与 assignment01 的 common.h 同源,
// 增补了 bf16 与带宽相关的部分。
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",               \
                    cudaGetErrorName(err_), __FILE__, __LINE__,           \
                    cudaGetErrorString(err_));                            \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

#define CUDA_CHECK_KERNEL()                        \
    do {                                           \
        CUDA_CHECK(cudaGetLastError());            \
        CUDA_CHECK(cudaDeviceSynchronize());       \
    } while (0)

struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// warmup 后取 iters 次平均,返回毫秒。测带宽和 TFLOPS 都用它。
template <typename F>
static inline float time_avg_ms(F&& launch, int iters, int warmup = 20) {
    for (int i = 0; i < warmup; i++) launch();
    GpuTimer t;
    t.start();
    for (int i = 0; i < iters; i++) launch();
    float ms = t.stop_ms();
    CUDA_CHECK(cudaGetLastError());
    return ms / iters;
}

// 有效带宽:bytes 是"必须经过 HBM 的字节数"(读 + 写),自己算清楚再传。
static inline double effective_gbps(double bytes, float ms) {
    return bytes / (ms * 1e6);
}

// 关于半精度对拍的 tolerance:
// tensor core 的累加顺序与 CPU 循环不同,fp16/bf16 输入、fp32 累加的
// GEMM 结果通常不能和 CPU 参考 bit-exact,相对误差按 K 的量级放宽
// (经验值 K=4096 时 rtol 取 1e-2 量级,bf16 输出再叠一层 2^-8 的
// 输出舍入)。纯整数或小整数构造的用例可以做到严格相等,判测里
// 用哪种口径,各题文件头会写明。
static inline int check_close(const float *got, const float *want, long n,
                              float rtol) {
    long bad = 0;
    for (long i = 0; i < n; i++) {
        float w = want[i];
        if (fabsf(got[i] - w) > rtol * (1.0f + fabsf(w))) {
            if (bad < 5)
                fprintf(stderr, "MISMATCH at %ld: got %f, want %f\n", i,
                        got[i], w);
            bad++;
        }
    }
    if (bad) fprintf(stderr, "total mismatches: %ld / %ld\n", bad, n);
    return bad == 0;
}
