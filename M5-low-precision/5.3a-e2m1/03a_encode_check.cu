// 问题 5.3(a) 的判测程序,不需要修改。B300(sm_100 家族)上运行。
//
// 真值就是硬件:kernel 里用 cuda_fp4.h 的 __nv_fp4x2_e2m1(单条
// F2FP.SATFINITE.E2M1 指令)转换每个候选值,和你在 e2m1_encode.h 里
// 写的编码器逐点比对。候选集合 = 全部格点中点及其邻域 + 均匀网格 +
// 随机采样。你的软件实现必须与硬件逐位一致。
#include <cuda_fp4.h>
#include <vector>
#include <random>
#include "../common.h"
#include "nvfp4_common.h"
#include "e2m1_encode.h"

__global__ void hw_encode_kernel(const float* in, uint8_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __nv_fp4x2_e2m1 p(make_float2(in[i], 0.f));
    out[i] = *reinterpret_cast<uint8_t*>(&p) & 0xF;  // 低 nibble = 第一个值
}

int main() {
    std::vector<float> cand;
    // 中点与邻域(两侧各偏 1 ulp 量级)
    const float mids[] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    for (float m : mids)
        for (float d : {-1e-6f, 0.f, 1e-6f}) cand.push_back(m + d);
    // 格点本身、饱和区、零
    for (float g : {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, 6.5f, 100.f})
        cand.push_back(g);
    // 均匀网格 + 随机
    for (int i = 0; i <= 1400; i++) cand.push_back(i * 0.005f);
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(0.f, 8.f);
    for (int i = 0; i < 100000; i++) cand.push_back(dist(rng));
    // 负值镜像
    size_t pos = cand.size();
    for (size_t i = 0; i < pos; i++) cand.push_back(-cand[i]);

    int n = (int)cand.size();
    float* din;
    uint8_t* dout;
    CUDA_CHECK(cudaMalloc(&din, n * 4));
    CUDA_CHECK(cudaMalloc(&dout, n));
    CUDA_CHECK(cudaMemcpy(din, cand.data(), n * 4, cudaMemcpyHostToDevice));
    hw_encode_kernel<<<(n + 255) / 256, 256>>>(din, dout, n);
    CUDA_CHECK_KERNEL();
    std::vector<uint8_t> hw(n);
    CUDA_CHECK(cudaMemcpy(hw.data(), dout, n, cudaMemcpyDeviceToHost));

    long bad = 0;
    for (int i = 0; i < n; i++) {
        uint8_t mine = e2m1_encode(cand[i]) & 0xF;
        if (mine != hw[i]) {
            if (bad < 8)
                printf("MISMATCH x=%.7f: yours=0x%x hw=0x%x (decode %.1f vs %.1f)\n",
                       cand[i], mine, hw[i], e2m1_decode(mine),
                       e2m1_decode(hw[i]));
            bad++;
        }
    }
    if (bad)
        printf("FAIL: %ld / %d mismatches\n", bad, n);
    else
        printf("PASS: %d values match hardware\n", n);
    return bad != 0;
}
