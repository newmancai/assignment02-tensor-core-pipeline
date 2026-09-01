// 问题 4.5(EXPERIMENT,负结果):瘦 GEMM——tensor core 什么时候帮不上忙。
//
// 形状取自 vLLM 主树 Kimi K3 的 decode GEMM dispatch 表
// (vllm/models/kimi_k3/nvidia/low_latency_gemm.py,针对 SM103 BF16):
// 每行是一个真实投影层的 (N, K);M 是这一步 forward 处理的 token 数
// ——decode 时 = batch(1-16 量级),chunked prefill 时 = chunk 大小
// (几千到几万),权重形状 N/K 与 context 长度无关。上游的做法本身
// 就是本题的题设:M<=16 时 vLLM 放弃 cuBLAS/tensor core 路径,换成
// 直接用 CUDA core FMA 的 skinny kernel(绕开 TMA 与 tensor core 的
// setup),微基准提升 8%-100%。
//
// 做法(先预测再实测):
//   1. 对表中每个 (M, N, K) 先手算 arithmetic intensity
//      AI = 2MNK / (2MK + 2NK + 2MN) [flop/byte],与你 0.2 算的机器
//      平衡点比较,预判 memory-bound 还是 compute-bound,预填每行的
//      理论上限(compute 侧 = 峰值 TFLOPS;memory 侧 = AI × 峰值带宽)
//   2. 跑本程序(cuBLAS bf16,f32 累加,bf16 输出),把实测列填进表
//   3. 回答 handout 4.5 的问题:M 多小时达成率塌掉;塌掉的那部分时间
//      去了哪(双分母:对 tensor core 峰值的达成率、对带宽 roofline 的
//      达成率,各自说明什么)
//
// 用法:./05_thin_gemm [峰值TFLOPS 峰值GB/s]
//   给出两个峰值参数(用你 0.2 推导的数)时,额外打印两列达成率。
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../common.h"

struct Shape {
    int n, k;
    const char* name;
};

// low_latency_gemm.py 的 KIMI_K3_PROJECTIONS 选出的代表形状(TP 切分
// 后的本地形状),每行都能从 K3 config 推出(hidden=7168,96 头,
// q_lora=1536,kv_lora=512,rope=64,dense intermediate=33792):
//   f_b_proj   门控 rank-128 瓶颈的第二段,12288/TP8,K=128 是极端小 K
//   q_b_proj   96x(128+64)=18432/TP8
//   o_proj     96x128=12288/TP8 进 hidden
//   fused_qkv_a_proj  1536+512+64=2112,不切分
//   in_proj_qkvgfab   KDA 输入投影(q+k+v+g 主体)/TP8——团队题 C1
//   dense_down_proj   33792/TP4(仅 layer 0 是 dense)
//   dense_gate_up_proj 2x33792/TP4
static const Shape SHAPES[] = {
    {1536, 128, "f_b_proj"},
    {2304, 1536, "q_b_proj"},
    {7168, 1536, "o_proj"},
    {2112, 7168, "fused_qkv_a_proj"},
    {6288, 7168, "in_proj_qkvgfab"},
    {7168, 8448, "dense_down_proj"},
    {16896, 7168, "dense_gate_up_proj"},
};
// M 轴覆盖 decode 到 prefill 全谱:1-16 是 decode batch(skinny kernel
// 接管区),64-256 是过渡,1024-65536 对应 chunked prefill 每步的
// token 数(K3 是 1M 上下文模型,但再长的 prompt 也按 chunk 进 GEMM,
// 每步的 M = chunk 大小;大 M 端用来看达成率的饱和平台)。
static const int MS[] = {1, 8, 16, 64, 256, 1024, 4096, 16384, 65536};

int main(int argc, char** argv) {
    double peak_tflops = argc > 2 ? atof(argv[1]) : 0;
    double peak_gbps = argc > 2 ? atof(argv[2]) : 0;

    int maxM = 65536, maxN = 0, maxK = 0;
    for (auto& s : SHAPES) {
        if (s.n > maxN) maxN = s.n;
        if (s.k > maxK) maxK = s.k;
    }
    size_t nA = (size_t)maxM * maxK, nW = (size_t)maxN * maxK,
           nD = (size_t)maxM * maxN;
    // 数据内容不影响耗时,用便宜的 xorshift 填个非零值即可
    std::vector<__nv_bfloat16> hA(nA);
    uint32_t x = 0x12345678;
    for (auto& v : hA) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        v = __float2bfloat16((float)(int)(x % 7) - 3.f);
    }
    __nv_bfloat16 *dA, *dW, *dD;
    CUDA_CHECK(cudaMalloc(&dA, nA * 2));
    CUDA_CHECK(cudaMalloc(&dW, nW * 2));
    CUDA_CHECK(cudaMalloc(&dD, nD * 2));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * 2, cudaMemcpyHostToDevice));
    // 权重缓冲大,用 A 的内容平铺填充
    for (size_t off = 0; off < nW; off += nA)
        CUDA_CHECK(cudaMemcpy(dW + off, dA,
                              (nW - off < nA ? nW - off : nA) * 2,
                              cudaMemcpyDeviceToDevice));

    cublasHandle_t h;
    cublasCreate(&h);
    float alpha = 1.f, beta = 0.f;

    printf("%-20s %5s %6s %6s %9s %9s %9s %7s", "layer", "M", "N", "K",
           "us", "TFLOPS", "GB/s", "AI");
    if (peak_tflops > 0) printf(" %8s %8s", "%TCpeak", "%BW");
    printf("\n");
    for (auto& s : SHAPES) {
        for (int M : MS) {
            // D[M,N] 行主序:C_col[N,M] = W_col[K,N]^T x A_col[K,M]
            auto launch = [&] {
                cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, s.n, M, s.k, &alpha,
                             dW, CUDA_R_16BF, s.k, dA, CUDA_R_16BF, s.k,
                             &beta, dD, CUDA_R_16BF, s.n, CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT);
            };
            launch();
            CUDA_CHECK(cudaDeviceSynchronize());
            int iters = M <= 256 ? 200 : (M <= 4096 ? 50 : 20);
            float ms = time_avg_ms(launch, iters);
            double flop = 2.0 * M * s.n * s.k;
            double bytes = 2.0 * ((double)M * s.k + (double)s.n * s.k +
                                  (double)M * s.n);
            double tflops = flop / (ms * 1e9);
            double gbps = bytes / (ms * 1e6);
            double ai = flop / bytes;
            printf("%-20s %5d %6d %6d %9.1f %9.1f %9.1f %7.1f", s.name, M,
                   s.n, s.k, ms * 1e3, tflops, gbps, ai);
            if (peak_tflops > 0)
                printf(" %7.1f%% %7.1f%%", 100.0 * tflops / peak_tflops,
                       100.0 * gbps / peak_gbps);
            printf("\n");
        }
        printf("\n");
    }
    cublasDestroy(h);
    return 0;
}
