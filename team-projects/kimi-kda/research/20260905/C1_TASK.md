# C1:FlashKDA——官方 kernel 停在 SM80 MMA

## 背景

Kimi K3 的线性注意力部分是 KDA(Kimi Delta Attention)。Moonshot 开源的
FlashKDA(本目录 `FlashKDA/` 快照)是其高性能 forward kernel,在 GB200 上
对 fla 的 Triton `chunk_kda` 有 1.7-3.3 倍加速(`BENCHMARK_GB200.md`)。
值得注意的是:这份 kernel 的矩阵乘用的是 SM80 世代的 `mma.sync`,而不是
wgmma(SM90)或 tcgen05(SM100)——在 GB200/B300 上运行时同样如此。
作者在 `docs/20260420-flashkda-v1-deep-dive.md` 里解释了设计,但"为什么
停在 SM80 指令"这个决策的量化论证是留白的。这道题就是把这份论证做出来
——或者推翻它。

## 任务(三层)

1. 复现:在 B300 上装起 FlashKDA,跑通官方 benchmark(`benchmarks/`,
   形状对照 `BENCHMARK_GB200.md`),用 ncu/SASS 确认计算主路径确实是
   SM80 MMA(`benchmarks/ncu.sh` 是官方的 ncu 模板)。
2. 分析:下面的讨论点逐个给出"结论 + 证据"。量化类的先纸面推算,
   再用 microbench 验证。
3. 挑战:选 SM100 路线的任一切面动手——只换指令不动算法、大 CHUNK +
   rescale、并行度重构,任选其一。正确性对 `fla_kda_ref/` 的实现对拍
   (`naive.py` 是纯 PyTorch 朴素参考,`chunk.py` 是 Triton 参照),
   性能对 FlashKDA 本体。做不出正收益也算完成:把"官方停在 SM80 是
   对的"论证扎实,就是讨论点 6 的另一半答案。

交付:代码 + 报告 + 答辩。

## 讨论点

1. CHUNK=16 的三个理由——bf16 数值范围、16×16 Neumann 级数求逆、
   SM80 MMA 形状匹配——各自量化:CHUNK=32/64 时哪个先破,代价多大?
2. tcgen05 的最小 tile 与 CHUNK=16 的形状匹配吗?不动 CHUNK 只换指令
   有没有收益——先纸上算,再 microbench 验证。
3. 递推在 chunk 间有状态依赖,并行度还能从哪来(多 head 进一个 CTA /
   persistent kernel / 2-CTA)?列出候选方案,互相找反例。
4. 这个负载在你们的卡上是 compute-bound 还是 memory-bound?用哪几个
   ncu metric 回答?(assignment 4.5 的瘦 GEMM 表是现成的参照系,
   `in_proj_qkvgfab` 就是 KDA 的输入投影。)
5. 状态存 bf16 的精度验证怎么设计?官方只说内部测试通过,拿出你们的
   验证方案和数据。
6. 假设你们是作者:v2 出不出 sm100a 专版?把可移植性与峰值两边的论据
   都写全,给出你们的结论。

## 材料

- `FlashKDA/`:官方仓库快照,pin commit `1ce47ea`(2026-07-29)。
  cutlass 子模块未含在快照里,构建时用
  `git clone --recurse-submodules https://github.com/MoonshotAI/FlashKDA`
  后 `git checkout 1ce47ea`(cutlass pin `5c149f5`)。
  重点文件:`docs/20260420-flashkda-v1-deep-dive.md`(设计文档)、
  `BENCHMARK_GB200.md`(官方数据表)、`csrc/smxx/`(kernel 本体)、
  `benchmarks/`(bench 与 ncu 脚本)、README(chunk_kda 调用约定与
  dispatch 调试方法)。
- `fla_kda_ref/`:fla-org/flash-linear-attention 的 `fla/ops/kda/` 快照,
  pin commit `a3edffc`。`naive.py` 纯 PyTorch 参考;`chunk.py` 及其依赖
  是 Triton 参照;`backends/flash_kda.py` 是 fla 调用 FlashKDA 的适配层
  (两边张量约定的对照表)。
- 形状:K3 的 KDA 配置是 96 头 × head_dim 128(93 层中 69 层 KDA、
  24 层全注意力);官方 benchmark 的 `T=8192, H=96, D=128` 即此,
  H=64 组只是附加对照形状。注意 TP 部署下每卡头数是 96/TP(TP8 为
  12)——讨论并行度时用每卡数;GEMM 侧形状见 assignment 4.5。
