# 关键结果复验指南

## 1. 环境边界

正式结果绑定 NVIDIA B300 SXM6 AC、148 个物理 SM、`sm_103a` 构建目标和归档中的实现身份。复验前先核对源文件、构建产物和二进制哈希；设备或工具链变化时，应将结果视为新 scope。

## 2. 推荐复验顺序

1. **官方身份**：用 `04_evidence/official/BUILD_MANIFEST.json` 与 `SHA256SUMS` 核对构建，再检查 `sass_opcode_summary.csv`。
2. **正确性**：运行 `05_reproduction/common/check_h3_gpu_correctness.py` 或对应路径的 public pair；同时检查 output 与 final state。
3. **H3 正式比较**：使用 `bench_h3_isolated_process.py`，确保每个 worker 只加载一种实现。
4. **CAKE/official 合并**：分别采集隔离 worker 结果，再由 `combine_isolated_cake_official.py` 合并。
5. **第八轮双分支**：HMMA 与 tcgen05 使用各自目录中的 README、source 和 batch 文件；预算以 `04_evidence/agent_rounds/round8_budget_receipt.json` 为准。
6. **证据分析**：使用 `analyze_round_evidence.py` 检查 block median、speedup、correctness 和晋级状态。
7. **Stage 12 HMMA**：按 `05_reproduction/round9_hmma/README.md` 构建 paired-warps，并与官方及当前 ValueSlice baseline 做 fresh-worker public-full 比较。
8. **Stage 12 tcgen 布局**：按 `05_reproduction/round9_tcgen/README.md` 比较 logical 与 producer-ready B；必须使用修正 swizzle 与 K-非退化输入的 Job 25380 口径，Job 25317 已被 supersede。
9. **Stage 12 P3/P4 lifecycle**：按 `05_reproduction/round10_tcgen/README.md` 运行正确的 shared-carrier 探针；先要求 inner=1/2/4 逐 bit PASS，再计时 grids 12/96 和 inner 1/64。该项仍是 mechanism probe，不得升级成 public-full 结论。
10. **闭包审计**：运行 108 项 agent-framework 测试，并核对 `04_evidence/agent_rounds/round9_closure_certificate.json` 的 envelope digest、缺失列表和 correctness-rejected 候选。

## 3. 正式计时要求

- public-full GPU span；
- JIT 与一次性分配在计时外；
- 每调用必要的 layout、workspace 和 state copy-back 在计时内；
- 非零 BF16 初始 state；
- cold L2；
- 20 次 warmup；
- 每 block 100 次正式样本；
- 4 个 balanced/ABBA blocks；
- 每个 worker 只加载一个 extension；
- 保存原始样本、block median、设备、route、source/build/binary identity。

开发 screen 可以减少重复次数，但不得替代 headline qualification。

## 4. 正确性要求

- 不改变 BF16 舍入 DAG 的候选，应优先要求 output/state bitwise equal；
- 改变运算顺序的算法等价候选，应在看结果前冻结共同 reference、误差指标与阈值；
- 至少覆盖 fixed、tail、multichunk、packed 和非零初始 state；
- 对递推路径增加连续两次调用，验证 state 回写确实可被下次调用消费。

## 5. 结果判读

- microprobe 只回答局部机制；
- kernel-only 不等于 public-full；
- 一个正例可以推翻“所有迁移都无收益”，但不能证明所有 profile 都应迁移；
- 一个负例只停止相同候选，除非合法性证据证明整个候选族不成立；
- 比较 HMMA 与 tcgen05 时，先确认两者是否处于同一 profile、同一 public-call 边界和同一正确性合同。

## 6. 未归档的大文件

为保持交付包精简，本目录没有复制完整编译缓存、全部原始日志和大体积 NCU `.ncu-rep`。它们仍保留在原实验树中，摘要和哈希由 `04_evidence/official/SHA256SUMS`、各 manifest 与 `SOURCE_MAP.tsv` 指向。需要做 profiler 深挖时再从原位置读取，不应仅凭归档中的汇总 CSV 推断全部 stall 原因。
