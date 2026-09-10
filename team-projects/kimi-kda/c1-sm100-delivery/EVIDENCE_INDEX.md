# 结论—证据—适用范围索引

本索引用于回答两个问题：每句话由哪份证据支持，以及这句话最多能说到哪里。

## 1. 复现与身份

| 结论 | 主要证据 | 测量范围 | 允许表述 | 禁止外推 |
|---|---|---|---|---|
| 官方 recurrence 在 B300 上使用 HMMA | `04_evidence/official/sass_opcode_summary.csv`、`BUILD_MANIFEST.json` | 课程 pin、实际构建二进制、SM103a | recurrence SASS 有 3,640 条静态 HMMA，TCGEN/UTCMMA 为 0 | 整个 kernel 都是 SM80；所有上游版本均相同 |
| 测量对象可追溯 | `BUILD_MANIFEST.json`、`SHA256SUMS`、各路径 manifest | source/build/binary | 结论绑定冻结实现与哈希 | 只凭 backend 名称认定真实 route |

## 2. 正确性与计时合同

| 结论 | 主要证据 | 测量范围 | 允许表述 | 禁止外推 |
|---|---|---|---|---|
| 正式 gate 覆盖 output 与 final state | `04_evidence/cake/b300_stage3_public_paired.json`、`04_evidence/h3/h3_gpu_correctness.json` | public KDA、非零 state、所列 cases | 递推算子的正确性包含输出和更新后状态 | 只对一次 output 即证明跨调用正确 |
| 正式 magnitude 使用隔离 worker | `04_evidence/h3/h3_b300_isolated_process_cupti.json`、`04_evidence/cake/b300_stage12_cake_official_isolated_pair.json` | 每进程只加载一种 extension | 避免同名 CUDA symbol 互插影响对比 | 同进程旧结果仍作为 headline |
| CUPTI headline 采用统一协议 | 同上 JSON 的 `timing_policy` | cold L2、20 warmup、100 repeats、4 blocks | H3 与 CAKE 的正式结果具有统一 public-call 计时边界 | 将 microprobe 与 public-full 时间直接串联 |

## 3. HMMA 分支

| 结论 | 主要证据 | 测量范围 | 允许表述 | 禁止外推 |
|---|---|---|---|---|
| H12 存在独立工作不足 | `03_reports/SM100_MAINLINE_DELIVERY.md`、官方 NCU summary | TP8/H12、12 CTA、148 SM | 该 profile 更接近并行度/占用/关键路径受限 | 所有 H96、packed 和并发 workload 都相同 |
| ValueSlice 可改善特定 H12 profile | `04_evidence/direct_tcgen/stage11_mma_migration_memory.json`、agent writeback | 指定 H12/fixed 或 nseq profile | 增加独立 CTA 是有效优化方向 | V16/V32/V64 在所有 profile 永远最优 |
| HMMA nseq=3 的 V32/V64 边界 | `04_evidence/agent_rounds/round6_writeback.json`、`round7_writeback.json` | balanced/skew nseq=3 | V32 相对 V64 约 1.076× 并跨偏斜复现 | 把 nseq=3 的 winner 推到 nseq≥4 |
| nseq=4 出现反转 | `round6_writeback.json` | balanced nseq=4 | V32/V64 为 0.9618×，需 profile dispatch | 得出 V32 路线整体失败 |
| L3 lookahead 没有新增收益 | `round8_writeback.json` | H12 fixed8192、当前 public extension | L3/L2 为 0.999978，当前候选不晋级 | 所有 future lookahead/fragment reuse 永远无效 |
| 压缩 compute warps 明显回退 | `04_evidence/agent_rounds/round9_hmma/summary.json` | H12 balanced nseq3 V32、nseq6 V64，public-full | paired/baseline 为 0.8461x/0.8907x；保留当前 mapping | 所有 warp 重组都无效 |

## 4. tcgen05 分支

| 结论 | 主要证据 | 测量范围 | 允许表述 | 禁止外推 |
|---|---|---|---|---|
| Direct V128 tcgen05 更慢 | `04_evidence/direct_tcgen/03_tcgen05_probe_17937.csv`、`analysis_tcgen05.md` | Phase-6、V128/grid12、inner64 microprobe | 该 instruction-only candidate 为 0.919742× | tcgen05 整体不值得 |
| V16 core 有潜力 | `stage11_v16_tcgen_probe_24111.csv`、迁移 memory | V16 preferred-layout L0 | 该 core 为 1.615021× | 等同于完整 KDA 或生产级收益 |
| scalar rematerialization 吞掉收益 | 同上 | 当前 L1 integration probe | 集成后为 0.778523×，布局/转换是实际成本 | 已证明唯一瓶颈就是 layout |
| preferred-layout staging 修复部分成本 | `04_evidence/agent_rounds/round11_tcgen_revalidation/round8_v16_preferred_25379.csv` | D128/K16/V16、grid96、inner64 probe，修正 swizzle/验证 | preferred/scalar 为 1.2107× | 直接当成 T1 public-full 结果 |
| preferred probe 仍未赢 HMMA | 同上 | 同一 probe | preferred/HMMA 为 0.9564× | 否定跨 phase TMEM carrier |
| producer-ready global B 没有追回差距 | `04_evidence/agent_rounds/round11_tcgen_revalidation/` | Phase-6 V16 probe，inner=1/64 | physical/logical 为 0.99994x/0.99263x；inner64 physical/HMMA 为 0.94606x | P4 原位产生 layout 或跨阶段 TMEM 无效 |
| 长驻留 P3/P4 tcgen lifecycle 可超过 HMMA | `04_evidence/agent_rounds/round10_p34/` | 双 MMA mechanism，grids 12/96，inner64 | shared-carrier tcgen/HMMA 为 1.4460x/1.4303x | 当作 full public-call speedup |
| 短驻留不足以摊薄 lifecycle | 同上 | 双 MMA mechanism，inner1 | tcgen/HMMA 为 0.7651x/0.7089x | tcgen 在所有 profile 都慢 |
| D-fragment 不能无变换直通 A-TMEM | `04_evidence/agent_rounds/round12_tmem_carrier_rejection/` | 两种 compiled layout，inner1/2/4 | 两者均在计时前失败 24,576/24,576 输出 | 不存在任何合法 D→A 布局变换 |
| load-minimax 不是 latency 目标 | `round7_writeback.json` | exact tcgen schedule profile | MAX169/default 为 0.9383× | 所有负载均衡或调度优化无效 |

## 5. 完整挑战路径

| 结论 | 主要证据 | 测量范围 | 允许表述 | 禁止外推 |
|---|---|---|---|---|
| H3 应停止 | `04_evidence/h3/h3_b300_isolated_process_cupti.json`、`h3_gpu_correctness.json` | H12 六形状 + H96 三形状、public-full | H12/H96 geomean 为 0.905526×/0.758097× | 用一个 1.0219× case 宣布 H3 成功；推广到所有算法重排 |
| 完整 SM100 路径值得 | `04_evidence/cake/b300_stage12_cake_official_isolated_pair.json` | 同一 public contract、九个 profiles | 对官方 H12/H96 geomean 为 2.482262×/2.253190×，九个均正 | 归因成 tcgen05 单指令收益 |
| CAKE 是 full-stack treatment | `cake_sm100_full_stack.json`、`cake_source_binary_manifest.json` | source-grounded partial IR | 收益来自 tcgen/TMEM、数据流、角色、流水和 dispatch 的组合 | 已分离每个因素的边际贡献 |
| CAKE 对强化 HMMA 仍有优势迹象 | `04_evidence/agent_rounds/b300_ir_knowledge_smoke_summary.json` | 三形状探索性 smoke、历史 correctness contract | 三个已测形状分别约 1.530×/2.682×/2.391× | 当作最终共同 profile H1/T1/T2 qualification |

## 6. Agent 与 IR 方法

| 结论 | 主要证据 | 适用范围 | 允许表述 | 禁止外推 |
|---|---|---|---|---|
| 双分支闭环已运行八轮 | `round1_writeback.json` 至 `round8_writeback.json` | 窄宽度 HMMA/tcgen05 实验 | 框架产生候选、执行 gate、写回正负知识 | 多智能体一定优于单智能体 |
| 第八轮使用 matched opportunity | `round8_budget_receipt.json` | 每 lane 一个结构假说和统一上限 | 相同新增实验机会和晋级规则 | 历史成熟度、LOC、人工与 wall time 完全相等 |
| 默认知识达到局部饱和 | `03_reports/DEFAULT_KNOWLEDGE_SATURATION_REVIEW_20260910.md` | 冻结 B300、现有 source/carrier、当前 IR | 相同知识下没有新的高价值可执行候选 | 理论最优、全局搜索空间饱和 |
| 结构化经验能降低误用风险 | `IR_KNOWLEDGE_REVIEW_20260910.md`、路径 index | 设计与历史审计层 | scope、反例和 reopen key 改善可审计性 | 已证明固定预算搜索性能优于普通记忆 |
| Typed IR 能组合互补角色 | `06_agent_framework/kda_ir/design_ir.py`、`portfolio_search.py`、108 项测试 | 当前有限 KDA grammar | layout/residency/scheduler/parallelism/numerics patch 可验证组合，D 与 A-TMEM 布局不会被误合并 | 任意 CUDA 变换都已可表示 |
| Stage 12 局部域已闭包 | `04_evidence/agent_rounds/round9_closure_certificate.json` | 证书声明的三个有限域和 profile | 所列 semantic design 无缺失观测，错误候选在计时前隔离 | 全局 GPU 最优或完整 SM100 算法闭包 |

## 7. 结论等级

- **L0 事实**：源文件、构建、二进制、SASS、设备身份。
- **L1 局部观察**：某 profile 下的原始时间和正确性。
- **L2 机制支持**：由 profiler、阶段定位或受控干预支持的解释。
- **L3 路径结论**：经过 public-full profile 矩阵和资格 gate 的路线判断。
- **L4 部署建议**：由多个 L3 结果支持、包含 guard 与 fallback 的工程决策。

答辩 headline 使用 L3/L4；microprobe 主要用于解释 L2，不单独承担产品级结论。
