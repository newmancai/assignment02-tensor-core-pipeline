# 双层 IR Agent：两轮闭环复盘

> 历史阶段文档：本页只覆盖 R1--R2。R3--R8 的证据、已执行结构候选与当前停止边界以
> `DEFAULT_KNOWLEDGE_SATURATION_REVIEW_20260910.md` 为准；下面的“下一轮优先级”已经完成或被新证据取代。

## 这次是否真正用了论文里的 agent 架构

是。这里复用的是论文中的闭环结构，而不是旧 Stage9 的固定参数树：

1. HMMA proposer 与 tcgen05 proposer 各自读取带作用域的历史经验；
2. 两侧自由提出下层程序/物理 IR 候选；
3. critic 检查公平契约、候选身份、错误迁移和可证伪性；
4. executor 只把现有载体能表达的候选送入隔离 B300 测量；
5. 结果写回下层实现 receipt 和上层经验 claim；
6. 第二轮必须引用第一轮 claim，再提出边界测试。

旧 `run_stage9_agents.py` 没有直接复用，因为它的输出 schema 只允许
`chunks_per_cta` 决策树，会把 HMMA 与 tcgen05 的自由探索错误压成同一个参数空间。
本轮保留角色、预算、验证、测量和回写结构，候选 IR 改为开放的双分支表达。

## 预算与契约

每轮、每条路线只执行一个新 challenger；两轮都不新增编译。每个实现独立进程，
每个 block 8 次 warmup、20 次 cold-L2 CUPTI sample，两个平衡 block。输入使用非零
BF16 初态，public call 包含 state 更新或 evolution 的同流 copy-back。

共同准入契约是 output 和 final state 对同 profile 官方 peer 同时 finite 且满足
`atol=rtol=0.01`。HMMA 的 bitwise identity 另行记录。这个契约允许当前 HMMA 与 CAKE
共同参与探索，但不是独立高精度 oracle；特别是部分 H12/H96 CAKE 结果不满足新拟的
normalized-Linf 1% 指标，不能把这两种门槛混用。

## 第一轮

HMMA agent 提出在 H12 packed mixed6 上比较现有 V64 与 V128。它没有把旧的 17.4%
结果当结论，只把它当作复测优先级。结果如下：

| profile | official | HMMA V128 | HMMA V64 | CAKE | 新候选相对 incumbent |
|---|---:|---:|---:|---:|---:|
| H12 mixed6, total 8192 | 0.595834 ms | 0.592031 ms | **0.486405 ms** | 0.240242 ms | **1.21716x** |

V64 两个 block 都与官方 output/state bitwise equal。旧经验在当前强化二进制和新测量
epoch 上复现，因此可以晋级为 exact-profile scoped prior。它仍不授权所有 packed shape。

tcgen05 agent 从调度 IR 推出：H64 mixed 的 152-CTA schedule 可把最大 stride 从 126
降到 114，并复用已生成的 `m128_h64_p1_s114`。合法性检查证明 schedule work 守恒，
route receipt 也确认实际加载了声明的 sm103a variant；性能却明显回归：

| profile | Cake | evolution 148 CTA / s126 | virtual 152 CTA / s114 | 新候选相对 incumbent |
|---|---:|---:|---:|---:|
| H64 mixed6 | 0.400542 ms | **0.387072 ms** | 0.655116 ms | **0.59085x** |

这否定了 exact candidate，没有否定“负载均衡”或 tcgen05。新的关键经验是：
`stride` 不能脱离物理 SM 数与 wave 结构排序。152 个逻辑 CTA 在 148-SM B300 上越过
单波容量，降低 12 个 stride 仍可能远不够偿还第二波尾巴。因为 CTA 数和 kernel stride
同时变化，“第二波”仍是 leading hypothesis，不是已经隔离的单因素因果。

## 第二轮

HMMA proposer 固定 H12、total=8192、nseq=6 和同一二进制，只把 mixed6 改成
balanced6。这比重跑已知的 8x1024 sentinel 少改变一个变量：

| profile | official | HMMA V128 | HMMA V64 | CAKE | 新候选相对 incumbent |
|---|---:|---:|---:|---:|---:|
| H12 balanced6, total 8192 | 0.321537 ms | 0.321000 ms | **0.273819 ms** | 0.149353 ms | **1.17231x** |

两个 block 的 V128/V64 比值为 1.17089x 与 1.17372x，正确性仍 bitwise。由此可排除
“Round 1 收益只来自 mixed 长度偏斜”的简单解释。经验现在可以写成：当前二进制的
两个 H12、六序列、total8192 profiles 上 V64 都优于 V128。下一步 guard 仍要联合检查
`nseq`、total chunks、max chunks、长度离散度和由 ValueSlice 产生的 CTA waves。

tcgen05 proposer 把同一 148→152 CTA 干预迁移到 H96 mixed，实际 route 是
`m128_h96_p1_s173` 对 `m128_h96_p1_s166`：

| profile | Cake | evolution 148 CTA / s173 | virtual 152 CTA / s166 | 新候选相对 incumbent |
|---|---:|---:|---:|---:|
| H96 mixed6 | 0.571154 ms | **0.550993 ms** | 0.945165 ms | **0.58296x** |

两个 block 的比值为 0.58469x 与 0.58123x。跨 H64/H96 的重复负结果使“148-SM 上
不要为了较小 stride 把 grid 推过一波”成为更强的调度先验，但仍不应提升为 verifier ban。
在物理 152-SM 设备上，或能固定 kernel/stride 单独改变 CTA 数时，应该重开。

第二轮还暴露了一个有用的执行错误。初次 H96 运行时，声明的两个 evolution arm 实际
都被 deployment guard 回退到 Cake，因此得到约 1.00x。identity gate 将它标成
`diagnostic_only_superseded`；打开精确 search-only shape、确认加载 s173/s166 后才产生
上表有效结果。这说明 `requested_backend` 远远不够，IR 必须记录 `actual_route_id`。

此外，search-only 的 H96 s173 本轮比 Cake 快 1.03659x。它只有两 block、profile 已知，
不能直接激活 deployment，但足以重开正式四 block qualification。

## 对主线问题的新认识

两轮没有推翻“完整 SM100 路径值得做”，反而把比较变得更诚实。

1. HMMA 侧确有未吃完的优化空间。H12 packed mixed6 和 balanced6 的 V64 分别比当前
   V128 快 1.21716x 和 1.17231x，所以“原 HMMA 完全没优化”是不准确的；更准确的说法是
   HMMA 已有强化载体，但 packed dispatch 仍有明显策略空洞。
2. 找到更强 HMMA 后，CAKE 在相同 H12 profile 上仍快 2.02465x 和 1.83336x。这个差距
   比“CAKE 对官方”的 2.48014x 和 2.15286x 小，说明过去用官方作分母确实高估了相对
   强 HMMA 的优势；差距依然很大。
3. tcgen05 的优化也不是单调的。减少 schedule stride 的候选两次严重失败；已有的
   H64 s126 与 search-only H96 s173 却分别优于 Cake。有效经验是 route × hardware ×
   wave × resource receipt 的组合，而不是“新 ISA 更快”或“更均衡更快”。
4. 本轮每条路线拥有相同的 proposal/GPU challenger 数，能回答“小规模同等新增预算下，
   现有载体还能否找到改进”。它不能补平两条路线历史工程成熟度，因而仍不是最终的
   “同等总人工预算下最优 HMMA vs 最优 tcgen05”因果实验。

当前最稳妥的主线表述是：

> 在两个新测的 H12 packed profiles 上，修正 HMMA dispatch 后，完整 tcgen05/SM100
> 路径仍有 1.83–2.02x 的 public-full 优势；HMMA 侧的可优化空间真实存在，因此最终
> 论文应以 stronger-HMMA 为分母，并把 CAKE 的收益称为 SM100 full-stack treatment，
> 不能称为 tcgen05 opcode 的单因素收益。

## 双层 IR 应怎样改

下层程序/物理 IR 至少增加：

- `requested_backend`、`actual_route_id`、dispatch guard 和 fallback reason；
- physical SM count、logical CTA count、`ceil(grid_x / sm_count)` wave 数；
- schedule stride、平均/最大 bin load、任务守恒 receipt；
- instruction family、tile/layout、state/storage、warp roles、barrier/pipeline；
- source/build/binary identity；
- `changed_subtrees` 与同时变化的变量 bundle，防止把联合干预伪装成单因素。

上层经验 IR 至少增加：

- 精确 proposition，而不是一句“V64 好”或“s114 差”；
- source scope、target scope、matched/mismatched/unknown preconditions；
- `observation / localization / controlled_intervention` 机制等级；
- competing mechanisms、invalidation keys、reopen conditions；
- `ranking_prior / requires_retest / deployment_evidence` 的允许用途；
- agent retrieval receipt：哪些 claim 影响了提案、拒绝和测量；
- requested identity 与 observed identity；fallback 测量必须自动降为 diagnostic；
- 每轮 proposal、compile、GPU block 和 model 使用预算。

## 下一轮优先级

1. 给 H12 HMMA 做小型 dispatch frontier：保持现有 carrier，覆盖 nseq=4/6/8、balanced/
   skew、相同 total chunks，学习 V64/V128 guard；不要重测 V16 已知负区。
2. tcgen05 schedule 搜索先加硬约束 `grid_x <= physical_sm_count`，再在 148 CTA 内优化
   binning；不要把最小 stride 当目标函数。
3. 对 H96 mixed 的 s173 对 Cake 做独立四 block qualification 和置信区间；只有通过才
   讨论加入 deployment activation set。
4. 下一笔结构预算分别给 HMMA Phase-1 L3 lookahead 和 tcgen05 V16 preferred-layout
   bridge。两者都需要真实 build，并继续保持每 lane 相同 proposal/compile/GPU slots。
5. 最终论文比较应选共同 profiles，把 H1（强 HMMA）、T1（强 tcgen05）和 T2（guarded
   deployment）并列；报告每个 profile 的 winner 和 applicability，不只报总 geomean。

机器可读提案和回写位于同目录的 `round1_agent_ir.json`、`round1_writeback.json`、
`round2_agent_ir.json` 与 `round2_writeback.json`。完整远端 samples 保存在各 writeback
记录的 B300 路径中。
