# FlashKDA 主线增量：按初态合约选择 Phase 1 lookahead

2026-09-05。结论：**保留 V128 `mma.sync` fallback + guarded ValueSlice 的主线，新增一个经过干净构建验证、默认关闭的 Phase 1 调度候选。** 不引入 `tcgen05` 全面替换，不缩小模型维度，不改变状态精度，也不把无效初始化实验带入交付。

本轮修改、构建和 GPU 检查均已完成；没有安装覆盖现有包、修改原 dirty 仓库、提交 commit/PR 或发布。它是单请求延迟导向的可复现候选，**不是所有生产负载均无回归的通用替代**。

## 1. 保留下来的实现

在已有 Phase6 StatePrefetch=4 的 V16 路径内，Phase1 的 `k @ state` / `q @ state` 增加 k/q/state 三组 fragment ring：无 initial state 用 lookahead 4，有 initial state 用 2。两个消费者完成后才覆盖 slot，按原来的 k 递增顺序更新累加器。初始化、后续 state FMA、BF16 舍入点、布局、barrier 和公开调用参数不变；不增加零状态 tensor、额外 kernel 或 host 同步。

准入仍为：显式 SM103a 编译、两个优化 flag 同时开启、已有策略选中 V16、D128/C16、非 FP32 public state、N1/H12、2048≤T≤8192。fixed/合法 packed single 使用同一 HasStateIn 规则。其它场景保持原路由；Python 的设备资源/ValueSlice 策略未改。该 guard **不会检测运行时并发**，不能把 N1 误解为 GPU 上只有一个请求。

生产尺寸依据仍是 [C1 任务约束](../../C1_TASK.md#L59)：模型 H96、D128，TP8 对应每卡 H12。H12 是生产分片形状，不是把模型砍小后的 benchmark。H24/48/96、FP32、batch/packed 多序列保留兼容性检查和回退；本轮不声称给这些形状提供同等加速。

交付：[0005 增量补丁](clean-phase1.patch)、[构建与回滚说明](CLEAN_PHASE1.md)、[完整身份清单](BUILD_MANIFEST.json)。补丁按 0001→0002→0003→0004→0005 应用；前四项的精确路径和 hash 在清单中。

## 2. 干净版本的真实收益与边界

**以下只用 Job19934 的同输入、同作业配对结果。** 参照为此前四补丁 Phase6 P4 的独立二进制；第三组是同作业内候选的强制 V128。每个形状三轮随机变体顺序，表中为三轮 median 的中位数。不是将不同作业的最好数字拼起来。

H12/B1/T8192/D128/C16，fixed，BF16，有 final state：

| 初态合约 | 旧 P4 full forward | 新 Phase1 full forward | 对旧 P4 延迟减少 | 同作业 V128 | 对 V128 总延迟减少 |
|---|---:|---:|---:|---:|---:|
| 无 initial state：首段 prefill | 0.950848 ms | 0.862208 ms | **9.32%** | 1.373680 ms | **37.23%** |
| 有 BF16 initial state：续段 | 0.796144 ms | 0.743968 ms | **6.55%** | 1.365632 ms | **45.52%** |

这里是实际不变 Python wrapper 的 eager CUDA-event 区间，包含 K1+K2 的 forward，不是单 K2 或完整模型服务延迟。另测 Graph replay、每次调用前的 256 MiB cache perturbation，以及调用前后同步的 host wall；四种口径不混合。Graph 不计每次 Python 调度，cache perturbation 的写入在计时外，host wall 包含同步开销。

40 个性能点包括 24 个 fixed/packed single × 四种 state-in/out 合约 × T2048/4096/8192，6 个尾长、4 个 T3072/6144 留出点，以及 6 个范围外对照。**34 个优化域点在四种口径的中位数上都更快**；eager 增量收益约 5.15–12.40%。6 个回退点的四口径中位数偏移约在 ±0.9% 内，不能据此声称完全零漂移或测尽区间内每个整数长度。完整轮次、最坏配对值与逐形状数据见 [独立验收报告](CLEAN_FINDINGS.md)。

逐轮核查也保留：优化域的 408 个配对轮次收益全部为正，最小为 3.72%；但回退对照的单轮最坏回归为 1.32%，所以“中位数近持平”不等于每轮都在 ±0.9% 内。

必须公开的负结果：两 stream 同时执行两个 T8192 完整请求时，有初态 pair 近乎持平；**无初态 pair 从 1.147440 增至 1.164832 ms，回归 1.52%**，三轮方向一致。这是 joined pair 时间，不除以二当作单请求 latency，也不声称 serving throughput 提升。吞吐导向部署应暂留旧 P4 构建，候选仅适合已接受此取舍的低并发延迟路径。

Job19934 开头 GPU 时钟采样为 1095 MHz；独立 Job19935 profile 的 GPC 平均约 1.08 GHz，而前面的 Job19918 约 1.91 GHz。频率差异是本轮绝对时延高于早先实验的直接线索，但不是全程受控 clock 实验。因此不把 Job19924 的 0.494/0.427 ms 当作干净候选的实测数，更不按频率比例“校正”成对外宣传值。

## 3. 验证不是只有一个计时点

新 `.so`：`flash_kda_phase1_C`，SHA256 `f6f80fa402cc1dc00b09a8082b10806bbe17c0e533d067e931dd774d270b9270`。原 `setup.py` 在独立源/构建目录编译；未带实验 ID 或初始化消融分支。GPU 为 B300 SXM6 AC、148 SM、SM10.3；原 CUDA/PyTorch/CUTLASS 环境复用。

| 验收项 | 结果与口径 |
|---|---|
| 源码及 CPU 契约 | 原 K2 除 Phase1 新分支/模板外可逐字还原；20 项构建配置、234 项实际 C++ selector、旧构建契约 13 项通过 |
| 跨路径逐位回归 | 120 主比较 + 14 尾块/状态补比较，out 和存在的 final state 均 finite/bitwise PASS；包含 auto/force16/off、FP32、H24/48/96、多 batch/packed 及 gate 极端值 |
| 状态传递 | 原带初态链 3 步 + 真正从 None 开始、随后携带状态的链 3 步 PASS |
| Graph 与计时后检查 | 80 跨路径 Graph 检查 + 40 V128 自检；另有 80 次计时后跨路径检查 PASS。自检和重复检查不冒充独立形状 |
| 并发正确性 | 两种初态合约 × 两请求，共 4 次跨实现比较 PASS；性能取舍如上 |
| 入口 hardening | binding、错位 beta、parity、非默认 stream/Graph、CPU device 拒绝共 5 case PASS；multi-GPU 与单独 alias 扩展 SKIP |
| Memcheck / synccheck | 各 20 次对照完成，退出 0，`ERROR SUMMARY: 0 errors`；是定向矩阵，不是全面 racecheck |
| 性能完整性 | 40 shapes × 3 variants × 3 rounds = 360 行，另 12 行双流 pair；原始日志保留 |
| 实际编译/launch 机制 | 精确 SASS 与 4 个独立 NCU K2 profile，16 passes/次；不把 replay passes 当独立统计样本 |

原始证据：[构建日志](build_clean.log)、[主验证](clean_19934.log)、[memcheck](clean_19934_memcheck.log)、[synccheck](clean_19934_synccheck.log)。数值验证是对既有实现的逐位回归，不是新的高精度算法 oracle 或全模型质量评估。

## 4. 为什么这不是为了创新而创新

本轮有明确的反证链，而不是只留下赢的配置：

1. **先消除状态值混淆。** 同一 q/k/v/g/beta 下，None 和显式 BF16 零初态数学上相同，但原 state-present 路径依然更快，说明差距不只是“零与非零状态”的数据差异。显式零 tensor 仅作诊断，没有隐藏到新实现里。见 [matched 结果](MATCHED_FINDINGS.md)。
2. **否决看似直观的清零优化。** 向量清零无明显收益，单 warp 清零 eager 改善不足 1%；统一 runtime 初始化分支甚至让原有初态快路径慢约 20%。三者均不合入。见 [初始化消融](ABLATION_FINDINGS.md)。
3. **定位并改变 load/use 调度。** 原 SASS 的 Phase1 有 shared matrix load 后立即依赖消费的片段。三组 fragment ring 保持原算术，改变可用的独立工作窗口。ring4 会让有初态路径退化，所以保留 HI=false→4、HI=true→2 的最小规则。packed 单点偏好 ring2，但 ring4 已有稳定正收益，不为一个百分点继续增加分支维度。见 [两深度消融](PHASE1_FINDINGS.md)。
4. **反汇编不替性能找漂亮借口。** clean 两实例的矩阵/FMA 工作量未减少。无初态版本有 8 B stack、真实 LDL/STL，循环内 S2R 也未减少；不能宣称“消除 spill”或“修复全部不变量 hoist”。NCU 却显示 short-scoreboard 下降、eligible/issue 提高，说明有额外控制开销仍可净胜。带初态获胜版本为 72 registers、无 local spill。见 [SASS 与计数器互证](CLEAN_SASS.md)。

可站得住的 insight 是：**在固定生产维度与状态数值合约下，ValueSlice、分阶段预取和模板导致的寄存器生命周期需要一起设计；换成更新的 MMA 指令不是充分条件。** 这是本作业的可测工程贡献，不声称 fragment prefetch 是全球首次算法发明。SASS/NCU 支持调度机制，但不把全部节省归因到某一个 PC，也不把 stall ratio 当 wall-time 百分比相加。

## 5. 后续只做有收益门槛的主线工作

优先级一是解决已暴露的**低并发 latency 与并发 throughput 分歧**：用真实请求分布、实际服务并发和单独受控环境复测，决定是否需要调用方明确选择 latency 模式。暂不加入猜测 GPU 上并发数的隐式 dispatcher，也不把更宽 guard 当成进展。

第二是围绕真实循环控制 spill / load-use 做单因素候选，要求继续胜过这份新基线，并且不能扩大双流回归。不能仅因为静态寄存器数下降、spill 清零或某个局部 K2 计数变好就合入。

第三是交付集成验收：多 GPU guard、单独 alias build、实际包安装/回滚、完整模型/服务的正确性与时延。当前未获授权去覆盖原安装或发布，所以保留独立产物与完整复现链。主线外 Loop Transformer/广泛 `tcgen05` 重构仍只作补充，不挤占这些准入项。
