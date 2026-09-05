# FlashKDA 主线推进：ValueSlice 之后，优化单 warp 的状态流水线

2026-09-05 · B300 / SM103 · 实现与实测已完成，交付隔离的 opt-in release candidate；未发布、未替换现有安装、未修改已有脏工作树。

## 1. 结论

**本轮找到了值得保留的主线增量：在既有 V16 上，将 Phase 6 的状态预取窗口从 1 个 keyblock 调为 4 个。** 保留 `mma.sync`、D128/C16、完整 128×128 state、原 FP32 更新与 BF16 舍入；不改模型，不扩大时间 chunk，也不引入额外输出同步方案。

正式候选在 H12/T8192、带 BF16 初态并输出最终 state 的真实 wrapper 中，从旧 V16 的 **0.569712 ms 降到 0.459184 ms，新增降时 19.40%**。同一 job 的 V128 为 0.784816 ms，累计降时 **41.49%**，约 1.71×；两个降时百分比不能相加。

但生产口径必须分开：**无初始 state 的首次 prefill，新增收益约 7–9%；带初态的已测形状约 17–20%；双流并发的两请求总完成时间只改善约 0.47%。** 因此结论是“低并发长 prefill 的受限优化”，不是通用吞吐提升，更不是整个模型加速 41%。

主线发布判断据此更新为：

> 保留 V128 `mma.sync` fallback 与既有 guarded ValueSlice；在 B300 已验证的 H12、单序列、BF16、T2048–8192 域内，增加编译期 opt-in 的 V16 Prefetch4。全面机械替换 `tcgen05` 仍无必要；新的指令/布局候选必须击败这个更强基线。

## 2. 这个增量为何契合题目与生产，而非为了创新

[C1 题目](../../C1_TASK.md) 明确允许并行度重构；模型为 H96、head_dim128，TP8 每卡 H12。本轮所有优化保持这些生产维度。关于 Kimi 论文、C16 与模型 head dimension 的区别，延续[前一轮主线审查](../kda-review-20260905/MAINLINE_REVIEW.md#L27)，没有将论文中的另一 chunk 实现机械套到当前 kernel。

既有 ValueSlice 用 8 个 Value 子片将 H12 的 K2 从 12 CTA 展开为 96 CTA，但 V16 每 CTA 只有一个 compute warp，chunk 之间仍串行依赖 state。本轮继续利用的是**同一 chunk 内八个独立 keyblock 的读取/计算重叠**，不是改变模型时间依赖。

四槽环的操作为：预装 keyblock0–3；消费0–3时分别补入4–7；随后排空4–7。每个 state 元素仍执行原来的 MMA、gate FMA、BF16 转换和写回，矩阵指令及算术次序未改。[局部依赖与环索引证明](policy/PREFETCH_REVIEW.md)

可准确陈述的贡献是：**在生产维度与舍入契约固定时，联合选择 CTA 的 Value 并行粒度和 CTA 内部状态预取窗口，并用正式入口、负例和回退边界完成验证。** 环形预取本身不是新的算法，原源码已具备环框架；本轮不是宣称全球首创，也不把一个常量调优包装成新的注意力模型。

## 3. 用消融选择实现，而不是把实验全部合并

Job19896 使用独立实验二进制。同一形状 B1/H12/T8192、BF16 双 state、完整 raw forward（K1+K2，含 beta 转置，workspace 预分配）的三轮中位数：

| V16 内部变体 | 相对原 V16 的延迟减少 | 处理 |
|---|---:|---|
| 仅延后输出 buffer acquire | −5.56% | 不保留 |
| 提前发布非最后 tile 的输出 | −1.44% | 不保留 |
| 状态预取2 | +7.72% | 不保留；该目标实例出现寄存器 spill |
| 预取2 + 延后 acquire | +10.76% | 不保留；额外调度复杂度无优势 |
| **状态预取4** | **+19.43%** | **唯一进入干净候选的性能改动** |

该 job 的 225 条跨实现输出/state 比较均逐位一致；旧 V16 与实验二进制中的原 V16 在主形状基本一致（0.569984 / 0.569952 ms），避免把构建变化误认为算法收益。完整样本与反例见[独立消融复核](analysis/EXPERIMENT_FINDINGS.md)。实验 selector 和输出调度代码只留在 `experiment.patch` 中，**不能作为发布补丁安装**。

## 4. 正式 wrapper 的结果与边界

Job19901 从原 `setup.py` 在全新 build 目录构建干净四补丁候选；旧扩展独立加载，未被覆盖。正式 Python wrapper 未改，旧/新 wrapper 分别绑定各自的 raw 扩展。以下是各轮中位数的再中位数，不是置信区间。

| H12/T8192 场景 | 旧 auto / ms | 候选 auto / ms | 新增降时 |
|---|---:|---:|---:|
| 固定单序列，BF16 initial + final，eager | 0.569712 | 0.459184 | 19.40% |
| 同上，CUDA Graph replay | 0.566816 | 0.454304 | 19.85% |
| 同上，调用前写256 MiB缓存扰动缓冲 | 0.571376 | 0.459808 | 19.53% |
| 无 initial、输出 BF16 final，eager | 0.601840 | 0.547568 | 9.02% |
| Packed 单序列，BF16 initial + final，eager | 0.573472 | 0.463152 | 19.24% |
| 两个独立 stream 各一个请求，总完成时间 | 0.672208 | 0.669040 | 0.47% |

固定单序列、双 state 的 T2048/3072/4096/6144/8192，eager 新增降时分别为 **17.27 / 18.32 / 18.42 / 19.16 / 19.40%**。3072/6144 是原标定点之间的留出长度，不把它们称作真实流量 trace。

计时范围必须读对：eager 是 CUDA events 包围真实 wrapper 调用，执行了 dispatch 与 workspace 分配；可能包含 host 提交造成的 GPU 空隙，但**不是 CPU wall time、服务端请求延迟或模型端到端延迟**。Graph replay 不包含 Python；缓存扰动发生在 start event 前，且 K1 随后会生产 workspace，不等价于 K2 全冷缓存。

域外复测保持原策略：B2/H12 用原 V32，T16384 和 packed多序列按原条件 fallback；FP32 public state 不进 P4。已测域外最大负向轮次约0.39%，量级很小但保留原数据；未开启 P4 的 packed短序列 eager 约1%差异不能归因于新预取。实验中强制 P4 在 B4/H12 比原最佳 V64 慢约35%，进一步说明不能扩大启用范围。[Job19901 原日志](release_19901.log)、[完整审计汇总](analysis/19901_summary.md)

### 状态接口补测：不能拿 continuation 代表首次 prefill

Job19903 单独覆盖四种 state 模式（both/in/out/none）、T2048/4096/8192、fixed/packed单序列，共24种组合；再测 T2049/4095/8191 三个非整块长度。各3轮，和19901不合并统计。

- 带初态（both/in）的已测 eager 新增降时约 **16.8–19.6%**。
- 无初态（out/none）的已测 eager 新增降时约 **7.4–9.1%**；缓存扰动最小约 **7.0%**，仍有正收益。
- 三个非整块长度的 eager 新增降时约 **17.8 / 18.1 / 19.4%**。
- 81条检查通过，其中 **54条是跨路径对拍，27条只是 V128 reference 自比较的 finite/sanity 检查**，不算81个独立跨实现验证。

该差异不应解释为“零初态比非零初态的数学工作更多”：它们编译为不同模板实例，寄存器分配和生成代码也不同。目标 P4 在 HasStateIn=true 时为70 registers、false时为63，均无spill；具体为何后者较慢，需要继续针对生成代码取证。固定序列的 out−both eager 差值随 T2048/4096/8192 为20.576/43.152/90.048 µs，近似随长度增长，提示值得检查稳态代码；这还不是同输入受控因果实验。[状态补测原始日志](state_matrix_19903.log)、[独立状态复核](analysis/STATE_FINDINGS.md)

## 5. NCU 支持什么，不支持什么

Job19901 的 K2 profile 确认真正运行了末尾模板参数为4的 P4 实例，而非只换模块名。两者 grid 都是 `(1,12,8)`、block都为96 threads、dynamic shared 都为48.640 KB（另有1.024 KB driver shared）。

| 同形状 BF16 双 state K2 | 旧 V16/P1 | V16/P4 |
|---|---:|---:|
| registers/thread | 54 | 70 |
| register 上限允许的 blocks/SM | 12 | 9 |
| shared-memory 上限允许的 blocks/SM | 4 | 4 |
| achieved occupancy | 4.690% | 4.688% |
| issue active | 14.72% | 18.12% |
| eligible warps/active cycle | 0.1472 | 0.1812 |
| short-scoreboard cycles/issued instruction | 0.9010 | 0.4649 |
| sleeping cycles/issued instruction | 2.2596 | 1.2915 |
| long-scoreboard cycles/issued instruction | 0.9334 | 1.0478 |
| 执行的 warp-level instructions | 41,818,128 | 40,144,608 |
| profile K2 duration / µs | 521.088 | 409.984 |

这里最有价值的证据不是 occupancy 上升——它几乎没变——而是**在没有跨过实际共享内存驻留上限的前提下，以更多寄存器换来了更连续的发射、更低的短依赖等待，以及约4%的动态指令减少**。这与软件预取改善依赖隐藏一致，不支持“收益来自减少 HBM 字节”或“所有 stall 都下降”。

WarpStateStats 数值是每条已发射指令对应的平均周期，不是百分比；stall 下降也不自动构成 Phase6 指令级因果证明。[NVIDIA 指标定义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules)

两份 profile 各做16-pass kernel replay，未固定 clocks、未控制 caches；只有一份每变体样本，**不能替代三轮完整 forward 计时，也不能把 sleeping 全归因于某一个 warp**。目标 P4 的8个已编译 BF16模板实例都无stack/spill；整个扩展其他既有变体仍有spill，不能写“release全局无spill”。[旧 V16 指标](release_19901_baseline_ncu.csv)、[P4 指标](release_19901_release_ncu.csv)、[独立 NCU 复核](analysis/NCU_FINDINGS.md)

## 6. 可交付的代码、保护与验收

补丁顺序为已有0001/0002，再加：

1. [0003 入口硬化](hardening/0003-release-entry-hardening.patch)：输入同设备检查、CUDAGuard、实际beta基址16B对齐修复、C++/nvcc默认slice宏一致性。原有合法对齐路径不增加CUDA kernel。
2. [0004 guarded V16 Prefetch4](release/0004-guarded-v16-prefetch4.patch)：只改kernel模板参数、launch细分和编译开关；没有实验selector或新输出同步。

P4 默认编译关闭。开启要求显式 `FLASH_KDA_CUDA_ARCHS=103a`、`FLASH_KDA_ENABLE_V16_PREFETCH4=1`。运行时须已选V16、D128、非FP32、N1/H12、T_total在2048–8192；正常auto仍先经过原B300 SM/L2策略保护。**N1不表示GPU上只有一个活跃请求，guard没有实时并发感知。** Raw/强制V16会绕过Python硬件策略，不能称为硬件全面防护。

P4接受区间内所有整数长度，不能把有限采样称作穷尽验收；性能仍有明确外推部分。`explain_k2_dispatch()`还描述旧ValueSlice模型，未标定P4延迟/资源；它不是新内核性能证明。

验收记录：

- CPU：78个原policy/wrapper契约、51个真实C++选择器分支、13个build flags检查通过；汇总程序另有合成日志回归测试。
- GPU：225条实验跨实现对拍；干净release的120条out/final逐位对拍，含尾块、FP32 fallback、packed、四state组合和门控极值；3步状态传递与2个并发请求输出检查通过。
- 正式release入口硬化5个单GPU子用例通过；未构建alias、未获得双GPU allocation，相关项保持SKIP。
- Memcheck和synccheck分别对T2049 fixed双state及T2048 packed无初态/final进行检查，均退出0、ERROR SUMMARY为0；不是全矩阵sanitizer或racecheck证明。
- 新构建使用原setup、既有CUDA/PyTorch/CUTLASS及Python headers；未安装依赖，未覆盖旧扩展。二进制SHA及四补丁顺序见[构建清单](BUILD_MANIFEST.json)。

运行时强制 `FLASH_KDA_K2_VALUE_SLICE=128` 可退回V128/P1。编译后仅在运行时取消 `FLASH_KDA_ENABLE_V16_PREFETCH4` 无效；如要恢复旧V16，需使用未开启P4的独立二进制。强制slice优先于`DISPATCH=off`。[构建、使用与回滚说明](release/README.md)

## 7. 之后优先做什么

本轮的实现、验证、负例与候选补丁已形成闭环，不需要再堆一个创新名词。下一阶段按实际价值排序：

1. **优先研究无初态模板的生成代码差异。** 首次prefill更接近这一接口；当前新增收益只有约9%，且P4无初态比带初态的绝对时间更长。先做同输入“无初态/显式零初态”语义与计时对照、PC级stall/SASS分析，定位差异；只有完整wrapper净降时且维持零初态契约的实现才保留。这比再扩大MMA微基准更贴近生产。
2. **取得真实prefill调用分布，确定是否该启用本候选。** 至少区分有无初态、每卡heads、单卡同时活跃请求、fixed/packed和长度。两个stream已显示收益几乎消失；在取得调度器可提供的并发信息前，不把“低并发”伪装成已有在线检测能力。端到端收益须按实测critical path占比验证。
3. **完善可观察性，再考虑扩域。** 让诊断明确给出实际prefetch子变体和构建身份，重新标定P4；随后才评估H范围、T16384和FP32。当前域外fallback是设计决定，不能拿实验里强制P4的正点冒充auto已支持。

Cluster/multicast或跨phase TCGEN/TMEM仍只作为后续候选：基线升级为当前最佳guarded路径，必须计入生产者/消费者布局、状态转换、同步和并发反例，取得完整forward净收益才进入主线。Loop transformer 不属于本轮新增交付，也没有用它来稀释FlashKDA的发布问题。
