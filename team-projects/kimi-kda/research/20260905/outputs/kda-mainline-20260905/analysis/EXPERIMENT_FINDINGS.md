# Job19896：V16 state-prefetch 消融独立复核

本报告只读本地归档日志、测量脚本和构建记录，不修改 kernel、不运行 GPU。数据来源为 [mainline_19896.log](../mainline_19896.log)，统计由 [summarize_mainline.py](summarize_mainline.py) 复算。当前结论针对包含多种消融的实验 binary；干净生产候选与公共 wrapper 的结果需要另行复验。

## 已证实的结果

**V16 的 Phase6 四块 state 预取（P4）产生了真实完整 raw forward 收益。** 在 fixed B1/H12、BF16 输入及双 state、T2048/3072/4096/6144/8192 的测量域，P4 相对同 binary 原 V16 的 eager 降时为 **17.28–19.43%**，CUDA Graph 为 **18.34–19.71%**。这些点的原始 slice 最优者均为 V16，因此同样优于在这四个原始 slice 中重新挑选的最佳者。

这次正结果不来自重播相同 operands 的 Phase6 微基准。计时覆盖真正的 K1+K2 forward 和 beta 转置；workspace 预分配复用，**不包含 Python 公共 wrapper 分配和 auto dispatcher**。实验仍使用 D128/C16、原 MMA atom、原 state 更新和 BF16 舍入语义。`216/416` 是实验选择码，不是模型 value dimension。

最有代表性的 T8192：

- Eager：同 binary V16 `0.569952→0.459216 ms`，降低 **19.429%**；同 binary V128 为 `0.780064 ms`，总降时 **41.131%**。
- Graph：V16 `0.566976→0.455248 ms`，降低 **19.706%**；V128 为 `0.776896 ms`，总降时 **41.402%**。

不能把旧 ValueSlice 的约 27% 与本次约 19% 相加为 46%。完整改善应直接用相同测量口径的 V128 与 P4 时间相除。

## 完成性、正确性与测量覆盖

日志包含唯一 `kind=complete`（第 819 行）、225 条 `correctness` PASS 和对应 `correctness_complete comparison_rows=225`（第 308 行），不存在计数缺口。225 是 **25 个输入 case × 9 个同 binary variant**；总计 414 个 output/final-state tensor 比较，全部 bitwise equal 且 finite。没有 final_state 的 case 只比较 output，因此不能写成 450 个 tensor 比较。

独立 correctness 对照对象是现有 legacy 扩展的 V128，包括短序列/tail、B2、BF16/FP32 public state、四种 state 输入/输出组合、含空序列的 packed 输入、长序列、长记忆 gate、H96 和 packed 单/多序列。这里证明的是新旧执行路径一致；不意味着重新证明了数学参考、修复了原 FP32 public state 的内部 BF16 语义，或证明模型级精度。

性能覆盖 **15 个 shape × 11 个 variant × 3 repeat = 495 条 performance JSON 行**，每行含 eager/graph 两种口径，即 990 个单轮时间汇总。每种口径每轮测 60 次调用，前置预热 10 次；主表取三个 repeat median 的中位数。每轮 variant 顺序以固定种子打乱。p10/p90 是单轮调用延迟的描述性分位数，**不是置信区间**；三个 repeat 也不是三台 GPU 或三个独立环境。

Entry hardening suite 报告 PASS，但必须标记 **PASS_WITH_SKIPS**：binding、alignment、parity、stream_graph、cpu_rejection 通过；alias_default 因未提供 alias 扩展而 SKIP，multi_gpu 因只有一张可见卡而 SKIP。该作业不能声称验证了双卡 CUDAGuard 行为或独立 alias build。

实验 binary SHA-256（日志第 82 行）：`bcb240e738279ea4e9b06ce73e086a129a24bc9b622d9cbacd0f47e40afbada8`。硬件为 B300/CC10.3、148 SM；PyTorch 2.10.0+cu130。

## 固定 H12 长度扫描及非标定 anchor

单位 ms；所有降时相对于同一 binary、相同 shape/计时口径的原 V16。

| T | Eager V16 | Eager P4 | Eager 降时 | Graph V16 | Graph P4 | Graph 降时 |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 0.154272 | 0.127616 | 17.279% | 0.151168 | 0.122496 | 18.967% |
| 3072 | 0.223904 | 0.184992 | 17.379% | 0.222736 | 0.179872 | 19.244% |
| 4096 | 0.293680 | 0.239984 | 18.284% | 0.290496 | 0.237216 | 18.341% |
| 6144 | 0.432624 | 0.348960 | 19.339% | 0.429728 | 0.345904 | 19.506% |
| 8192 | 0.569952 | 0.459216 | 19.429% | 0.566976 | 0.455248 | 19.706% |

3072/6144 不是原 dispatcher 的 2048/4096/8192 标定 anchor；它们提供了插值区间的留出点证据。这里的“留出”只是相对于原标定 anchor，而不是独立分布、盲测模型或统计泛化保证。此次五点扫描没有证明每个中间整数 T、所有 H≤18 或所有 public-state 模式的性能均改善。

每个目标点的三轮同编号比较也均为正。Eager 各点最小逐轮降时分别为 `17.241/17.359/18.246/19.316/19.411%`；Graph 分别为 `18.956/19.213/18.330/19.309/19.697%`。这比仅展示聚合 median 更有力，但仍是同一作业内的短期稳定性证据。

另外两个有价值但不能直接扩大默认发布域的结果：

- fixed B1/H12/T16384：eager `1.119728→0.895680 ms`，降低 **20.009%**；graph 降低 **20.085%**。原 dispatcher 对此长度属于域外，不能只凭该单点无条件扩大插值域。
- packed 单序列 T8192：eager `0.576112→0.463648 ms`，降低 **19.521%**；graph `0.569024→0.458400 ms`，降低 **19.441%**。这支持研究 packed-single 复用策略，但不能将其推广到多序列 varlen。

## 反例：优化 V16 不等于该选 V16

下表的 P4 与最佳原始 slice 均来自同一 binary。正的“慢于最佳”表示退化，绝不是加速。

| Shape | 最佳原始 slice | 最佳 Eager | P4 Eager | P4 相对 V16 Eager 降时 | P4 慢于最佳 Eager | P4 慢于最佳 Graph |
|---|---|---:|---:|---:|---:|---:|
| B1/H24/T8192 | V32 | 0.619168 | 0.641088 | +0.408% | 3.540% | 3.745% |
| B2/H12/T8192 | V32 | 0.619168 | 0.641520 | +0.331% | 3.610% | 3.631% |
| B4/H12/T8192 | V64 | 0.735952 | 0.993344 | −2.708% | 34.974% | 35.266% |
| B1/H48/T8192 | V64 | 0.737744 | 0.992848 | −2.421% | 34.579% | 34.964% |
| B1/H96/T8192 | V64 | 1.018592 | 2.285664 | −0.243% | 124.394% | 124.512% |
| packed ragged6 | V64 | 0.350640 | 0.381536 | +2.570% | 8.811% | 8.966% |
| packed 8×1024 | V64 | 0.153984 | 0.313744 | −0.020% | 103.751% | 108.660% |
| packed 32×256 | V128 | 0.125392 | 0.258848 | −0.006% | 106.431% | 112.221% |

本次 ragged6 长度是 `[16,32,512,1024,2512,4096]`，与 C1 最终 campaign 的旧 ragged6 分布不同；不能把两个作业中同名 ragged6 的绝对时间直接拼成进展曲线。

就 P4 对原 V16 本身而言，15 个 shape 中最坏聚合退化发生在 B4/H12：eager **2.708%**，最坏同编号 repeat **2.850%**；graph 聚合/最坏 repeat 均为 **2.873%**。它不是在所有 shape 都免费的替换。

H12 的 V16 grid 为 96 CTA，而 H24 或 B2/H12 为 192 CTA，跨过 148 SM 的一层 CTA 覆盖。这与正收益域明显变窄的现象一致；但尚未采新 NCU，不能把 crossover 全部归因于 CTA layer、寄存器或缓存中的某一个因素。

## 不能忽略 legacy 与同 binary 对照

本实验同时测 legacy16/legacy128 与候选 binary 的原 v16/v128，能检查“重建 binary 本身变快/变慢”是否污染结论。15 个 shape 上，同 binary 相对 legacy 的时间变化范围如下（正值为新 binary 更慢）：

| 原始路径 | Eager 范围 | Graph 范围 |
|---|---:|---:|
| V16 | −0.056% 至 +0.365% | −0.106% 至 +0.940% |
| V128 | −0.483% 至 +0.228% | −0.686% 至 +0.085% |

这些差异远小于目标域 P4 的 17–20% 收益，支持正结果不是 baseline 编译差异的产物。但是小差异并非零，也不构成跨构建统计等价证明。比如 T2048/Graph 的新 V16 比 legacy16 慢约 0.94%，所以 P4 相对同 binary V16 为 18.967%，相对 legacy16 为 18.205%；两者都应保留，不能只挑更大的数字。

T8192 的 P4 相对 legacy16 降时为 eager **19.434%**、graph **19.706%**，与同 binary 对照一致。主报告建议把“相对同 binary V16 的新增收益”作为第一口径，“相对同 binary V128 的完整收益”作为第二口径；legacy 作为构建差异控制。legacy 扩展本身是此前集成版本，不能重新命名为本次 freshly-built untouched upstream。

## 为什么 P4 值得推进，P2 却不能只看参数名解释

目标五个固定 shape、两种计时口径的新增降时范围：

| 消融 | 相对同 binary V16 的降时范围 | 判断 |
|---|---:|---|
| P2 | +6.41% 至 +7.72% | 有效，但明显弱于 P4 |
| P2 + delayed acquire | +9.31% 至 +10.95% | 有效，但不如 P4，且机制含编译变化 |
| delayed acquire 单独 | −5.67% 至 −4.66% | 目标域反而变慢 |
| early output publication | −1.84% 至 −0.94% | 目标域反而变慢 |
| P4 | +17.28% 至 +19.71% | 当前最强候选 |

构建证据来自 [build_with_headers.log](../build_with_headers.log) 第 351–380 行，定位到 **V16、96 threads、HasStateIn=true、HasStateOut=true、StateFP32=false、IsVarlen=false** 的具体 kernel 实例：

| 消融 | Registers/thread | Stack frame | PTXAS spill stores / loads | Barriers |
|---|---:|---:|---:|---:|
| 原 V16（P1，schedule0） | 54 | 0 B | 0 / 0 B | 9 |
| early publish（P1，schedule2） | 58 | 0 B | 0 / 0 B | 9 |
| P4（schedule0） | 70 | 0 B | 0 / 0 B | 9 |
| P2 + delayed（schedule1） | 56 | 0 B | 0 / 0 B | 9 |
| P2（schedule0） | 56 | 8 B | 12 / 8 B | 9 |
| delayed（P1，schedule1） | 48 | 16 B | 20 / 20 B | 9 |

这些 spill 字节是 PTXAS 的静态报告，不能当作 profiler 测得的每次完整调用流量。不能把某一个实例的“P4 无 spill”扩大为所有 state dtype/varlen 特化都无 spill；生产编译还需核对自己的资源记录。

P4 在 Phase6 用更多寄存器维持多个独立 state/kr/g 块，让 LDS/LDSM 与计算有更多重叠机会；原 V16 在已存在的 register-ring 代码里仅预取一块。D128/C16 下 Phase6 有八个独立 state 行块，四块 ring 是这条已知数据流上的局部调度选择，无需改变算术归约次序。这个机制与测量正收益一致。

但资源表现明显非单调：P2 只有 56 registers 却产生 spill，P4 用 70 registers 反而没有 spill；仅改变 output acquire 时机也让寄存器分配与 spill 改变。因此这次创新的真实价值是**在保持数值语义的约束下，联合调整预取距离、寄存器驻留和编译调度，缩短低并发 recurrence 的执行时间**。它不是“把预取数字调大就必然更快”，也不是已证明了精确的 stall 根因。

early output publication 的负结果也有价值：即使提前发布在依赖关系上安全，原来两级输出流水可能已经隐藏了 store；新增同步/fence 和编译变化会抵消重叠机会。不能只凭“理论上多重叠”把该消融一起合并。

## 严格的推进边界

当前可以推进一个 **P4-only、明确 shape guard 的干净生产候选**；先验证与实验 binary 的编译资源、同口径完整 forward 以及公共 wrapper 行为一致，再决定提交。直接证据最强的是 B1/H12、BF16 双 state、T2048..8192 的五个已测长度及 packed 单 T8192。更宽的 H、state 组合、长度插值范围或多序列策略需要新增测量；不能简单地把“原 dispatcher 选 V16”都替换为 P4。

本次不能声称：

- 已获得模型 TTFT、TPOT、吞吐、SLO goodput 或功耗改善；没有 checkpoint、serving scheduler、跨卡通信或模型测量。
- 已验证公共 auto wrapper 的相同收益；此日志直接调用原始扩展并复用 workspace。
- 已证明全部 sequence/head/长度组合都会加速；B2/B4 和 packed 多序列给出了明确反例。
- 已证明新的 attention 算法、降低计算复杂度、改变 state 精度或解决跨 chunk 数值边界；这是保持 D128/C16 与原 BF16 舍入的 kernel 调度改进。
- 已证明多 GPU 入口或 alias build 通过；这两个部分在本作业中 SKIP。
- 已建立跨 GPU/时段置信区间，或将 p10/p90 作为统计显著性；当前是同一 B300 作业中的三轮测量。
- 已从 NCU 证明 P4 的某类 stall 减少；当前只有真实性能和编译资源交叉证据，新的 profiler 因果验证尚未完成。

在这些边界内，P4 的收益规模、非 anchor 长度上的一致性、同 binary/legacy 双重控制、bitwise correctness 和负例都支持继续推进生产候选。其强度已超过单纯 Phase6 microbench 的研究信号，但尚不能跳过独立生产构建与 wrapper 验收。
