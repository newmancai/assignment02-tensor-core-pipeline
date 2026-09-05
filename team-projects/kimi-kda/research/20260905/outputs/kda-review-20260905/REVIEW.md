# KDA 深度审查：从可交付算子到循环推理系统

2026-09-05 · 已连接 `b300-login` · B300 实测 Jobs **19844 / 19845**

## 判断

**ValueSlice 的核心优化站得住；当前实现的生产集成边界还没有封闭；`tcgen05` 的研究空间比旧报告表现出来的更大。** 接下来最有价值的工作，是把状态身份、数值舍入边界和硬件驻留生命周期一起设计。

本次审查没有改动原作业、kernel、Git 分支或已有补丁。新增独立探针、原始日志和本报告。对 loop transformer，采用“共享 block 沿深度重复执行”的含义；没有训练或运行完整 looped 模型，没有测 Kimi 端到端 serving。

## 1. 需要优先处理的代码问题

| 优先级 | 问题 | 证据等级 | 来源与影响 |
|---|---|---|---|
| P1 | 合法 contiguous 单 token view 可以因 TMA 地址不对齐使整个进程 abort | **B300 两次复现，含 clone 对照** | 上游继承；流式切片接口的实际故障 |
| P1 | 缺少 CUDA device guard，输入设备与当前设备可能不一致 | 静态确认，未做双卡复现 | 上游继承；多 GPU 集成需修复 |
| P2 | V16 profiling alias 的编译宏没有传给 C++ binding，默认仍为 V128 | AST 工厂检查已复现 | 本补丁新增；显式传 slice 的现有主实验不受影响 |
| P2 | FP32 buffer 不保留 FP32 state 精度，空序列也量化初态 | **B300 复现** | 上游继承；状态精度与空序列契约需明确 |
| P2 | 非连续 `cu_seqlens` 未拒绝，按连续裸指针读取 | 静态确认 | 上游继承；文档原本要求 contiguous，这是输入拒绝缺口 |

### 1.1 最直接的故障：contiguous 不保证 TMA 地址对齐

[上游 C++ 入口第 131 行](../../implementation/upstream/csrc/flash_kda.cpp#L131) 使用 `beta_2d.t().contiguous()`。当 `T_total=1` 时，转置结果可能已经 contiguous，因此这一步保留原 view 的 storage offset。TMA 的 base address 却必须满足 16 字节对齐。

Job19844 在把 T256/H2 按 token 切片执行时触发 CUTLASS host assertion。Job19845 用独立子进程建立了最小对照：

| beta 输入 | `is_contiguous()` | `data_ptr()%16` | 结果 |
|---|---:|---:|---|
| 从四元素 BF16 storage 取 `[1:3]` 并 reshape 为 `[1,1,2]` | True | 2 | **SIGABRT，子进程 returncode −6** |
| 同一 view 的 `.clone()` | True | 0 | 完成，输出 finite |

这不是向 kernel 喂任意非法 stride。输入满足当前文档声明的 CUDA、dtype、shape、contiguous 条件。H12 的普通逐 token beta 切片也可能有 `24 mod 16 = 8` 的偏移；H 为 8 的倍数会掩盖问题。

建议在最终 `beta_t` 上检查指针对齐，不齐才复制到对齐 storage；再调一次 `.contiguous()` 没有效果。其他 TMA 输入也应有一致契约。输出与 final-state 若使用 scratch，必须写回原 buffer，保持 in-place 语义。`A_log`、`cu_seqlens` 不是同类 TMA base，不应无端给它们增加 16 字节限制。

### 1.2 执行设备与输入设备必须一致

[C++ 入口第 136 行](../../implementation/upstream/csrc/flash_kda.cpp#L136) 调用无 device 参数的 `getCurrentCUDAStream()`，没有持有 `CUDAGuard(q.device())`。若 q 在 GPU1、当前设备仍为 GPU0，后续自定义 launch 会使用错误的执行设备/stream。dispatcher 查询属性时的临时切设备在返回前已经恢复，不能保护实际 launch。

这是静态确定的错误设备路径；具体表现可能受 peer access/UVA 影响。本次只申请单 GPU，未声称已经双卡复现。修复应在 C++ 入口持有 device guard，并验证输入、输出、workspace、state 和 offsets 位于同一设备。

### 1.3 编译期 V16 alias 名称与实际默认值不一致

[补丁中的 extension factory](../../../../patches/0001-k2-value-slice-and-dispatch.patch#L893) 把 `-DK2_VALUE_SLICE=16` 只交给 nvcc。读取该宏设置 binding 默认值的 `flash_kda.cpp` 却由 C++ 编译器处理。AST 检查得到 `cxx_flags=['-O3','-Wno-psabi']`，所以直接调用 `flash_kda_vsplit16_C.fwd()` 并依赖默认参数时仍走 V128。

应让 binding 编译获得相同 define，或由 alias 显式传 slice。**不能据此否定已有约 27% 的主结果**：主实验显式传 slice，最终 wrapper 也传入运行时选择。

完整代码依据、指针矩阵和修复建议见 [code-review.md](code-review.md)。final-state TMA 末尾缺显式 wait 的观察仍是待核验疑点，没有当成已证实 bug。

## 2. 新实测重新打开了 tcgen05 的一个窗口

旧报告 `inner=64` 的量是 `T(64)/64`。若 `T(n)=a+bn`，这个量仍包含 `a/64`；它不能自动代表稳态成本。原始两点数据已经提示 TCGEN 的增量斜率可能更低，因此本次用服务器原有 release 二进制，扩大到 `inner=1,4,16,64,96,128,256,512`。

每点 warmup20、iters100、repeats5，交替两条路径，整个 sweep 重复两轮。V128，L0，速度比定义为 `mma_time/tcgen_time`，大于 1 表示 TCGEN 更快：

| grid / inner | 第一轮 | 第二轮 |
|---|---:|---:|
| 12 / 64 | 0.982446× | 0.983072× |
| 12 / 96 | 1.023575× | 1.027044× |
| 12 / 128 | 1.078886× | 1.079218× |
| 12 / 512 | **1.211895×** | **1.211583×** |
| 148 / 64 | 0.903874× | 0.903890× |
| 148 / 96 | 0.963235× | 0.963190× |
| 148 / 128 | 1.012949× | 1.012921× |
| 148 / 512 | **1.137683×** | **1.137703×** |

实际采样交叉区间为 grid12 的 `(64,96]`、grid148 的 `(96,128]`；没有声称找到精确交叉点。两轮在同一 job、同一卡完成，证明同次实验复现，不是跨机器置信区间。新旧 batch 的绝对时间不同，不能拼接它们拟合更精确的曲线。

**应该更新的是研究判断，而不是直接上线新指令。** L0 使用不变的 A/B、每轮重算 MMA、最后一轮才写 global；它没有完整 KDA 的变化输入和 state 依赖。包含 state/gate 与保守 scalar U 重排的 L1，即使 inner512 仍只有约 **0.261×**，说明当前物化路径代价很大。新验证在 grid12 对 L1 的 inner1/2/4/128 通过；原探针 L0 的逐元素正确性检查仍只覆盖 inner1，长 inner timing 不应包装成完整正确性矩阵。

还有一处归因问题：TCGEN L0 的 V128 global 写出在 warp 内跨行，scalar store 的相邻 lane 相隔 512 B；其 B staging 也按转置访问逻辑矩阵。MMA 的 global 访存布局不同。因此固定项不能全归给 TMEM alloc/commit，也不能把这两段 standalone microkernel 称为所有集成实现的严格性能下界。

我的下一道 gate 是：**在真实变化的 U/state、相同舍入点和相同生产者/消费者边界下，跨 Phase 1/3/4/6 复用布局与 TMEM 生命周期。** 最终要击败的是当前最佳 ValueSlice 路径，包括 V16，而不只是旧 V128。这个实验的潜力已经被看见，完整 K2 净收益尚未得到证明。

详见 [性能审查](performance-review.md)、[第一轮 CSV](tcgen_sweep_19844_1.csv)、[第二轮 CSV](tcgen_sweep_19844_2.csv)。

## 3. ValueSlice 仍然有效，但调度模型的适用域要讲清楚

Job19845 对 fixed `B1,T8192,H12,D128`、BF16 state 做完整 forward 对照。表中每条路径取三轮 median 的中位数：

| 测量方式 | V128 / ms | V16 / ms | 降时 |
|---|---:|---:|---:|
| eager，预热 | 0.782784 | 0.569008 | **27.31%** |
| eager，每次调用前写 256 MiB 缓冲 | 0.783408 | 0.570272 | **27.21%** |
| CUDA Graph，预热 | 0.779904 | 0.566784 | **27.33%** |
| CUDA Graph，每次 replay 前写 256 MiB 缓冲 | 0.780272 | 0.567168 | **27.31%** |

auto wrapper 的输出与 final state 相对 V128 逐位一致。这个正结果补上了同口径 graph baseline/variant 对照。

256 MiB 写操作在计时区间之前，用于扰动调用前缓存；**它不保证 K2 进入时冷缓存**，因为同一 forward 中 K1 会重新生产并预热 workspace。这里也没有模拟其他 stream 持续竞争 L2 或 SM。实验支持“收益不依赖当前这一个 eager/hot-buffer测量方式”，不能声称已经覆盖生产并发。

当前 dispatcher 本质上是特定 B300 拓扑上的离线拟合：`reuse_over_l2` 仅用于诊断，resident-block 数主要用于候选可行性；它没有读取实时 cache 压力、其他请求或其他 stream 的并发。3% + 5 µs 是预测收益阈值，不是统计置信界。

Cluster/multicast 的 81.3% 是理想 source-request 减少比例。它不是 HBM 实测减少，更不是 latency 收益。在确定投入完整 cluster 实现之前，应测实际 L2/TMA/DRAM traffic、等待和 residency。

一个更直接的源码机会是：K1 已生成 `ws_tile_prefix`，K2 却对每个序列再次扫描所有前置 `cu_seqlens`。以 CTA 逻辑工作量计，这条路径为 `H × (128/V) × N(N−1)/2`；多短 packed 请求下值得复用现成 prefix，去掉二次扫描。其收益需要按 N 与长度分布测量，不能从单序列 prefill 推断。

## 4. 调度方式会改变数值语义

Job19845 比较同一 T256/H2 输入的整段执行、16-token 分段、17+239 分段和逐 token 执行。所有分段输入使用新分配的对齐 storage；初态相同，final state 串接。随机门控和 raw gate=-8 的长记忆门控均测试 V128/V16。

| 门控 / 分段 | output 相对整段的 relative RMSE | final state 相对整段的 relative RMSE |
|---|---:|---:|
| 随机 / 16×16 | 0，逐位一致 | 0，逐位一致 |
| 随机 / 17+239 | 0.581% | 0.474% |
| 随机 / 256×1 | 0.637% | 0.456% |
| 长记忆 / 16×16 | 0，逐位一致 | 0，逐位一致 |
| 长记忆 / 17+239 | 0.744% | 0.909% |
| 长记忆 / 256×1 | **1.708%** | **2.246%** |

V16 与 V128 在上述比较中结果一致。状态 in-place alias 的两个门控 case 也与独立输出 buffer 逐位一致。

2.246% 是两种 FlashKDA 执行分段之间的差异，**不是相对 FP32 ground truth 的误差，也不是模型精度下降**。它说明，改变 chunk 边界会改变累加/舍入组织；现有 bitwise 证明只覆盖相同组织下的 ValueSlice。未来 time×depth 流水如果随意把 chunk 切成 token，数值上不再自动等价。16-token 对齐是本次小样本得到支持的起点，不是对所有形状的普遍证明。

另一个更小但精确的状态边界：packed offsets 为 `[0,0,17]` 时，第一条空序列的 FP32 初态 `1.00100004673` 被返回为 `1.0`，V128/V16 相同。因为内部 state 先变成 BF16，FP32 public buffer 只扩宽最终值。若 loop API 承诺空步恒等或精确高精度 state carry，这个承诺目前不成立。

## 5. KDA 与 loop transformer 的边界

[LT2（2026-05-20）](https://arxiv.org/html/2605.20670v1) 已实际研究 looped KDA。“KDA 加 loop”已有直接先例。值得继续做的贡献，是精确状态接口、可验证调度和记忆/计算的成本边界。

### 5.1 同时存在时间和深度两条递推

以 FLA 的 `S∈R^(K×V)` 约定：

\[
S_t^{(r)}=A_t^{(r)}S_{t-1}^{(r)}+B_t^{(r)},\qquad
(A_t^{(r)},B_t^{(r)})=P_\theta(h_t^{(r-1)}).
\]

一个 `(token t, loop r)` 同时依赖上一 token 的同-loop state，以及同-token 上一 loop 的 hidden。共享 θ 不使 `S^(r)` 变成同一个变量。精确流式推理通常需要以 `(request, physical layer, loop)` 区分矩阵 state 和短卷积 state。

**已经运行的精确反例：**两个 token、两个 loop，输入为 `[1,2]`。正确独立状态得到第二轮 `[0.25,0.75]`。把第一轮整段 final state 当第二轮 initial，则首输出成为 `0.875`；只把未来 token 从 2 改为 4，首输出变成 `1.375`。正确实现的首输出仍是 `0.25`。这是未来信息泄漏，不能用“小量数值误差”解释。标准库 Fraction 探针见 [loop_boundary_probe.py](loop_boundary_probe.py)。

可以设计跨 depth 共享状态的不同模型，但必须重新定义训练、prefill 和 decode 语义。不能把它当成精确缓存复用。

### 5.2 固定大小 state 的优势仍有成本边界

H12/D128 下，每物理层、每请求、每 loop 的矩阵 state 是 BF16 **384 KiB**，FP32 **768 KiB**。常规不重算的精确 decode，32 loops 即 **12 MiB / 24 MiB**，还要乘物理层和请求数，并加上卷积及其他 attention 的缓存。这不是所有算法的空间下界；重算可以交换时间和空间。

更多 loop 可以提高对已保留信息的计算能力。如果两个历史已经被压成完全相同的全部可用状态，后续确定性计算无法重新区分它们。重新读原 token、稀疏 KV 或外部记忆可以增加可访问信息，但此时应把读取成本也计入模型。

### 5.3 当前 beta 参数域不能直接承接所有表达力定理

对单位 key，当前数学参数域为 `β∈[0,1]`，`A=(I−βkkᵀ)Diag(α)`。因此 `det(A)=(1−β)∏α≥0`。有限个这样的 transition 相乘，不能精确构造 determinant 为 −1 的 Householder 反射。

LT2 附录的反射构造允许 β=2；本地 FlashKDA 使用普通 sigmoid。这是前提差异。该代数结论约束的是理想归一化下的线性 state transition，不能当作 BF16 舍入保持该性质的保证，也不否定完整 nonlinear block 的任务能力。改变 beta 参数域应视为新的模型/数值路线。[论文相关推导](https://arxiv.org/html/2605.20670v1#A2.SS1)

### 5.4 时间递归可以 scan，完整 depth loop 一般不能直接 scan

固定投影后，仿射 pair 的合成 `(A2,B2)∘(A1,B1)=(A2A1,A2B1+B2)` 有结合律，因此 token/chunk 方向原则上能 prefix scan。障碍是合成后秩增长或稠密化、额外算量、workspace 和通信，不能称为数学上不可并行。

深度方向的下一轮投影依赖上一轮 nonlinear 输出，通常不能预先整理成相同的固定仿射 scan。值得测试的是 token-chunk×depth 的合法 wavefront，让多个已就绪阶段并发；但这会改变实际资源竞争与 ValueSlice 最佳选择。loop 数不能直接当作独立 batch 数。

完整文献、公式与公开实现核查见 [loop-boundary.md](loop-boundary.md)。其中对 LT2 公开 KDA generation/cache 路径的观察仅限所读代码，不等于复现或否定该论文的全部模型实验。

## 6. 我建议的后续顺序

1. **接口正确性收口。** 对齐、device guard、编译默认值；明确空 state、alias、分块与 dtype 契约。新增测试按具体触发条件组织。
2. **建立精确 loop-state reference。** 完整序列/流式/随机分块对拍，未来 token 扰动测试，每层每 loop 独立 state 和卷积缓存。先用小型共享 block 让语义可审计。
3. **开启跨 phase TMEM 原型。** 变化的 U/state、真实 BF16 舍入点，包含 layout 转换和消费者，比较完整 K2 与最佳 ValueSlice。L0 crossover 作为立项线索，不作为性能承诺。
4. **按工作负载验证调度。** held-out T3072/T6144、head 阈值、packed 长度分布、并发 stream、graph；看 auto 对最佳候选的 regret 和相对 V128 的最坏退化。
5. **再验证 memory×compute 边界。** 对同一已训练小模型扫序列长度、loop 数、state dtype，以实际毫秒、状态内存、检索与推理正确率作判断；将 wavefront、ValueSlice、检索混合逐个消融。

我最看重的研究问题是：**给定相同答案质量与因果语义，哪一组状态组织、舍入粒度和驻留生命周期，能在 B300 上得到最低 time-to-answer？** 这个问题连接了当前已经有效的 ValueSlice 与 looped reasoning，也能被明确的反例和实验否证。

## 证据与复现

- [Job19844 原始日志](review_19844.log)：Phase-6 两轮 sweep 完成；后续数值测试发现对齐问题后 abort。不能把整个 job 标记成全通过。
- [Job19845 原始日志](followup_19845.log)：完整补充测试运行到 `kind=complete`；未对齐输入在隔离子进程中按预期 −6 退出，clone 对照正常。任务起止 UTC `01:10:26–01:10:34`。
- [最终审查探针](review_probe.py)、[首次 job 脚本](run_review.sbatch)、[补充 job 脚本](run_followup.sbatch)。首次数值探针使用 `.contiguous()`，最终版本改为 `.clone()` 完成分块比较，并保留独立 alignment 复现模式。原始生产实现未改。
- [性能复算](summarize_review.py)、[编译宏证明](code-review-proof.py)、[精确 loop 反例](loop_boundary_probe.py)。
- 本次服务器 final wrapper SHA-256：`c638962a3d333680e923884ba47ffcd1cc4f26b1db4b2097af9bfa01b0b4f50f`，与本地 current 快照相同。dispatch SHA-256：`74e59195d1bdad5a68f3ad9793d722c8195d4f5de3266f8609526d2360ac59b8`。
- 服务器基线 commit：`1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b`；使用现成 patched 扩展和现成 release microbench；GPU CC10.3/148SM，PyTorch2.10.0+cu130，driver580.126.09。

未完成、也未冒充已完成的验证：双卡 device-mismatch、完整 looped 模型训练/质量、真实 decode、并发服务流量、跨 phase TCGEN 实现、Cluster/multicast 实现和端到端 Kimi serving。
