# 无初态特化：最小消融设计与 canonical 草案审查

范围：只审查四补丁干净源 `/private/tmp/kda-release-prefetch4.DStLoA` 的 K2 初始化、主循环和 launch；生产仍为 D128 / C16，V16 / Prefetch4 仍在已有编译 opt-in 与 shape guard 内。本文不扩展 loop transformer，不改变数值递推或 Python policy。

状态：源码只读审查完成；按主代理追加授权，在独享副本实现了 **canonical 策略 3 草案**，没有改原树或四补丁基线。只完成补丁正反向可应用检查，**未编译 CUDA、未运行 GPU**。向量零填与单 warp 零填是交给主代理实现/验证的消融设计，不能视为已完成优化。

## 1. 当前最强结论：不能把差距直接解释成零填开销

- `HasStateIn` 在 K2 中只影响初始化：模板参数与两条 `if constexpr` 分支；不出现在 recurrence 主循环。主循环从第 458 行开始，Prefetch4 的状态行 ring 在第 697 行开始。[K2 初始化](../../implementation/phase6/csrc/smxx/fwd_kernel2.cuh#L263)，[主循环](../../implementation/phase6/csrc/smxx/fwd_kernel2.cuh#L458)，[ring](../../implementation/phase6/csrc/smxx/fwd_kernel2.cuh#L697)
- 无初态当前由整个 CTA 执行 BF16 标量零填，然后 generic→async fence、CTA barrier，再经过公共 CTA barrier。有初态由 LOAD warp 的一个 elected lane 发起 TMA，整个 CTA 等待 transaction barrier，随后到达公共 barrier。前者少了 global state load，源码层面却不能保证编译后主循环指令顺序或寄存器分配相同。[标量零填](../../implementation/phase6/csrc/smxx/fwd_kernel2.cuh#L330)，[初态 TMA](../../implementation/phase6/csrc/smxx/fwd_kernel2.cuh#L265)
- `build_release.log` 已直接记录 BF16、D128、V16、Prefetch4、96 threads、非 varlen、`HasStateOut=true` 的 `HasStateIn=false` 使用 **63 registers / 9 barriers**，`HasStateIn=true` 使用 **70 registers / 9 barriers**；packed 版本也分别是 63/70。不是仅由模板名字猜测资源差异。[无初态 ptxas](../kda-mainline-20260905/build_release.log#L152)，[有初态 ptxas](../kda-mainline-20260905/build_release.log#L222)，[packed 无初态](../kda-mainline-20260905/build_release.log#L317)，[packed 有初态](../kda-mainline-20260905/build_release.log#L387)
- 上轮 B1/H12/T8192 的 release wrapper eager 中位数为有初态 0.459184 ms、无初态但写 final state 0.547568 ms；**这些原测试不是严格共用 q/k/v/g/beta 的 NoState 对 explicit-zeroState 因果实验**。可用来定位现象，不能由这两个数字单独证明根因。[汇总有初态](../kda-mainline-20260905/analysis/19901_summary.md#L21)，[汇总无初态](../kda-mainline-20260905/analysis/19901_summary.md#L39)

V16 的 launch 使用 1 个 compute warp + 1 个 LOAD warp + 1 个 STORE warp，共 96 threads；B1/H12 的 K2 网格共 `1 × 12 × (128/16) = 96` 个 CTA，而记录的 B300 有 148 个 SM。因此“少 7 个寄存器必然带来更高有效占用率/更快”不成立：该网格本就平均不足一个 CTA/SM。应优先检查循环 ILP、LDSM/状态更新依赖链、调度和代码布局，而不是先压寄存器上限。[threads/grid](../../implementation/phase6/csrc/smxx/fwd_launch.cu#L192)

一次初始化的逻辑状态量仅 `V × D × sizeof(BF16)`，V16/D128 为 4 KiB/CTA；实现必须仍按 `cute::cosize_v<StateSmemLayout>` 清除物理范围。候选构建可增加静态断言核对实际 cosize，而不可凭逻辑形状忽略 padding。若它确为 2048 个 BF16 元素，当前每个线程仅执行 21 或 22 次标量 store；而 T8192 有 512 次 recurrence tile。**推断**：若严格同输入下差值随 T 明显增长，更符合特化影响循环吞吐；固定截距型差值才更支持一次初始化成本。尚未把此推断标记为 GPU 结论。

## 2. 三种有因果意义的最小消融

所有消融使用同一个工具链、同一个实验二进制、相同 Prefetch4 和 HasStateOut；不改 state 更新、输入/输出 pipeline、barrier 个数、主循环或 dtype。策略编号 1/2 是建议；主代理负责 selector 映射。

| 消融 | 唯一主要改变 | 与谁比较 | 可回答的问题 |
|---|---|---|---|
| 1：全 CTA 128-bit/vector 零填 | 把 BF16 标量 store 变成物理连续向量 store，参与线程仍是全 CTA | 原无初态策略 0 | 标量指令/初始化循环形式是否重要？ |
| 2：只由 LOAD warp 执行相同向量零填 | 与策略 1 使用同一 store helper，但 writer 仅 32 lanes | 策略 1，而非只与原始标量比较 | compute/STORE warp 参与初始化是否改变代码生成或启动路径？ |
| 3：canonical + uniform runtime 初态分支 | true/false 共用同一 K2 实例；零填仍保留原始标量实现 | 策略 0 两种初态；策略 3 内部 NoState 对 explicit-zeroState | 去掉 HasStateIn 内核特化差异后，性能差距是否消失？ |

### 消融 1：全 CTA 向量零填

局部替换位置仅为旧 K2 第 332–336 行；fence 与两个已有 CTA barrier 原样保留。伪代码：

```cpp
constexpr int kBytes = cute::cosize_v<StateSmemLayout> * sizeof(BF16);
static_assert(kBytes % 16 == 0);
auto* words = reinterpret_cast<uint4*>(shared_storage.state_acc.begin());
for (int i = threadIdx.x; i < kBytes / 16; i += NumThreads) {
    words[i] = make_uint4(0, 0, 0, 0);
}
```

这是实现轮廓，不是已经编译验证的补丁。`state_acc` 本身已有 `alignas(128)`；16-byte stride 保持对齐。应检查实际 PTX/SASS 是否生成向量 store，而不能把 `uint4` 源码自动等同于 `STS.128`。如果需要明确控制指令或避免类型重解释问题，可使用经编译验证的 shared-memory vector helper / `st.shared.v4.b32`，不要顺带加入 volatile 或修改 shared layout。PTX 的地址对齐要求见 [NVIDIA PTX ISA — Addresses as Operands](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#addresses-as-operands)。

全零位模式与现有 `BF16(0)` 的正零一致；不允许改成“第一轮不做 k@s/q@s”等算术捷径。这里只清 `state_acc` 的物理范围，不能 memset 整个 `SharedStorageK2`：pipeline/barrier 已经在初始化前构建，且 FP32 buffer 与 pipeline buffers 共享 union。[shared storage](../../implementation/phase6/csrc/smxx/fwd_kernel2.cuh#L97)，[pipeline 构建](../../implementation/phase6/csrc/smxx/fwd_kernel2.cuh#L213)

### 消融 2：LOAD warp-only vector 零填

复用消融 1 的 vector helper，循环改为：

```cpp
if (warp_role == WarpRole::LOAD_QKG) {
    for (int i = threadIdx.x % 32; i < kBytes / 16; i += 32) {
        words[i] = make_uint4(0, 0, 0, 0);
    }
}
// All CTA threads still reach the original fence and __syncthreads().
```

不能加 `&& lane_predicate`：那会把 writer 从 32 lanes 缩成一个 lane，并破坏此循环的覆盖。也不能把含 `__syncthreads()` 的整个零填 helper 只放进 LOAD warp 分支；只能限制 **stores** 的参与者，CTA barrier 仍由所有线程到达。

在 4 KiB 状态下，每 lane 8 次向量 store；并不预设它比全 CTA 的每线程 2–3 次快。价值是让 compute warp 不执行初始化 store 循环，尝试接近有初态路径的 warp 职责划分。它没有提前启动 workspace TMA 或隐藏初始化延迟；公共 barrier 未动。

### 消融 3：canonical specialization（已交付草案）

独享副本：`/private/tmp/kda-canonical-draft.03NaCh`。

补丁：`canonical-draft.patch`，基于四补丁源；SHA-256 `ad1ff278692c93b07a5df0b1d99f0159c63e738e139e02557a38217962e79987`。

草案只修改 `csrc/smxx/fwd_kernel2.cuh` 与 `csrc/smxx/fwd_launch.cu`：

1. K2 模板追加 `int InitStrategy = 0`；kernel 最后追加 `bool load_initial_state = true`。
2. `launch_fwd_impl` 模板追加同名默认参数；策略 3 把传给 **kernel 模板** 的 HasStateIn 固定为 true，并把真实的 host `HasStateIn` 作为最后一个 kernel bool argument。Host descriptor 构造仍按真实 HasStateIn 选择 initial_state 或原 dummy，完全未改。
3. BF16 初始化分支中用 `if (InitStrategy != 3 || load_initial_state)` 包围原 TMA 初始化、copy、CTA barrier、transaction wait 与 fence。false 路径调用原标量零填逻辑。
4. 标量零填抽成局部 `zero_initial_state` lambda，由原无初态 else 和策略 3 的 runtime-false 路径共用。它保留原 fence 和 CTA barrier；公共 barrier 也未删改。
5. `static_assert(InitStrategy != 3 || !StateFP32)` 防止本消融意外扩到 FP32 descriptor / conversion 分支。策略 0 仍按原 HasStateIn 和 StateFP32 决策；没有 selector/setup 映射，所以草案本身尚未选择策略 3。

| InitStrategy | Host HasStateIn | Kernel HasStateIn | Kernel bool | 初始化 |
|---:|---:|---:|---:|---|
| 0 | false | false | false（不使用） | 原标量零填 |
| 0 | true | true | true（不使用） | 原 TMA load |
| 3 | false | true | false | 标量零填；不访问 dummy descriptor |
| 3 | true | true | true | 原 TMA load |

当 D/V/Prefetch/HasStateOut/StateFP32/IsVarlen 相同，BF16 TMA descriptor 的 C++ 类型也相同时，策略 3 的 true/false 必须指向 **同一个 kernel 实例**，而不是两个仅代码相似的实例。bool 来自 host 已知的 optional-state 存在性，不需要读取 device tensor、`.item()`、host sync 或新 allocation。主代理应在 SASS/符号或 profiler 中验证确实 canonicalized；不能只看它们都打印策略 3。

不应把 `launch_fwd_impl` 的真实 HasStateIn 一起改成 true，否则 descriptor 构造会把空 initial_state_ptr 当成有效初态。当前原 BF16 无初态 descriptor 使用 out_ptr 作为 dummy；它不是合法“零状态源”，其描述的 state 形状也不是 output buffer 的使用契约。runtime-false 必须绕开整个 state TMA 分支，特别不能 unconditional `arrive_and_expect_tx` 或 `wait(0)`，否则可能读取 dummy、等待不存在的事务或死锁。[descriptor 构造](../../implementation/phase6/csrc/smxx/fwd_launch.cu#L136)

本草案保证的是策略 0 的分支语义、零填数值和同步顺序不变，**不保证重新编译后策略 0 的 SASS/寄存器绝对不变**：增加参数与局部 lambda 本身也可能触发代码生成变化。应保留原四补丁二进制作为策略 0 的外部对照，并检查是否出现 helper CALL/新 spill。若外部对照显著变动，不能把实验新二进制与旧二进制的差值全归于策略 3。

## 3. 同步与数值红线

- bool 是全 kernel 一致参数，所有 CTA 线程选择同一初始化大分支；`__syncthreads()` 不能被置于 elected lane 或单 warp 条件内。
- 零填后必须保留每个 writer 的 generic→async 可见性和 CTA 汇合。尤其空 packed segment 可以不执行任何 recurrence tile，随后直接 TMA store final state；不能依赖第一轮 compute 的 fence 来发布初始零。NVIDIA 的示例同样把各线程 shared writes 的 proxy fence 放在 CTA 同步前，再由一线程发起 async copy。[NVIDIA Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
- 本轮不删第 340/345 行看似相邻的两个 CTA barrier，不改 TMA pipeline 初始化或 final-state store 完成语义。把删 barrier 与零填/特化混在一个候选会失去因果可辨性。
- 不改第一轮递推、不跳过零参与的乘法、不改变 BF16 cast 或 FP32 state 转换；这避免改变 NaN/Inf、signed-zero 或现有舍入路径。正零状态与 NoState 的逐 bit 相等仍需在严格相同其余输入下运行验证。
- 保持 HasStateOut 固定来对比；有无 final-state 输出也是独立模板维度，不能把其资源变化混入 HasStateIn 实验。
- 仅在主代理已有实验 selector 中选择这些消融；最终发布 guard 不因某一内核改善而自动扩大。特别不把 FP32、D/C、架构支持范围放宽。

## 4. 最小验证矩阵与判定标准

主代理正在执行严格同输入验证；这里不重复提交 GPU 作业。建议每个 N1/H12/T 的 q/k/v/g/beta、参数、workspace 策略完全一致，final_state 固定 BF16、单独输出缓冲；显式零状态预先分配/清零且不与 final_state alias。固定 V16，验证实际 Prefetch4 路径。输入准备、zero tensor 创建不计入 GPU kernel 因果对比，但另行注明它们不是服务端成本为零。

- 核心 T：2048、4096、8192；每种初态/策略交错随机顺序多轮测量，分别记录 K2-only、wrapper CUDA events、graph replay。拟合/比较 `time(T) ≈ intercept + slope × ceil(T/16)`，不单看 T8192 的一个值。
- 必需对照：策略 0 NoState vs 策略 0 explicit-zeroState；策略 1/2 NoState vs 0 NoState；策略 3 NoState vs 3 explicit-zeroState；策略 3 非零初态 vs 策略 0 相同非零初态。前两项分辨初始化实现，第三项剥离不同 kernel 实例，最后一项防止误跳过真实初态。
- 逐 bit 检查 out 和存在的 final_state；增加输出 buffer poison（例如预置 NaN）后 NoState vs explicit-zeroState 检查。仅 memcheck 不能证明没有读 dummy：out_ptr 本身可能指向已分配但语义错误的数据。
- 重跑 nondefault stream 和 CUDA Graph；分别 capture 不同真实初态存在性的调用，不能假设已捕获 graph 的 bool parameter 会随运行时 Python 环境自动改变。
- 主线合法 packed single `[T]` 与 state output absent 必须有代表性执行；公共零填 helper 回归应包含 guard 外的合法空 packed segment。若策略 3 保持 N1/T>=2048 的现有选择范围，空 segment 无法直接覆盖该 fast-path，须明确记录而不是扩大发布 guard 或伪造非法输入。
- sanitizer 重点检查共享内存/同步；确认每个 variant 的 descriptor false 分支安全、TMA transaction wait 均有对应发起者。NCU 应核验 matched kernel 的寄存器、spill、shared-memory footprint、实际 loop 指令/延迟来源；不要先把 63→70 或低 occupancy 本身写成原因。

解释规则：若策略 1/2 的变化主要是固定微小截距，初始化带宽不是约 0.09 ms 长序列差距的充分解释；若策略 3 的两种初态趋同且循环 slope 改善，支持特化/代码生成假说，但需 SASS/NCU 解释具体机制；若策略 3 两种初态相同寄存器仍差距很大，就继续查 TMA 数据路径、输入/测量控制与调度，不得宣布“寄存器已修复”。三种消融都允许失败，它们的目标首先是给出可证伪结论。

## 5. 本子任务交付与验证范围

- `canonical-draft.patch`：两个源文件的最小实验 patch；正向 `git apply --check` 在四补丁基线上 PASS，反向检查在独享草案副本上 PASS。
- 可供主代理继续加策略 1/2 的源副本：`/private/tmp/kda-canonical-draft.03NaCh`。未附 CUTLASS 依赖；沿用既定依赖版本。
- 未改四补丁源、现有发布补丁、Python package、setup 或主代理实验目录。
- 尚未编译 CUDA、验证 lambda 是否 inline、检查新 SASS、运行 GPU correctness/performance/sanitizer。因此没有将草案称为新的发布优化。
