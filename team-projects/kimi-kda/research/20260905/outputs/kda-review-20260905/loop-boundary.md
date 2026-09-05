# KDA × Looped Transformer：可证明边界与下一轮实验

日期：2026-09-05。范围：以共享 Transformer block 沿深度重复执行为 looped Transformer；以当前本地 FlashKDA forward 和 FLA naive 为算子语义。不把当前算子测量冒充模型推理实验。本文没有在远端执行训练或 benchmark；已运行下述本地精确有理数 CPU 反例。

## 判断

最有价值的问题已经从“能否把 KDA 放进 loop”移动到：**在精确因果语义下，能否用更少的状态和更合适的 GPU 调度换取有效推理深度，同时保住可检索的信息和数值稳定性？**

这也划清三类独立对象：训练好的共享权重是慢变量，KDA 的矩阵记忆是沿 token 更新的快变量，每个 token 的 latent hidden 是沿 depth 更新的快变量。三者都叫 recurrent，并不使它们成为同一个缓存。

## 1. 已有工作已经到哪里

- [Universal Transformers](https://arxiv.org/abs/1807.03819) 建立了共享参数的深度递归和按位置动态停机；它不是 KDA 的时间递归。
- [Kimi Linear 官方论文](https://arxiv.org/html/2510.26692v1) 给出 KDA 的细粒度 gate 和 delta-rule recurrence；其模型是 KDA 与 MLA 混合，不能把整模型称为固定大小记忆的纯 RNN。[官方仓库](https://github.com/MoonshotAI/Kimi-Linear)、[FlashKDA](https://github.com/MoonshotAI/FlashKDA)。
- [Huginn / Scaling up Test-Time Compute with Latent Reasoning](https://arxiv.org/html/2502.05171v1) 的 §6.1–6.2 试验过缺失 depth 的 KV 使用最深可用条目、循环覆盖 KV cache；这是有任务评估的近似，不是参数共享推出的逐元素等价。它还在每轮注入输入并采用截断反传。[官方代码](https://github.com/seal-rg/recurrent-pretraining)。
- **直接相关的新工作已经存在：**[LT2: Linear-Time Looped Transformers，2026-05-20](https://arxiv.org/html/2605.20670v1) 的 Table 2 包含 looped KDA；实验尺度为 0.6B/1.3B、4 loops。[官方 LT2 代码](https://github.com/chili-lab/LT2) 已开放。因此不能把“loop + KDA”本身作为新颖贡献。

以下是从方程、本地代码与上述公开实现独立推导的边界，非论文替我们证实的效果。

## 2. 二维因果依赖：权重能共享，精确状态通常须分 depth

采用 FLA 参考的数学布局 `S ∈ R^(d_k × d_v)`；FlashKDA API 使用转置后的 `[B,H,V,K]`，不能直接混用。

\[
D_t=\mathrm{Diag}(e^{g_t}),\quad
A_t=(I-\beta_t k_t k_t^\top)D_t,\quad
S_t=A_t S_{t-1}+\beta_t k_t v_t^\top,\quad o_t=S_t^\top q_t.
\]

本地依据：naive.py:60–63（历史临时路径，参见本目录实验补丁）：先 decay，后用衰减后的状态计算 delta residual。

将一个 KDA block 沿 depth 执行 `R` 次时，标准语义为

\[
(q_t^{(r)},k_t^{(r)},v_t^{(r)},g_t^{(r)},\beta_t^{(r)})
=P_\theta(h_t^{(r-1)}),\qquad
S_t^{(r)}=A_t^{(r)}S_{t-1}^{(r)}+B_t^{(r)}.
\]

`(t,r)` 同时依赖 `(t−1,r)` 的矩阵记忆与 `(t,r−1)` 的隐变量。训练时按 depth 扫完整序列、decode 时按 token 扫所有 depth，都是这个 DAG 的合法调度，但必须保留相同状态身份。缓存 key 至少包含 `(request, physical_layer, loop_index)`，短卷积 state 也同样须分 loop。

**可手算反例。**取一个 scalar KDA，`q=k=1, β=1/2, α=1`，`v=h_previous`，省略 residual/MLP，两个 token 输入 `1,2`。逐 depth 独立状态的第一轮输出为 `0.5,1.25`，第二轮为 `0.25,0.75`。如果 token-major 执行时所有 depth 共用一个 state，第二轮输出变成 `0.5,1.25`。同样的共享权重没有带来同样的结果。

更危险的是整序列跑完第一轮后，将 `S_L^(1)` 当作下一轮 `S_0^(2)`：例中第二轮首 token 变为 `0.875`，已经依赖第二个输入。对 causal LM，这是未来信息泄漏，不仅是精度误差。

## 3. “线性时间”不等于无限记忆，也不等于 loop 免费

忽略投影、MLP、卷积和额外混合层，固定宽度下：

| 语义 | 总序列 mixer 算量 | 精确流式 decode 的记忆 |
|---|---:|---:|
| 单轮 KDA | `O(L H d_k d_v)` | `O(H d_k d_v)` |
| `R` 轮独立状态 KDA | `O(R L H d_k d_v)` | `O(R H d_k d_v)` |
| `R` 轮 full attention | `O(R L² H d)` | `O(R L H d)` |
| `R` 轮共享同一 KDA state | 算量仍随 `R` 增长 | 状态可不随 `R` 增长，但模型语义已改变 |

精确 decode 保存 `R` 个 state 是常规不重算方案，不是严格的所有算法空间下界；愿意重算历史或假设特殊结构时可以交换空间和时间。训练无梯度 prefill 可以 depth-major 扫描，仅保留当前扫描 state 和前后 hidden；这不代表生成下一个 token 时仍可免费丢弃其他 depth 的历史 state。

按当前代表形状 `H=12,d_k=d_v=128`，每物理 KDA 层、每请求、每 loop 的矩阵 state 为 BF16 `384 KiB`，FP32 `768 KiB`；`R=32` 即 `12 MiB` / `24 MiB`。还未乘物理层数、请求数，也没计卷积 state、激活及混合 MLA 的 KV。权重共享减少参数/优化器成本，不自动减少这些项。

在 `p` 位有限精度、仅保留一个 `d_k×d_v` state 的模型里，矩阵能编码的离散状态数至多为 `2^(p d_k d_v)`，这是信息容量的粗上界，**不是**“只能记 d_k 个 token”的结论。若两个不同历史已映到完全相同的所有可用状态，后续只读取这些状态的确定性 loop 无法区分它们。更多 depth 能改善对保留信息的计算；如果每轮重新读取原 token 或访问外部 KV，就也在重新访问更多信息，不能把增益全归功于固定 state 的表达力。

## 4. 当前 FlashKDA 的参数域限制了哪些表达力论证

本地 [fwd_kernel2.cuh:586](../../implementation/upstream/csrc/smxx/fwd_kernel2.cuh#L586) 用普通 sigmoid 得到 beta，约落在 `[0,1]`；[Python 接口](../../implementation/upstream/flash_kda/__init__.py#L13) 也明确接受 beta logits。LT2 附录 B.1 的反射构造取 `β=2, α=1`。这个差异会改变可表示变换，而非仅改变数值误差。[LT2 Appendix B.1](https://arxiv.org/html/2605.20670v1#A2.SS1)

对单位 key，`0≤β≤1`、正对角 `D`，矩阵行列式引理给出

\[
\det(A)=(1-\beta)\prod_i\alpha_i\ge0.
\]

因此任意有限个这样的线性 state-transition 乘积也不会有负行列式；无法精确等于行列式为 `−1` 的 Householder 反射。**这只约束这类矩阵 transition，不否定带非线性投影/FFN 的整模型可能实现相关任务。** 若探索 `2·sigmoid`，须视为新算法/模型参数域，重新训练或迁移验证，当前 forward kernel 的兼容性不能假定成立。

另一个代数审查点：LT2 Table 1 使用与 KDA 一致的 `(I−βkkᵀ)D`，但正文 Eq.(4)、附录 Eq.(10) 写作 `D(I−βkkᵀ)`。二者只有在相应因子可交换时等价。取 `k=(1,1)/√2, β=1, D=diag(1/2,1)`，前者为 `[[.25,−.5],[−.25,.5]]`，后者为 `[[.25,−.25],[−.5,.5]]`。引用推导前应统一约定；不能据此直接断言其全部实验实现错误。

**“R 个 rank-1 factor”也不自动等于标准 depth-loop 的单个 rank-R state 更新。**乘积论证成立的前提是这些 factor 依次作用于同一个 state；标准 depth-loop 则有多个 `S^(r)`。即使指定共享 state，在有非标量 gate 时，省略 gate 后的相同 key / 正交 key 简化也不能无条件套回去。`rank(A−I)` 在一般 diagonal gate 下甚至可以是满秩；应讨论相对适当 diagonal 基准的低秩修正。

## 5. 公开实现给出的一个直接可审查接口缺口

2026-09-05 读取的 LT2 `main` 中，`KDALinearAttentionBlock.forward` 固定 `past_key_values=None,use_cache=False`，外层对该 block 只传 hidden；见 [transformer.py](https://github.com/chili-lab/LT2/blob/main/apps/LT2/transformer.py#L922)（约 L922–941、L2376）。这对应整段序列独立扫描，未将 KDA state 跨 depth 串成一个矩阵。[generate.py](https://github.com/chili-lab/LT2/blob/main/apps/LT2/generate.py#L77) 的缓存安装针对 `Attention` 模块，随后逐 token 调 model。由这两个文件推断：直接使用此通用 generation 路径时，KDA 历史状态没有被接上，需要先做完整序列与流式 logits 对拍。此发现仅针对所读公开路径，不断言论文使用的所有 benchmark 或其他部署实现都有此问题。

对我们更有意义的交付是一个**明确、可测试的 loop-state API**：既能精确处理每层每 loop 的矩阵和卷积 state，又能显式标注近似 sharing/halting 策略。先保证 `full-sequence == token-streaming`，再谈吞吐。

## 6. 因果线性 state 可 scan；非线性 depth 一般不能同样 scan

固定一层的投影后，KDA 是仿射 recurrence。定义组合

\[
(A_2,B_2)\circ(A_1,B_1)=(A_2A_1, A_2B_1+B_2).
\]

该运算满足结合律，所以 token/chunk 方向原则上可做 parallel prefix scan。“chunk 依赖永远无法并行”不是正确的不可能性结论。代价是一般合成会增加秩或稠密化，显式 dense `A` 合成约需 `O(d_k³)`，同时增加 workspace、同步和数据搬运。当前 K2 的长 CTA 串行链是具体成本选择。

下一轮 depth 的 q/k/v/g/beta 依赖上一轮输出和非线性，因而一般无法把 `R` 轮提前整理成同一个固定 affine scan。合法的并行空间是二维 DAG 的已就绪 wavefront、多个请求、head、value slice；把 `R` 直接乘进独立 batch 是错误调度。

**值得测的系统假设：**以 token microchunk 沿 `time×depth` 对角线 wavefront 流水，给低 head 场景创造多个已就绪 depth stage，可能用流水并发改善 underfill。但这必须计入 microchunk 搬运、全层投影/FFN 依赖、launch/同步，以及与现有 ValueSlice 的竞争；只是可证伪假设。

## 7. ValueSlice 的可迁移边界与误差边界

固定 q/k/v/g/beta 时，`S[:,J]` 各 value 子集的更新彼此独立；这是 ValueSlice 不需要 reduction 的数学依据。它仍适用于每一次 loop 的 KDA mixer。因此能复用该切分原则。

然而下一 block 的输出投影和 FFN 通常会混合 value/head 通道，之后的 q/k/v 也随之变化；不能将 `R` 次完整 loop 各自保持为永久独立的 value slices，除非重新约束权重结构。当前 report 的约 `27%` forward 降时也只覆盖指定长 prefill 和低并发形状。若受优化部分占整轮时间比例 `f`，整体速度比为 `1/(1−0.27f)`；增加 `R` 只重复这个代价，不会变成 `R` 倍加速。decode `T=1` 应单独评估 fused recurrent 路径，不能套用 `T=8192` 调度阈值。

固定投影、单位 key、`β∈[0,2]`、`0≤α≤1` 时：

\[
\|A_t\|_2\le\|I-\beta_t k_tk_t^\top\|_2\|D_t\|_2\le1.
\]

这保证精确线性状态转移非扩张，但不保证严格收缩。若单步舍入扰动为 `ε_t`，最坏仍可能线性积累；有一致收缩 `ρ<1` 时才有几何级数误差界。更不保证含 residual、归一化、MLP、数据依赖投影的 depth Jacobian 也非扩张。

应分别测 `time horizon L` 与 `loop depth R` 的误差面、状态范数、gate 饱和、Jacobian-vector 放大和任务正确率。只测最后 tensor 的 global RMSE，可能掩盖少数 token/head 的大误差；测 hidden 迭代差很小，也可能是错误固定点、表示坍缩或周期轨道，不足以证明推理完成。

## 8. 下一步最值得执行的四个实验

| 优先级 | 实验与控制 | 判定标准 |
|---|---|---|
| 1 | 构建一小块共享 KDA 的 FP32/FP64 reference；完整序列、逐 token、随机分块；独立 loop state、共享 state、整序列跨 loop 续 state 三种策略 | 精确策略对拍一致；后两种的偏离与泄漏可被捕获。加扰动未来 token 的因果测试 |
| 2 | 固定训练算力/参数/数据分别比较 KDA、GDN、full、KDA+稀疏检索；`R=1,2,4,8,16`；扫记忆条目数与推理步数两个轴 | 找到“信息已丢失，增加 R 无补救”和“信息仍在，增加 R 改善计算”的分界；不能只报告平均任务分数 |
| 3 | 对同一已训练小模型扫 `(L,R,state dtype)`，记录 per-token/head 最大误差、logits KL、任务精度与实际毫秒 | 数值稳定与推理收益同时成立；把纯算子 bitwise 对拍和模型质量结果分开 |
| 4 | 同一精确模型比较 V128/V16、普通逐 depth 扫描与 microchunk wavefront，并单独测 decode | 以端到端 time-to-answer、峰值 state/activation 内存与正确率绘制 Pareto；所有缓存、同步、投影/FFN计时 |

自适应停机适合放在这些控制实验之后：某 token 提前停机，其后继 token 的更深 loop 所需历史 state 不能凭空存在。可以选择重算、以冻结 hidden 补写深度 state，或近似共享；三种方法的算量与模型语义不同。Huginn 的经验说明近似可能有效，但 KDA 的历史已聚合进 state，不能像逐 token KV 那样替换一个条目而不重放后续状态。

## 本地证据入口

- [可运行 CPU 反例](loop_boundary_probe.py)，命令 `python3 outputs/kda-review-20260905/loop_boundary_probe.py`，仅 Python 标准库，全部 Fraction 精确算术；2026-09-05 执行 PASS。独立 depth state 的完整序列与 token-streaming 都得到第二轮 `[0.25,0.75]`；改变未来 token `2→4`，正确首输出始终 `0.25`，错误跨 loop 续整段 final-state 则由 `0.875→1.375`。同一探针还验证非交换矩阵例、beta 单位区间行列式恒等式，以及 `β=2` 可构造行列式 `−1` 的反射。探针是 mixer-only 反例，不是训练模型表现。
- [最终报告](../../../../docs/c1-final/FINAL_REPORT.md)
- FLA naive 语义（历史临时路径，参见本目录实验补丁）
- [FlashKDA Python state 布局](../../implementation/upstream/flash_kda/__init__.py)
- [ValueSlice 补丁](../../../../patches/0001-k2-value-slice-and-dispatch.patch)

公开源码引用的是读取时 `main`，未固定 commit；发表级复现须 pin commit、依赖和模型 checkpoint。本文重点是给出可验证问题与限制，不声称已完成 looped 模型训练或验证其质量收益。
