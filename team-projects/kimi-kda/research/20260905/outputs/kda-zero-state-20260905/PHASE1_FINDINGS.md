# Job 19924：Phase1 lookahead 的独立审计与 guard 边界

## 结论

**本轮出现了不同于初始化微调的清晰净收益。** 固定 H12/B1/BF16、请求终态的已测点上，无初态选 Phase1 ring4、有初态选 ring2，均在 eager GPU Event、Graph、pre-call cache perturbation、同步 host wall 四种口径胜过外部 release 和同 binary base。T8192 无初态的 eager 延迟 **0.548432 → 0.493776 ms，减少 9.966%**；有初态 **0.459152 → 0.427328 ms，减少 6.931%**。

但不能直接把它写成“所有 2048..8192、所有 state/packing 合约均已验证”的规则：

- **ring4 在全部已测有初态 `both` 点退化**，对 same-binary base 的四口径中位数退化约 5.829–7.713%。
- **packed single、无初态 T8192 的 ring2 比 ring4 更快**，四口径都成立；所以 `HasStateIn` 不是最优 depth 的唯一分界。
- `initial-only=in` 没有性能数据；`state_mode=none` 仅固定 T8192；packed 仅 `out` T8192 有性能。当前无法为这些分支完整区间背书。
- T2049/4095/8191 的无初态尾块均明显正收益，**没有实测证据要求加 `T % 16 == 0` guard**；T<2048 仅正确性，不应据此扩大性能下界。

应继续做独立干净候选和边界补测，而非把实验 selector/三个已否决的 init 变体一起带进生产。此审计没有执行 GPU 或修改既有源码。

## 1. 完整性与比较口径

唯一数据：[phase1_19924.log](phase1_19924.log#L4)。主日志复用 `zero-init-ablation` 标签，但 `strategies` 明确为 base=16、phase1_ring2=40016、phase1_ring4=50016，不能按上一轮 4 种 initializer 数量去汇总。

| 核查项 | 实际结果 |
|---|---|
| 正确性 | **93 PASS = 31 shapes × 3 variants**；exact shape 和全部应有 out/final_state 字段均正确，bitwise/finite 全 true |
| 数值 reference | 候选强制 V16，经实验 adapter 映射 selector；reference 是外部既有 release 强制 V128 |
| Graph correctness | **44 PASS = 11 shapes × 4 variants**，其中 **11 个 legacy 自比较**、33 个非自比较 |
| 性能 | **132 行 = 11 shapes × 4 variants × 3 rounds**；每个 case/name 的 repeat=0/1/2 恰好各一次 |
| 四口径 count | eager 60、graph 60、cache_perturbed 30、wall_sync 20，每行均符合；数值 finite/positive、quantile 顺序合法 |
| 性能 dispatch | 所有 132 行 `decision.value_slice=16`；这是 adapter 前的 ValueSlice 解释，不是对每个 GPU 模板的 profiler 认证 |
| 终止 | 1 environment；correctness_complete(31/93)、11 shape_complete、performance_complete(11/132)、唯一 ablation_complete，均齐全 |

31 个正确性 shapes 和 11 个性能 shapes 与脚本定义逐一匹配，包含状态缺失、短长边界/尾块、FP32、packed 含空段、B2、H24 的正确性控制。但这些额外正确性点不都是性能实验，也没有在本日志中执行 sanitizer、独立 post-timing 检查或多 GPU 测试。完成标记：[correctness](phase1_19924.log#L98)、[performance](phase1_19924.log#L286)、[main](phase1_19924.log#L287)。

候选 SHA256 为 `2d7e3e4c61936bd1fa3b567492de53d38dac8964787e2582052ade1eb9ee1a0f`；外部 `legacy` SHA256 为 `34e2c68cf80de0bd24278afb035fdcd44e4a6205241dfcb9aab4ee95891bc486`。**legacy 是已有 Phase6 P4 release，不是最初 P1 或固定 V128 基线。** 所有收益均为本轮同 job 比较，未与 19918/19920 拼接采样。

指标定义：每个 shape/variant/scope 先有 3 个 round median，再取其中位数；收益为 `100 × (1 − candidate / named_baseline)`，正表示更快，负表示退化。下表四元组依次为 **eager / graph / cache_perturbed / wall_sync**，单位 %。p10/p90 和 shape 范围均不是 CI。

## 2. 对外部既有 release 的完整收益与回归

`out`=无初态、有终态输出；`both`=初态/终态均有；`none`=均无。除 packed 行外均 fixed、H12/B1/BF16。

| Shape | ring2 vs legacy：E / G / P / W | ring4 vs legacy：E / G / P / W |
|---|---:|---:|
| T2048 out | +6.927 / +7.152 / +7.350 / +6.696 | +8.299 / +8.477 / +9.261 / +7.971 |
| T2048 both | +6.406 / +8.196 / +6.308 / +5.022 | −6.431 / −4.959 / −6.247 / −6.766 |
| T4096 out | +7.968 / +8.039 / +7.981 / +7.539 | +9.334 / +9.501 / +9.391 / +8.663 |
| T4096 both | +6.628 / +6.878 / +6.354 / +6.204 | −7.029 / −6.946 / −7.329 / −6.640 |
| T8192 out | +8.592 / +8.824 / +8.542 / +8.277 | +9.966 / +10.326 / +10.032 / +9.683 |
| T8192 both | +6.931 / +6.537 / +6.678 / +6.594 | −7.408 / −7.713 / −7.531 / −7.285 |
| T8192 none | +8.703 / +8.702 / +8.572 / +8.436 | +9.468 / +9.547 / +9.326 / +9.157 |
| packed single T8192 out | +12.925 / +13.079 / +12.917 / +12.715 | +11.815 / +11.959 / +11.790 / +11.719 |
| tail T2049 out | +8.151 / +7.059 / +6.878 / +6.924 | +8.258 / +8.471 / +8.095 / +7.832 |
| tail T4095 out | +8.271 / +8.696 / +7.877 / +7.497 | +9.727 / +9.427 / +9.316 / +8.892 |
| tail T8191 out | +8.677 / +8.643 / +8.389 / +8.286 | +10.008 / +10.119 / +9.678 / +9.717 |

## 3. 对同 binary base 的完整收益与回归

| Shape | ring2 vs base：E / G / P / W | ring4 vs base：E / G / P / W |
|---|---:|---:|
| T2048 out | +6.967 / +7.152 / +7.145 / +6.704 | +8.339 / +8.477 / +9.060 / +7.979 |
| T2048 both | +6.418 / +7.435 / +6.308 / +5.349 | −6.418 / −5.829 / −6.247 / −6.399 |
| T4096 out | +7.968 / +8.039 / +7.950 / +7.739 | +9.334 / +9.501 / +9.360 / +8.861 |
| T4096 both | +6.052 / +6.891 / +6.046 / +6.167 | −7.689 / −6.931 / −7.682 / −6.682 |
| T8192 out | +8.602 / +8.816 / +8.537 / +8.373 | +9.976 / +10.318 / +10.027 / +9.777 |
| T8192 both | +6.944 / +6.537 / +6.688 / +6.620 | −7.393 / −7.713 / −7.519 / −7.255 |
| T8192 none | +8.658 / +8.718 / +8.572 / +8.439 | +9.423 / +9.563 / +9.326 / +9.159 |
| packed single T8192 out | +12.938 / +13.074 / +12.912 / +12.783 | +11.828 / +11.954 / +11.785 / +11.788 |
| tail T2049 out | +8.161 / +8.353 / +6.869 / +7.047 | +8.268 / +9.745 / +8.085 / +7.955 |
| tail T4095 out | +7.945 / +8.039 / +7.944 / +7.581 | +9.406 / +8.776 / +9.383 / +8.975 |
| tail T8191 out | +8.647 / +8.622 / +8.399 / +8.297 | +9.979 / +10.098 / +9.688 / +9.729 |

关键结果的原始 ms：

| Shape / variant | Eager | Graph | Perturbed | Synced wall |
|---|---:|---:|---:|---:|
| T8192 out legacy | 0.548432 | 0.545440 | 0.549920 | 0.560356 |
| T8192 out base | 0.548496 | 0.545392 | 0.549888 | 0.560940 |
| T8192 out ring4 | 0.493776 | 0.489120 | 0.494752 | 0.506096 |
| T8192 both legacy | 0.459152 | 0.454304 | 0.459776 | 0.470835 |
| T8192 both base | 0.459216 | 0.454304 | 0.459824 | 0.470966 |
| T8192 both ring2 | 0.427328 | 0.424608 | 0.429072 | 0.439786 |
| packed T8192 out ring2 | 0.481920 | 0.476800 | 0.483344 | 0.492571 |
| packed T8192 out ring4 | 0.488064 | 0.482944 | 0.489600 | 0.498191 |

`ring4` 并非“更深所以总更快”：fixed 无初态偏好 ring4；packed 无初态的这个已测点偏好 ring2；fixed 有初态 ring4 全部回归。packed ring4 仍是明显正收益，只是约落后 ring2 1.1 个相对 legacy 的百分点；这是否值得独立 IsVarlen selector，须补 packed 多长度后再定，不能凭单点建立复杂 lookup。

## 4. 三轮最坏结果与 baseline 漂移

所建议的已测分支组合没有隐藏 losing round：

- fixed 无初态 ring4：六个 out 长度（2048/4096/8192/2049/4095/8191）及 none=8192，全部 scope、全部 paired round 都较两个基线更快。全组最坏 paired reduction：对 legacy **7.679%**，对 base **7.620%**，均出现在 T2048 out 的同步 wall。
- fixed both ring2：三个长度、全部 scope/round 正收益。最坏 paired reduction：对 legacy **4.772%**、对 base **5.233%**，为 T2048 both 的同步 wall。
- packed out ring2，仅 T8192：全部 scope/round 正收益；最坏 paired reduction 对 legacy **12.707%**、对 base **12.683%**。
- fixed both ring4：最坏 paired regression 对 legacy **7.792%**、对 base **7.769%**，均为 T8192 graph。不能启用无条件 ring4。

同 binary base 对 legacy 仍存在局部漂移，不能混为 lookahead 收益。全 11 shapes 的中位数 `100×(1−base/legacy)` 范围：eager **−0.043% ～ +0.613%**；graph **−1.412% ～ +0.822%**；perturbed **−0.073% ～ +0.327%**；wall **−0.345% ～ +0.040%**。

尤其 T2049 out graph，legacy **0.145056 ms**，base **0.147104 ms**；所以 ring4 相对 base 的 **9.745%** 高于相对 legacy 的 **8.471%**。应保留两列，不拿较慢 base 夸大结果。T4096 both eager，base **0.238464 ms** 比 legacy **0.239936 ms** 更快，ring2 对 base **6.052%**、对 legacy **6.628%**。正结果对两个基线都成立，没有依赖选择有利分母。

## 5. 编译资源不是“无 spill 更快”的单调故事

`phase1-draft.patch` 保持 k 递增的 accumulator 更新顺序，为 k/q/state 三组 raw fragments 建立 lookahead ring；两个 GEMM 消费后才覆盖同一 slot，避免越界尾部 load。源码依据：[三组 ring](phase1-draft.patch#L14)、[消费/复用顺序](phase1-draft.patch#L34)。InitStrategy=4/5 走原标量零分支，不启用 vector/onewarp/unified 初始化。

从本次 `build_phase1.log` 精确匹配 C16/D128/V16/96-thread、BF16、HasStateOut=true 的模板，资源如下。spill 数字是 ptxas 报告的静态 bytes，不是每个 tile 的动态流量：

| IsVarlen / HasStateIn | Variant | Registers | Stack B | Spill store/load B |
|---|---|---:|---:|---:|
| fixed / false | base | 63 | 0 | 0 / 0 |
| fixed / false | ring2 | 56 | 16 | 24 / 20 |
| fixed / false | ring4 | 56 | 8 | 16 / 12 |
| fixed / true | base | 70 | 0 | 0 / 0 |
| fixed / true | ring2 | 72 | 0 | 0 / 0 |
| fixed / true | ring4 | 56 | 8 | 16 / 12 |
| packed / false | base | 63 | 0 | 0 / 0 |
| packed / false | ring2 | 63 | 0 | 0 / 0 |
| packed / false | ring4 | 62 | 0 | 0 / 0 |

来源：[fixed no-state](build_phase1.log#L173)、[fixed state-present](build_phase1.log#L288)、[packed no-state](build_phase1.log#L428)。

所以 fixed 无初态的获胜 ring4 **有 spill**，而原 base 没有；已有初态获胜 ring2 为 72 registers、无 spill。不能把净收益写成“消除了 spill”，也不能凭 register 数量推断 occupancy 或停顿原因。数据支持阶段性 lookahead 改变了有效调度，但各阶段收益/资源惩罚的因果份额仍需目标实例 SASS/NCU，不能由静态 byte 数直接拆账。

## 6. Guard：哪些是证据，哪些仍是候选

保留已有 V16/P4 候选前提：B300 对应构建/硬件策略、D128/C16、非 FP32 public state、N1/H12、已选择 V16、2048≤T≤8192。不能把强制实验 selector 在正确性测试中通过其它 H/N/FP32，当作扩展生产性能范围的依据。

| 分支 | 本轮直接性能证据 | 当前建议 |
|---|---|---|
| fixed、HI=false、HO=true | 2048/4096/8192 + 2049/4095/8191，ring4 四口径/三轮全部正 | ring4 是清晰候选；补 3072/6144 留出点，再验证独立干净 build |
| fixed、HI=true、HO=true | 2048/4096/8192，ring2 全部正、ring4 全部负 | ring2 是清晰候选；禁用通用 ring4；补 3072/6144 与有初态尾块 |
| fixed、HI=false、HO=false | 仅 T8192，ring4 全部正且优于 ring2 | 不足以覆盖整段长度；补 2048/4096/3072/6144；在此之前无终态分支保持基线是保守选择 |
| fixed、HI=true、HO=false（in） | 只有正确性、没有性能 | UNVERIFIED；不要直接沿用 both 的 ring2 收益数字 |
| packed single、HI=false、HO=true | 仅 T8192；ring2 > ring4 > base | 两者都可作候选，但不能声称 HI-only rule 最优；补 2048/4096/3072/6144 |
| packed single 的 both/in/none | 仅部分正确性、无性能 | UNVERIFIED；补矩阵前不扩大该分支性能承诺 |
| T<2048、T>8192、其它 H/N、FP32 | 正确性控制，不是性能准入 | 保持现有回退，不扩大 guard |

不存在仅由这 11 shapes 推出的“2048..8192 每一个整数长度都已可靠”的结论。实际可做的是预先定义小而合理的候选域，再增加留出/尾块/状态矩阵，记录最坏轮次；避免为了当前 benchmark 结果制造仅含几个 token 点的永久 lookup。

若必须在补测前界定最保守的可验证候选，只支持 **fixed + final-state requested** 两个分支：HI=false ring4，HI=true ring2，其它分支回退。该建议是证据边界，不是已获准发布；干净 production patch 去除诊断 selectors/旧 init 分支后可能改变编译，应重新构建并复验默认漂移、位相等、同步/内存安全和净收益。短尾块不需要仅因“不整除 16”而回退，但已有初态尾块/无终态短长度仍需补证据。

最后，eager 是完整 wrapper 的 GPU Event 区间；graph 是 replay；perturbation 在 K1 前，非 cold-K2；wall_sync 是每次调用前后同步的 host wall，不是异步并发吞吐。这里的 6–13% 是相对已有 P4 release 的增量，不能与先前相对 V128 的百分比直接相加，也没有证明模型端到端或多请求 serving 获同等收益。
