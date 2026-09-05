# Job 19920：初始化消融的负结果与基线漂移审计

## 结论先行

**本轮三个初始化变体均不应发布。** `vector` 没有形成清晰收益；`onewarp` 在无初态路径有小幅、若干 case 跨轮稳定的改善，但没有恢复同输入显式零初态对照揭示的约 14–17% 机会；`unified` 没有显著加快无初态，却把已有初态的快路径拖慢约 18–20%。因此不能保留这些初始化微调，并把下一轮真正有效的改善混记为它们的贡献。

一个必要修正：`onewarp` 不是“所有结果都小于 1%”。T2048 `out` 的 cache-perturbed 中位数改善 **1.228%**，最坏 paired round 仍改善 **1.155%**。其无初态 eager 最大 **0.718%**、graph 最大 **0.765%**、同步 host-wall 最大 **0.665%**。这是真实观测到的小收益，不能一概称为噪声；但它不足以支持当前主线想恢复的稳态收益，也不能推翻其余状态路径的反例。

独立证据仅来自 [ablation_19920.log](ablation_19920.log#L4) 及相应源码。未与 Job 19918/19903 合并时延，未运行 GPU 或改动任何 kernel/probe。

## 1. 完成和数值门槛：通过，不等于性能准入

检查了 exact shape metadata、case/name/repeat 覆盖、输出字段、所有 finite/bitwise 标记、timing count/quantile 顺序、默认 dispatch 与终止标记；无重复/缺失：

| 项目 | 实测 | 严格解释 |
|---|---:|---|
| 主 environment | 1 | `zero-init-ablation`；候选与 legacy 都记录了路径/SHA256 |
| `correctness` | 124 PASS | 31 shapes × base/vector/onewarp/unified；不是把 legacy reference 自比较算进来的 124 行 |
| `correctness_complete` | 1 | shapes=31、rows=124，吻合实际 |
| `graph_correctness` | 55 PASS | 11 shapes × 5 variants，**其中 11 行为 legacy 自比较**；44 行非自比较 |
| `performance` | 165 | 11 shapes × 5 variants × 3 rounds，严格每组 repeat=0/1/2 各一次 |
| 四种 timing scope | 全部完整 | eager=60、graph=60、cache_perturbed=30、wall_sync=20 samples/round |
| `shape_complete` | 11 | case/shape 一一对应且唯一 |
| `performance_complete` / `ablation_complete` | 各 1 | shapes=11/rows=165，随后唯一主完成标记 |

所有应有的 out/final_state 均 bitwise=true、finite=true；没有请求 final_state 的 case 仅要求 out。124 行 correctness 在 candidate 侧强制 V16（经 adapter 映射到诊断 selector），对 **既有 release 的强制 V128**；不是对更早 upstream 构建的 V128。Graph correctness 对既有 release 的 auto 输出。见 [ablation_probe.py:56](ablation_probe.py#L56)、[graph checks](ablation_probe.py#L89)。没有独立的 post-timing correctness 或 sanitizer 结果记录在此日志中。

数值覆盖含短序列/尾块、T8193、state both/in/out/none、gate=-8/+12、packed single、含空段的多序列、B2/T33、FP32 state、H24。它们用于正确性，不代表这些形状都做了性能准入。实验 selector 对非 FP32 的诊断路径直接使用 P4；它不是生产 shape guard 的新承诺。

终止标记：[correctness_complete](ablation_19920.log#L129)、[performance_complete](ablation_19920.log#L361)、[ablation_complete](ablation_19920.log#L362)。

## 2. 基线是什么：必须保留 same-binary base

- `legacy`：此前干净 release/P4 binary，SHA256 `34e2c68cf80de0bd24278afb035fdcd44e4a6205241dfcb9aab4ee95891bc486`。**此处不是此前 mainline 报告中的旧 P1/legacy V16。**
- `base`：新的实验 binary 中 InitStrategy=0 默认路径，SHA256 `42cf4fef69bfc018c7269fe3d468ae944492f3cde879b01f6e41e0051cdf7ac3`。
- `vector`：全 CTA 以 uint4 写零；`onewarp`：仅 warp0 以 uint4 写零；`unified`：编译 HasStateIn=true 的新实例，并用 CTA-uniform runtime bool 选择 TMA 初态加载或标量写零。

三个候选与 base 同 binary、同 inputs、同 wrapper adapter。诊断 ID 10016/20016/30016 只是实验 selector，不是 D/V 维度或新的公开 dispatch policy。Adapter 仅在已有 wrapper 选择 V16 时替换 raw selector，见 [ablation_probe.py:19](ablation_probe.py#L19)。165 行性能记录的 `decision.value_slice` 全为 16；该说明是 adapter 之前的 ValueSlice，不是新模板全部身份的直接 profiler 证明。

新的 base 本身也不是源码字节未变的旧实现：实验 patch 将初始化提取为 lambda、加入 kernel 参数和模板参数。这正是保留 `legacy` control 的必要性。候选的性能主比较采用同 binary **base**；跨 binary legacy 比较只作为漂移/整合成本对照。

### 默认 base 相对 legacy 的差异

下表是 `100 × (1 − base / legacy)`；正值表示新 base 更快。每个数由同一个 Job 中三轮 median 再取 median，范围覆盖 11 个已测性能 shapes，不是 CI。

| Scope | shape 中位数差异范围 | 最坏 paired-round 差异 |
|---|---:|---:|
| Eager GPU Event | −0.025% ～ +0.320% | −0.143% |
| Graph replay | −0.225% ～ +1.645% | −0.327% |
| Cache perturbation | −0.162% ～ +0.010% | −0.464% |
| Synchronized host wall | −0.276% ～ +0.059% | −0.351% |

Graph 有不能忽略的局部基线差异：T2048 `both` 的 legacy 三轮均为 **0.124480 ms**，base 三轮均为 **0.122432 ms**，差 **1.645%**；T2049 `out` 为 **0.147008 → 0.144864 ms**，差 **1.458%**。这不是可以擅自宣布为随机噪声的事实，也没有在本轮隔离出其原因。用跨 binary graph 的 1–2% 差异冒领初始化改动收益是不成立的。

反过来，T8192 `both` 的 base/legacy eager 为 **0.459152 / 0.459168 ms**，差仅约 **0.0035%**；graph 为 **0.456160 / 0.455136 ms**，新 base 慢 **0.225%**。这些差异远不能解释 unified 在同 binary 下约 **20%** 的退化，所以后者不是基线漂移制造的假负结果。

## 3. 无初态：局部清零优化没有恢复稳态 gap

下面全部相对 same-binary base，正值为延迟减少。每格为 **eager / graph / cache perturbation / synchronized wall**，单位 %。`out` 表示无初态、有终态输出；`none` 表示两端 state 都不请求。

| Shape | vector | onewarp | unified |
|---|---:|---:|---:|
| fixed T2048 out | −0.011 / 0.000 / −0.010 / +0.062 | +0.011 / 0.000 / +1.228 / +0.455 | −0.670 / 0.000 / −0.094 / −0.096 |
| fixed T4096 out | −0.338 / +0.034 / −0.023 / +0.068 | +0.372 / +0.765 / +0.700 / +0.545 | −0.344 / −0.503 / −0.039 / −0.199 |
| fixed T8192 out | 0.000 / −0.376 / +0.003 / −0.086 | +0.708 / +0.373 / +0.428 / +0.480 | −0.023 / −0.376 / −0.017 / −0.344 |
| fixed T8192 none | +0.017 / −0.006 / +0.012 / −0.044 | +0.411 / +0.555 / +0.361 / +0.512 | +0.338 / +0.176 / +0.335 / +0.227 |
| packed single T8192 out | +0.020 / +0.204 / +0.190 / +0.278 | +0.373 / +0.569 / +0.533 / +0.557 | −0.003 / +0.009 / +0.182 / +0.258 |
| tail T2049 out | −0.053 / −0.066 / −0.021 / −0.034 | +0.096 / 0.000 / +0.021 / +0.526 | −0.064 / −0.210 / 0.000 / −0.335 |
| tail T4095 out | 0.000 / 0.000 / 0.000 / +0.074 | +0.718 / +0.726 / +0.333 / +0.665 | +0.011 / 0.000 / +0.017 / −0.071 |
| tail T8191 out | −0.023 / 0.000 / −0.250 / +0.014 | +0.370 / +0.363 / +0.387 / +0.472 | −0.358 / −0.375 / −0.355 / −0.333 |

T8192 out 的原始中位数说明数量级：

| Variant | Eager ms | Graph ms | Perturbed ms | Synced wall ms |
|---|---:|---:|---:|---:|
| legacy | 0.549168 | 0.544256 | 0.549952 | 0.560499 |
| base | 0.549232 | 0.544224 | 0.549968 | 0.560358 |
| vector | 0.549232 | 0.546272 | 0.549952 | 0.560838 |
| onewarp | 0.545344 | 0.542192 | 0.547616 | 0.557669 |
| unified | 0.549360 | 0.546272 | 0.550064 | 0.562283 |

onewarp 的 eager 节省 **3.888 µs**，与 matched control 里显式零状态约 **89 µs** 的机会不在一个量级。两个数字来自不同 job，不能相减作为精确因果拆账；这里仅说明改动没有靠近目标数量级。

## 4. 已有初态：unified 明确破坏快路径

`unified` 不是简单复用旧有初态 kernel：它改变该函数的控制流，使同一新实例包含标量清零和 TMA 加载两个初始化分支。将模板 bool 设成 true 不保证编译器保留原 fast-path 稳态代码。代码位置：[runtime 分支](experiment.patch#L75)、[模板 bool 映射](experiment.patch#L122)。

| both-state Shape | Base eager → unified ms | Eager 退化 | Graph 退化 | Perturbed 退化 | Synced wall 退化 |
|---|---:|---:|---:|---:|---:|
| T2048 | 0.127552 → 0.150080 | 17.662% | 18.427% | 17.347% | 15.695% |
| T4096 | 0.239920 → 0.284992 | 18.786% | 18.138% | 18.891% | 17.842% |
| T8192 | 0.459152 → 0.551344 | 20.079% | 20.011% | 20.049% | 19.407% |

T8192 both 的 worst paired-round 退化：eager **20.089%**、graph **20.168%**、perturbed **20.053%**、wall **19.487%**。这不是从一个偶发离群值挑出的反例。

另外，onewarp 在 T2048 both 的 graph 为 **0.124160 ms**，相对 base **0.122432 ms** 慢 **1.411%**，最坏 paired round 慢 **1.529%**。该 state-present 路径本不需要执行清零，不能直接归因于清零工作本身；它再次提示编译/capture/layout 层面的 residual 差异。没有证据支持把 onewarp 宣称为对所有状态无退化。

## 5. 四种口径与下一步约束

`wall_sync` 是新加入的真实 host-wall scope：每个样本先 synchronize，再 `perf_counter_ns`，执行完整 wrapper，再 synchronize；每轮 20 个样本。它包含 CPU 调度/adapter/分配路径、GPU 工作和尾部同步等待，但其请求模式被逐次同步改变，不能叫异步 serving 吞吐。见 [ablation_probe.py:31](ablation_probe.py#L31)。它与 eager CUDA Event 分列，不能互相替代，也不能据此倒推 Job 19918 的 CPU 零张量分配成本。

每轮 variant 顺序随机化；各 shape 五臂共享输入，seed 为 20260907。p10/p90 是单轮描述性分位数，不是 CI。cache perturbation 在 start event 前写独立 256 MiB buffer，K1 仍可暖 K2 workspace。没有跨 job pooling，没有模型/训练/并发吞吐结论。

本轮排除的是**这三种具体初始化改法足以恢复主要 gap**的假设，不是证明所有初始化设计永远无价值，也不是用负结果直接证明 Phase1 lookahead 一定有效。静态 SASS 已经指向循环不变量与 Phase1 load/use 排序；下一轮应保持 original scalar-zero，独立消融 Phase1 triple-ring 的 depth=2/4，并同时保留：

1. 原 release 与 same-binary 默认路径，用于检查编译/捕获漂移。
2. 有初态 both/in 和无初态 out/none；不能只拿无初态的正结果而隐藏快路径损失。
3. 完整 wrapper eager/graph/perturbed 和单列同步 wall；原 rounding、D128/C16、state layout 不变。
4. 逐 bit correctness、真实模板/寄存器/spill 与热循环 SASS 证据；不能仅凭 source ring 深度或寄存器数量声称延迟隐藏成功。

在新候选出现之前，主线应保留既有 P4 release，三个 init 变体仅作为被否决的实验记录。实验完成、数值通过、性能改进成立，是三个不同的门槛。
