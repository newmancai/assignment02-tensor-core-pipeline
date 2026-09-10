# Runtime Profile Agent

## 核心观点

**Kernel is cheap; a trustworthy profile-to-policy decision is expensive.**

这里的 “cheap” 不是说 GPU 执行时间不重要，而是说在生成式工具和模板化
kernel 已经普及后，生成更多候选并不稀缺。真正困难的是：知道当前 workload
走了哪条物理 route、为什么慢、哪些参数应随硬件和请求变化，以及一个候选是否
有足够证据安全进入 dispatcher。

本工具把这条决策链变成可检查的普通程序，而不是让 Agent 直接改 CUDA 后凭
一次 benchmark 宣布成功。

## 主线：先用 Agent 完成复现、测量与迁移判断

Agent 不是挑战阶段的附加包装。`MmaMigrationProfile` 先把同一 B300
workload 的 SASS opcode、recurrence grid、NCU throughput/duration、tile 几何、
compiled residency 和 `tcgen05` 隔离探针绑定成一个证据对象。
`build_c1_mma_profile.py` 直接从原始 CSV 取数，`assess_mma_migration`
随后依次输出：

1. 指令路径是否被 SASS 证实；
2. 物理瓶颈是 compute、HBM、grid/recurrence 还是 issue latency；
3. 在挑战之前给出“是否全面迁移”的结论及证据；
4. 将后续 MMA 工作排序为 `KEEP / MEASURE / STOP`。

对当前 H12 证据，Agent 先得到 `12 CTA / 148 SM`、SM/DRAM
throughput 为 `2.64% / 1.24%`，并由 kernel 语义确认 128 个 Value
行是不改变单元素归约顺序的独立维度；然后结合 V128 `tcgen05`
L0/L1 为 `0.920× / 0.256×`，主导 M=16 与 `tcgen05` 最小 M=64
存在几何错配，且 V128 active blocks/SM 在 L0/L1 从 `12/5` 降为
`1/1`，因此停止 Phase-6 direct swap。下一轮优先保留
`mma.sync.m16n8k16` 并优化并行分解和 issue overlap；只把跨 phase
TMEM-resident 数据流作为需要重新实测的 `tcgen05` 候选。
挑战后的 ValueSlice/P4/Phase-1 结果只用于 verify/measure 和回写
memory，不反过来充当挑战前的选题依据。

可重放主线判断：

```bash
cd team-projects/kimi-kda
PYTHONPATH=tools/runtime-profile-agent python3 \
  tools/runtime-profile-agent/build_c1_mma_profile.py \
  --sass experiments/data/sass_opcode_summary.csv \
  --ncu experiments/final_campaign/data/raw/05_targeted_ncu_summary_17965.csv \
  --tcgen experiments/final_campaign/data/raw/03_tcgen05_probe_17937.csv \
  --semantic experiments/final_campaign/analysis_tcgen05.md \
  --output experiments/runtime_profile_evolution/c1_b300_h12_mma_profile.json
PYTHONPATH=tools/runtime-profile-agent python3 \
  tools/runtime-profile-agent/run_c1_mma_assessment.py \
  --profile experiments/runtime_profile_evolution/c1_b300_h12_mma_profile.json \
  --output experiments/runtime_profile_evolution/c1_b300_h12_mma_assessment.json
```

## 后续：Agent 自优化闭环

`AutonomousRuntimeProfileAgent` 按轮执行 `propose → verify → screen → qualify →
activate`。每一轮把 verifier 拒绝、实测 latency、置信区间和 activation 状态写入
conclusions-only memory；下一轮 proposal source 读取这些结论，避开已知无效结构，
继续提出 typed schedule。候选按 schedule 内容生成 canonical ID，语义重复项只测
一次；只有正确性、scope、正的置信下界和 fallback 同时成立，incumbent 才会更新。

循环在没有新候选、screen budget 用尽、证据连续 plateau 或达到最大轮数时停止。
因此这里的“自优化”是**用可重放的物理证据改进下一轮搜索策略**，不是在线训练
模型权重，也不是让 LLM 绕过 gate 自动修改 production dispatcher。

## 输入与输出

Runtime inputs 包括 workload profile、compiled-kernel resource receipt、typed
schedule proposal、固定测量预算，以及 correctness/scope/timing/fallback
evidence。工具输出：

- `RuntimeWorkloadProfile`：total/max chunks、有效序列并行度和 skew；
- `PrepareGridGeometry`：候选 cpc 的 CTA、wave 和 tail geometry；
- resident-grid-capacity recommendation；
- canonical candidate ID、verifier rejection、screen/qualification receipt、
  conclusions-only memory 和停止原因；
- `SHADOW_ONLY/QUALIFIED/ACTIVE` resolution 和命名的回退原因码。

## 为什么它与 C1 有关

C1 表面问题是要不要把 SM80 `mma.sync` 换成 SM100 `tcgen05`。profile 工具
把问题改写为可验证的决策：当前瓶颈究竟是指令吞吐、grid underfill、prepare
work decomposition、recurrence critical path、资源驻留还是运行时并发？
只有 profiler 与前瞻实验支持某个物理假设后，才把对应参数写入带 fallback 的
策略。

ValueSlice、Phase-6/Phase-1 prefetch 和 cpc capacity rule 因而不是三组互不
相关的技巧：它们分别来自跨 CTA underfill、CTA 内 issue latency 和 prepare
resident-grid capacity 三种 profile 结论。`tcgen05` 的负实验同样是工具链
产出的有效 stop decision。

## 最小检查

```bash
cd team-projects/kimi-kda/tools/runtime-profile-agent
PYTHONPATH=. python3 -m unittest discover -s tests -q
```

当前测试集共 44 项，覆盖原始证据取数、主线 MMA 迁移判断、typed schedule、验证器、runtime profile、自优化闭环 Agent、
等预算消融协议与 proof-carrying shadow policy。GPU 性能证据不由这些 CPU 测试
替代；前瞻 B300 证书见
[`../../experiments/runtime_profile_evolution/`](../../experiments/runtime_profile_evolution/)。

## 当前边界

容量策略只在同一张 B300、H12、BT16 prepare/chain route 和完全一致的 resource
receipt 上通过前瞻验证。默认仍是 `SHADOW_ONLY`：receipt、ABI、route、kernel
image、occupancy 或设备变化时，recommendation 不自动生效。
