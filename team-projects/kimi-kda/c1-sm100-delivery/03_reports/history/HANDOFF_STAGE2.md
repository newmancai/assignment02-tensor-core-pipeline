# FlashKDA Hardware-Explicit Typed IR - Stage 2 Handoff

> 用途：将本文作为下一对话的工作上下文。继续推进时不要从头调研，也不要重复已经通过的 source-to-IR、typed corpus 和 B300 方向性校准。

## 1. 战略命题

目标不是复刻 CAKE，也不是让 Agent 在 CUDA/PTX 空间盲搜。我们正在构建一层 AI-native kernel control plane：

```text
真实专家 kernel
  -> source-grounded hardware-explicit typed IR
  -> verifier / proof obligations
  -> typed semantic-delta search
  -> B300 calibration + compute accounting
  -> deployment tactic
  -> 将失败沉淀为 rule / primitive，将收益沉淀为 calibration / tactic
```

核心差异：typed IR 不只用于从零生成 kernel，也用于解释、验证和改造生产专家 kernel。候选生成成功不等于优化完成；只有经过相同 public API scope 的激活与复测，候选才能从 `Calibrated` 升级为 `Dispatchable`。

## 2. 当前阶段结论

Stage 2 已完成：

- 从真实冻结 FlashKDA BF16 fused M128 kernel 恢复 hardware-explicit IR。
- 识别 6 类 warp role、23 个 barrier group、97 个带源码行号的 barrier/elect 事件。
- 为真实 Q/K 边建立 completion-dominance 路径。
- 将 29 个 frozen evolution candidate 解析为 29 个唯一 typed semantic fingerprints：
  - 7 个 `scalar_tile`
  - 21 个 `value_tile`
  - 1 个 `value64_split`
- 在 NVIDIA B300 SXM6 AC / SM103a 上完成六形状 correctness 与 CUPTI 校准。
- 远端和本地 typed-IR 回归均为 `20/20 PASS`。

方向性 B300 结果：

| Shape | CAKE public | Prepared evolution | Speedup | Tactic |
|---|---:|---:|---:|---|
| H96 fixed 8192 | 432.838 us | 360.853 us | 1.199x | adapter 验证后激活 |
| H96 mixed | 331.634 us | 330.997 us | 1.002x | 保留 baseline |
| H96 uniform | 372.884 us | 301.669 us | 1.236x | adapter 验证后激活 |
| H64 fixed 8192 | 431.254 us | 334.372 us | 1.290x | adapter 验证后激活 |
| H64 mixed | 231.380 us | 229.363 us | 1.009x | adapter 验证后激活 |
| H64 uniform | 250.963 us | 206.835 us | 1.213x | adapter 验证后激活 |

- 六形状全部 correctness 通过。
- 几何平均方向性提升：`1.152641x`。
- 最大提升：`1.289743x`。
- 5/6 形状的 bootstrap 95% speedup 下界高于 1。
- H96 mixed 的下界低于 1，因此明确保留 baseline。

必须保留的结论边界：

- CAKE baseline 通过 public `recurrent_kda` 测量。
- evolution candidate 通过 prepared evolution launch 测量。
- 两者都是 CUPTI cold-L2 GPU timing，但调用 scope 尚未完全相同。
- 因此 `1.153x` 是选择下一轮 activation 的方向性 kernel evidence，不是 production API speedup。

## 3. 最重要的新发现：Deployment gap

对 `backend=auto` 使用同一个 public benchmark 复测时，六个形状全部仍解析为：

```text
resolved_backend = cake
variant = m128 / m64
```

也就是说，frozen evolution winner 已存在且有方向性收益，但没有进入 public dispatcher。这是当前最有价值的工作边界：

```text
Expressible -> Verified -> Calibrated -> Dispatchable
                                      ^
                                      当前停在这里
```

下一阶段不要继续扩展候选数量；优先打通第一个 `verified candidate -> explicit public backend -> scope-identical benchmark -> guarded activation` 闭环。

## 4. 真实 Q/K 协议事实

冻结源：

```text
<REMOTE_HOME>/flashinfer-cake-sm103-20260908/
  csrc/kda/flashkda_generated_bf16_fused_m128_a1418fd1ae.cu
```

SHA256：

```text
88cd99e24b95baa973b33027760cffb940273e5669889959f9651e5f698c7cb5
```

硬件与资源：

- 1024 threads / 32 warps
- 117376 bytes SMEM
- 240 TMEM columns
- `chunk_tokens=16`
- `value_rows=128`
- 5-stage ring

物理 warp ranges：

| Role | Warp range |
|---|---|
| `compute` | 0-3 |
| `epilogue` | 4-7 |
| `beta_prefetch` | 8 |
| `mma` | 9 |
| `aux_mma` | 10-11 |
| `prep` | 12-31 |

Q/K first edge：

- `aux_mma` 通过 `elect_one` 发布 `qk_full(count=1)`。
- consumers 为 `compute + mma`。
- `compute` 的 4 个 warp 分别 leader-arrive，形成 `smem_free(count=4)`。
- arrival mode 为新增的 `ELECT_ONE_PER_WARP`。
- reuse waiter 是 `prep`。
- `mma` completion 经 SM100 `elect_commit(final_ready)` 汇入 completion path。

已固化的规则/primitive：

| Code / primitive | 含义 |
|---|---|
| `KIR107` | 拒绝物理 warp role 重叠 |
| `KIR713` | 拒绝无法由物理参与者解释的 arrival count |
| `KIR714` | 拒绝 consumer 到 reuse 缺失 completion path |
| `ArrivalMode.ELECT_ONE_PER_WARP` | 表达每个 warp 各由 elected leader arrive |
| `CompletionJoin` | 区分逻辑 consumers、物理 completion participants 与 reuse waiter |

## 5. B300 算力与实验账本

硬件观测：

- GPU：NVIDIA B300 SXM6 AC
- Compute capability：10.3 / SM103a
- SM count：148
- 显存：275040 MiB
- Power limit：1100 W
- PyTorch：2.10.0+cu130
- CUDA：13.0

本阶段保留下来的成功实验：

- 4 次单卡 B300 allocation
- `55.322929 GPU-seconds`
- `0.922049 GPU-minutes`
- `1,980` 个 CUPTI timing samples
- CUPTI sample duration 累计 `0.639589 seconds`
- 估算 allocation energy：`12.72509 kJ`
- 峰值显存约 4.01 GiB

这是可核验下界，不是全部历史用量。早期三个 pre-kernel 环境失败复用了同一个输出路径，账本被后续成功结果覆盖，因此没有被加入 55.32 GPU-s。不要根据 SSH 命令墙钟反推它们的 GPU 用量。

`account_b300_run.py` 已升级为 append-only attempt ledger。未来无论成功还是失败，都应保留：

- UTC start/end
- wall seconds / GPU-seconds
- GPU UUID、型号、显存、power limit
- power/utilization/memory samples
- estimated allocation energy
- Slurm job id/name/node
- command 与 exit code

基础设施经验：远端 JIT 工具链必须使用绝对路径。相对的 `.runtime-deps/bin` 会在 Ninja build directory 中失效。

## 6. 工作区与关键路径

本地 typed-IR 项目：

```text
<PROJECT_ROOT>
```

远端 typed-IR 镜像：

```text
<REMOTE_HOME>/kda-typed-ir-20260908
```

远端 FlashInfer / frozen evolution checkout：

```text
<REMOTE_HOME>/flashinfer-cake-sm103-20260908
```

Python：

```text
<REMOTE_HOME>/FlashKDA/.venv/bin/python
```

远端运行环境必须包含：

```bash
export PATH=<REMOTE_HOME>/flashinfer-cake-sm103-20260908/.runtime-deps/bin:$PATH
export PYTHONPATH=<REMOTE_HOME>/flashinfer-cake-sm103-20260908/.runtime-deps:<REMOTE_HOME>/flashinfer-cake-sm103-20260908
```

Slurm：

- GPU partition：`gpu`
- 分区时限：1 小时
- 当前节点共 8 张 GPU；实验统一申请单卡，除非后续明确需要并行候选。

## 7. 关键代码

| 文件 | 作用 |
|---|---|
| `kda_ir/model.py` | hardware-explicit schedule types、warp placement、fanout、CompletionJoin |
| `kda_ir/frozen_source.py` | 真实 CUDA 源码 role/barrier/elect 事件导入 |
| `kda_ir/kda_first_edge.py` | Q/K first edge、source graph、completion proof、schedule 构造 |
| `kda_ir/verify.py` | KIR verifier rules |
| `kda_ir/cutlass_ts.py` | typed IR 到 CUTLASS Task Scheduling plan |
| `kda_ir/semantic_delta.py` | 29 个 Blackwell candidate 的 typed semantic representation |
| `inspect_frozen_source.py` | source-to-IR certificate 入口 |
| `inspect_evolution_corpus.py` | typed corpus 生成入口 |
| `calibrate_b300.py` | correctness、bootstrap、tactic、算力账本合并 |
| `account_b300_run.py` | append-only B300 experiment accounting |
| `tests/test_vertical_slice.py` | source-to-IR 与 verifier 纵向回归 |
| `tests/test_semantic_delta.py` | typed candidate corpus 回归 |

## 8. 关键证据文件

均位于本地和远端项目的 `evidence/`：

| 文件 | 内容 |
|---|---|
| `source_to_ir_a1418fd1ae.json` | 真实 kernel 导入、warp ranges、97 个事件、completion paths |
| `evolution_typed_corpus.json` | 29 个 typed semantic fingerprints |
| `b300_evolution6_results.json` | 初始 6-shape candidate correctness/timing |
| `b300_evolution6_st_stable.json` | 20 dry-run + 100 measured candidate evidence |
| `b300_legacy_cake.json` | public CAKE baseline |
| `b300_legacy_auto.json` | public auto routing control；证明仍走 CAKE |
| `b300_*_accounting.json` | 四次成功实验资源账本 |
| `b300_stage2_calibration.json` | Stage 2 汇总、置信区间、activation tactic、compute accounting |
| `b300_leader_split_consumer.json` | elect-one + CompletionJoin B300 smoke |
| `b300_cooperative_split_consumer.json` | cooperative split-consumer B300 smoke |

内部交付 PDF：

```text
output/pdf/flashkda_typed_ir_stage2_internal.pdf
```

PDF 共 10 页，已渲染检查并完成文本一致性验证。

## 9. 下一对话的唯一主线

### Stage 3 目标

完成第一个 proof-carrying production activation：

```text
typed winner
  -> verifier-clean certificate
  -> explicit public evolution backend/adapter
  -> identical output/final-state semantics
  -> identical rotating-state CUPTI benchmark
  -> per-shape activate/retain tactic
  -> fallback to CAKE
```

建议执行顺序：

1. 审查 `flashinfer/kda.py`、`flashinfer/kda_prefill.py` 和 `flashinfer/kda_evolution.py` 的 public state-update contract。
2. 不要静默改变 `backend=auto`；先增加显式、可审阅的 evolution backend/adapter。
3. 保持 output、initial/final state、cu_seqlens、state rotations 和 CUPTI scope 与 CAKE baseline 完全一致。
4. correctness 先覆盖六形状，随后使用相同 `20 + 100` 协议复测。
5. tactic 初始策略：5 个置信下界高于 1 的形状候选激活；H96 mixed 保留 CAKE。
6. 如果 scope parity 后收益消失，保留结论并将原因分类为 calibration/tactic，不要为追求正结果篡改口径。
7. public activation 闭环成立后，再进入真正的新 schedule 搜索：warp-role reallocation、completion topology rewrite、stage/resource 联合优化。

### Stage 3 验收条件

- 显式 public adapter 存在且有回退路径。
- 六形状 correctness 全通过。
- candidate/baseline 完全同口径。
- 每个 shape 产生 bootstrap confidence 与 activate/retain 决策。
- 所有 B300 尝试都有 append-only 账本。
- 只有满足以上条件后，才报告 production speedup。

## 10. 恢复工作命令

本地回归：

```bash
cd <PROJECT_ROOT>
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest discover -s tests -q
```

远端回归：

```bash
ssh <B300_LOGIN_HOST>
cd <REMOTE_HOME>/kda-typed-ir-20260908
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest discover -s tests -q
```

重建 typed corpus：

```bash
python3 inspect_evolution_corpus.py \
  <REMOTE_HOME>/flashinfer-cake-sm103-20260908/csrc/kda \
  --output evidence/evolution_typed_corpus.json
```

重建 Stage 2 calibration：

```bash
python3 calibrate_b300.py evidence \
  --output evidence/b300_stage2_calibration.json
```

已验证的最终摘要应为：

```json
{
  "activation_eligible": 5,
  "all_correct": true,
  "allocated_gpu_seconds": 55.32292857486755,
  "geomean_speedup": 1.1526412835968947,
  "public_auto_inactive": true
}
```

## 11. 外部依据

- CAKE: Coding Agents for Kernel Exploration: <https://arxiv.org/html/2608.12629>
- CUTLASS Task Scheduling introduction: <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/ts_general/ts_introduction.html>
- CUTLASS validation and verification gaps: <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/ts_general/ts_validation.html>

## 12. 下一对话建议首句

> 请读取 `HANDOFF_STAGE2.md` 并直接推进 Stage 3：把 verifier-clean evolution winner 接入显式 public adapter，建立与 CAKE 完全同口径的 B300 correctness + CUPTI + append-only accounting 闭环。在 scope parity 完成前不要宣称 production speedup，也不要重新做 Stage 1/2 已完成的调研和回归。
