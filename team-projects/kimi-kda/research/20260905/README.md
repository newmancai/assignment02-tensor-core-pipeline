# C1：FlashKDA 从 SM80 MMA 迁移到 SM100，是否值得？

## 简要回答

**不值得全面机械替换成 `tcgen05`，值得做受保护的 B300/SM103 专用优化。** 官方 FlashKDA 的 `mma.sync` 与 C16、小矩阵求逆和现有融合布局协同设计；虽然 `tcgen05` 在部分长重复微基准中已获得正收益，但还没有完整 K2/forward 的净收益证据。原实现已经使用 TMA，“SM80 MMA”不等于整个 kernel 仅使用 SM80 能力。

已证明有效的路线是保持 D128、完整 128×128 state 和原舍入顺序，先用 ValueSlice 扩展独立 CTA，再优化 Phase6/Phase1 预取。最新同作业、H12/T8192/BF16 的完整 forward 相对 V128，无初态和有初态分别降时 **37.23% / 45.52%**；其中最新 Phase1 相对此前 P4 的新增降时为 **9.32% / 6.55%**。这些是算子级结果，不是完整模型加速。

因此推荐保留 V128 fallback，并以编译期 opt-in 提供已验证范围内的预取候选。双流无初态请求对仍有 **1.52% 回归**，不能默认替代吞吐导向路径。这里的贡献是生产约束下的并行度和调度设计，不是证明新指令永远无效，也不是全球首次提出软件预取。

## 从这里阅读

- [最新主线结论、边界与下一步](outputs/kda-zero-state-20260905/MAINLINE_RESULT.md)
- [40 形状干净候选验收](outputs/kda-zero-state-20260905/CLEAN_FINDINGS.md)
- [SASS 与 NCU 机制互证](outputs/kda-zero-state-20260905/CLEAN_SASS.md)
- [最新补丁、构建与回滚说明](outputs/kda-zero-state-20260905/CLEAN_PHASE1.md)
- [上一阶段 Phase6 P4 主线结果](outputs/kda-mainline-20260905/MAINLINE_RESULT.md)
- [TCGEN 长重复 crossover 与证据限制](outputs/kda-review-20260905/performance-review.md)
- [C1 原题](C1_TASK.md)、[原课程报告与答辩材料](../../docs/c1-final/README.md)

三个 `outputs/` 子目录完整保留本轮文本研究产物：代码审查、否决实验、原始计时/正确性日志、NCU CSV、SASS 提取、补丁、测试与作业脚本。Loop 边界探索只作附录，不作为主线发布的前置条件。未上传编译二进制、CUDA/CUTLASS 依赖、虚拟环境或 Python cache；这些不是本轮新增源码。

## 必须覆盖旧稿的两处口径修正

1. 不能再用 inner64 的负结果宣称 TCGEN“充分摊销后仍必然不如 MMA”。Job19844 的 grid12/inner512 L0 已约为 1.212×；但固定 operands 的 L0 不是实际 chunk recurrence，当前 L1 和完整数据流仍没有发布依据。
2. C32/C64 的 token18 FTZ/overflow 是最坏范围模型，不是改大 FlashKDA kernel 的 GPU 实测。FLA 大块小探针使用温和预激活 gate，`safe_gate=False`；其 finite 结果不证明极端门控下的 rescale 修复或对 C16 的性能优势。

旧报告中“下一步”“待验证”和“未发布”等表述是各阶段的历史状态，不应覆盖最新验收结论。本次 GitHub 上传是研究材料发布，不是安装候选扩展或宣告通用生产就绪。

## 代码与补丁链

`implementation/` 提供精确代码快照（不包含 CUTLASS），各自带上游 LICENSE：

| 快照 | 内容 |
|---|---|
| `upstream` | FlashKDA `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b` 的 C++/CUDA、Python 与 setup |
| `hardened` | 上游 + 0001/0002 ValueSlice/dispatch + 0003 入口硬化 |
| `phase6` | hardened + 0004 guarded StatePrefetch4 |
| `phase1` | phase6 + 0005 按初态选择 Phase1 lookahead；本轮最终候选 |

正式补丁顺序：

1. [0001](../../patches/0001-k2-value-slice-and-dispatch.patch)
2. [0002](../../patches/0002-dispatch-packed-single-sequence.patch)
3. [0003](outputs/kda-mainline-20260905/hardening/0003-release-entry-hardening.patch)
4. [0004](outputs/kda-mainline-20260905/release/0004-guarded-v16-prefetch4.patch)
5. [0005](outputs/kda-zero-state-20260905/clean-phase1.patch)

其它名为 `experiment.patch`、`canonical-draft.patch` 或 `phase1-draft.patch` 的文件是历史消融，不得叠加进上述发布补丁链。实验 selector 不在最终候选中。

在本目录运行 CPU 验证（不使用 GPU）：

```sh
python3 outputs/kda-zero-state-20260905/test_clean_phase1_contract.py implementation/phase6 implementation/phase1
python3 -m unittest discover -s outputs/kda-zero-state-20260905 -p 'test_summarize_clean.py'
python3 outputs/kda-zero-state-20260905/summarize_clean.py outputs/kda-zero-state-20260905/clean_19934.log --format markdown
```

GPU 构建需自行准备与清单匹配的 PyTorch/CUDA 环境、Python 开发头文件，以及 CUTLASS `5c149f52a436782210263fb2f19b354443a61c6a`，放在 `implementation/phase1/cutlass`。在该源码目录中构建隔离扩展，不安装覆盖已有包：

```sh
FLASH_KDA_CUDA_ARCHS=103a \
FLASH_KDA_ENABLE_V16_PREFETCH4=1 \
FLASH_KDA_ENABLE_V16_PHASE1_PREFETCH=1 \
FLASH_KDA_EXTENSION_NAME=flash_kda_phase1_C \
python setup.py build_ext --build-lib build-validation/lib --build-temp build-validation/temp --force
```

定制扩展名不会自动替换真实 wrapper 导入的 `flash_kda_C`。验证脚本显式将真实 wrapper 分别绑定到旧/新扩展，并记录实际 `.so` 身份；不要仅重命名二进制文件。历史 Slurm/build 脚本保留原复现布局，但服务器账号已改为 `YOUR_USER_ID`，需按本机路径配置后使用。

## 公开副本与原始证据的 hash

为了公开发布，本副本机械替换了个人工作站路径、服务器账号，并把 Markdown 本地链接改成仓库相对链接。数值、内核补丁和测试逻辑未作人为修饰；CUDA 源码快照与原候选保持一致。

[PUBLICATION_INVENTORY.json](PUBLICATION_INVENTORY.json) 记录每个导入文件的原始 SHA-256 和公开副本 SHA-256。历史 `BUILD_MANIFEST.json`、报告中的原始文件 hash 与 `.so` hash 仍保留为原实验身份，**不能误用它们校验经过路径规范化的公开日志**；公开文件用 `published_sha256` 校验。此清单不包含本 README 等新编写的发布说明。

未覆盖的项目仍包括真实模型质量、完整 serving 时延、多 GPU 验收、独立 alias GPU build 和持续混合并发。已完成的单卡算子验证不替代这些生产准入项。
