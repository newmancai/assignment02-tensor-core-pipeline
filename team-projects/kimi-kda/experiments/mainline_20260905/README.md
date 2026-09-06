# C1 主线补充实验（2026-09-05）

本目录归档 C1 在首轮 ValueSlice 结论之后完成的两层增量优化及其负例。所有 headline 均来自同一 Slurm job 内、同输入的配对比较；不同 job 的绝对毫秒不横向拼接。

公开归档已将机器相关绝对路径替换为 `repo://team-projects/kimi-kda/`、`artifact://local/` 与 `cluster://b300-home/` 占位符；二进制和补丁 SHA-256、Job ID、计时与硬件字段保持原值。

## 结论摘要

| 层级 | 参照 | B300/SM103 结果 | 交付状态 |
|---|---|---|---|
| Phase-6 StatePrefetch=4 | 旧 V16/P1 | 有初态新增降时 19.40%；无初态新增降时 9.02% | `0004`，显式 `sm_103a` 构建，默认关闭 |
| Phase-1 lookahead | 已有 Phase-6 P4 | Job19934 eager 增量 5.15%–12.40%；T8192 首段/续段分别为 9.32%/6.55% | `0005`，依赖 `0004`，默认关闭 |
| 完整候选 | 同 Job19934 V128 | T8192 首段 prefill 降时 37.23%；有初态续段降时 45.52% | 低并发 latency 候选，不是默认 serving 路径 |
| 双流负例 | Phase-6 P4 | 无初态 joined pair 由 1.147440 ms 增至 1.164832 ms，回归 1.52% | 阻止无条件默认启用 |

百分比不能相加。Job19934 的完整候选结果已经包含 ValueSlice、Phase-6 P4 和 Phase-1 lookahead 的累计贡献。Job19934 仅记录到一次 1095 MHz 时钟采样，早期作业频率不同，因此只使用同 job 比例。

## 代码顺序

从 FlashKDA commit `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b` 开始依次应用：

1. [`0001-k2-value-slice-and-dispatch.patch`](../../patches/0001-k2-value-slice-and-dispatch.patch)
2. [`0002-dispatch-packed-single-sequence.patch`](../../patches/0002-dispatch-packed-single-sequence.patch)
3. [`0003-release-entry-hardening.patch`](../../patches/0003-release-entry-hardening.patch)
4. [`0004-guarded-v16-prefetch4.patch`](../../patches/0004-guarded-v16-prefetch4.patch)
5. [`0005-guarded-phase1-lookahead.patch`](../../patches/0005-guarded-phase1-lookahead.patch)

`0004` 要求 `FLASH_KDA_CUDA_ARCHS=103a` 和 `FLASH_KDA_ENABLE_V16_PREFETCH4=1`。`0005` 还要求 `FLASH_KDA_ENABLE_V16_PHASE1_PREFETCH=1`。两项优化均为编译期开关；取消运行时环境变量不会改变已生成的二进制。运行时强制 V128 可回退兼容路径，恢复 Phase-6-only V16 需要单独二进制或重新构建。

准入域保持为已选 V16、D128/C16、BF16 public state、N=1、H=12、`2048 <= T_total <= 8192`。N=1 只描述一次调用的 sequence 数，当前 guard 不知道 GPU 上是否同时存在其他请求。

## 证据索引

| 证据 | 文件 | 含义 |
|---|---|---|
| Phase-6 正式候选 | [`release_19901.log`](data/release_19901.log) | 构建身份、正确性、计时、入口 hardening、sanitizer 退出状态 |
| Phase-6 状态矩阵 | [`state_matrix_19903.log`](data/state_matrix_19903.log) | fixed/packed、四种 state 合约、尾长与三种 CUDA-event 口径 |
| Phase-6 NCU | [`release_19901_baseline_ncu.csv`](data/release_19901_baseline_ncu.csv)、[`release_19901_release_ncu.csv`](data/release_19901_release_ncu.csv) | 旧 V16/P1 与 V16/P4 的匹配 profile |
| Phase-1 干净验收 | [`clean_19934.log`](data/clean_19934.log) | 40 形状主矩阵、尾块、state chain、Graph、计时后复查和双流负例 |
| Sanitizer | [`clean_19934_memcheck.log`](data/clean_19934_memcheck.log)、[`clean_19934_synccheck.log`](data/clean_19934_synccheck.log) | 定向检查均为 `ERROR SUMMARY: 0 errors` |
| Phase-1 profile | [`clean_profile_19935.log`](data/clean_profile_19935.log)、[`clean_19935_*_ncu.csv`](data/) | out/both × P4/Phase1 四条匹配 NCU 路径 |
| 身份与边界 | [`BUILD_MANIFEST.json`](data/BUILD_MANIFEST.json) | 源码、二进制、job 和已知限制 |

重新汇总 Job19934：

```bash
python3 scripts/summarize_clean.py data/clean_19934.log --format markdown
python3 -m unittest discover -s scripts -p 'test_summarize_clean.py'
```

主验证只证明单张 B300 上的已测 kernel/wrapper 合约。多 GPU guard、alias 独立构建、实际安装与回滚、完整 Kimi K3 checkpoint、TP8/NCCL 和真实 serving 并发矩阵仍未完成。
