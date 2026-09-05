# Mainline 实验日志汇总

`summarize_mainline.py` 仅依赖 Python 标准库，不导入 PyTorch、不访问 GPU，也不修改输入或写结果文件。每个输入日志必须对应一次 `mainline_probe.py` 执行；多个文件分别汇总，不跨 binary、作业或日志合并性能样本。

```bash
python3 outputs/kda-mainline-20260905/analysis/summarize_mainline.py /absolute/path/to/slurm.log
python3 outputs/kda-mainline-20260905/analysis/summarize_mainline.py /absolute/path/to/slurm.log --format markdown
python3 outputs/kda-mainline-20260905/analysis/summarize_mainline.py first.log second.log --format json
python3 outputs/kda-mainline-20260905/analysis/test_summarize_mainline.py
```

省略日志路径或传入 `-` 时从 stdin 读取。JSON/Markdown 均输出 stdout；需要归档时由调用方选择目标。默认要求每个 shape、每个模式正好有 repeat 0/1/2；如果主测量明确使用其他重复数，通过 `--expected-repeats N` 同步指定。

## 状态和统计口径

- 没有且仅有一个 `kind=complete`、缺 `shape_complete`、重复行、缺模式、缺重复轮、损坏 JSON、正确性行数与 `correctness_complete` 不一致，均标记 `INCOMPLETE`，不冒充 PASS。缺 `complete` 且已看到失败时，仍保留 `INCOMPLETE` 和 `failure_detected=true`。
- 完整日志的正确性检查失败，标 `FAIL`；没有记录正确性检查，标 `UNVERIFIED`。无法从缺失日志判断是否有意使用了 `--skip-correctness`，因此不会推断 PASS。
- `entry_hardening` 的 terminal suite、case/check 状态和 skip 原因独立汇总。SKIP 不计 PASS；成功但存在 skip 为 `PASS_WITH_SKIPS`。无 entry 日志为 `NOT_RECORDED`，不表示该 suite 已验证。
- 程序退出码：完整验证 PASS/PASS_WITH_SKIPS 为 0；INCOMPLETE、FAIL、UNVERIFIED 为 1。
- 每个 shape/name 的 eager、graph 分开统计。主值是各 repeat `median_ms` 的中位数，不是池化原始样本。每轮 median/min/p10/p90/count 均保留在 JSON；Markdown 显示每轮 median。
- 四个比较基准为同 binary V16、legacy16、同 binary V128，以及同 binary 原始 V16/V32/V64/V128 中聚合 median 最快者。最佳原始 slice 按 shape 和 eager/graph 分别选择，并在各 repeat 对照中固定该选择，避免偷偷使用逐轮 oracle。
- 增益是 `100 × (1 − candidate_ms / baseline_ms)`；正值为降时，负值为退化。另保留 speedup、同编号 repeat 对照、最坏轮次及最坏 shape 退化，不用平均收益掩盖反例。同编号 repeat 不是同时执行的统计配对实验。
- p10/p90 是单轮调用时间的描述性分位数，绝不是置信区间。高精度小数也不表示跨 GPU 或跨时间稳定性。

`test_summarize_mainline.py` 使用纯内存合成 fixture，覆盖成功、缺完成标记、缺模式/轮次、重复样本、正确性计数、伪 PASS、明确失败、skip、多 run 混入、损坏 JSON 和 Markdown 展示。它不需要实验结果文件。
