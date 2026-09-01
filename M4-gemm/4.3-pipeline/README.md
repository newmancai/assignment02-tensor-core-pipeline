# 4.3 · 多级 pipeline

状态：待 B 完成。`03_pipeline.cu` 的 NSTAGE 流水仍为 TODO。

- `sweep_stages.sh`：扫描 stage 数。
- `judge_ladder.sh`：性能阶梯判测入口。

报告应包含 producer/consumer 的 phase 变化、prefetch/compute 重叠时空图、
stage sweep 数据和最佳 stage 的原因。

