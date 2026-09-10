# C1：FlashKDA on B300

团队：**奶龙必胜**

## 最终交付

公开交付统一从 [`c1-sm100-delivery/README.md`](c1-sm100-delivery/README.md) 进入。该目录包含：

- 15 页答辩 PPTX，以及已嵌入字体的 PDF；
- 8 页论文 PDF、LaTeX 源文件和参考文献；
- 完整复盘、主线报告、IR 知识审查和阶段性交接记录；
- 结论对应的原始测量、SASS、正确性和预算凭据；
- HMMA/tcgen05 双分支 Program IR、Experience IR、Agent 闭环代码与 CPU 测试；
- B300 复现实验、CUDA probe、Slurm 作业和关键补丁；
- PIKE 与 Atrex Kernel Agent 的开源来源、锁定版本和许可证。

## 结论

在已测 B300 工作负载上，单独把 HMMA 替换为 tcgen05 没有形成稳定收益；围绕数据驻留、布局、并行度、流水和 dispatch 协同重组的 SM100 全栈路径获得明确收益。推荐对已认证 profile 启用受条件保护的 SM100 后端，其余场景回退 HMMA。

这里比较的是 FlashKDA public-call 范围内的 kernel 路径。完整 Kimi serving 的 tokens/s、TTFT、尾延迟和多卡收益仍需由对应系统测量决定。

## 三步主线

1. **复现与测量**：锁定 B300 执行身份、正确性、SASS/NCU 和完整接口计时边界。
2. **分析**：由双分支 Agent 在相近新增探索机会下生成候选，以双层 IR 保存“改了什么”和“结论在哪些条件下成立”。
3. **挑战**：用局部机制实验和隔离 public-full 测量决定候选晋级、停止或回退。

## 仓库中的其他目录

`docs/c1-final/`、`experiments/`、`patches/` 和 `tools/runtime-profile-agent/` 保留早期复现、挑战与框架演化记录。它们仍可用于追溯，但最终答辩口径、论文和公开审计以 `c1-sm100-delivery/` 为准。

