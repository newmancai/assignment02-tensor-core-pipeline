# 答辩速查

## 30 秒主线

官方 FlashKDA 在 B300 上可以编译运行，但递推矩阵乘的真实 SASS 仍是 HMMA。我们先建立 output + final state 的正确性合同和隔离进程 CUPTI 测量，再用 HMMA/tcgen05 双分支 agent 闭环探索多条迁移路径。结果表明：直接换指令会慢，HMMA 仍有 profile 相关优化空间，而完整 SM100 数据流在九个已测 profiles 上全部取得正收益。我们的建议是增加带 shape guard 的 SM100 专用后端，并保留 HMMA 回退。

## 必记数字

| 证据 | 数字 | 含义 |
|---|---:|---|
| 官方 SASS | HMMA 3,640；TCGEN/UTCMMA 0 | 确认真实执行身份 |
| Direct V128 tcgen05 | 0.919742× | 裸指令替换失败 |
| V16 preferred-layout core | 1.615021× | tcgen05 核心有局部潜力 |
| V16 加 scalar rematerialization | 0.778523× | 接入成本吞掉核心收益 |
| H3 vs official | H12 0.9055×；H96 0.7581× | 完整负结果，按 gate 停止 |
| CAKE vs official | H12 2.4823×；H96 2.2532× | 完整 SM100 路径值得 |
| HMMA nseq=3 | V32/V64 约 1.076× | HMMA 也必须公平优化 |
| tcgen round 8 | preferred/scalar 1.227×；preferred/HMMA 0.955× | 修复了机制，尚未形成生产 winner |

## MARPE 是什么

MARPE：**多智能体运行时画像演化**。它让 HMMA 与 tcgen05 两个分支在相近新增机会下独立提出候选，共用正确性、执行身份和性能 gate，并把正负结果写回双层 IR。

- Program IR 记录实际程序与物理路线；
- Experience IR 记录观察、机制强度、适用范围、允许用途和重开条件；
- 复用 PIKE 的分支搜索/候选分配和 Atrex Kernel Agent 的隔离执行/profiling/Git episode/ABBA；
- 本项目新增双分支 IR、公平预算与证明式晋级。

## 老师追问时的边界

- 2.48×/2.25× 是完整 SM100 engineering treatment，相对当前官方路径；不是 tcgen05 单因素收益。
- Round 8 是相同新增实验机会，不代表历史人工、代码成熟度与 wall time 完全相等。
- 当前没有单 agent 对照，因此不声称多智能体本身提高了最终性能。
- 产品决策是 guarded route + fallback；完整 H1/T1/T2 因果比较属于下一阶段。

## 最后一页可说的话

> 我们得到的不是“新指令一定更快”，而是一条可复核的迁移判定：先确认真实执行路径，再用完整语义和公平 gate 挑战多种方案。局部 tcgen05 会失败，优化后的 HMMA 仍有竞争力，但完整 SM100 数据流在已测 B300 workload 上稳定胜出。因此，FlashKDA 值得增加一个可回退、按 profile 启用的 SM100 专用路径。

