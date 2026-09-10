# Assignment 02 提交核对表

本表把课程题面最后一页的提交要求映射到仓库中的实际交付物。状态只表示技术
材料是否齐全，不代替课程方对独立作业、团队分工和公开许可的最终认定。

## 一、代码与判测

| 题面要求 | 仓库证据 | 状态 |
|---|---|---|
| 提交全部 Hands-on 实现 | M0–M6 对应目录；逐题入口见 [`README.md`](README.md) | 完整 |
| 保留 judge / PASS 输出 | 各题 README、实验记录与 [`docs/evidence/b300-results.md`](docs/evidence/b300-results.md) | 完整 |
| FROM-SCRATCH 题保留 PASS 记录 | [`docs/full-report.md`](docs/full-report.md) 的逐题输出和最终检查 | 完整 |
| DEBUG 题说明现象与修复 | 总报告各 DEBUG 小节包含修改前现象、根因和修复解释 | 完整 |

## 二、书面报告

| 题面要求 | 仓库证据 | 状态 |
|---|---|---|
| 回答纸面问题 | [`docs/full-report.md`](docs/full-report.md) 覆盖 M0–M6 | 完整 |
| 实验表格与性能归因 | 总报告及逐题 `EXPERIMENT.md` | 完整 |
| 所有数据注明 GPU | 总报告公共环境、逐题实验环境与 B300 证据页 | 完整 |
| 区分必做与选做 | M4.4、M5.3(d) 明确标为未纳入必做；Team C2 未开展 | 完整 |

M0–M6 在题面中属于非团队作业。若课程要求独立提交，每位成员仍应遵守课程的
独立完成与署名规则；本仓库的汇总报告用于进度协调、互审和归档。

## 三、Team C1 大作业

| 交付物 | 权威入口 | 状态 |
|---|---|---|
| 题目定义 | [`00_task/TASK.md`](team-projects/kimi-kda/c1-sm100-delivery/00_task/TASK.md) | 完整 |
| 代码与补丁 | [`08_patches/`](team-projects/kimi-kda/c1-sm100-delivery/08_patches/)、[`06_agent_framework/`](team-projects/kimi-kda/c1-sm100-delivery/06_agent_framework/) | 完整 |
| 正式报告 | [`SM100_MAINLINE_DELIVERY.md`](team-projects/kimi-kda/c1-sm100-delivery/03_reports/SM100_MAINLINE_DELIVERY.md) | 完整 |
| 研究复盘 | [`RETROSPECTIVE.md`](team-projects/kimi-kda/c1-sm100-delivery/RETROSPECTIVE.md) | 完整 |
| 答辩 PPTX/PDF | [`01_slides/`](team-projects/kimi-kda/c1-sm100-delivery/01_slides/) | 完整，已逐页检查 |
| 论文 PDF/源码 | [`02_paper/`](team-projects/kimi-kda/c1-sm100-delivery/02_paper/) | 完整，已逐页检查 |
| 原始/派生证据 | [`04_evidence/`](team-projects/kimi-kda/c1-sm100-delivery/04_evidence/) | 完整 |
| 复现说明 | [`REPRODUCTION.md`](team-projects/kimi-kda/c1-sm100-delivery/REPRODUCTION.md) | 完整 |
| 文件完整性 | [`MANIFEST.sha256`](team-projects/kimi-kda/c1-sm100-delivery/MANIFEST.sha256) | 823 项全部校验通过 |

## 四、视觉与可读性检查

- 题面 PDF 15 页已完整阅读并逐页检查；
- C1 答辩固定版 15 页已逐页检查，无裁切或不可读图表；
- C1 论文 8 页已逐页检查，无裁切或版式溢出；
- 3.3 barrier 代际图与 4.3 流水时空图已升级为泳道式 SVG/PNG，并核对依赖语义；
- 2026-09-10 冻结包是唯一最终入口，9 月 3–9 日版本明确标为历史材料；
- 根 README 先给提交入口和状态，再给目录与研究结论。

## 五、自动回归记录

- 本机：M1 fragment、M2 descriptor/swizzle 全部 PASS；
- 本机：C1 Agent 框架 `93 passed`；
- `b300-login` 隔离临时目录：M5 `7 passed`，量化 outlier 结果与报告一致；
- C1 冻结包：本次文档/论文改动后重新生成清单，`823/823` 项校验通过。

## 六、提交前人工项

- [ ] 成员 A、B 在 [`docs/full-report.md`](docs/full-report.md) 填写真实姓名；
- [ ] 若课程要求论文署名版，将公开 PDF 中的 `Anonymous Authors` 替换为真实成员；
- [ ] 确认课程允许公开 starter code 的衍生实现与答案；
- [ ] 决定许可证；
- [ ] 如集群账号标识必须从 Git 历史彻底移除，另行安排历史重写。
