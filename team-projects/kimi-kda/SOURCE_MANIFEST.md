# Source manifest

## 2026-09-10 公开冻结包

- 唯一入口：`c1-sm100-delivery/README.md`。
- 范围：最终 PPTX/PDF、论文、报告、双层 IR/Agent 源码与测试、B300 证据、复现实验、关键补丁及上游许可证。
- 完整性：`c1-sm100-delivery/MANIFEST.sha256` 覆盖冻结包内 822 个文件；清单文件本身不自校验。
- 脱敏：个人姓名、本机账户与绝对路径、集群登录别名均替换为团队名或占位符；GPU 型号、实验编号、结果和公开上游地址保留。
- 排除：约 237 MB 的可重建 `.pt` 张量快照、编译缓存、完整 profiler 二进制报告、第三方完整源码镜像和迭代旧稿。

## 基线与工作树

- Upstream：`https://github.com/MoonshotAI/FlashKDA.git`
- Baseline commit：`1ce47ea`
- 服务器与本机关键源码于 2026-09-01 逐文件核对，SHA-256 完全一致。
- 跟踪文件 diff：312 insertions、94 deletions；`git diff --check` 通过。

## 关键源码 SHA-256

| 文件 | SHA-256 |
|---|---|
| `csrc/flash_kda.cpp` | `6ea353ff4af3a1fad0f6f3c376c82fe58cceae23eba7c67f21fe0589883a5863` |
| `csrc/fwd.h` | `2612206023e4704f8a7cb91bc8bca0d1414bd124d4c953bb294fb58011261b1b` |
| `csrc/smxx/fwd_kernel2.cuh` | `70a7f7cda2cb1f9f5420b81d729a22d0cf23be4075e0c4d1e59b74eb22b9c1b8` |
| `csrc/smxx/fwd_launch.cu` | `c7d52dea8de32be7bc399423886f97c06ef0db655525c9a1120cd3102c71e99b` |
| `csrc/smxx/utils.cuh` | `b75cfa3b1fa35cc5dd268c0a6157025df0cf06b94f7511900fe0213dfec1e316` |
| `flash_kda/__init__.py` | `6e909391f49c198c593428dfbac99435718b462566ab4c8e97ba22df440f92b2` |
| `flash_kda/dispatch.py` | `74e59195d1bdad5a68f3ad9793d722c8195d4f5de3266f8609526d2360ac59b8` |
| `setup.py` | `2412204d9c44a63cd482abfc57ef6ffa762d9f8ba15b293125bf4e898d00e709` |
| `profile/k2-vsplit-opt/integrated_validation.py` | `fe9a17a0bc6757134d87e0992292b2f91ed2112aa47b010ab0763244749fe1f3` |

## 服务器证据

- 原始验证：Slurm Job 5195。
- 2026-09-01 独立复跑：Slurm Job 14592。
- 2026-09-02 第三次独立复跑：Slurm Job 15466；脱敏日志为
  `experiments/integrated_validation_20260902.log`。
- NCU resource/roofline：Jobs 5166、5173。
- 单扩展构建：Job 5191。
- Nsys V128/自动 ValueSlice 同进程 trace：Job 14991（2026-09-01）。
- Nsys 原始报告、四份 NCU 报告已在本机 `experiments/artifacts/` 归档并校验；因含服务器环境元数据不上传公开仓库。公开仓库保留已脱敏 CSV、紧凑 SASS 样例和复现脚本。

## Runtime Profile Agent 补充归档

- 工具源码：`tools/runtime-profile-agent/kda_ir/`，包含主线 MMA measurement assessment、workload profile、typed schedule、验证、自优化闭环 Agent、等预算协议和 proof-carrying runtime policy。`mma_evidence.py` 从已归档 SASS/NCU/tcgen05 CSV 自动生成物理 profile；`mma_migration.py` 基于 grid、throughput、tile 几何和 compiled residency 输出迁移结论与 MMA 候选排序；`autonomous_agent.py` 实现 conclusions-only memory、canonical 去重、screen/qualification、incumbent 更新和 budget/plateau stopping。
- C1 主线 receipt：`experiments/runtime_profile_evolution/c1_b300_h12_mma_profile.json` 与 `c1_b300_h12_mma_assessment.json`；复用已归档的 Jobs 17937/17965 与 SASS 数据，四份输入的 SHA-256 已写入 profile，未新增 GPU 运行。
- CPU 回归：`tools/runtime-profile-agent/tests/`，共 44 项；所需三份 source-to-IR/consumer evidence fixture 同步归档在 `tools/runtime-profile-agent/evidence/`。它验证工具逻辑，不替代 B300 性能证据。
- 前瞻资格：`experiments/runtime_profile_evolution/b300_prospective_capacity_certificate.json`。
- 资格后机制归因：`experiments/runtime_profile_evolution/b300_prepare_mechanism_certificate.json`。
- 适用边界：单张 B300、H12、BT16 CAKE-generated route 和固定 compiled-resource receipt；策略状态为 shadow-only。
