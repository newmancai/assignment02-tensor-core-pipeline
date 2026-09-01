# Source manifest

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
- NCU resource/roofline：Jobs 5166、5173。
- 单扩展构建：Job 5191。
