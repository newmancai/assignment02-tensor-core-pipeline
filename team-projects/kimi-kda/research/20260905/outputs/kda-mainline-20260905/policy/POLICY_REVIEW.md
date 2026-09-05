# Dispatcher 域与回滚契约回归

2026-09-05。本项遵循 [MAINLINE_REVIEW](../../kda-review-20260905/MAINLINE_REVIEW.md) 的发布收口范围；没有修改原 dispatcher、wrapper、kernel 或参数域。

**结果：78 / 78 PASS。**这是 CPU host-policy 和真实 wrapper 函数体的 stub 回归，不能替代 GPU 正确性与性能验收。

## 运行与产物

- [可运行回归脚本](test_policy_contract.py)：Python 标准库，无额外依赖。
- [逐案例结果与源码 SHA-256](policy_results.json)：78 条结果，含原因、选中变体、关键域和验证限制。

在工作区根目录运行：

```sh
python3 outputs/kda-mainline-20260905/policy/test_policy_contract.py --json-out outputs/kda-mainline-20260905/policy/policy_results.json
```

脚本直接用 `importlib` 加载 `dispatch.py`，不会触发 `flash_kda` 包入口。对 wrapper 使用 AST 提取原文件的函数体，注入显式的 torch 元数据与 extension stub，执行真实 `_dispatch_decision`、`explain_k2_dispatch`、`fwd`。没有导入 torch、加载 `.so`、创建 CUDA context 或联网；明确禁用 bytecode 写入，生成证据仅写本目录。

## 域覆盖结果

| 覆盖 | 实际行为 / 断言 |
|---|---|
| B300 基准 | CC10.3、148SM、L2=132,644,864 B；H12/T8192/BF16 选 V16 |
| 架构与 SM | CC10.0/10.1/9.0/12.0、147/149 SM 均以对应原因回退 V128 |
| L2 ±5% | 接受的整数边界为 126,012,621–139,277,107 B；边界外 1 B 与 L2=0 回退 |
| BF16 标定点 | H12，T2048/4096/8192 都进入完整候选打分并选 V16 |
| BF16 留出点 | H12，T3072/6144 都选 V16；预测分别等于两侧标定点预测的中点，仅证明插值实现一致 |
| BF16 域外 | T0/1/2047/8193/16384 回退 `recurrence_length_not_calibrated` |
| FP32 public state | 仅 T4096 进入打分；T2048/3072/4095/4097/6144/8192 回退 |
| B×H | 0、负数及 >96 乘积被回退；域内 1/12/24/48/96 得到 T8192 的 V16/V16/V32/V64/V128 |
| B×H=96 | 有预测分数，最后因收益 guard 回退；不能称为 sequence-head 域外 |
| 两层 CTA 限制 | V16 在 H37 仍打分、H38 排除；V32 在 H74 仍打分、H75 排除 |
| V128 资源不可行 | 将 fixture shared memory 降至 100,000 B，触发 `official_variant_not_feasible` |
| packed 多序列 | 2/6/32 sequences 直接回退 `varlen_not_calibrated`，先于架构和收益 guard |

这里的硬件参数是测试 fixture，不是本次实际查询设备的结果。域判断只使用 `B×H`，不承担所有非法输入验证；例如两项都为负但乘积为正不应由本报告推断为合法形状，张量 shape/API 校验另有职责。

## 开关、packed 与 raw-forward 契约

未设置环境变量时实际为 `auto`。`FLASH_KDA_K2_VALUE_SLICE=16/32/64/128` 的优先级高于 `FLASH_KDA_K2_DISPATCH=off`，甚至在 q/offsets 元数据不可读时也先返回 override。非法强制值即使同时设置 `off`，仍抛出 `ValueError`。`off`、`OFF`、`0`、`false`、`FaLsE` 在没有强制值时选择 V128。

**明确回滚方式是强制 `FLASH_KDA_K2_VALUE_SLICE=128`。**仅设置 `DISPATCH=off` 而遗留强制 V16，不会关闭切片。回滚依旧调用同一个 `_fwd_raw`，不是 FLA fallback，不承诺发生 CUDA/TMA 错误后重试。stub-forward 已验证强制 V128、off、默认 auto V16、强制 V16 高于 off、packed 多序列 auto V128 都正确传入 `k2_value_slice`，且保留 state/offset 对象与 workspace 参数。

packed 的 offsets 替身只允许 `.numel()`，任何索引、迭代或其他属性访问都会使测试失败。单序列结果与 fixed B1 完全一致，dispatch 只需一次 numel 元数据读取；多序列同样不读取 offset 内容。**这验证无数据读取的 policy，不验证实际 offsets 单调性、边界、设备、stride、alignment 或 C++ 输入契约。**

wrapper 的 public-state 判别也已覆盖：initial 或 final 任意一个是 FP32 就走 FP32 标定域；这不是内部 state 使用 FP32 的证据。

## 验证对象与尚缺的 GPU 验收

原文件 SHA-256 与 `MAINLINE_REVIEW` 所列一致：

- `dispatch.py`：`74e59195d1bdad5a68f3ad9793d722c8195d4f5de3266f8609526d2360ac59b8`
- `__init__.py`：`c638962a3d333680e923884ba47ffcd1cc4f26b1db4b2097af9bfa01b0b4f50f`

CPU PASS 证明这个源码快照的域/选择/回滚行为符合上述契约。下列项目尚不能由本项签字：实际 `.so` 身份和干净构建、GPU tensor 正确性、held-out T3072/6144 的 latency/regret、实测设备属性、真实 stream/device guard、TMA 对齐、持续并发、packed offsets 数据有效性。主代理的 GPU 实验应单独给出结果，不把本测试的 78 条 PASS 计作 GPU 判测。
