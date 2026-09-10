# MARPE 框架范围修正：主线、挑战与记忆

状态：自 2026-09-10 起作为后续 agent 探索的上位范围说明。

## 1. 唯一主线

> **FlashKDA 官方 kernel 当前使用 SM80 MMA，分析迁移到 SM100 是否值得。**

`chunks_per_prepare_cta`、runtime policy、多 agent 协同、并发、服务吞吐和 IR 建设都是
回答主线的工具、子问题或后续挑战，不能替换主线。一次局部优化成功也不能把论文问题
缩成“如何调好这个参数”。

## 2. 论文的挑战阶梯

为了把“值得迁移”与“tcgen05 单指令更快”分开，统一使用下面五层：

```text
H0 = 官方 HMMA
H1 = 给定明确且可比较的新增搜索预算后，最强 HMMA
T0 = 与 H1 尽量结构匹配的 tcgen05 + 必要协议包
T1 = 给定相同新增搜索预算后，最强 tcgen05 专属方案
T2 = 最强 guarded SM100 部署路径
```

主线工程问题比较 H0/H1 与 T2；因果挑战主要比较 H1/T0/T1。CAKE 是 T2/full-stack
treatment，不能直接充当 tcgen05 opcode 的单因素证据。每轮 equal-budget 只说明本轮新增
proposal、compile、GPU slots 可比，不能自动抹平两条路线此前投入的历史工程预算。

## 3. 开放 proposal plane

Stage9 的 depth-2 cpc tree 是一个已经执行过的 case-study grammar，不是框架总边界。
通用 proposal 可以改变算法分界、tile、layout、storage、HMMA fragment reuse、ValueSlice、
tcgen05/TMEM lifecycle、warp roles、barrier、pipeline、cluster、dispatch 或完整 carrier。

提案分四种表示状态：

- `executable_existing`：现有载体可直接运行；
- `patch_required`：语义清楚，需要源代码改动或新 build；
- `needs_ir_extension`：当前 IR 缺字段，但 idea 本身不因此错误；
- `prototype_first`：先用机制探针验证关键假设。

只有语义违法、声明/实际身份不符、正确性失败或非法执行可以确定性拒绝。IR 表达不了、
编译失败、相邻 profile 失败和机制解释冲突只能改变状态或优先级，不能升级为技术族禁令。

## 4. 双层 IR

下层 program/physical IR 记录候选是什么：

- semantic contract、lane、design id、parent relation；
- requested backend/route 与 actual route/fallback；
- logical/physical tile、instruction family、layout/storage；
- work partition、grid、physical SM、wave 数、warp roles；
- pipeline、prefetch、barrier、TMEM lifecycle；
- source/build/binary identity；
- changed subtrees 和 explicitly unknown fields。

上层 experience IR 记录证据能怎么用：

- 精确 proposition、比较双方、profile 与 correctness contract；
- matched、mismatched、unknown transfer preconditions；
- observation/localization/controlled-intervention 等级；
- competing mechanisms、invalidation keys、reopen conditions；
- `ranking_prior`、`requires_retest`、`qualification_candidate` 或 `deployment`；
- 影响 proposal/rejection 的 retrieval receipt 和本轮预算。

负结果只否定精确的 implementation × profile × contract。screen 最多写成 scoped prior；
没有新 winner 只说明给定 lane/profile/budget 的 plateau。

## 5. Python、扩展和头文件执行身份

以后每个可测实现都必须留下 execution identity receipt。仅记录 `backend="evolution"`
或顶层 `flashinfer/kda.py` 没有意义，因为实际执行可能 fallback，Python import 顺序也可能
选中另一份同名 `.so`。

最低记录项：

- `sys.executable`、完整 Python version、implementation、prefix；
- Python include/platinclude、`SOABI`、`EXT_SUFFIX`；
- 有序 `sys.path`；
- allowlist 环境：`CUDA_HOME`、`PYTHONPATH`、`LD_LIBRARY_PATH`、`PATH`；
- 每个关键 module 的请求名、实际 `module.__name__`、`module.__file__`、loader、package
  version 和文件 SHA-256；
- 所有已加载 native extension 的实际路径和 SHA-256；
- 关键 `.py`、`.cu`、`.cuh`、`.h/.hpp`、`.so` dependency 的路径、类型、存在性和哈希；
- requested backend/route、observed modules、actual route、target 和 fallback reason。

如果请求候选而实际运行 fallback，性能数据必须写成 `diagnostic_only` 或 `superseded`，
不得进入候选排名。工具入口为：

```bash
python record_execution_identity.py \
  --module flashinfer.kda \
  --module flashinfer.kda_evolution \
  --dependency-file path/to/carrier.cu \
  --dependency-file path/to/binding.cuh \
  --output execution_identity.json
```

执行前用 receipt 发现 import/path 错误；执行后以 observed route receipt 确认真实载体。
两者缺一不可。

## 6. 当前事实记忆

- official 路径是 HMMA；HMMA 已有 V16/prefetch 等强化，不能再称“未优化 HMMA”。
- H12 packed mixed6/balanced6 上，当前 V64 比 V128 分别快 1.2172x/1.1723x，且 bitwise；
  balanced7/8 的收益仅 1.0143x/1.0114x。V64 的 3% 实用边界因此位于 nseq=6 与
  nseq=7 之间，不是所有 packed 的全局规则。
- H12、total=8192 的 nseq=3 上，V32 相对 V64 的四块资格结果为 1.0764x；偏斜
  (4096,2048,2048) 的两块 screen 为 1.0774x。到 balanced nseq=4，V32 反而为
  0.9618x。可复用知识是 CTA-wave-aware crossover，不是固定 ValueSlice 常数。
- 相同 H12 profiles 上，CAKE 仍比 stronger HMMA 快 2.0246x/1.8334x；这是 full-stack
  comparison，不是纯 tcgen05 effect。
- 在 physical 148-SM B300 上，virtual152 persistent schedule 跨 H64/H96 两个 mixed
  profile 都严重回归。grid147、TAIL4、MINIMAX172 与具有紧致 load 下界的 MAX169
  也未过 gate；MAX169 慢 6.17%。这关闭的是当前载体的 load-minimax 调度目标。
- H96 mixed s173 对 Cake 的四块 qualification 为 1.0330x，已是 exact-profile 合格
  route；它没有自动成为所有 H96 或部署 profile 的 T1。
- HMMA V16 Phase-1 lookahead L3 对 L2 的两块结构 screen 为 0.99998x，且 output/state
  bitwise；该 exact H12 fixed8192 候选低于 3% gate。
- tcgen05 V16 preferred-layout contract probe 比 scalar-rematerialization 控制快 1.227x，
  但仍只有同探针 HMMA 的约 0.955x。它证明 layout materialization 是成本来源，不证明
  完整 K2/TMEM residency 已接通。

机器可读实现见 `kda_ir.research_memory`，执行身份见 `kda_ir.execution_identity`；两轮实例
见 `experiments/sm100_open_round1/dual_ir_agent_rounds/`。当前执行环境样例为：

- `evidence/execution_identity_local_framework_20260910.json`：本地框架/论文环境；
- `evidence/execution_identity_b300_framework_20260910.json`：B300 的 PyTorch、FlashInfer、
  evolution Python 源、binding header 与 H64/H96 schedule translation units。

R1--R8 的最终边界见
`experiments/sm100_open_round1/dual_ir_agent_rounds/DEFAULT_KNOWLEDGE_SATURATION_REVIEW_20260910.md`。
当前只签发冻结包络下的预算有界局部结构假说饱和；全局理论最优、完整双分支饱和和
最终 H1/T1/T2 迁移结论仍需新的 production carrier、lane oracle 与共同 public-full 资格实验。
