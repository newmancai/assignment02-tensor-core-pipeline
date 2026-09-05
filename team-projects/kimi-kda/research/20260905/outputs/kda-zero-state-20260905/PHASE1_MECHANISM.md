# Phase 1 triple-ring lookahead：机制初稿与 clean SASS 核验清单

2026-09-05。当前证据为实验源码 patch、Job 19924 日志、ptxas build log，以及此前同 release 的 SASS/匹配 NCU。**干净候选的实际 SASS 尚未核验**；本文不把实验二进制资源当作 clean build 的结果。仅读文件及经审批 SSH 读取 CUDA 官方本地 help，未改 flags、源或安装，未提交 GPU 作业。

## 结论：假说得到更直接支持，但边界是按 state presence 分化的

Phase 1 的 ring2/ring4 在不改变初始化方式、Phase 6 源码或算术顺序的实验中取得收益，方向上支持此前“Phase 1 load/use 调度是优化机会”的假说。它比显式全零 state 替代 None 的控制更集中，因为不再同时把共享内存清零切成初态 TMA 读取。

但是优化并不随 lookahead 单调改善：T8192 无初态时 ring4 比 ring2 更快；有初态时 ring2 更快，ring4 反而比原实现慢。实验中的 no-state ring2/ring4 都产生 local spill，而有初态 ring2 为 72 registers、无 spill。**存在 spill 的候选仍可能获益；无 spill/更少寄存器/更深预取都不是充分的性能判据。**

## 1. 已复核的实验结果及口径

Job 19924 candidate SHA-256 为 `2d7e3e4c61936bd1fa3b567492de53d38dac8964787e2582052ade1eb9ee1a0f`；external release SHA 为 `34e2c68cf80de0bd24278afb035fdcd44e4a6205241dfcb9aab4ee95891bc486`。

[phase1_19924.log](phase1_19924.log) 确认 31 shapes × 3 same-binary variants = **93 条 correctness PASS**，所有输出/存在的 final state 都 bitwise + finite；另有 44 条 graph correctness PASS（含 11 条 legacy 自检）、132 条性能记录、11 个 shape complete 及完整终止标记。93 条包含 base 对照与 FP32 等 fallback 情况，不是 93 次全部实际进入新 ring，也不是 93 个不同 shape 或训练/泛化正确性证明。

以下为 B1/H12/D128、T8192、非 packed、自动选择 V16 的 **3 轮 eager CUDA-event 中位数之中位数**，单位 ms；它计实际 wrapper 调用的 GPU timeline，不是单独 K2 NCU duration：

| State mode | external release | 同二进制 base | ring2 | ring4 |
| --- | ---: | ---: | ---: | ---: |
| out：无初态、有 final state | 0.548432 | 0.548496 | 0.501312 | **0.493776** |
| both：有初态、有 final state | 0.459152 | 0.459216 | **0.427328** | 0.493168 |

同二进制 base 与 external release 在此接近，支持未因整包重编译就出现同量级漂移；仍保留两种对照，不将其当同一个二进制。

相对于同二进制 base 的各轮配对结果：out/ring4 eager 改善 9.919–9.989%；both/ring2 改善 6.868–7.009%；both/ring4 **退步 7.258–7.400%**。这些是三轮观察范围，不是置信区间。两个优胜组合在本 shape 的 graph、cache-perturbed 与独立 synchronized-wall 口径也同向，但各口径不混合：

| 组合 | graph 中位数之中位数 | cache-perturbed | synchronized wall |
| --- | ---: | ---: | ---: |
| out/base | 0.545392 | 0.549888 | 0.560940 |
| out/ring4 | 0.489120 | 0.494752 | 0.506096 |
| both/base | 0.454304 | 0.459824 | 0.470966 |
| both/ring2 | 0.424608 | 0.429072 | 0.439786 |

这一段只复核父任务所指出的两个 T8192 主案例；未据此替所有 T/packing/state-output 情况作发布判断。测试脚本为 [ablation_probe.py](ablation_probe.py)，实验 adapter 仅把 wrapper 已选/强制的 V16 替换为诊断 ID。

## 2. 源级变化为何与原 SASS 假说一致

[phase1-draft.patch](phase1-draft.patch) 的策略 4/5 分别对应 ring2/ring4。它们保持 `HasStateIn` 原值，不启用先前 vector/onewarp/unified 初始化消融；仍用 Phase 6 StatePrefetch4。新分支限定 `V==16 && kBlocksPerWarp==1`。

每个 ring slot 拥有三个原始 BF16 fragment：`k_decayed[k]`、`q_decayed[k]` 和共同被两次 GEMM 使用的 `state[k]`。D128 的 8 个 key blocks：

1. Prologue 加载 0..L−1，L 为 2 或 4。
2. 在 k 时消费 slot `k%L`，按原顺序先做 k@state、再做 q@state；累加顺序仍严格 k=0..7。
3. 两次消费后才覆盖为 k+L，仅当 k+L<8；最后 L 次迭代 drain，不越界加载。

因此各 operand 的 8 块仍各加载一次，共 24 个源级 matrix-copy 调用；Phase 1 仍是 16 个 GEMM 调用（通常映射为原来的 32 个 m16n8k16 HMMA），没有减少数学项、改变舍入点或重排沿 k 的累加。

Input stage 直到整 tile 末尾才 release；state shared memory 在后面的 Phase 6 才写回。因此 Phase 1 预读未来 key block 不跨越源码上的写依赖，也没有新增 shared store、TMA 或 barrier。环形 fragment 的 owning-storage 与“两个消费者之后才覆盖”是其主要局部正确性约束；已有 bitwise PASS 支持这些测试输入上的实现结果，不替代尾部/状态矩阵/同步验证。

此前 [SASS_REVIEW.md](SASS_REVIEW.md) 已定位 no-state 热循环的额外不变量重算及 keyblock4 的 state LDSM→HMMA 零条间隔；[MATCHED_NCU.md](MATCHED_NCU.md) 给出相同零数学初态时 short-scoreboard、sleeping、issue 的对应变化。新 ring 直接增加源级可调度的独立加载/计算窗口，与此机会一致。

仍不能据此宣称 ptxas 完整保留 L=2/4 的预取距离，或性能只来自 Phase 1。更大 fragment 活跃区间会改变整个 kernel 的寄存器分配，Phase 6 即使源码不变也可能重排；spill/hoisting/地址计算的变化都可能参与收益。

## 3. 精确编译资源：区分固定长度与 packed

从 [build_phase1.log](build_phase1.log) 逐条匹配尾模板 `...,D128,V16,IS3,OS2,NT96,SI,SO,FP,VL,P4,InitStrategy`。以下均 SO=true、FP=false：

| 布局 / SI | 策略 | Registers | Stack B | Spill stores B | Spill loads B | build 行 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fixed / false | base 0 | 63 | 0 | 0 | 0 | 173 |
| fixed / false | ring2 = 4 | 56 | 16 | 24 | 20 | 183 |
| fixed / false | ring4 = 5 | 56 | 8 | 16 | 12 | 178 |
| fixed / true | base 0 | 70 | 0 | 0 | 0 | 288 |
| fixed / true | ring2 = 4 | 72 | 0 | 0 | 0 | 298 |
| fixed / true | ring4 = 5 | 56 | 8 | 16 | 12 | 293 |
| packed / false | ring2 = 4 | 63 | 0 | 0 | 0 | 438 |
| packed / false | ring4 = 5 | 62 | 0 | 0 | 0 | 433 |
| packed / true | ring2 = 4 | 72 | 0 | 0 | 0 | 553 |
| packed / true | ring4 = 5 | 62 | 0 | 0 | 0 | 548 |

所有这些条目为 9 barriers。spill stores/loads 是编译属性，不是本次请求动态总流量；不能将 16/12 B 直接乘 T 或线程数宣称运行流量，必须先定位发生在哪个路径、是否每 tile 执行。它们是真正的 ptxas spill 报告，与只凭 `MOV.SPILL` 指令名猜 local-memory spill 不同。

两个关键反例：

- fixed/no-state ring4 有 spill，仍在完整调用测量中比无 spill 的 base 快约 10%；“spill 必然使它慢于 base”已不符合此实验。
- fixed/有初态 ring4 也有 spill且变慢，而 ring2 无 spill且变快；这与资源/调度 trade-off 相容，却不证明该次退步全由 spill 导致。

不同模板可得相同 56 registers 却不同 stack、不同耗时；packed 实例又不 spill。不要用一个 fixed 实例的编译结果覆盖整个 selector，也不要把“深度越大寄存器越多”当作编译后定律。

## 4. 官方本地 flag 语义：不是“等级 10 用更少寄存器”

本次通过只读 SSH 现场读取 `/usr/local/cuda-13.0/bin/ptxas --help` 与 `nvcc --help`；二者版本 V13.0.88。ptxas 的 `--register-usage-level <0..10>` 默认 5：较高值更激进地优化，允许用额外寄存器换取潜在的生成代码改进；较低值抑制会显著增加寄存器使用的优化。它可与 maxrregcount/launch bounds 协同，属于 BETA，官方不保证不同 ptxas release 间实现一致。

因此当前 `--register-usage-level=10` 不是 10 个寄存器、不是寄存器最低/最高数量，也不是“无 spill”或“最大 ILP”的保证。`--maxrregcount` 则是上限，不是要求实际使用某个数量。本次没有改变这些 flags；clean build 应记录完全相同工具链，不能只写泛化的 CUDA 13.0 就假定代码生成稳定。

## 5. 干净候选 SASS 核验清单（待二进制）

### A. 身份、选择与对照

- 记录 clean `.so` 的真实加载路径/SHA、source patch hash、完整 nvcc/ptxas flags 与版本；与原 release、实验 ring 二进制分开。
- 从 resource dump 获取完整 symbol，再对精确实例 disassemble：D128/V16、Phase6 P4、state-out=true、fixed，SI=false 的候选 ring4 与 SI=true 的候选 ring2，分别配原 release 对照。clean 若采用独立 Phase1Lookahead 参数，重新解释尾模板，不能沿用实验 InitStrategy 4/5 的猜测。
- 实際 kernel name/launch 选择应由后续 profile 验证；`explain_k2_dispatch` 只证明 V16，不证明 Phase1 ring 被调用。guard 外应仍使用旧路径；不能把诊断 ID 留作生产维度。

### B. 算术与 shared 访问骨架

- 对齐 k/q/state 的 block 地址，确认 8 块都覆盖、每块两个消费者后再复用 slot，无 out-of-range tail preload；不要要求机器码按源码字面 prologue 顺序执行。
- 核对 Phase 1 两条累加链的 k 顺序和 BF16 operand/FP32 accumulator 类型；预期仍为 32 条 HMMA，整个 tile 原来为 52 HMMA。矩阵 load/store、conversion 的数量若改变，先解释编译变换，不能默认为数学等价或错误。
- 原整个 tile 为普通 LDSM 27、转置 LDSM 16、普通 STSM 1、转置 STSM 8；用它们作结构基准而非把 opcode 总数当动态工作量。Phase 6 源码未动也要复核写回、gate FMA/pack、barrier/fence/release 顺序。

### C. 真正检验调度假说

- 重新找 hot-loop 回边，计循环内 S2R、lane/CTA 地址计算及不变量保留；旧 PC 地址不可套用到 clean binary。
- 对相同 keyblock（尤其此前 block4）记录 k/q/state load→首次消费 HMMA 的指令间隔、邻近独立工作和寄存器覆盖；分别看 prologue、steady、drain，不用几个 load 提前就断言整体改善。
- 比较 Phase 6 的排序/活跃寄存器是否也改变；若变了，性能 attribution 要覆盖它，不称“只改变 Phase1 SASS”。
- 读取新 `REG/STACK/LOCAL`，定位真正 LDL/STL 的 PC、所在 warp 路径及循环次数。特别区分一次性初始化/producer 路径和每 tile compute spill；不要把 UR↔R 的 MOV.SPILL/R2UR.FILL 误当 local memory。

### D. 证据升级的边界

- 静态指令间隔不是 cycles，spill metadata 不是动态流量。若以后拿到 NCU，只将 issue/eligible/short-scoreboard/sleeping 的汇总变化作为旁证；PC/warp 角色归因需要对应定位数据。
- clean 抽取可能改变编译调度。即使源公式与实验相同，也需 clean 自身的 bitwise out/final-state、尾部、状态输入/输出组合、rollback/guard 和重复计时结果；不能直接继承 93 行 PASS 或上表性能。
- 在 clean SASS/验证到齐前，状态保持为“源级与实验机制相容，干净候选待验证”，不宣告完成发布或普遍最优。

## 证据指纹

| 文件 | SHA-256 |
| --- | --- |
| Phase1 增量 patch | `e10834179290cac2a833ee2642c11045ee4ee4bb07a54ffc41fa5696e0429179` |
| build_phase1.log | `5829063ef7b153c2a2637f6646d502db4e77e8d89ca1d29c28a00defaf7f9bd8` |
| phase1_19924.log | `62e5ae62e9bc7303c6ee162b5410d45e6cfe68ee22ca2e8bb3bdbc125d57f1e5` |
