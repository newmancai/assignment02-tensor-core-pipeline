# V16 StatePrefetch4 独立静态审查

2026-09-05。仅审查 release patch、候选 checkout 的 Phase 6、其前后 state/pipeline 生命周期和 launch guard；没有改动代码，没有运行 GPU，也没有延伸到模型或 loop 研究。

**判断：在当前 D128/V16、同步 `mma.sync`、合法输入及既有 CuTe 布局契约下，未发现 Prefetch4 新增的环形索引错误、旧 state 被提前覆盖、gate 配错或跨 tile 残留依赖。**索引和消费顺序可作局部证明。最终发布仍需完成分支/边界矩阵与二进制身份验证；单个随机长序列的 bitwise PASS 不覆盖这些义务。本项不独立核验所报告的 19.4% 性能收益。

## 1. 实际修改很小，生效范围必须读完整

[0004 patch](../release/0004-guarded-v16-prefetch4.patch) 没有重新写 Phase 6。它将原 `PREFETCH=1` 改为默认值为 1 的 template 参数，在已选 V16 时条件实例化 4。

候选 [launch guard](../../../implementation/phase6/csrc/smxx/fwd_launch.cu#L255) 为：编译期开启 `FLASH_KDA_ENABLE_V16_PREFETCH4`；`D==128 && !StateFP32`；`N==1 && H==12 && 2048<=T_total<=8192`；当前分支 `k2_value_slice==16`。默认构建、V32/V64/V128、FP32 public state、N>1、其他 H 和长度均保留 Prefetch1。release build 仅允许显式 `FLASH_KDA_CUDA_ARCHS=103a`。

`!StateFP32` 包括没有 state buffer 的情况，不是仅“两端都有 BF16 state”。FP32 initial-only、final-only 或两端 FP32 都被排除；两端 dtype 不一致由 C++ 入口拒绝。[state dtype 判别](../../../implementation/phase6/csrc/flash_kda.cpp#L118)

该 C++ guard 不自行查询 SM 数/L2；完整 Python auto 硬件 guard 与 build 约束是另两层。raw 或强制 V16 可以绕过 Python 的离线标定域，因此不要把 C++ guard 单独描述成实时 B300 拓扑验证。

## 2. 八个 keyblock、四个 slot 的不变量

`S_M_BLOCKS=D/16=8`。V16 对应 96 threads、一个 compute warp、每 warp 一个 value block。当前 CTA 的 state 物理覆盖完整 key128 × value16，全部八个 value CTA 合起来仍覆盖原完整 state。

进入第 m 次循环时，`slot=m%4` 必须持有同一 keyblock m 的：`k_restored` fragment、旧 BF16 state fragment、两个 gate 值。prologue 预装 keyblock 0–3。[Phase 6](../../../implementation/phase6/csrc/smxx/fwd_kernel2.cuh#L689)

| m | slot | 本轮消费并写回的 keyblock | slot 补入的未来 keyblock |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 4 |
| 1 | 1 | 1 | 5 |
| 2 | 2 | 2 | 6 |
| 3 | 3 | 3 | 7 |
| 4 | 0 | 4 | 无 |
| 5 | 1 | 5 | 无 |
| 6 | 2 | 6 | 无 |
| 7 | 3 | 7 | 无 |

每个 block 恰好读入一次、更新一次、写回一次。prologue 最大索引3，补入最大索引7，drain 不再预取，因此没有尾端读 keyblock8 的问题。`static_assert(1<=StatePrefetch<=D/16)` 保证 prologue 不越 key 维；算法并不要求8能被任意prefetch值整除，实际发布值4满足该界。

## 3. 读写顺序为何成立

**A fragment 生命周期。**本轮 `ring_A_kr[slot]` 在所有 bi 的同步 MMA 使用完后才被未来 A 覆盖（L725–733）。这是寄存器值依赖，不存在把未完成异步 MMA 的输入提前覆写。证明使用的是当前 SM80 同步 atom 语义。

**gate 配对。**本轮先把 `ring_g0/g1[slot]` 复制到局部 `g0/g1`（L721–722），再把环 slot 改成 m+4 的 gate（L735–736）。state FMA 使用的是局部旧 gate（L747–748），不读取刚更新的环 gate。`group_id` 取0–7，两个索引分别覆盖每个16-key block的前后8行；对应映射及FMA表达式相对P1未改。

**state 不被提前覆盖。**ring中保存本轮旧state，完成FMA与BF16舍入，再写回 keyblock m（L747–753）；之后才加载 keyblock m+4 到同一个寄存器slot（L755–757）。m+4尚未被本轮任何先前迭代更新，与刚写回m是不同的逻辑key区间。这个结论依赖既有 `StateSmemLayout`/`TransposedStateSmemLayout` 为正确互转且不同tile不别名；patch没有改变这些布局。

V16只有一个compute warp，且本warp负责的value16全程独占；跨CTA的value片也不重叠。故这里无新增跨warp state所有权交接。`ldmatrix`/`stmatrix`仍使用原同步warp指令，prefetch仅提前同一线程/warp将来要用的数据。

**数值操作组织。**每个state元素仍执行同一MMA结果与同一旧BF16 state、FP32 gate相乘相加，最后同一个BF16转换。各keyblock不是对同一个accumulator做reduction，因此调整block数据加载时刻没有改变数学归约顺序。源代码级结论仍需实际编译产物验证：寄存器压力、编译器重排/收缩、spill或不同工具链不能由C++文本证明逐位一致。

## 4. 跨 token tile 与状态入口/出口

ring数组与prologue均在当前token tile的循环体内。下一tile会重新装满四个slot，不读取上一tile留在寄存器里的block4–7。[tile循环与输入等待](../../../implementation/phase6/csrc/smxx/fwd_kernel2.cuh#L458)

每tile开始先等待该load stage；Phase6结束后保留compute barrier、async shared fence、output commit，再释放load stage（L762–769）。因此本tile的 `k_restored/g_total` 在全部环消费结束前不会被LOAD warp复用该stage覆盖。旧state的共享内存写回亦在下一tile读取之前经过既有同步。prefetch depth=4是片上keyblock窗口，**不是四个token tile的异步窗口**；不应与input stages=3混淆。

| HasStateIn / HasStateOut | Prefetch4 是否可能启用 | 需要独立验证的内容 |
|---|---|---|
| 无 / 无 | 是 | state零初始化，dummy state descriptor未实际使用，所有输出一致 |
| BF16 / 无 | 是 | 非零初态完整加载，不能由零初态测试替代 |
| 无 / BF16 | 是 | 零初始化及最终state输出；覆盖最后一次Phase6 |
| BF16 / BF16 | 是 | 非零初态、全部输出、最终state；最完整的数据依赖对照 |
| 任意一端 FP32 | 否 | 应确认实际退回P1；不能把P4 BF16证据写成FP32验收 |

初态TMA load/等待、无初态的zero-fill、最终BF16 state TMA store及FP32临时缓冲协议均沿用基线。该审查不扩大原有alias、empty-sequence或最终异步store的契约；没有在该patch中找到这些协议的新改动。

## 5. gscale 与尾块

`gate_scale` 仍只传给原K1；K1计算 `g_total` 后用原 `exp2` 路径恢复到FP32乘法gate，K2读取同一workspace。P4没有新增exp、gate缩放或转换，仅预取原有FP32 gate。其数值域、极端门控的FTZ/overflow行为仍继承原实现。

门控测试应让gate随key维和token tile变化。若每个keyblock的gate相同，即使环slot与gate错配，bitwise也可能恰好通过；`lower_bound=0`使衰减退化，更不能单独代表门控正确性。至少保留普通随机门控、长记忆接近1、强衰减、非零初态这些互补输入。

guard接受2048–8192中的**每一个整数T**，并没有要求T是16的倍数。D128的八个keyblock不受token尾块影响；但末tile的padding/`k_restored/U/g_total`正确性仍须由K1和尾部store保证。建议至少覆盖2049、4095、4097、8191，以及标定/留出点2048/3072/4096/6144/8192。packed N1还需合法 `[0,T]` 对照；仅numel=2不证明其值覆盖完整T，这一输入校验前提继承原接口。

## 6. 不能由漂亮的 bitwise 样本覆盖的发布义务

1. **同时比较out与final_state。**末tile的输出在Phase4/5已产生，最后一次Phase6才更新final state。若只比较out，最后一次Phase6全部错误也可能完全不可见。
2. **覆盖上表四个BF16有/无state模板组合。**一次“两端都有”的测试不能证明另一实例的zero-fill/dummy descriptor分支可用。FP32需验证未误进新实例。
3. **跨pipeline wrap及尾块。**长序列覆盖输入3-stage/output2-stage多次环绕；另用非16对齐T覆盖尾部，而非把所有长样本都取整齐长度。
4. **确认真正运行了P4。**Python仍仅报告V16，不暴露StatePrefetch。记录opt-in构建参数、实际加载.so的hash/路径、必要时kernel实例符号；源码hash或模块名不足以证明命中新实例。
5. **P1 baseline不能被新二进制污染。**在opt-in二进制的guard域，强制V16也会执行P4。原V16基线需独立P1构建/二进制；强制V128仍是原V128/P1，但不能代替对“比旧V16快”的验证。
6. **旧policy诊断不等于新kernel资源标定。**Python未改，V16 `predicted_ms`与register/resource表仍描述P1。P4增加fragment/gate寄存器驻留，需保留ptxas资源/spill记录及实际净收益；不要把旧预测当新性能证明。state容量、共享内存布局和请求次数并未因此降低。
7. **性能与正确性分开。**ring不变量支持所有合法D128 keyblocks的消费语义；它不能推出2048–8192全区间都有净收益，也不能证明持续竞争下的19.4%。发布guard内未测点仍是外推。

上述是明确的验证义务，不是声称发现了新的错误，也不是要求改写kernel。若release矩阵满足这些条件，本次静态审查支持将P4作为编译期opt-in、在选中V16后的受限细化；不支持将它宣传为任意shape或任意state dtype的通用优化。

## 审查对象身份

- patch SHA-256：`246a6aa347a1779215d5cdb72d84f2be18357ea0b77be8df987c8c46118b9a96`
- candidate `fwd_kernel2.cuh`：`78e21ed05cfeada41d04b0018232ea98a3478aa26b7bc7f3a38a4d5b29877546`
- candidate `fwd_launch.cu`：`0c03beaa93c072418bd89646a245b7b686f77200e89460af6e997e3f87171051`

路径 `/private/tmp/kda-release-prefetch4.DStLoA` 为本次读取的候选checkout；长期可复现入口应使用已保存的patch与固定基线，而非依赖临时目录继续存在。
