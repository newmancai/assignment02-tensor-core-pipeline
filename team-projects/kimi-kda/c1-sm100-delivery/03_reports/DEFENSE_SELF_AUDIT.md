# SM100 迁移主线答辩自审

更新日期：2026-09-10。

## 结论先行

论文前提在可辩护的表述下成立：官方 FlashKDA 在 B300 上的递推核心仍使用
HMMA，完整、协同设计的 SM100 路径在已测 H12/H96 public-full profiles 上值得；
但“把 HMMA 指令机械替换成 tcgen05 就会更快”不成立。正式隔离进程 CUPTI
结果中，CAKE 相对官方在 H12 六形状几何平均 2.4823x，在 H96 三形状几何平均
2.2532x，九个形状均为正收益。直接 V128 指令替换为 0.9197x，H3 局部代数重排
为 H12 0.9055x、H96 0.7581x。所以主线答案是：**值得做架构级迁移，不值得把
“迁移”简化成 opcode substitution。**

状态词含义：`已回答` 表示现有证据直接支持；`有边界` 表示答案成立但外推受限；
`已实验关闭` 表示曾是开放问题，现已用实验决定停止或推进；`后续挑战` 表示不影响
当前主线闭合，但可成为论文下一阶段。

## 一、问题和前提

### Q1. 我们究竟在回答什么？【已回答】

回答官方 FlashKDA 的 SM80-MMA 实现迁移到 SM100 是否值得。对象是同一 recurrent
KDA prefill 语义和 public output/final-state contract；不是一般 GEMM，也不是只比较
某一条指令峰值。

### Q2. “官方使用 SM80 MMA”是否只是源码命名？【已回答】

不是。归档 SASS receipt 在递推路径中计到 3,640 条静态 HMMA，TCGEN/UTCMMA 为
0。源码和机器码证据相互印证。

### Q3. 官方实现是否完全没有利用新架构能力？【有边界】

不能这样说。官方路径已有 TMA 等现代数据搬运机制；“SM80”准确指其 MMA atom/
递推矩阵乘部分，不应扩写成整个 kernel 都停留在 Ampere。

### Q4. SM80 指令在 B300 上是否天然低效？【已回答】

不是。B300 能高效执行 HMMA；性能取决于 shape、并行度、布局转换、流水与资源占用。
直接 V128 tcgen05 probe 反而只有 0.9197x，说明 ISA 更新不是自动收益。

### Q5. 论文前提是否被直接替换失败推翻？【已实验关闭】

没有。直接替换只否定一个具体路径。CAKE 作为技术上显著不同的完整 SM100 路径，
在九个 profiles 上全部胜出，反驳了“SM100 迁移整体没有价值”。

### Q6. 最准确的一句话结论是什么？【已回答】

在一张 B300/SM103a 上，SM100 全栈协同迁移对已测 H12/H96 KDA profiles 值得，
而 instruction-only 或未经全成本验证的局部迁移不值得默认采用。

## 二、基线、硬件与研究范围

### Q7. 官方基线是否可复现？【已回答】

官方 commit、CUTLASS identity、构建与历史 benchmark/SASS/NCU 已归档；本轮隔离
计时还记录官方扩展 SHA-256。正式 H3 和 CAKE 对比都使用同一官方二进制 identity。

### Q8. 为什么主要在 B300 而不是 5090 上回答？【已回答】

目标是 SM100/Blackwell datacenter 路径；实际设备为 NVIDIA B300 SXM6 AC，compute
capability 10.3，构建目标 SM103a。旧 5090/SM120 环境只产生环境或汇编尝试，不能
作为 B300 性能结论。

### Q9. 为什么 H12 是重要 profile？【已回答】

Kimi-K3 TP8 对应 12 个 local heads。官方递推由此只有 12 CTA，而 B300 有 148 SM；
历史 NCU 中 SM/DRAM throughput 仅 2.64%/1.24%，低并行度是明确的结构性动机。

### Q10. 只测 H12 是否足以代表 KDA？【已实验关闭】

不足。因此本轮把正式矩阵扩到 H96。CAKE 在 H96 三形状仍为 1.8819--2.6222x，
几何平均 2.2532x；H3 在 H96 则降到 0.7581x。head-count 外推问题已从“完全未测”
收窄为 H12/H96 两族，但不覆盖所有 H。

### Q11. 为什么还保留 fixed、packed、mixed、uniform 和 tail？【已回答】

它们改变有效序列并行度、最长递推链、tail 浪费和 dispatch route。H3 的 fixed T8192
小胜但 packed/uniform 大败，证明单一长序列不能替代形状矩阵。

### Q12. 这是 kernel microbenchmark 还是完整模型结果？【有边界】

是 public KDA forward kernel scope，包含每次调用所需的转换、workspace 流量及公共
state copy-back，但不包含整个模型、通信、调度和服务排队。不能直接写成模型 token/s
或端到端延迟提升。

### Q13. 是否覆盖 B200、另一张 B300 或跨工具链复现？【后续挑战】

没有。本地主结论只属于一张 B300/SM103a 和所记录工具链。外部 B200 结果只能作
旁证，不能与本地 B300 数字混算。

## 三、语义与数值正确性

### Q14. 不同实现比较的是同一个 KDA 问题吗？【已回答】

是。输入 q/k/v/g/beta、A_log、dt_bias、初始 state、packed 边界及 output/final-state
语义保持一致。允许内部融合、布局、分块和舍入树不同，但必须明确兼容级别。

### Q15. 为什么既检查 output 又检查 final state？【已回答】

KDA 是递推算子；只对当前 output 正确不保证下一次调用正确。所有正式 correctness
gate 都把 final state 纳入，连续调用 state handoff 也在 H3 oracle 中检查。

### Q16. H3 的数学改写是什么，实数域上成立吗？【已回答】

令 `R=INV*diag(beta)`，官方为 `U=R*(V-Kd*S)`；H3 预计算 `P=R*V`、
`W=R*Kd`，再令 `U=P-W*S`。由分配律，实数域上严格等价。

### Q17. 数学等价是否意味着 bitwise 等价？【已回答】

不意味着。H3 改变 BF16 舍入 DAG，因此明确标为
`algorithm_equivalent_not_official_bitwise`。CPU 80-case、32-scenario、18-case
input-oracle 与 B300 8-case gate 都通过，但 H3 与官方通常不 BF16-equal。

### Q18. H3 的误差阈值是否看完结果才放宽？【已回答】

没有。single-chunk、multi-chunk 与 input-level gate 都在执行前冻结，并同时与同一个
FP64/FLA-style reference 比较官方和 H3。文档没有把探索用阈值冒充生产容差。

### Q19. H3 CPU oracle 能否替代 GPU correctness？【已回答】

不能。CPU 近似不复现 CUDA 的 `tanh.approx`、`ex2.approx.ftz` 和 warp reduction
顺序，所以只授权实现；随后在 B300 上对 fixed、tail、multichunk、packed 8 case
检查 output 和 final state，8/8 通过。

### Q20. CAKE 的正确性与性能是否来自同一份证据？【有边界】

不是同一份 artifact。correctness 来自既有 public paired 证据和 Stage12 H96/H64 peer
checks；新的隔离进程 artifact 专门修正性能测量隔离。答辩时应分别引用，不能说每个
计时 worker 又跑了一次独立 oracle。

### Q21. packed sequence 是否可能串 state？【已回答】

语义 contract 明确保留 packed boundaries；H3 input oracle 覆盖多组非等长 packed
序列，B300 gate 覆盖 `[4,17,33]` 等 case。当前证据支持已测样例，不宣称形式化证明
任意长度输入。

### Q22. 是否覆盖强衰减、弱衰减、抵消和非零初态？【已回答】

H3 recurrence screen 覆盖 strong/mixed/weak decay、first-chunk cancellation、非零
state、2/4/16/64 chunks 及 two-call handoff；这是针对数值风险的主动反例搜索。

## 四、性能测量是否公平

### Q23. headline 的计时边界是什么？【已回答】

public-full：预分配和 JIT 在计时外；每次 forward 必需的 beta/layout/workspace 操作和
同 stream state copy-back 在计时内。K1/K2 单独时间只允许作诊断，不能替代完整跨度。

### Q24. 初始 state 是否对每次调用相同？【已回答】

使用预初始化、轮转的 state slots；变体接收相同输入和非零初态，避免递推 state 在
重复测量中逐次漂移。

### Q25. cache 和测量顺序是否控制？【已回答】

正式结果采用 cold-L2 CUPTI、20 warmup、每 block 100 samples、4 个 balanced-order
blocks。每个 case 保存原始 samples、block medians 和 bootstrap interval。

### Q26. 为什么不能继续使用最初的同进程对比？【已实验关闭】

`nm -D` 发现官方/H3 扩展导出同名 weak `launch_fwd<...>` 和 global `fwd`；反转加载
顺序会显著改变绝对时间，即使 `sys.getdlopenflags()==RTLD_LOCAL`。因此旧 CUDA-event
和同进程 CUPTI 文件仅作诊断，正式结果改为每个 worker 只加载一个实现。

### Q27. 隔离进程是否又引入不同输入或测量配置？【已回答】

没有。父 runner 固定同一 case、seed、shape 和计时 policy，以平衡顺序分别启动
official/candidate worker；进程隔离只移除 ELF/CUDA symbol interposition。

### Q28. 为什么用几何平均而不是挑最好 case？【已回答】

比较量是速度比，几何平均适合汇总乘性 ratio，且避免大绝对延迟 case 独占汇总。
同时保留每个形状的原始值和区间，不用 geomean 掩盖 regression。

### Q29. CAKE 的收益是否统计稳定？【已回答】

九个 per-shape bootstrap 95% interval 均完全高于 1.0；最小点估计是 H96 uniform 的
1.8819x，其区间约 [1.8812,1.8827]。这远离测量噪声边界。

### Q30. H3 fixed T8192 的 1.0219x 是否足以保留？【已实验关闭】

不足。预注册 practical gate 要求 H12 geomean >1.05、无 case 超过 2% regression。
实际 H12 geomean 0.9055x，最差 0.7587x，H96 0.7581x；一个局部 winner 不能推翻
跨形状 stop rule。

### Q31. 是否把 workspace allocation 时间算进去了？【有边界】

复用 allocation 本身在计时外，符合稳定部署；但每次调用实际 workspace 写读在计时
内。H3 每 chunk/head allocation ledger 从 13,824 增至 17,408 bytes（+25.9%）。
logical touched bytes 不是实测 HBM bytes，不能拿它冒充 DRAM counter。

### Q32. 是否可能是 GPU clock、排队或跨 job 差异？【有边界】

正式 comparison 不跨 job 合并绝对时间；每个 case 的两变体在同一 allocation 的平衡
blocks 内完成。单卡单次实验仍不等于跨节点复现，后者留作外部有效性挑战。

## 五、多条迁移路径分别告诉了我们什么

### Q33. 直接 V128 tcgen05 路径回答了什么？【已实验关闭】

在 B300、H12、CHUNK16、Phase-6、V128、grid12、inner64、L0 的精确适用键上，
instruction-only 路径为 0.919742x。它是 scoped failure card，不是禁止所有 tcgen05。

### Q34. V16 probe 为什么重要？【已回答】

它区分计算核心与表示成本：preferred-layout L0 为 1.615x，而含 scalar
rematerialization 的 L1 为 0.779x。说明薄 N、更多 CTA 有潜力，但布局物化足以吃掉
全部收益。

### Q35. V16 是否已经证明 production ValueSlice tcgen 路径可用？【有边界】

没有。它是 mechanism probe，未实现完整 production ValueSlice pipeline。现有结论只
支持“需要消除 transfer”，不支持宣布一个可部署 tcgen ValueSlice winner。

### Q36. H3 为什么值得实现，即使最后失败？【已回答】

它把“只换指令”之外的算法/数据流假设落到真实 K1/K2、workspace、SM103a 编译和
B300 correctness/full-cost measurement，验证了 Typed IR 能表达并证伪实质不同路径。

### Q37. H3 为什么失败？【有边界】

已测事实是完整路径跨形状变慢；+25.9% workspace、K1 新增 W/P 工作、packed 和高 H
放大成本是与结果一致的解释。因为没有正式 K1/K2 phase attribution，不能把任一项写
成单独的已证实主因。

### Q38. 为什么不继续把 H3 改成 tcgen05 看看？【已实验关闭】

预注册规则要求 matched-HMMA parent 先证明 full-path usefulness。它明显失败，继续
只会把一个负的数据流父节点与新的 ISA 变量混在一起，既浪费 GPU 预算又破坏归因。

### Q39. P3/P4 cross-phase TMEM 为什么仍未实现？【有边界】

Typed design 已表达 logical/physical transpose、TMEM lifetime 和复用，但 repo 中没有
可直接复用的 `tcgen05.st` packed-BF16 carrier lane mapping。未做非对称数据的编译/GPU
验证前，不能称为完成路径；它也不是主线结论所必需的缺口。

### Q40. CAKE 为什么算“完整 SM100 路径”？【已回答】

实际生成源码包含 `tcgen05.mma`、TMEM alloc/load/store，并协同改变 fusion/split、
work partition、warp roles、barrier topology、pipeline、layout 和 shape dispatch；测量调用
完整 public recurrent KDA 路径，而非单条 MMA microbenchmark。

### Q41. CAKE 的 2.482x/2.253x 能否全部归因于 tcgen05？【已回答】

不能。这些是 full-stack treatment effects。要识别 tcgen05 单因素，需要结构匹配的
HMMA counterfactual，并控制 fusion、roles、layout、resources 与 pipeline；当前没有这样
的完整消融。

### Q42. CAKE 是不是最终最强部署路径？【有边界】

不是所有形状都停在原始 CAKE。既有 Stage3 guarded evolution 在 H96/H64 六形状中对
五个激活、一个回退 CAKE，部署 geomean 比 CAKE 再高 1.142x。它证明实际建议应是
“CAKE/演化 guarded route”，但该结果不属于 tcgen05 因果消融，也未覆盖本轮 H12 矩阵。

## 六、Typed IR、可复核性和复现

### Q43. 多路径 Typed IR 是否只是几段说明文字？【已回答】

不是。`paths/index.json` 逐路径绑定 parent、implementation、IR、correctness、measurement
scope 和 outcome；direct probe 使用 migration v2；H3 有数学、workspace、物理实现和
证据链；CAKE 有 source-grounded partial reconstruction。

### Q44. 为什么 CAKE IR 只称 partial？【已回答】

它已记录语义 contract、真实 dispatch routes、tcgen/TMEM family，以及 fused-m128 的
threads、smem、TMEM columns、warp roles、pipelines 和 barrier counts；但尚缺全部 math
sites 的 MNK、每 route descriptor layout、prepare-chain-m64 完整 barrier graph 和 per-route
SASS opcode counts。明确缺口比伪造“完整 IR”更可辩护。
`cake_source_binary_manifest.json` 另行冻结实际 dispatch/JIT 源码、七个观测到的
SM103a cached route binaries、正式 measurement 和 job log 的哈希；由于计时 artifact
本身没有逐 case 回写 loaded-binary hash，这一点仍被明确列为复现边界。

### Q45. H3 的源码能否从证据恢复？【已回答】

`h3_source_manifest.json` 对 8 个修改文件记录 official/H3 SHA-256，并保存远端扩展
binary hash；build log、correctness 和正式 performance artifact 均已归档。它足以验证
本次被测对象 identity，虽不等于把远端完整 toolchain 打包成可移植容器。

### Q46. 失败路径是否会被 verifier 永久禁止？【已回答】

不会。性能 failure card 是带 applicability key 的 scoped prior，不能升级成语义规则。
只有 hardware/toolchain、layout、grid 或跨阶段 dataflow 等关键条件改变，才有理由重开。

### Q47. 旧证据和新证据冲突时如何处理？【已回答】

不删除旧 artifact；在 IR/index/doc 中标为 historical diagnostic 或 superseded，并把正式
结论指向 isolated-process CUPTI。这样保留研究轨迹，也避免 cherry-pick。

### Q48. 现有 schema 是否覆盖一切新 idea？【有边界】

不覆盖，也不需要假装覆盖。Typed IR 是可扩展的证据载体；表达不了的 carrier、route
或数值契约应登记为 capability gap，而不是把 idea 判错或用自由文本冒充已验证字段。

## 七、因果、外推与答辩中的攻击面

### Q49. 我们能否说“tcgen05 比 HMMA 快 2.48x”？【已回答】

不能。正确说法是“包含 tcgen05/TMEM 的 CAKE 全栈 SM100 路径，相对官方 HMMA
完整路径快 2.48x(H12)/2.25x(H96)”。direct swap 负例正说明两句话不是一回事。

### Q50. 官方低利用率是否证明了 CAKE 的因果机制？【有边界】

低 CTA 和 NCU throughput 证明官方存在结构性 underfill，ValueSlice 的历史收益也与此
一致；但 CAKE 同时改变多因素，不能仅凭相关性把其全部收益归因于 occupancy。

### Q51. 为什么不把 optimized HMMA 作为每条路径的强制基线？【已回答】

对可部署 winner 应比较最强 incumbent；但 H3 相对最基础、scope-identical 官方路径已
大幅失败，按预注册 stop rule 无需再花预算证明它也输给更强路径。CAKE 后续本身已有
guarded evolution 作为更强部署对照。

### Q52. shape dispatch 是否意味着 cherry-pick？【已回答】

若只报 winner 会是。当前保留所有九个 CAKE/official case、所有九个 H3 case和一个
Stage3 fallback；guard 必须对不合格 shape 回退，不能把 fallback 静默计为优化实现。

### Q53. 是否证明了“所有 SM100 迁移都值得”？【已回答】

没有，也不需要。研究结论是条件化决策：全栈协同路径值得，direct/local 路径需逐项
验证。负路径和正路径共同构成结论，而非只展示最快 kernel。

### Q54. 是否证明了对所有输入数值都正确？【有边界】

没有形式化证明。已有官方/FLA-style reference、随机与对抗场景、packed/tail/state
handoff 和 GPU checks，足以支持已测 contract；更广 dtype、极端动态范围和长期模型
质量需单列验证。

### Q55. 是否证明了实际服务吞吐提升？【后续挑战】

没有。多 stream、多请求竞争、并发调度、缓存压力、模型其他层和通信可能改变收益。
这些正是主线之后的 challenge，不能反向要求 kernel 主线先回答所有系统问题。

### Q56. 最大的内部有效性风险是什么，是否处理了？【已实验关闭】

最大的新发现是同进程扩展符号互插。已经用 reverse-load diagnostic 暴露，并用 one-
implementation-per-process 的正式 CUPTI 设计处理；新结论不依赖旧同进程绝对值。

### Q57. 最大的外部有效性风险是什么？【有边界】

单 B300、有限 H12/H96 shapes、BF16 public KDA kernel scope。它限制外推范围，但不
否定“这些已测条件下值得”的主线回答。

### Q58. 如果答辩老师说“你们只是证明 CAKE 快”，如何回答？【已回答】

回答三层证据：首先 SASS/NCU 确认迁移动机；其次 direct V128、V16 layout crossover、
H3 full-cost 负例解释为什么局部迁移不够；最后 CAKE full-stack 正例证明存在值得的
SM100 组织。贡献是有边界的迁移判断和多路径证伪链，不是只报一个最快数字。

### Q59. 如果追问“为什么论文标题还能成立”？【已回答】

因为标题问“是否值得”，并未预设“任何迁移都更快”。最强答案恰好是条件化的：存在
稳定且幅度大的完整迁移收益，同时存在可复现的直接/局部失败，说明问题值得研究且
需要 Typed IR 和 full-cost evidence 来做决策。

### Q60. 现在主线还有哪项未决会阻止进入论文优化挑战？【已回答】

没有阻塞项。仍可深化的 tcgen 单因素因果、完整 CAKE per-route IR、跨 B300/B200
复现、长期数值与并发都是真实开放问题，但它们分别属于机制消融、表示完善、外部
有效性和系统挑战，不改变当前“完整迁移值得、裸替换不保证值得”的主线结论。

## 八、公平性追问：到底在比较 ISA 还是完整实现

### Q61. V128 HMMA/tcgen05 的 0.9197x 是否是“双方都充分优化”的公平对比？【有边界】

不是。它对一个窄问题较公平：相同 logical Phase-6 contraction、固定 V128/grid12/
inner64，并在 preferred-layout L0 下比较 instruction-family replacement，且将 tcgen
protocol 以 64 次复用摊销。它能否定“换 ISA 自动更快”，但 HMMA/tcgen05 两边都还有
tile、warp roles、residency、pipeline 和 launch-topology 优化空间，不能据此排名两种
ISA 的全局上限。

### Q62. CAKE/官方 2.482x、2.253x 的比较公平在哪里、不公平在哪里？【已回答】

对工程决策公平：两边解决同一 KDA、使用相同 shapes/seeds、public output/state contract、
完整每调用成本、隔离进程和同一 CUPTI policy。因此它能回答“用完整 SM100 路径替换
当前官方实现是否值得”。对机制归因不公平：CAKE 同时改变 fusion、work partition、
warp roles、layout、TMEM residency、barriers、pipeline 和 dispatch，不能把总差值叫作
tcgen05 的 ISA speedup。

### Q63. CAKE 是在官方 FlashKDA kernel 上继续打补丁优化吗？【已回答】

不宜这样描述。CAKE 是面向同一 KDA public semantics 的生成式/重组织 SM100a
full-stack route，物理 kernel boundary 和执行组织与 MoonshotAI 官方 HMMA 路径有实质
差异；它不是“只把官方 kernel 某几行改成 tcgen05”的增量补丁。之后 Stage3 的
`evolution` 才是在 CAKE checkout/route 基础上做进一步 schedule、dispatch 与 guarded
activation 优化，并对不合格 shape 回退 CAKE。

### Q64. 真正回答“tcgen05 相对最优 HMMA 的边际价值”还缺什么？【后续挑战】

需要结构匹配的 causal ladder，而不是只比较两个端点：

1. 官方 HMMA；
2. ValueSlice + phase-prefetch 等 optimized HMMA；
3. 尽可能保留 CAKE fusion/layout/roles/pipeline、仅将 tensor-core implementation 换成
   HMMA 的 matched counterfactual；
4. 原 CAKE tcgen05/TMEM route；
5. CAKE 上的 guarded evolution。

应分别报告 `official -> optimized HMMA` 的软件组织收益、`matched HMMA -> tcgen05`
的近似 ISA 边际收益、`official -> CAKE` 的总迁移 treatment effect，以及
`CAKE -> evolution` 的后续优化收益。如果由于 TMEM/warp-specialization 与 tcgen05
不可分而无法构造完全匹配的 HMMA 路径，应把 estimand 明确改为“ISA+必要协议包”的
联合效应，而不是宣称纯 ISA 因果。

## 九、证据总表与答辩红线

| 命题 | 正式证据 | 可以说 | 不可以说 |
|---|---|---|---|
| 官方 recurrence 使用 HMMA | archived SASS/NCU receipt | 3,640 static HMMA，0 TCGEN | 整个官方 kernel 都是 SM80 技术 |
| direct V128 失败 | Stage11 failure card | 精确条件下 0.919742x | tcgen05 永远不值得 |
| V16 layout crossover | job 24111 evidence | L0 1.615x、L1 0.779x | production ValueSlice tcgen 已完成 |
| H3 数值成立 | CPU gates + B300 8/8 | algorithm-equivalent，GPU gate 通过 | official-bitwise |
| H3 性能失败 | `h3_b300_isolated_process_cupti.json` | H12 0.9055x、H96 0.7581x，stop | workspace 已被实测证明为唯一原因 |
| CAKE 完整路径胜出 | `b300_stage12_cake_official_isolated_pair.json` + correctness refs | H12 2.482x、H96 2.253x，九形状正收益 | tcgen05 单因素带来这些全部收益 |
| 最强实际路线 | Stage3 activation certificate | guarded evolution 比 CAKE 六形状 geo 1.142x | 该增益是 tcgen05 增益或已覆盖 H12 |

答辩红线：不混用 B200/B300；不把 microprobe 写成 full forward；不把 algorithm-
equivalent 写成 bitwise；不引用同进程 artifact 作为正式 magnitude；不把 full-stack
treatment effect 写成 tcgen05 main effect；不把单卡 kernel 结果外推为模型/服务结果。

## 十、进入下一阶段前的判断

主线已达到“可答辩闭合”而非“研究空间穷尽”：有官方基线、两类机制 probe、一个真实
实现但失败的算法/数据流路径、一个跨 H12/H96 成功的完整 SM100 路径、同 scope 的
正确性和隔离性能证据、路径级 Typed IR、失败复盘与清楚的外推边界。下一阶段可以把
注意力转到论文优化挑战；最有价值的方向是对 CAKE/evolution 做结构匹配消融、并发/
服务场景验证和更完整的 route-level IR，而不是重新打开已经失败且没有新假设的 H3。
