# 作业二：Tensor Core & Pipeline

## 基本信息

| 项目 | 内容 |
|---|---|
| 课程 | Weiming HPC Training Camp × LCPU AI Infra Seminars |
| 作业 | Assignment 02 |
| 成员 A | 【姓名】 |
| 成员 B | 【姓名】 |
| 成员 C | 【姓名】 |
| 完成日期 | 【日期】 |
| 代码目录 | `assignment02/` |

> 注意：M0–M6 在题面中属于非团队作业。如果课程要求独立提交，本模板只用于进度协调和互审，最终代码与报告应按课程要求独立完成。

## 分工与进度

| 范围 | 负责人 | 代码 | 判测 | 报告 | 实验数据 |
|---|---|---|---|---|---|
| M0 环境与峰值 | A | ☑ | ☑ | ☑ | ☑ |
| M1 fragment 与 `mma.sync` | A | ☑ | ☑ | ☑ | ☑ |
| M2 descriptor 与 swizzle | A | ☑ | ☑ | ☑ | 不适用（Host 判测） |
| M3 `tcgen05` | B | ☑ | ☑ | ☑ | ☑ |
| M4.1–M4.3 完整 GEMM | B | ☑ | ☑ | ☑ | ☑ |
| M4.5 thin GEMM | C | ☐ | ☐ | ☐ | ☐ |
| M5 低精度与 block scaling | C | ☐ | ☐ | ☐ | ☐ |
| M6 TileLang 对照 | A | ☑ | ☑ | ☑ | ☑ |

## 公共实验环境

| 项目 | 配置 |
|---|---|
| GPU 1 | NVIDIA B300 SXM6 AC，275040 MiB（约 270 GiB 可用） |
| GPU 2 | 【型号、显存容量】 |
| CUDA | 13.0 |
| Driver | 580.126.09 |
| NVCC | 13.0.88（`/usr/local/cuda-13.0/bin/nvcc`） |
| Nsight Compute | 【版本】 |
| 操作系统 | 【版本】 |
| 默认 ARCH | `100f` |
| 其他 ARCH | 【例如 `120a`】 |

性能实验前后均确认 GPU 没有其他计算进程：

```bash
nvidia-smi --query-compute-apps=pid,name --format=csv
```

## 公共参数与交接信息

该表由 A 填写，B、C 直接读取。

| 参数 | 数值 | 口径/来源 |
|---|---:|---|
| RTX 5090 BF16 峰值 | 209.5 TFLOPS | 官方 dense、boost、FP32 accumulate |
| B300 BF16 峰值 | 2250 TFLOPS | 官方 dense，每 GPU |
| B300 FP8 峰值 | 4500 TFLOPS | 官方 dense，每 GPU |
| B300 FP4 峰值 | 13500 TFLOPS | 官方 dense NVFP4，每 GPU |
| B300 显存带宽 | 8000 GB/s | 官方 up to，每 GPU |
| B300 BF16 机器平衡点 | 281.25 FLOP/byte | 2250 TFLOPS / 8000 GB/s |
| 2.2 descriptor 判测 | ☑ PASS / ☐ FAIL | 3/3 场景通过 |
| 2.3 swizzle 判测 | ☑ PASS / ☐ FAIL | 128B / 64B / 32B 通过 |

### M2 → M3 交接

```text
场景 1 LBO/SBO/layout：128 / 1024 / 0
场景 2 LBO/SBO/layout：0 / 1024 / 2
场景 3 LBO/SBO/layout：0 / 1024 / 2
descriptor 编码规则：(saddr>>4) | ((LBO>>4)<<16) | ((SBO>>4)<<32) | (1<<46) | (layout<<61)
128B swizzle 地址公式：chunk' = (colByte>>4) XOR (row&7)，offset=row*128+chunk'*16+(colByte&15)
```

# M0 环境与峰值（A）

## Prob 0.1 最小 Tensor Core 程序

### 实验 GPU 与命令

```text
GPU：NVIDIA B300 SXM6 AC（compute capability 10.3）
ARCH：100f
命令：make -B NVCC=/usr/local/cuda-13.0/bin/nvcc run/m0_env/01_first_mma
```

### 正常运行结果

```text
D[0][0]=2 D[0][7]=2 D[15][0]=2 D[15][7]=2
PASS
```

### ARCH 不匹配实验

```text
使用的 ARCH：120a（RTX 5090），在 B300 上运行
观察到的现象：编译成功，kernel 启动失败
关键报错：cudaErrorNoKernelImageForDevice: no kernel image is available for execution on the device
```

### 原因解释

`-gencode arch=compute_120a,code=sm_120a` 只把面向 sm_120a 的机器码
装进 fatbin。B300 属于 sm_10x family，无法执行 sm_120a SASS；fatbin 中又
没有可供 B300 JIT 的兼容 PTX，因此运行时找不到 kernel image。正确的
`sm_100f` family 机器码可在本机 B300（CC 10.3）上运行。

## Prob 0.2 理论峰值与机器平衡点

### 计算口径

- Dense / sparse：dense；sparse 数据只用于对照
- Base / boost：5090 用官方 boost；B300 同时给出实卡最大应用时钟推导与官方额定值
- FMA 是否按 2 FLOP：是
- 数据来源：NVIDIA RTX Blackwell whitepaper、CUTLASS SM100 文档、NVIDIA HGX/Blackwell Ultra 官方资料

### 推导过程

```text
RTX 5090 BF16 = 512 × 170 × 2.407 GHz = 209.505 TFLOPS
RTX 5090 FP8/FP4 位宽估算 = 419.011 / 838.021 TFLOPS
B300 BF16 = 8192 × 148 × 2.032 GHz = 2463.629 TFLOPS
B300 FP8/FP4 位宽估算 = 4927.259 / 9854.517 TFLOPS
RTX 5090 机器平衡点 = 209.505 TFLOPS / 1792 GB/s = 116.91 FLOP/byte
B300 实卡上界机器平衡点 = 2463.629 TFLOPS / 7672.32 GB/s = 321.11 FLOP/byte
B300 官方 roofline = 2250 TFLOPS / 8000 GB/s = 281.25 FLOP/byte
```

### 结果表

| 参数 | RTX 5090 | B300 |
|---|---:|---:|
| BF16 FLOP/cycle/SM | 512 | 8192 |
| BF16 峰值（TFLOPS） | 209.5 | 2463.63（实卡最大时钟上界） |
| FP8 峰值（TFLOPS） | 419.0（位宽估算） | 4927.26（位宽估算） |
| FP4 峰值（TFLOPS） | 838.0（位宽估算） | 9854.52（位宽估算） |
| 显存带宽（GB/s） | 1792 | 7672.32（实卡接口推导） |
| BF16 机器平衡点（FLOP/byte） | 116.91 | 321.11（实卡上界） |
| 官方值及口径差异 | BF16 209.5、FP8 419、FP4 1676；FP4 专用通路高于简单位宽估算 | BF16 2250、FP8 4500、FP4 13500、带宽 up to 8000；系统额定值不同于最大时钟上界 |

### 与单条 MMA 的比较

单条 MMA 的 3.2 FLOP/byte 仅为 RTX 5090 平衡点的 1/36.53、B300 官方
平衡点的 1/87.89。要接近 Tensor Core 峰值，必须让从 HBM 搬入的 tile 在
shared memory、寄存器和 TMEM 中被多次复用，并通过 swizzle、TMA 与多级
pipeline 降低冲突、减少搬运开销、重叠访存和计算。这里比较的是单条指令的
局部强度；完整 GEMM 可以依靠跨多条 MMA 的数据复用提高 kernel 级强度。

## Prob 0.3 概念判断

| 小题 | 判断 | 一句话理由 |
|---|---|---|
| (a) | 对 | 按 S016 的单条指令口径，分子是 `2MNK`，分母统计 A/B 读入和 D 写回字节。 |
| (b) | 对 | `mma.sync` 由整个 warp 协作完成，所有 lane 必须一致执行，否则行为未定义。 |
| (c) | 错 | 大 shape 可能提高局部计算强度，但会增加寄存器、供数和调度压力，并受 ISA 支持形状限制。 |
| (d) | 错 | 单条 MMA 的局部强度不等于 kernel 强度；完整 GEMM 可跨多条 MMA 复用 tile。 |

# M1 fragment 与 mma.sync（A）

## Prob 1.1 FP8 fragment 映射

### 映射公式

```text
令 group=lane>>2，tig=lane&3，寄存器内 byte 序号 i。
A.row = group + 8*((i>>2)&1)
A.col = 4*tig + (i&3) + 16*(i>>3)
B.k   = 4*tig + (i&3) + 16*(i>>2)
B.n   = group
```

### 判测结果

```text
PASS
```

### 附加问题

同一个 b32 中的 4 个 FP8 都沿 K 方向连续。A 的 K 是列方向；B 在
n-major staging 中 K 也是连续字节。因此可以把 4 个 FP8 看作两个相邻
b16，让 `ldmatrix` 直接装成 MMA 需要的 b32；A 用 `.x4`，B 用 `.x2`，
两者均不需要 `.trans`。

## Prob 1.2 fragment bug

### 修改前现象

```text
D 中错误的位置：只出现在第 8–15 行；实测 59/128 个元素不等。
错误值示例：D[8][0] got -1 / want -11；D[8][1] got -5 / want 5。
与正确部分的关系：错误版本的下半输出逐元素复制上半 0–7 行的输出；
其中 5 个位置因随机整数数据数值偶合而恰好等于参考值。
```

### 修改内容

修正 A fragment 的 `a2/a3/a6/a7`：它们的 row 从 `group` 改为
`group+8`，K 坐标保持不变。

### 根因解释

`a0/a1/a4/a5` 属于输出上半行，原本正确；表示下半行的
`a2/a3/a6/a7` 却重复读取了上半行 A，因此 Tensor Core 给下半 D 计算的
实际也是上半 A×B，恰好产生“下半复制上半”的规律。

### 修复后判测

```text
PASS
```

## Prob 1.3 单 tile FP8 MMA

### 实现说明

- MMA shape：`m16n8k32.row.col`
- 输入类型：A/B 均为 `e4m3`
- 累加类型：f32
- fragment 装载方法：按 1.1 公式逐 byte 手工装载并打包为 b32，不使用 `ldmatrix`
- seed 参数处理：命令行读取 unsigned seed，使用 `std::mt19937` 生成可被 E4M3 精确表示的小整数

### 五个 seed 判测

```text
PASS seed=1
PASS seed=7
PASS seed=42
PASS seed=1234
PASS seed=99999
JUDGE: PASS
```

## Prob 1.4 ldmatrix

### 两条装载路径

| 路径 | 正确性 | 装载指令数 | 地址计算指令数 |
|---|---|---:|---:|
| 手工装载 | ☑ PASS | 24×`ld.shared.u8` | 10 条索引/地址算术；另有 36 条移位/拼接 |
| `ldmatrix` | ☑ PASS | 1×`.x4` + 1×`.x2` | 10 条索引/地址算术，无手工拼接 |

### 分析

1. `ldmatrix` 省掉了逐 byte load、跨 lane 的 fragment 收集以及把 4 个
   FP8 移位/拼成 b32 的工作；硬件按矩阵布局一次直接写入目标寄存器。
2. 普通 `ld.shared` 是每 lane 独立的标量装载，不知道 MMA fragment 的
   warp 级分布，因此软件必须自己算每个 lane 的地址并完成打包。

## Prob 1.5 stride 与 bank conflict

### 预测与实测

| 行跨度 | 预测 wavefront | 比例 | 实测 wavefront | 实测 conflict | 平均 cycle |
|---|---:|---:|---:|---:|---:|
| 32 B | 2 | 1× | 16384 | 8192 | 9.72 |
| 64 B | 4 | 2× | 32768 | 24576 | 10.75 |
| 128 B | 8 | 4× | 65536 | 57344 | 16.06 |
| 128+16 B | 1 | 0.5× | 8192 | 0 | 9.23 |

### 结果解释

128 B stride 相对 32 B 使 wavefront 增加到 4 倍，但耗时只从 9.72
增加到 16.06 cycle（约 1.65 倍）。一个 block 有 8 个活跃 warp，某个
warp 因 bank conflict 被拆成多个 wavefront 时，LSU 可以交错服务其他 warp；
同时循环还包含发射、依赖与指令流水开销，所以总耗时不会按单个访问的
wavefront 数线性放大。144 B padding 打散了 bank 映射，实测 conflict 为 0。

# M2 descriptor 与 swizzle（A）

## Prob 2.1 异步排序

### 正确顺序

```text
st.shared → fence.proxy.async → wgmma.fence → wgmma.mma_async →
wgmma.commit_group → wgmma.wait_group
```

### 每一步的作用

| 操作 | 避免的乱序 |
|---|---|
| `st.shared` | 先由 generic proxy 写入 wgmma 将要读取的 shared-memory 数据。 |
| `fence.proxy.async` | 建立 generic proxy 写与 async proxy 读之间的可见性。 |
| `wgmma.fence` | 将先前寄存器访问排在后续异步 MMA 之前，防止累加器/输入寄存器乱序。 |
| `wgmma.mma_async` | 发射异步矩阵乘，结果尚未保证完成。 |
| `wgmma.commit_group` | 把已发射的 MMA 组成一个等待组；本身不等待执行完成。 |
| `wgmma.wait_group` | 等待规定数量的组完成，之后才安全读取/复用累加器。 |

### 判断题

| 小题 | 判断 | 理由 |
|---|---|---|
| 1 | 错 | 它解决 generic/async proxy 的跨代理可见性，TMA、tcgen05 等跨 proxy 场景也需要。 |
| 2 | 错 | `commit_group` 只划分异步操作组，不阻塞等待；等待由 `wait_group` 完成。 |
| 3 | 对 | `st.shared` 经 generic proxy，wgmma 经 async proxy；缺少 proxy fence 可能看到旧值。 |

## Prob 2.2 SM100 descriptor

### 位域编码

```text
start_address：`(saddr >> 4) & 0x3fff`，放入 bit [0,14)
LBO：`((lbo >> 4) & 0x3fff) << 16`
SBO：`((sbo >> 4) & 0x3fff) << 32`
version：`1ull << 46`
layout_type：`(layout & 7ull) << 61`
```

### 三个场景

| 场景 | LBO | SBO | layout | 判测 |
|---|---:|---:|---:|---|
| K-major，无 swizzle | 128 B | 1024 B | 0 | ☑ PASS |
| K-major，128B | 0 B | 1024 B | 2 | ☑ PASS |
| MN-major，128B | 0 B | 1024 B | 2 | ☑ PASS |

### 附加问题

区别体现在 descriptor 之外的数据 staging 与坐标解释：哪一个逻辑维度
连续、逻辑 `(row,col)` 如何写入 swizzled shared memory 由生产者决定。
这两组参数的 64 位 descriptor 数值相同，只说明消费者看到的物理 core-matrix
步长和 swizzle 类型相同，不说明上层矩阵的 K/MN 语义相同。

## Prob 2.3 swizzle

### 地址映射公式

```text
令 chunk=colByte>>4，inner=colByte&15。
128B：offset=row*rowBytes+((chunk XOR (row&7))<<4)+inner
64B： offset=row*rowBytes+((chunk XOR (row&3))<<4)+inner
32B： offset=row*rowBytes+((chunk XOR (row&1))<<4)+inner
```

### 判测结果

```text
128B PASS
 64B PASS
 32B PASS
```

### 双射与 bank conflict 说明

对固定 row，`chunk XOR 常数` 是自身可逆的置换，因此不会产生地址碰撞，
整个映射保持双射。row 的低 3/2/1 bit 分别参与 128/64/32B 模式的 XOR，
让相邻行原本落到同一 chunk/bank 的列访问分散到不同物理 chunk。

# M3 SM100 tcgen05（B）

## Prob 3.1 概念判断

| 小题 | 判断 | 理由/计算 |
|---|---|---|
| (a) | 对 | TMEM 地址高 16 bit 表示 lane 偏移；各 warp 可用 `tcgen05.ld` 读取对应的 32 条 lane。 |
| (b) | 对 | MMA 由 elected lane 发射，硬件异步执行，累加结果写入 TMEM。 |
| (c) | 错 | TMEM 不能直接作为 TMA 的 global 写回源；需 `tcgen05.ld` 到寄存器后再写回。 |
| (d) | 对 | TMEM 总容量 `128×512×4=256 KiB`；m128n256 FP32 accumulator 为 `128×256×4=128 KiB`，恰为一半。 |
| (e) | 错 | `tcgen05.commit` 不等于阻塞等待；必须等待其 mbarrier 完成后才能安全读取 TMEM。 |

## Prob 3.2 tcgen05 单 tile GEMM

### 实现流程

```text
global → smem → tcgen05.mma → TMEM → tcgen05.ld → global
```

kernel 为 A/B 建立 128B swizzle shared-memory descriptor，初始化 mbarrier，
分配 64 个 TMEM column。K=64 被拆成四个 k16 MMA：elected lane 发射
`tcgen05.mma`，随后以 `tcgen05.commit` 把完成事件关联到 barrier。等待完成后，
四个 warp 用 `tcgen05.ld` 读取各自负责的 32 条 lane，将结果写回 global，最后
同步并释放 TMEM。

### 判测结果

seeds 1、7、42、1234、99999 均为 `PASS`。

### 移除 fence.proxy.async 的现象

定义 `OMIT_PROXY_FENCE` 移除 `fence.proxy.async.shared::cta` 后，上述五组 seed
在本次 B300 实验中仍全部 PASS。这个阴性结果不证明 fence 可以删除：普通
shared-memory 写与 async proxy 之间的可见性仍由该 fence 建立，缺少 fence
属于模型上未保证的写法，结果可能随调度、编译器或负载变化。

## Prob 3.3 mbarrier bug

### 修改前实验

| rounds | 结果 | 是否超时 | 观察 |
|---:|---|---|---|
| 1 | PASS | 否 | 第一代 phase 0 与错误等待恰好一致 |
| 2 | FAIL 或卡住 | 部分运行超时 | 第二代实际需要 phase 1，固定等 phase 0 已失配 |
| 4 | FAIL 或卡住 | 部分运行超时 | 代际错误继续累积，20 s 限时内不能可靠完成 |

### 修复说明

barrier 初始化 arrival count 为 1；每轮 commit 后等待 `round & 1`，正确相位
依次为 0、1、0、1。原错误版固定等待 0，从第二轮开始可能等待旧 generation，
导致提前读取仍在更新的 TMEM，或在错误代际上无限等待。

### 状态变化图

```text
init: phase 0 pending
round 0 commit -> phase 0 complete -> reset phase 1
round 1 commit -> phase 1 complete -> reset phase 0
round 2 commit -> phase 0 complete -> reset phase 1
```

### 修复后判测

修复版 rounds=1/2/4、seeds=42/7 的六种组合全部 `PASS`。

## Prob 3.4 CTA pair

| 指标 | cta_group::1 | cta_group::2 |
|---|---:|---:|
| 每 CTA 的 B shared memory | 8 KiB | 4 KiB |
| 每 CTA 的 TMEM | m128n64，64-column 半区 | m128n64，64-column 半区 |
| shared memory/block | 24588 B | 20492 B |
| NCU shared-store wavefront | 778 | 650 |
| 正确性 | PASS | PASS |

### 分析

1. group 2 让 cluster 内两个 CTA 各 staging 一半 B，再用 group-2 MMA 和
   multicast commit 协作。B shared memory 减半，但每 CTA 仍对应 m128n64
   的 64-column TMEM 半区，因此 TMEM 占用不随 B staging 减半。节省出的
   shared memory 可用于更多 pipeline stage。
2. 该实现依赖 CTA cluster、cluster rank 映射、跨 CTA 协作的 TMEM
   alloc/MMA/commit/dealloc 机制。
3. 数据中心 GPU 更强调大矩阵吞吐、cluster 协作与片上数据复用，也更能以
   足够多的并行工作摊薄 cluster 同步开销。

# M4 完整 GEMM（B/C）

## GEMM 性能阶梯表

| 实现 | TFLOPS | 对 cuBLAS 达成率 | 时间主要花在哪里 |
|---|---:|---:|---|
| naive（assignment01，FP32，1024³） | 3.129 | — | 标量 FMA 与 global 访问 |
| 4.1 tiled | 49.9 | 5.1% | 普通 staging 指令及 load/MMA 串行 |
| 4.2 TMA | 279.5 | 28.6% | 单缓冲下 TMA 与 MMA 串行 |
| 4.3 pipeline（S=3） | 287.8 | 29.6% | shared-memory 容量、occupancy 与等待 |
| cuBLAS | 约 972–979 | 100% | 高度优化的生产级实现 |

## Prob 4.1 tiled GEMM（B）

### 实现说明

grid 覆盖 128×64 输出 tile；K 维以 64 为步长。128 个线程用普通 global
load/shared store staging A/B，执行 async-proxy fence 后发射 MMA，并用单缓冲
同步保证下一次覆写前消费完成。FP32 在 TMEM 累加，最后转为 BF16 写回。

### 正确性与性能

4096³：2.753 ms，49.9 TFLOPS；同输入 cuBLAS 为 0.141 ms、976.2 TFLOPS。
BF16 输出逐元素位级比较为 `exact PASS`。

### 瓶颈判断

4096³ BF16 GEMM 的理想算术强度约为 1365.3 FLOP/byte，高于 B300 官方
机器平衡点 281.25 FLOP/byte；理想充分复用实现应为 compute-bound，但 4.1
没有持续向 Tensor Core 供数。普通线程需要执行大量地址计算、global load 和
shared store，而且 staging 与 MMA 严格串行，因此前端指令与数据搬运无法被隐藏。

Nsight Compute 2025.3.1 `detailed` profile（Job 14904）显示：Compute/SM
46.62%、Memory 30.97%、DRAM 0.38%、Issue Slots Busy 41.40%、L2 hit rate
88.47%，并有约 13% excessive global sectors。DRAM 远未饱和，说明瓶颈是
普通 staging 的指令/地址开销、访问合并和延迟，而非 HBM 或 Tensor Core 屋顶。

## Prob 4.2 TMA（B）

### 修改内容

host 用 `cuTensorMapEncodeTiled` 建立 A/B tensor map；kernel 以
`mbarrier.arrive.expect_tx` 声明传输字节数，并用两条 2D TMA load 搬运 tile。
`full` barrier 表示 TMA 完成，`empty` barrier 表示 MMA 已消费，避免同一个
barrier 同时承担两种 completion generation。

### 正确性与性能

4096³：0.492 ms，279.5 TFLOPS；cuBLAS 为 0.140 ms、978.8 TFLOPS；
逐元素位级比较 `exact PASS`。

### 4.1 与 4.2 对比

4.1 的线程负责地址生成、global load、寄存器暂存和 shared store；4.2 将这些
批量搬运工作交给 TMA，吞吐提升约 5.6 倍。4.2 仍是单缓冲，因此尚未隐藏
TMA 与 MMA 之间的串行等待。上述 4.1 NCU 数据进一步排除了 HBM 带宽饱和，
验证了 TMA 所替代的普通地址/load/store 供数路径是主要开销。

## Prob 4.3 多级 pipeline（B）

### Stage sweep

| 形状 | S=2 | S=3 | S=4 | S=6 |
|---|---:|---:|---:|---:|
| 4096³ | 301.8 | 288.5 | 253.3 | 183.3 |
| 256×4096×16384 | 168.5 | 207.4 | 189.4 | 210.5 |

### 流水时空图

```text
时间 →       t0          t1          t2          t3
TMA producer load S0     load S1     load S2     refill S0
MMA consumer             use S0      use S1      use S2
barrier       full0       full1       full2       empty0→reuse
```

每个 stage 有 `full[s]`（producer→consumer 的 RAW 保护）和 `empty[s]`
（consumer→producer 的 WAR 保护）；消费完成后才能覆写环形槽位。

### 分析

1. CUDA occupancy API 在 B300 上对 S=2/3/4/6 均返回 1 block/SM。4096³
   有 2048 CTA，即约 13.8 waves，仍可依靠 block 间调度隐藏延迟，S=2 最好；
   M=256 只有 128 CTA，少于 148 SM，更依赖单 block 流水，因此 S=3/S=6
   优于 S=2。性能变化不能解释为 blocks/SM 从 4 逐级降到 1。
2. 4.1 主要受普通 staging 指令限制；4.2 消除大部分线程搬运工作；4.3
   重叠 TMA/MMA 后，瓶颈转向 shared-memory 容量、occupancy 和同步气泡。
3. 4.1→4.2 减少地址生成/load/store 指令，4.2→4.3 隐藏单缓冲串行等待。
4. 每 stage 的 A/B tile 约 24 KiB；S=2/3/4/6 约占 48/72/96/144 KiB，
   occupancy API 的实际最大 blocks/SM 均为 1。设备每 SM 有 233472 B shared
   memory；仅按总字节数整除得到 4/3/2/1，但 occupancy API 的实际值均为 1。
   S=2 的 NCU basic profile（Job 14933）也将 6.2% theoretical occupancy 的
   限制归因于 required shared memory，说明简单字节除法忽略了当前动态 smem
   配置/硬件分配约束。继续增加 stage 时 shared memory 仍会先于当前
   64-column TMEM accumulator 成为硬容量限制。

## Prob 4.5 thin GEMM（C）

### 理论 Roof

\[
AI=\frac{2MNK}{2MK+2NK+2MN}
\]

【插入不同 M、N、K 的 AI、compute roof 和 memory roof 计算表。】

### 实测结果

【粘贴程序输出或整理后的表格。】

### 分析

1. 性能从哪个 M 开始明显下降：【】
2. 从哪个 M 开始进入平台：【】
3. 平台达成率：【】
4. 哪些形状主要受显存带宽限制：【】
5. `f_b_proj(K=128)` 的主要限制：【】
6. vLLM 在 M≤16 使用 skinny CUDA Core kernel 的原因：【】

# M5 低精度与 block scaling（C）

## Prob 5.1 FP8 outlier

| 采样点 | ≈0.5 | 0.1 | 0.01 | 0.005 | 3000 |
|---|---:|---:|---:|---:|---:|
| 相对误差 | 【】 | 【】 | 【】 | 【】 | 【】 |

1. 移除 outlier 后，x≈0.5 的误差变化：【】倍
2. 量化为 0 的阈值及其与 scale 的关系：【】
3. 使用 1×128 per-block scale 后的变化：【】

## Prob 5.2 block scaling

### 判测

```text
【粘贴 pytest 输出】
```

### 代数说明

```text
可以提出 scale 的条件：
不能提出 scale 的情况：
```

### 分析

1. 为什么分组沿 K 方向：【】
2. 为什么使用 1×128、128×128 或 1×16：【】
3. 从 128 缩小至 16 的精度收益和 scale 开销：【】

## Prob 5.3 NVFP4 量化通路

### 5.3(a) E2M1 编码器

```text
【粘贴 03a_encode_check PASS】
```

### 5.3(b) NVFP4 quant kernel

```text
【粘贴逐 byte 判测和 test_fp4_gemm 输出】
```

```text
maxrel：
SF 布局说明：
```

### 5.3(c) Ceiling probe

| 指标 | 数值 |
|---|---:|
| ceiling probe GB/s | 【】 |
| quant kernel GB/s | 【】 |
| quant / ceiling | 【】 |

【结合 Nsight Compute 判断剩余差距来自访存还是计算。】

## Prob 5.4 融合 RMSNorm + NVFP4

### 实现与调优

```text
融合 kernel 配置：
两步基线配置：
正确性结果：
```

### 逐形状结果

| M | K | 两步时间 | 融合时间 | 加速比 | 相对 ceiling |
|---:|---:|---:|---:|---:|---:|
| 【】 | 【】 | 【】 | 【】 | 【】 | 【】 |

【继续补全十个测试形状。】

### 性能归因

【解释不同 M 范围的限制，以及实测结果与 2.56× 理论上限的差距。】

## Prob 5.5 W4A16 与 NVFP4

### (a) 量化类别

【分别说明存储量化或计算量化。】

### (b) 节省的资源

【比较显存容量、显存带宽和计算吞吐。】

### (c) 小 batch decode

【说明哪类量化收益更直接及原因。】

# M6 TileLang 对照（A）

## Prob 6.1 lowering 对照

| 项目 | sm_90a | sm_100a |
|---|---|---|
| Tensor Core 指令 | `wgmma.m64n128k16`（`tl::wgmma_ss`） | TileLang 0.1.13 实际回退到 `mma.sync.m16n8k16`，并生成 `ldmatrix`；未选 tcgen05 |
| descriptor 在哪里、由谁生成 | lowering 在 kernel 内生成 `GmmaDescriptor`；TMA `CUtensorMap` 由 host/runtime 生成 | `mma.sync` 使用寄存器 fragment，因此没有 tcgen05 smem descriptor；TMA TensorMap 仍由 host/runtime 生成 |
| swizzle 在哪一步确定 | layout inference/lowering 根据 TMA/WGMMA 与 tile 约束确定 | layout inference/lowering 根据 TMA staging 与 `ldmatrix` 约束确定 |
| 谁将数据搬入 smem | 编译器生成 `tl::tma_load`、mbarrier 和三阶段 producer/consumer pipeline | 同左 |

### 生成文件

```text
sm_90a CUDA 源码：assignment02/m6_lowering_output/sm_90a_generated.cu
sm_90a lowering：assignment02/m6_lowering_output/sm_90a_lowered_tir.txt
sm_100a CUDA 源码：assignment02/m6_lowering_output/sm_100a_generated.cu
sm_100a lowering：assignment02/m6_lowering_output/sm_100a_lowered_tir.txt
```

### 对照分析

1. DSL 自动完成的硬件相关决策：Tensor Core 指令族、fragment/ldmatrix
   映射、WGMMA descriptor、smem staging 布局、TMA TensorMap、mbarrier
   phase、同步与 producer/consumer pipeline。
2. 程序员仍需决定的参数：BM/BN/BK、threads、stages、dtype/accum dtype、
   transpose、grid 映射、目标架构，以及是否 benchmark/autotune。
3. “谁负责”表新增内容：`Tensor Core 指令选择与供数布局`——CUDA 手写时
   由程序员负责；TileLang 中由 lowering 根据 target/dtype/tile 自动生成，
   程序员通过 tile、threads、stages 和 target 间接约束。

固定版本核实结论：字面 `sm_100a` 和文档推荐的
`sm_100f + code=[sm_100a]` 都能完成 cubin 编译，但对本次普通 FP16
`T.gemm` 均回退到 `mma.sync + ldmatrix`。报告以生成源码为准，不把
tcgen05 支持能力误写成本次实际选择结果。

# 最终检查

## 代码与判测

- [ ] 所有动手题代码已保存
- [ ] 所有 FROM-SCRATCH 题均有 PASS 记录
- [ ] 所有 DEBUG 题均有修改前现象和修复说明
- [ ] 所有 EXPERIMENT 均有预测、实测和解释
- [ ] 所有性能数据均注明 GPU
- [ ] 不包含 4.4 Optional
- [ ] 不包含 5.3(d) Optional
- [ ] 不把 5.4 末尾额外优化误认为必做
- [ ] C1/C2 团队题未混入非团队部分

## 报告完整性

- [ ] 目录与题号完整
- [ ] 表格单位完整
- [ ] 图片清晰且有标题
- [ ] 命令和关键输出已保留
- [ ] 引用的峰值、带宽注明口径和来源
- [ ] 三部分报告格式一致
- [ ] 已完成最终通读和交叉检查
