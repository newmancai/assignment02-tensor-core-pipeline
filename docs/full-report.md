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
| M3 `tcgen05` | B | ☐ | ☐ | ☐ | ☐ |
| M4.1–M4.3 完整 GEMM | B | ☐ | ☐ | ☐ | ☐ |
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
| (a) | 对 / 错 | 【】 |
| (b) | 对 / 错 | 【】 |
| (c) | 对 / 错 | 【】 |
| (d) | 对 / 错 | 【写出计算过程】 |
| (e) | 对 / 错 | 【】 |

## Prob 3.2 tcgen05 单 tile GEMM

### 实现流程

```text
global → smem → tcgen05.mma → TMEM → tcgen05.ld → global
```

【说明 descriptor、swizzle、TMEM alloc、mbarrier、idesc、mma 和 ld 的实现。】

### 判测结果

```text
【粘贴 make 与 judge_tile.sh 的 PASS 输出】
```

### 移除 fence.proxy.async 的现象

```text
观察现象：
原因：
```

## Prob 3.3 mbarrier bug

### 修改前实验

| rounds | 结果 | 是否超时 | 观察 |
|---:|---|---|---|
| 1 | 【】 | 【】 | 【】 |
| 2 | 【】 | 【】 | 【】 |
| 4 | 【】 | 【】 | 【】 |

### 修复说明

【描述 phase、arrival count 的修改以及错误等待何时过早或过晚放行。】

### 状态变化图

【插入修改前后 mbarrier 状态图。】

### 修复后判测

```text
【粘贴 judge_mbar.sh 输出】
```

## Prob 3.4 CTA pair

| 指标 | cta_group::1 | cta_group::2 |
|---|---:|---:|
| 每 CTA 的 B shared memory | 【】 | 【】 |
| TMEM 占用 | 【】 | 【】 |
| shared memory 总流量 | 【】 | 【】 |

### 分析

1. 节省的 shared memory 在 M4 pipeline 中可以用于：【】
2. 依赖的硬件机制：【】
3. 为什么更常出现在数据中心 GPU：【】

# M4 完整 GEMM（B/C）

## GEMM 性能阶梯表

| 实现 | TFLOPS | 对 cuBLAS 达成率 | 时间主要花在哪里 |
|---|---:|---:|---|
| naive（assignment01，FP32） | 【】 | — | 【】 |
| 4.1 tiled | 【】 | 【】 | 【】 |
| 4.2 TMA | 【】 | 【】 | 【】 |
| 4.3 pipeline（S=3） | 【】 | 【】 | 【】 |
| cuBLAS | 【】 | 100% | 【】 |

## Prob 4.1 tiled GEMM（B）

### 实现说明

【说明 grid/tile 映射、K 循环、同步和累加。】

### 正确性与性能

```text
判测：
时间：
TFLOPS：
```

### 瓶颈判断

【结合 0.2 的机器平衡点说明主要受哪一环节限制。】

## Prob 4.2 TMA（B）

### 修改内容

【说明 tensor map、cp.async.bulk.tensor、expect_tx 和同步方式。】

### 正确性与性能

```text
判测：
时间：
TFLOPS：
```

### 4.1 与 4.2 对比

【说明 4.1 staging 的组成，以及哪些工作改用 TMA 后不再由普通 CUDA 指令执行。】

## Prob 4.3 多级 pipeline（B）

### Stage sweep

| 形状 | S=2 | S=3 | S=4 | S=6 |
|---|---:|---:|---:|---:|
| 4096³ | 【】 | 【】 | 【】 | 【】 |
| 256×4096×16384 | 【】 | 【】 | 【】 | 【】 |

### 流水时空图

【任选一个 STAGES，插入稳态阶段 TMA 与 MMA 的时空图。】

### 分析

1. 两个形状对 STAGES 的敏感程度：【】
2. 从 4.1 到 4.3 的瓶颈变化：【】
3. 每一级优化减少的开销：【】
4. 继续增加 tile/stage 时先遇到的容量限制：【】

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
