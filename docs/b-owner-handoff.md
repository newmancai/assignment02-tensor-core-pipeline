# Assignment 02 B 部分交接说明

更新时间：2026-09-01
负责范围：3.1–3.4、4.1–4.3
目标硬件：NVIDIA B300，CUDA 13.0，`compute_100f/sm_100f`

## 一、结论

3.1–3.4、4.1–4.3 已全部完成。3.2、3.3 修复版、3.4、4.1、4.2、4.3
均已在 B300 上完成最终一致回归。
4.1–4.3 还加入了同输入 cuBLAS BF16 参考结果，按 BF16 位模式逐元素严格
比较，4096³ 均为 `exact PASS`。

原理、代码入口、B300 数据与结论已经分别写入 3.1–4.3 的小题 README，
汇总结果同步进 `docs/full-report.md`。最终回归为 Slurm Job 14793；完整文本
记录在 `docs/evidence/b300-results.md`。当前环境没有可用浏览器运行时，因此
没有生成 PNG 截图；交接以可复制、可搜索的原始终端输出为证据。

## 二、逐题状态

| 题目 | 代码状态 | B300 验证 | 文档状态 |
|---|---|---|---|
| 3.1 | 判断、理由和容量计算完成 | 不需要 GPU | 已写入 README |
| 3.2 | 单 tile tcgen05 GEMM 完成 | 五组 seed 均 PASS；无 fence 对照已运行 | 已写入 README |
| 3.3 | phase 修复及 `BUGGY_PHASE` 复现开关完成 | 修复版六种组合 PASS；错误版已复现失败/超时 | 已写入 README |
| 3.4 | cta_group::1/2 两个实现完成 | 两种实现 PASS；shared-memory 与 NCU 数据齐全 | 已写入 README |
| 4.1 | 普通 load/store + swizzle tiled GEMM 完成 | 4096³ exact PASS | 已写入 README |
| 4.2 | TMA 单缓冲 GEMM完成 | 4096³ exact PASS | 已写入 README |
| 4.3 | STAGES 环形缓冲、full/empty barrier 完成 | S=2/3/4/6 两种形状均 exact PASS | 已写入 README |

## 三、已修改文件

- `M3-tcgen05/3.2-single-tile/02_single_tile.cu`
- `M3-tcgen05/3.3-mbarrier-debug/03_bug_mbarrier.cu`
- `M3-tcgen05/3.4-cta-pair/04_cta_pair.cu`
- `M4-gemm/4.1-tiled/01_tiled.cu`
- `M4-gemm/4.2-tma/02_tma.cu`
- `M4-gemm/4.3-pipeline/03_pipeline.cu`
- `M4-gemm/4.3-pipeline/judge_ladder.sh`
- `M4-gemm/4.3-pipeline/sweep_stages.sh`

上述改动已随本次 B 部分 commit 归档。原有用户改动没有执行 reset、checkout
或覆盖式回退。

## 四、关键实现与修复

### 3.2：单 tile tcgen05

实现的数据路径为：

`global -> swizzled shared memory -> tcgen05.mma -> TMEM -> tcgen05.ld -> global`

流程包括 mbarrier 初始化、64 个 TMEM column 分配、四条 k16 MMA、
`tcgen05.commit`、barrier 等待、四个 warp 分别读取 32 条 TMEM lane，
最后同步并释放 TMEM。

### 3.3：barrier phase

正确等待相位为 `round & 1`，即 0、1、0、1。正常构建使用正确相位；
定义 `BUGGY_PHASE` 时固定等待 phase 0，用于复现原题错误。

### 3.4：CTA pair

`cta_group::1` 使用两个独立 CTA，各自 staging 完整 B；`cta_group::2`
使用一个两 CTA cluster，每个 CTA 只 staging 一半 B，并使用 group-2
alloc、MMA、multicast commit 和 dealloc。

### 4.1：普通 staging

grid 覆盖全部 128×64 输出 tile，K 以 64 为步长。128 个线程通过普通
global load/shared store 完成 A/B staging，随后执行 proxy fence、MMA
和单缓冲同步。这一版本有意保留地址计算与普通 load/store 指令开销。

### 4.2：TMA 单缓冲

host 使用 `cuTensorMapEncodeTiled` 创建 A/B tensor map，kernel 使用
`mbarrier.arrive.expect_tx` 和两条 2D TMA load。TMA 完成与 MMA 完成
使用独立的 `full`、`empty` barrier。

曾尝试让同一个 barrier 交替承担 TMA transaction completion 和 MMA
commit completion；小形状可以通过，但 4096³ 大 grid 会挂死。拆分成
双 barrier 后 4096³ 稳定 PASS。

### 4.3：多级 pipeline

每个 stage 使用一对 barrier：

- `full[s]`：TMA producer 到 MMA consumer 的 RAW 保护；
- `empty[s]`：MMA consumer 到 TMA producer 的 WAR 保护。

每 stage 的 A/B tile 合计约 24 KiB。S=3 已超过 48 KiB 静态 shared
memory 上限，因此 ring buffer 使用 opt-in dynamic shared memory，并通过
`cudaFuncSetAttribute(...MaxDynamicSharedMemorySize...)` 设置允许值。

预热时填满可用 stage；消费 stage 后必须等 `empty[s]`，才允许 TMA
覆写该 stage。此前候选代码中两个关键 wait 被乱码注释吞掉，K tile 开始
复用时会出现 `unspecified launch failure`，现已修复。

## 五、B300 实测结果

### 3.2 / 3.3

- 3.2：seeds 1、7、42、1234、99999 全部 PASS。
- 3.3 修复版：
  - rounds=1，seeds 42/7：PASS；
  - rounds=2，seeds 42/7：PASS；
  - rounds=4，seeds 42/7：PASS。

### 3.4

正确性与容量：

| 实现 | shared memory/block | 正确性 |
|---|---:|---|
| cta_group::1 | 24588 B | PASS |
| cta_group::2 | 20492 B | PASS |

Nsight Compute 指标
`l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum`：

| 实现 | shared-store wavefront |
|---|---:|
| cta_group::1 | 778 |
| cta_group::2 | 650 |

两者单 tile 时间处于噪声范围，不能据此判断优劣。有效结论是 group 2
每 CTA 的 B staging 从 8 KiB 降为 4 KiB，总 shared staging 流量随之下降，
但 A staging 不变，所以总量不会减半。

### 4.1–4.3 性能梯子

4096³，BF16 输入/输出、FP32 累加；每个自定义 kernel 都与 cuBLAS BF16
输出进行逐元素位级比较：

| 实现 | TFLOPS | cuBLAS 达成率 | 正确性 | 主要开销 |
|---|---:|---:|---|---|
| assignment01 naive FP32（1024³） | 3.129 | 仅比较量级 | PASS | 标量 FP32 FMA 与 global 访问 |
| 4.1 tiled | 49.9 | 5.1% | exact PASS | 普通 staging 指令和 MMA 严格串行 |
| 4.2 TMA | 279.5 | 28.6% | exact PASS | TMA 与 MMA 仍为单缓冲串行 |
| 4.3 pipeline，S=3 | 287.8 | 29.6% | exact PASS | shared-memory 容量/occupancy 与流水等待 |
| cuBLAS | 约 972–979 | 100% | 参考 | 高度优化实现 |

这些数字会受 GPU 时钟与同卡负载影响，README 中应注明是一次 B300
实测，不要把最后一位视为稳定值。

4.1 另在 Slurm Job 14904 使用 Nsight Compute 2025.3.1 `detailed` set
采集一次 kernel：SM 46.62%、Memory 30.97%、DRAM 0.38%、Issue Slots Busy
41.40%，并有约 13% excessive global sectors。该证据排除了 HBM 饱和，
支持“普通 staging 的地址/指令开销、访问合并与延迟”为主要瓶颈。

### 4.3 stage sweep

单位：TFLOPS。

| 形状 | S=2 | S=3 | S=4 | S=6 |
|---|---:|---:|---:|---:|
| 4096³ | 301.8 | 288.5 | 253.3 | 183.3 |
| 256×4096×16384 | 168.5 | 207.4 | 189.4 | 210.5 |

解释：

- occupancy API 对 S=2/3/4/6 均返回 1 block/SM。4096³ 有 2048 CTA，
  即约 13.8 waves，可由 block 间调度隐藏延迟，因此 S=2/S=3 更好；性能
  回落不能解释为 blocks/SM 从 4 逐级降到 1。
- M=256 的形状只有 128 CTA，少于 B300 的 148 SM；更深的单 block 流水
  更重要，因此 S=3/S=6 优于 S=2。S=4 的回落表明 stage 管理、片上容量
  压力、调度和测量波动仍会影响结果。
- 每 stage 约 24 KiB，S=2/3/4/6 分别约 48/72/96/144 KiB。随着 stage
  增加，shared memory 会先于当前 64-column TMEM accumulator 成为硬容量
  限制。实际 blocks/SM 在四种配置下均为 1；3.4 的 group-2 B 切分仍可为
  更深流水腾出 shared-memory 空间。

## 六、3.1 最终答案

- (a) 正确。TMEM 地址高 16 bit 表示 lane 偏移；每个 warp 用
  `tcgen05.ld` 读取自己对应的 32 条 lane。
- (b) 正确。一条 tcgen05 MMA 由 elected lane 发射，硬件异步执行，结果
  位于 TMEM。
- (c) 错误。TMEM 结果需要先由 `tcgen05.ld` 进入寄存器，再写回 global；
  不能直接用 TMA 从 TMEM 搬到 global。
- (d) 正确。总容量为 `128×512×4 = 256 KiB`；m128n256 FP32 accumulator
  为 `128×256×4 = 128 KiB`，恰好一半。
- (e) 错误。`tcgen05.commit` 本身不阻塞到 MMA 完成；需要等待 commit
  到达的 mbarrier 后才能安全读取 TMEM。

## 七、3.3 状态机与错误版实验

正确状态循环：

```text
init: phase=0, pending=1
commit round 0 completes -> phase 0 complete -> reset to phase 1
commit round 1 completes -> phase 1 complete -> reset to phase 0
commit round 2 completes -> phase 0 complete -> reset to phase 1
```

正确等待序列是 0、1、0、1。错误版本固定等待 phase 0，从第二轮开始等待
的不是当前 generation，可能提前放行，使 `tcgen05.ld` 与仍在执行的 MMA
发生竞态；也可能在错误代际上等待。实测 rounds=1 PASS；rounds=2/4 出现
失败或 20 秒超时，完整现象已记录到 3.3 README。

## 八、后续可选收口

1. 在有浏览器或终端截图能力的环境中，将原始日志渲染为 PNG，若课程明确要求
   截图再补入报告。
2. 提交前填写成员姓名、日期等个人信息，并确认课程对公开代码的要求。
3. 若需重新测性能，保持同一节点、空闲 GPU 和相同编译参数；绝对 TFLOPS 会
   随时钟与负载波动，应同时保留 cuBLAS 达成率。

## 九、B300 复现命令

所有 GPU 命令都应通过 Slurm 执行，不要直接在登录节点运行。以下命令假定
当前位于项目根目录：

```bash
export NVCC=/usr/local/cuda-13.0/bin/nvcc
export GENCODE="-gencode arch=compute_100f,code=sm_100f"
mkdir -p /tmp/assignment02-b

$NVCC -O3 -std=c++17 $GENCODE \
  M3-tcgen05/3.2-single-tile/02_single_tile.cu \
  -o /tmp/assignment02-b/m32

$NVCC -O3 -std=c++17 $GENCODE \
  M3-tcgen05/3.3-mbarrier-debug/03_bug_mbarrier.cu \
  -o /tmp/assignment02-b/m33

$NVCC -O3 -std=c++17 $GENCODE \
  M3-tcgen05/3.4-cta-pair/04_cta_pair.cu \
  -o /tmp/assignment02-b/m34

$NVCC -O3 -std=c++17 $GENCODE \
  M4-gemm/4.1-tiled/01_tiled.cu -lcublas -lcuda \
  -o /tmp/assignment02-b/m41

$NVCC -O3 -std=c++17 $GENCODE \
  M4-gemm/4.2-tma/02_tma.cu -lcublas -lcuda \
  -o /tmp/assignment02-b/m42

$NVCC -O3 -std=c++17 $GENCODE \
  M4-gemm/4.3-pipeline/03_pipeline.cu -lcublas -lcuda \
  -o /tmp/assignment02-b/m43
```

编译建议在登录节点完成，运行使用：

```bash
srun -G 1 --time 00:15:00 bash -lc '
  /tmp/assignment02-b/m32 42 &&
  /tmp/assignment02-b/m33 42 4 &&
  /tmp/assignment02-b/m34 &&
  /tmp/assignment02-b/m41 4096 4096 4096 &&
  /tmp/assignment02-b/m42 4096 4096 4096 &&
  /tmp/assignment02-b/m43 4096 4096 4096
'
```

3.3 错误版单独编译：

```bash
$NVCC -O3 -std=c++17 -DBUGGY_PHASE $GENCODE \
  M3-tcgen05/3.3-mbarrier-debug/03_bug_mbarrier.cu \
  -o /tmp/assignment02-b/m33-bug
```

stage sweep：

```bash
cd M4-gemm/4.3-pipeline
NVCC=/usr/local/cuda-13.0/bin/nvcc bash sweep_stages.sh
```

## 十、临时环境

B300 用户目录下曾创建两个测试目录：

- `codex_assignment02_candidate`
- `codex_assignment02_target`

它们仅用于真机候选代码与目标代码验证，不属于项目交付物。完成最终回归并
确认日志不再需要后，可以在确认绝对路径无误的前提下清理。
