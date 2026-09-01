# B300 复现流程

## 环境

- NVIDIA B300 SXM6 AC，CC 10.3，148 SM。
- CUDA 13.0；编译目标由 `FLASH_KDA_CUDA_ARCHS=103a` 指定。
- 运行前确认 GPU 无其他计算进程，性能测试使用独占的一张卡。

## 1. 准备源码

```bash
git clone --recurse-submodules https://github.com/MoonshotAI/FlashKDA.git
cd FlashKDA
git checkout 1ce47ea
git apply /path/to/0001-k2-value-slice-and-dispatch.patch
```

## 2. 构建单一扩展

四个 K2 变体必须编译进同一个 `flash_kda_C`，否则 Python 可能误加载旧扩展，造成“代码改了但跑的不是它”的假验证。

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export FLASH_KDA_CUDA_ARCHS=103a
export NVCC_THREADS=8
python setup.py build_ext --build-lib build/integrated/lib \
  --build-temp build/integrated/temp --force
```

构建后检查：

```bash
PYTHONPATH=build/integrated/lib:. python - <<'PY'
import flash_kda_C
print(flash_kda_C.__file__)
print(flash_kda_C.get_device_characteristics())
PY
```

## 3. 正确性与集成性能

在 Slurm 集群中先申请单卡，再把新扩展放在 `PYTHONPATH` 首位：

```bash
srun --partition=gpu --gres=gpu:1 --time=00:30:00 --pty bash
export PYTHONPATH="$PWD/build/integrated/lib:$PWD"
unset FLASH_KDA_K2_VALUE_SLICE
unset FLASH_KDA_K2_DISPATCH
python profile/k2-vsplit-opt/integrated_validation.py \
  --warmup 10 --iterations 50
```

验证门槛：

1. fixed BF16、FP32 state、ragged varlen 的 output/final state 与 V128 bitwise equal；
2. V16/V32/V64/V128 policy 边界符合已标定区间；
3. CUDA Graph capture/replay 通过；
4. H75 fallback 选择 V128，差异只应是计时抖动；
5. state-carrying prefill + decode trace 保持正收益。

## 4. 性能口径

每个形状 10 次 warmup、50 次 CUDA Event 计时。比较对象是同一个扩展强制 V128 与自动 dispatcher，不跨 build 比较。NCU profiler duration 只用于诊断，不能代替正常 benchmark 延迟。

## 5. 两次关键结果

| Run | BF16 高价值区间 | FP32 高价值区间 | Stateful trace |
|---|---:|---:|---:|
| Job 5195（2026-08-21） | 14.66%–23.30% | 9.13%–21.67% | 5.45% |
| Job 14592（2026-09-01） | 14.99%–26.10% | 9.37%–21.97% | 5.68% |

第二次运行 GPU 时钟较低，绝对延迟整体变大，但 dispatcher 的边界、bitwise correctness 和相对收益稳定复现。

## 6. Trace/profile 归档

- `BOTTLENECK_ANALYSIS.md`：完整瓶颈判断与答辩解释。
- `figures/kimi_kda_b300_bottleneck.png`：基于实测数据的浓缩图片。
- `figures/kimi_kda_b300_bottleneck.svg`：可编辑矢量版。
- `artifacts/nsys/`：本机保存 Nsight Systems 原始报告、SQLite 与导出统计；这些文件含服务器元数据，已被 Git 忽略。
- `artifacts/ncu/`：本机保存四个 K2 对照的 Nsight Compute 原始报告；已被 Git 忽略。
- `data/k2_nsys_timeline.csv`、`data/k2_nsys_ranges.csv`：时间线紧凑数据。
- `data/k2_ncu_metrics.csv`：duration、throughput、L2、scheduler、occupancy 与 launch resource 指标。
- `data/sass_opcode_summary.csv`、`artifacts/sass/`：recurrence 指令族汇总与抽样 SASS；本次为 3,640 条 HMMA、0 条 TCGEN/UTCMMA。
- `extract_nsys_timeline.py`、`export_ncu_metrics.py`、`make_bottleneck_figure.py`、`render_figure_png.mjs`：从报告到图片的可复现链路。

采集 H12 的同进程 V128/V16 trace：

```bash
sbatch run_nsys_trace.sbatch
```

图片不是生成式图像；它由绘图脚本读取 CSV 后确定性生成，因此每个柱、数值和时间段都能回到归档报告。
