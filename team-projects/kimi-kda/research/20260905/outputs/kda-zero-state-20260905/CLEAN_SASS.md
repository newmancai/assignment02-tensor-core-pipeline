# Clean Phase 1：SASS 核验与最小 NCU 互证

2026-09-05。经审批只读 SSH 使用 CUDA 13.0 `cuobjdump` 提取现有 `.so` 的两个精确实例，再读取本地 Job 19935 NCU；未改远端文件、源或安装，未提交 GPU 作业。输出 [clean_sass_targets.json](clean_sass_targets.json) 包含完整符号、每条 PC/指令、原始 SASS hash、资源及核验摘要；原始机器码编码未保留。

## 结论

干净二进制确实生成并在新 NCU 中调用了 **无初态 P4/L4、有初态 P4/L2**，不是实验 selector 混入生产接口。两个实例保留原 P4 的矩阵计算结构；寄存器 provenance 的局部追踪也确认 Phase 1 同 k 的 operand 配对、沿 k=0..7 的累加次序。

重要限定：无初态 L4 **没有解决全部循环不变量重算**，循环内 S2R 从旧版 2 条增至 4 条，并真实 spill 了两个控制值。它仍在部分 key block 获得更长 load/use 间隔；NCU 显示发射效率改善、short-scoreboard 降低、单次 K2 更快。更强的精确互证是：SASS 预测 245,760 个 local spill 请求，NCU 恰好记录 245,760。不能再将这条路径称为无 spill，也不能将收益解释为“减少了动态指令”。

有初态 L2 无 spill、循环内仍无 S2R。它不是每个 load/use 间隔都更长：此前重点 block4 的间隔与旧有初态 P4 一样；改善分布在其它位置及整个 kernel 的调度。Phase 6 也重新排过，不能说只有 Phase 1 的机器码改变。

## 1. 二进制与模板身份

```text
clean binary:
 /home/lcpu/YOUR_USER_ID/kda-zero-state-20260905/clean_build/lib/
 flash_kda_phase1_C.cpython-312-x86_64-linux-gnu.so
SHA256 f6f80fa402cc1dc00b09a8082b10806bbe17c0e533d067e931dd774d270b9270
owner UID USER_UID = 已登录账户 YOUR_USER_ID

old P4 SHA256:
 34e2c68cf80de0bd24278afb035fdcd44e4a6205241dfcb9aab4ee95891bc486
```

依据 [干净源码模板](../../implementation/phase1/csrc/smxx/fwd_kernel2.cuh#L130) 解码最后参数，而非匹配类型树中的任意整数：

```text
NoState: ...ELi16ELi128ELi16ELi3ELi2ELi96ELb0ELb1ELb0ELb0ELi4ELi4EEv...
StateIn: ...ELi16ELi128ELi16ELi3ELi2ELi96ELb1ELb1ELb0ELb0ELi4ELi2EEv...
            C16  D128  V16  IS3 OS2 NT96  SI   SO   FP   VL   P4   L
```

固定 SO=true、FP=false、VL=false；clean 的最后参数是 `Phase1Prefetch`，**不是实验 InitStrategy 4/5**。绑定/launch 已恢复普通 selector；源 guard 在已选 V16、D128/BF16、N1/H12/T2048..8192 内按 SI 选择 L2/L4。本次不宣称已反汇编所有 state-output/varlen/fallback 组合。

[build_clean.log:59](build_clean.log#L59) 使用 sm_103a、`--register-usage-level=10`、原 fast-math，以及 `FLASH_KDA_ENABLE_V16_PREFETCH4=1` / `FLASH_KDA_ENABLE_V16_PHASE1_PREFETCH=1`。build log 152、222 行的目标属性与实际 cubin 一致。

## 2. 资源与数学骨架

| 指标 | 旧无初态 P4 | clean 无初态 L4 | 旧有初态 P4 | clean 有初态 L2 |
| --- | ---: | ---: | ---: | ---: |
| Registers | 63 | 56 | 70 | 72 |
| Stack B | 0 | 8 | 0 | 0 |
| ptxas spill stores / loads B | 0 / 0 | 16 / 12 | 0 / 0 | 0 / 0 |
| 全函数静态指令数 | 1472 | 1480 | 1456 | 1448 |
| 主回边对应线性区域指令数 | 453 | 458 | 425 | 421 |
| 主循环内 S2R | 2 | 4 | 0 | 0 |
| Phase 1 matrix LDSM | 24 | 24 | 24 | 24 |
| Phase 1 HMMA | 32 | 32 | 32 | 32 |
| 全 tile HMMA | 52 | 52 | 52 | 52 |
| 全 tile normal / transposed LDSM | 27 / 16 | 27 / 16 | 27 / 16 | 27 / 16 |
| 全 tile normal / transposed STSM | 1 / 8 | 1 / 8 | 1 / 8 | 1 / 8 |
| 全 tile FFMA / BF16 pack | 66 / 49 | 66 / 49 | 66 / 49 | 66 / 49 |

两个 clean cubin 的 `SHARED:1024` 不包括 launch 的 dynamic shared，不是 CTA 总共享内存。特别注意 clean 无初态 cubin 虽报 `LOCAL:0`，却有 stack 与真实 LDL/STL；**LOCAL 字段为零不能单独证明无 spill**。

线性区域按回边目标到回边计数，含条件分支和 trap，不含跳出的等待 helper，不能当精确动态指令数。clean 无初态回边 `0x4600→0x2970`；有初态 `0x4400→0x29c0`。

局部寄存器标签追踪将 LDSM 标为 k/q/state、block 与 word，再沿 MOV/HMMA 追踪：四实例各 24 个 Phase 1 operand load、32 条同 k 配对的 HMMA，抽象累加链均按 0..7 递推。它支持数学结构保持，不是整 kernel 的形式证明或替代 out/final-state GPU 逐位验证。

## 3. Load/use：确有改进，但不均匀

比较相同 block4，表中为 load 与第一条对应半块消费 HMMA **之间夹着的静态指令数**，不是 cycles：

| 间隔 | 旧无初态 | clean 无初态 | 旧有初态 | clean 有初态 |
| --- | ---: | ---: | ---: | ---: |
| k4 → k-GEMM | 1 | 4 | 4 | 4 |
| state4 → k-GEMM | 0 | 1 | 7 | 7 |
| q4 → q-GEMM | 0 | 3 | 4 | 4 |

clean 无初态的局部证据：

```text
2ec0 LDSM.16.M88.4 R16, [R24+0xa80]    // k4
2ef0 LDSM.16.M88.4 R8,  [R33+0x800]    // S4
2f00 LDSM.16.M88.4 R12, [R24+0x1a80]   // q4
2f10 HMMA.16816.F32.BF16 R40, R16.reuse, R8, R40
2f40 HMMA.16816.F32.BF16 R36, R12.reuse, R8, R36
```

此处 input 基址已被编译器提前加了 `0x1000`；已依据地址计算归一化，与旧 P4 的 `k4+0x1a80 / q4+0x2a80` 对齐，不能仅比较字面 offset。

反例也保留：clean 无初态 k6 load 后仍紧接其 HMMA（0 条间隔），而 state6 很早预读；clean 有初态 k7 对应 state load 后紧接消费。源级 L4/L2 **不保证每个 operand 都获得四/两块的物理预取距离**。有初态 block2 的 k/q load 间隔从 5/5 变为 9/8，而 block4 不变；不能挑选一个 PC 就概括全部变化。

## 4. 不变量与 spill 精确位置

无初态 clean loop 的 S2R 在 `0x2970/0x29e0/0x2c50/0x3170`（TID、CTA、CTA、TID），旧版循环只有 2 条。因此“clean 消除了每 tile 重算”的假说被本次 SASS 否定；有初态两版循环都没有 S2R。

无初态 8 B stack 的数据用途可直接追踪：

| Slot | 写入/读取与用途 |
| --- | --- |
| `[R1]` | 初始为 0；0x2ab0 读出后左移 31 位用于 load-pipeline phase wait；tile 结束按 stage wrap 异或更新 |
| `[R1+4]` | 初始化为负的 tile 次数；结束时读出、加 1、比较零以决定回边 |

```text
初始化：2850 STL [R1], RZ
        2960 STL [R1+0x4], R2
每 tile：2ab0 LDL R0, [R1]
         44d0 LDL.LU R2, [R1+0x4]
         4500 LDL.LU R5, [R1]
非末 tile：45c0 STL [R1], R4
           45f0 STL [R1+0x4], R3
```

这些是 compute warp 的循环控制 spill，不是 ring 的整个矩阵 fragment 落到 local memory。其位置意味着不能把 8 B stack 当作只在初始化付一次的代价；也不能把 ptxas 的 16/12 B 当整次请求总流量。有初态 clean L2 全函数没有 LDL/STL。

## 5. Phase 6 也发生了重排

clean 无初态 Phase 6 为 `0x3410..0x4380`，有初态为 `0x32a0..0x4210`；与各自旧版均为 248 条指令。各有 16 转置 LDSM、16 scalar LDS、16 HMMA、64 FFMA、32 BF16 pack、8 转置 STSM。

但相同 opcode 数量不等于相同排序。例如有初态的 state block2 load，旧版位于 Phase 6 起点后 `0x80`，clean 位于起点后 `0x210`；其它 load/MMA/store 也重排。无初态 old/clean 的 SHF/IMAD 数量仍为 32/32，有初态为 33/31；都不表示寄存器活跃区间相同。结尾 compute barrier、MEMBAR、async fence，再 commit/release 的结构仍可定位。没有将整个优化归因于 Phase 1 的充分证据。

## 6. Job 19935 NCU：最小互证

独立读取 4 份 CSV，均 3 行、644 列、1 个目标 launch、16 replay passes、0 warmup passes。profile 代码固定 B1/H12/T8192/D128/C16；在独立进程中使用同 seed/生成顺序，每个 contrast 固定 state mode。实际 kernel 尾模板与上述 SASS 一致，两个 `.so` SHA 也一致。每对的 grid=(1,12,8)、block=(96,1,1)、49.664 Kbyte shared、SMEM CTA limit=4 均相同，achieved occupancy 均约 4.69%。

| 指标 | out：旧 P4 → clean L4 | both：旧 P4 → clean L2 |
| --- | ---: | ---: |
| K2 duration µs | 883.680 → 792.000（−10.3748%） | 718.880 → 676.000（−5.9648%） |
| Issue active % | 15.895675 → 17.772315 | 18.078663 → 19.399433 |
| Eligible warps / active cycle | 0.158884 → 0.177338 | 0.180776 → 0.193987 |
| Short scoreboard cycles / issued instruction | 0.697670 → 0.461213 | 0.467719 → 0.341831 |
| Sleeping cycles / issued instruction | 1.834549 → 1.519628 | 1.340814 → 1.086790 |
| Long scoreboard cycles / issued instruction | 0.953134 → 0.975016 | 1.028060 → 1.016731 |
| Wait cycles / issued instruction | 1.190575 → 1.175948 | 1.160557 → 1.141424 |
| `inst_executed` | 42,580,616 → 42,876,332（+0.6945%） | 40,160,044 → 39,963,408（−0.4896%） |
| `derived__local_spilling_requests` | 0 → **245,760** | 0 → 0 |
| GPC average elapsed frequency GHz | 1.083899 → 1.083622 | 1.089368 → 1.080725 |

Stall ratio 是每条发射指令对应的平均 cycles，不是 wall-time 百分比；不将它们相加分摊耗时。[NVIDIA WarpStateStats](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules) 这组数据支持发射效率/等待结构改善；out 的动态指令反而增加，因此不能将它解释为指令数减少，更不能称优化消除了所有依赖等待。

### Spill 请求数与 SASS 精确吻合

每 CTA 一个 compute warp，T8192/C16=512 个 tile，96 个 CTA。SASS 有初始化 2 次 store、每 tile 3 次 load、非末 tile 2 次 store，故预期 warp 级请求数为：

```text
96 × [2 + 3×512 + 2×(512−1)] = 245,760
```

恰与 NCU 的 `derived__local_spilling_requests` 相同，直接支持以上 spill 位置/频次解释；**不是 245,760 bytes、DRAM misses 或 spill 带来的时间损失**。4 份 CSV 的 `derived__local_spilling_requests_pct` 都是 `no data`，不能填成 0% 或 100%。

### 频率与实验边界

Job 19935 的 GPC 平均 elapsed frequency 约 1.08 GHz；已核实 Job 19918 的两份对应字段为 1.906496/1.906890 GHz。这是跨 job 绝对 µs 数明显不同的直接背景线索，不能将旧 job 与本轮绝对时延直接拼接或认定性能回退。该计数器也不是证明全程锁频；两轮 NCU 均未控制 clocks/caches，replay 非独立统计样本。

更不能根据这四份 profile 假设 Job 19934 的完整 benchmark 全程固定在 1.08 GHz，也不将 duration 乘频率就“校正”成已测时间。主线 clean 重复计时/正确性由另一条验证链给出；这里的单点 K2 duration 只作机制旁证。现有数据没有 PC/warp 角色级 stall 归因，故尚不能量化 Phase 1、Phase 6、spill 各自贡献。

## 证据与剩余边界

局部结构核验通过；本报告未重新运行数学/GPU测试，未验证全部 packed、state-output absent 或 guard 外二进制。发布判断仍需 clean 自身的完整正确性、state chaining、sanitizer、rollback/域外控制和重复性能结果，不能继承实验包的 93 行 PASS。

原始文件：[out P4 CSV](clean_19935_out_p4_ncu.csv)、[out clean CSV](clean_19935_out_phase1_ncu.csv)、[both P4 CSV](clean_19935_both_p4_ncu.csv)、[both clean CSV](clean_19935_both_phase1_ncu.csv)、[profile 脚本](run_clean_profile.sbatch)、[旧 SASS evidence](sass_targets.json)。

| 材料 | SHA-256 |
| --- | --- |
| clean-phase1.patch | `fae72eccda8eea94d5609fd30df75f9855dc9c9c00231300b2af00f89da910d1` |
| build_clean.log | `751cea0609e94afbce291432aa88dbbeb2551acad4f8b5e829f30e29f8bf232f` |
| out P4 CSV | `c81f18eeec26bcc96bc6b1e1c707b77a18286d1b4c45ba25c4076bdbaddb9f5b` |
| out clean CSV | `e21a53f513209bec7391212c12a1e53c47dbad2347f0c91c69ac97c369b45b18` |
| both P4 CSV | `04b9656e6e31cc685c1d203e73a43a3b1e0e70e1017d7413bfc61089152d1a20` |
| both clean CSV | `2f533955f1d9568a755fc321367450dc4dcaf280f7bf8b8661403cc8d3787b07` |
