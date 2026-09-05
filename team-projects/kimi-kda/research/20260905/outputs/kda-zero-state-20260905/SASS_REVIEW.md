# HasStateIn=false 的 V16/P4 SASS 独立复核

2026-09-05。只读源码及现有 release 二进制；经审批 SSH 到 `b300-login`，没有 GPU 作业，没有改动远端源、安装或生成远端报告。先读取 `cuobjdump --help`、`nvdisasm --help`，实际使用 CUDA 13.0.85 `cuobjdump --dump-resource-usage / --dump-sass --function`；反汇编由 CPU 完成。

## 最重要的发现

**两个实例的稳态 compute loop 并不相同，不能把差异归为一次性的初态加载/清零。** 无初态版本不仅少用 7 个寄存器，还在每个 token tile 内重取线程/CTA ID、重算部分不变量；Phase 1 多处 shared matrix load 紧接依赖它的 HMMA。有初态版本将更多不变量放在循环外，并在若干同一 key block 上给 load 更长的指令间隔。

两边每 tile 的矩阵计算量相同；Phase 6 都有 248 条线性 SASS 指令，HMMA、LDSM、STSM、FFMA 数量相同，但调度不同。这是“全函数编译改变了稳态调度/寄存器活跃区间”的直接静态证据，是性能差距的候选机制；**不是已测出的因果分解，也不能证明全部差距来自这一机制**。

## 精确对象与定位方法

远端账户 `YOUR_USER_ID`（UID USER_UID），二进制 owner UID USER_UID 匹配：

```text
/home/lcpu/YOUR_USER_ID/kda-mainline-20260905/release_build/lib/
  flash_kda_release_C.cpython-312-x86_64-linux-gnu.so
SHA256 34e2c68cf80de0bd24278afb035fdcd44e4a6205241dfcb9aab4ee95891bc486
ELF    fwd_launch.sm_103a.cubin
```

从 resource dump 取得完整 mangled symbol，再逐个传给 `--function`，没有用易混淆的中间 `bool` 或文件名匹配：

```text
无初态：...ELi16ELi128ELi16ELi3ELi2ELi96ELb0ELb1ELb0ELb0ELi4EEv...
有初态：...ELi16ELi128ELi16ELi3ELi2ELi96ELb1ELb1ELb0ELb0ELi4EEv...
            C16  D128  V16  IS3 OS2 NT96  SI   SO   FP   VL   P4
```

固定 `HasStateOut=true, StateFP32=false, IsVarlen=false`；只变 `HasStateIn`。完整符号、资源、原始 SASS 文本 hash、全函数 opcode 计数、初始化及 compute 的逐 PC 指令已保存到 [sass_targets.json](sass_targets.json)。未保留整函数冗长的 CuTe header/机器码；可用完整符号重新提取。以下阶段范围由已读源结构、矩阵操作及 loop 回边定位，未声称有 nvdisasm source-line annotation。

## 资源与矩阵工作量：更少寄存器不等于更快

| 指标 | HasStateIn=false | HasStateIn=true |
| --- | ---: | ---: |
| Cubin `REG` | 63 | 70 |
| Cubin `STACK / LOCAL` | 0 / 0 | 0 / 0 |
| Cubin `SHARED` | 1024 B | 1024 B |
| 全函数静态指令数 | 1472 | 1456 |
| Compute 主回边 | 0x4580 → 0x2940 | 0x4460 → 0x29e0 |
| 回边对应线性区域大小 | 453 指令 | 425 指令 |
| 循环内 `S2R` | 2 | 0 |
| 循环内 `HMMA.16816.F32.BF16` | 52 | 52 |
| 循环内 `LDSM.16.M88.4` | 27 | 27 |
| 循环内 `LDSM.16.MT88.4` | 16 | 16 |
| 循环内 `STSM.16.M88.4` | 1 | 1 |
| 循环内 `STSM.16.MT88.4` | 8 | 8 |
| 循环内 `FFMA.FTZ` | 66 | 66 |
| 循环内 `F2FP.BF16.F32.PACK_AB` | 49 | 49 |

这里的 `SHARED:1024` 是 cubin resource dump 字段，不是包含 launch-time dynamic shared 的 CTA 总量；不能据此称该 kernel 只用 1 KiB shared。主回边线性区域包含条件分支、不可达 trap 和退出路径，不含跳到区域外的 wait helper；453/425 是静态布局比较，不是每 tile 精确动态发射数量。不能把其 6.59% 差异直接当作延迟变化。

## 1. 无初态版本把循环不变量重计算放进热循环

无初态的回边每次跳到下面开头：

```text
2940 S2R R4, SR_TID.X
2950 IMAD.MOV.U32 R0, RZ, RZ, 0x1
2970 ISETP.GE.U32.AND P0, PT, R4.reuse, 0x40, PT
2980 ISETP.GE.U32.AND P1, PT, R4, 0x20, PT
2990..29f0  SEL / ISETP / LOP3 计算 role 相关条件
2a40 S2R R28, SR_CgaCtaId
2a50 MOV R0, 0x400
2a80 LEA R28, R28, R0, 0x18
...
2b90 IMAD.SHL.U32 R3, R4.reuse, 0x10, RZ
2bb0 IMAD.SHL.U32 R0, R4, 0x4, RZ
2bc0..2c90  lane、group、state/load 地址位运算
```

相对地，有初态版本在 **回边目标 0x29e0 之前** 的 0x27f0–0x29d0 已建立 lane/地址等不变量，例如 `R65/R67` 与 state 地址 `R34`；循环用保留的 `R0/R64/R65/...`，没有 `S2R`。这与 70 而非 63 个寄存器的资源结果相容。

源代码 [fwd_kernel2.cuh:451](../../implementation/phase6/csrc/smxx/fwd_kernel2.cuh#L451) 开始 compute 区，`HasStateIn` 并未在 tile 循环中另写一套公式。源码共用并不保证两个模板实例生成相同 steady-state body；这次二进制明确没有做到。

## 2. Phase 1 load→HMMA 间隔的具体反例

以下对齐同一 key block 4：k 的 shared offset 为 `0x1a80`、q 为 `0x2a80`、state 为 `0x800`。寄存器与基址不同是正常的寄存器分配结果。

```text
HasStateIn=false
2f60 LDSM.16.M88.4 R8,  [R32+0x1a80]   // k4
2f70 LDSM.16.M88.4 R20, [R35+0x800]    // S4
2f80 HMMA.16816.F32.BF16 R36, R8.reuse, R20, R36
2f90 HMMA.16816.F32.BF16 R24, R8,       R22, R24
2fa0 LDSM.16.M88.4 R8,  [R32+0x2a80]   // q4
2fb0 HMMA.16816.F32.BF16 R16, R8.reuse, R20, R16

HasStateIn=true（省略不相关的中间指令）
2de0 LDSM.16.M88.4 R16, [R34+0x800]    // S4
2e10 LDSM.16.M88.4 R8,  [R6+0x1a80]    // k4
2e40 LDSM.16.M88.4 R20, [R6+0x2a80]    // q4
2e50 LDSM.16.M88.4 R12, [R34+0xa00]    // 提前加载 S5
2e60 HMMA.16816.F32.BF16 R36, R8.reuse, R16, R36
2e90 HMMA.16816.F32.BF16 R24, R20,      R16, R24
```

| 到上述首条消费 HMMA 的间隔 | 无初态 | 有初态 |
| --- | ---: | ---: |
| S4 load 后夹着的指令数 | 0 | 7 |
| k4 load 后夹着的指令数 | 1 | 4 |
| q4 load 后夹着的指令数 | 0 | 4 |

这是 PC 间的静态指令间隔，**不是等待 cycles 或实测延迟**；没有解码 instruction control fields，也没有 PC 级 NCU 计数器。它精确表明相同计算存在不同的 load/use 安排，因此比“寄存器多所以快”的泛化更接近可检验假设。

## 3. Phase 6 P4 仍在，但编译后的排序不同

无初态：`0x3400..0x4370`；有初态：`0x32b0..0x4220`。两段均 248 条、无分支，且各有：16 条转置 LDSM、16 条 scalar LDS（gate）、16 条 HMMA、64 条 FFMA、32 条 BF16 pack、8 条转置 STSM。SHF/IMAD 的等价转换组合分别为 32/32 与 33/31；不是逐字节相同。

例如无初态在 `0x3930` 才载入 `k_restored` block 3（offset `0x3880`），在 `0x39f0` 开始消费；有初态对应 `0x36c0→0x38c0`，间隔更长。首个 state STSM 分别在 `0x3b90`、`0x3a50`。这些位置说明源级 P4 ring 的加载/计算/写回被编译器重新交错，而不能把“源码 PREFETCH=4”理解为硬件严格执行四块完整 prologue 后才开始 compute。

这一段没有出现额外 state store、少掉 MMA、改用其它精度或 P1 模板的证据；是否哪一种重排在目标输入上更好，仍需要匹配状态语义的 runtime 对比。

## 4. 初始化与 spill 的边界

[源码:265](../../implementation/phase6/csrc/smxx/fwd_kernel2.cuh#L265) 的有初态路径用 TMA + barrier 获取初态；[源码:330](../../implementation/phase6/csrc/smxx/fwd_kernel2.cuh#L330) 的无初态路径用 96 线程跨步写 BF16 零，随后 fence + `__syncthreads()`。SASS 无初态全函数有 29 个静态 `STS.U16` site，包含分段展开与残余循环；有初态没有这些清零 stores。它们在 tile 循环之前，不能仅凭数量解释随长序列重复累积的差距；另一方面，静态分析也不能排除一次性初始化开销及它对 pipeline 起步的影响。

两边 resource `STACK/LOCAL=0`，整函数均未找到 `LDL`/`STL`。不要把下列名字误报为 local-memory spill：

```text
无初态 1240 MOV.SPILL R18, UR47
       1350 R2UR.FILL UR47, R18
```

这些可见操作数是 uniform/general 寄存器间转移，全函数有 42/33 对对应 opcode site，但选中的 compute 主循环两侧都没有 `MOV.SPILL/R2UR.FILL`。此前构建日志的两个 P4 目标也均为 0 bytes spill stores/loads；这里没有发现新的内存 spill 解释。

## 对主线实验的行动建议

1. 首先让 `initial_state=None` 与显式 BF16 全零 initial state 使用完全相同的 q/k/v/g/beta、state-out、shape 与强制/auto slice；逐位比较 out/final，再做交错重复 timing。否则非零 vs 零初态与入口差异混在一起，不能归因。
2. 优先验证“稳定的循环不变量保留 + Phase 1 load/use lookahead”，而非只减少清零指令。判定候选是否实现目标，要检查新 binary 的回边和 load/use 排序，不能只看源代码把变量写到了循环外或 register count 变小。
3. 可将显式零初态走既有有初态 kernel 作为诊断对照；不要据此直接引入生产用 dummy buffer/额外 allocation/TMA 路由。额外缓冲、生命周期、graph capture、packed 和并发语义都需要单独验证。
4. 把当前归因写成“两个模板实例有可定位的编译调度差异，正在进行匹配输入验证”。本复核没有 timing、PC stall、clock/cache 控制或任何 GPU 执行证据，不能承诺修复收益，更不能从 fixed/state-out=true 推广到其它组合。

## 复现命令骨架

完整符号见本地 JSON；以下均是只读 dump，不需 `-xelf` 提取远端文件：

```text
/usr/local/cuda-13.0/bin/cuobjdump --help
/usr/local/cuda-13.0/bin/nvdisasm --help
/usr/local/cuda-13.0/bin/cuobjdump --list-elf <existing-release.so>
/usr/local/cuda-13.0/bin/cuobjdump --dump-resource-usage <existing-release.so>
/usr/local/cuda-13.0/bin/cuobjdump --dump-sass --function <exact-symbol> <existing-release.so>
```
