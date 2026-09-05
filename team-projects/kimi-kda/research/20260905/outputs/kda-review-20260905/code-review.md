# KDA 独立代码审查（2026-09-05）

范围：`assignment02-work/team/c1_flashkda/FlashKDA` 上游树、提交包 `0001` / `0002` 补丁，以及 `experiments/final_campaign/implementation/current/flash_kda`。独立审查者未修改生产代码、未连接远端；本报告的远端实测证据由主审在 B300 Job 19844 执行并反馈。把上游源文件复制到 `/private/tmp/kda-review-code.PDkQDX` 并成功应用两份补丁；得到的 Python wrapper 与提交包 current 快照逐字一致。下文明确区分本地已复现、主审 GPU 已复现、静态可推导行为与仍待验证的疑点。

## 1. [P2，本补丁引入，已在本地复现] 编译时 V16 默认值没有传到 C++ binding

位置：`assignment02-github/team-projects/kimi-kda/patches/0001-k2-value-slice-and-dispatch.patch:893`、`:918`，对应应用补丁后 `setup.py:83`、`:95`、`:120`。读取宏的代码位于补丁开头及应用后 `csrc/flash_kda.cpp:5`。

`make_cuda_extension(..., ["-DK2_VALUE_SLICE=16"])` 只把 experiment flags 加入 `extra_compile_args['nvcc']`；`cxx` 始终是 `['-O3', '-Wno-psabi']`。但 `kDefaultK2ValueSlice` 的宏分支在由 C++ 编译器处理的 `csrc/flash_kda.cpp` 中。因此使用 `FLASH_KDA_BUILD_VSPLIT16=1` 构建的 alias，或者使用编译期 `FLASH_KDA_K2_VALUE_SLICE=16` 构建的扩展，binding 默认值依然是 128。nvcc 编译单元内没有读取该宏来改变运行时默认选择。

触发：直接调用 `flash_kda_vsplit16_C.fwd(...)`，不显式传 `k2_value_slice`，依赖文档声明的 V16 默认值。结果会走 V128。若运行时显式传 `k2_value_slice=16`，或 Python wrapper 读取运行时环境变量并传值，则不受影响。不能据此推翻已明确记录 kernel 模板实参或显式传 slice 的生产 benchmark。

本地可重复验证：

```sh
python3 outputs/kda-review-20260905/code-review-proof.py /private/tmp/kda-review-code.PDkQDX
```

输出要点：`cxx_flags=["-O3","-Wno-psabi"]`，`nvcc_value_slice_flags=["-DK2_VALUE_SLICE=16"]`，`binding_without_macro_defaults_128=true`，`finding_reproduced=true`。该脚本仅抽取 AST 中的工厂函数并使用字典 stub，不运行 setup、不 import PyTorch、不构建 CUDA。

修复方向：把用于 binding 默认值的 define 同时传入 `cxx`，或取消这种编译期默认值机制，让 alias wrapper 显式传 `k2_value_slice=16`。构建回归应检查 `fwd.__doc__` 的默认参数或 profiler 中实际实例化的 V，而不能只检查模块名。

## 2. [P1，上游继承，静态确定，GPU 复现交主审] CUDA device guard 缺失

位置：上游 `assignment02-work/team/c1_flashkda/FlashKDA/csrc/flash_kda.cpp:136`；应用补丁后 `csrc/flash_kda.cpp:180`。检查入口只有 `is_cuda()`，没有对所有 tensors 的 device 一致性检查，也没有 `CUDAGuard(q.device())`。

`at::cuda::getCurrentCUDAStream()` 不传设备参数，得到的是当前 CUDA device 的 stream。若全部合法连续输入均位于 `cuda:1`，但 Python 当前设备为 `cuda:0`，ATen 的 beta 转置可以在 tensor 所属设备处理后恢复设备，后续自定义 CUDA launch 却仍使用当前 device 0 的 stream 和 device 1 的裸指针。peer access、UVA、设备异构与 stream 配置不同会改变具体症状，因此不能把错误模式限定成某一种异常；确定的是执行设备和依赖 stream 已错误。

新增 dispatcher 中 `_device_characteristics` 的 `with torch.cuda.device(device_index)` 仅覆盖属性查询；离开该函数后恢复原设备，对 `_fwd_raw` 没有保护。普通的“同设备 non-default stream”本身不因这一点出错，不能将问题泛化为所有非默认 stream 都错。

复现要点：先在 device 1 创建所有输入并同步，将当前设备切回 0，再调用 wrapper；与整个调用位于 `with torch.cuda.device(1)` 内的结果对比。应在独立子进程做，避免失败 CUDA context 影响其他实验。修复应在 C++ 入口持有 `CUDAGuard` 并校验所有数据/状态/cu_seqlens/workspace 设备一致性，不能只在 Python 属性查询处切换设备。

## 3. [P2，上游继承，B300 Job 19844 已复现] 空 packed 序列会改变 FP32 初始 state

位置：上游 `csrc/smxx/fwd_kernel2.cuh:300` / `:811` 附近；应用补丁后 `csrc/smxx/fwd_kernel2.cuh:322` 与 `:838`。

所有 FP32 初始 state 在进入 recurrence 前先转为 BF16，最终 FP32 state 再由 BF16 扩回 FP32。这意味着“支持 FP32 state buffer”不等于“用 FP32 保存 recurrent state 精度”。这是上游既定数值实现，并非 ValueSlice 改变了舍入顺序。

一个清晰边界是 `cu_seqlens=[0,0,16]`：第一条序列没有 token，recurrence 的恒等语义应为 `S_out[0]=S_in[0]`，但初态值 `1.001f` 会变成 `1.0f`。现有代码明确处理 `t_tiles==0` 的 state store，因此不能简单认为空序列永远无法到达。

主审在 B300 Job 19844 已确认空 FP32 state 从 `1.00100004673` 变为 `1.0`，V128 与 V16 行为相同，符合上述静态推导。复现应比较空条目 `final_state[0]` 和 `initial_state[0]` 的逐位恒等，而不要用全 tensor RMSE 淹没空序列的变化。可用的最小形状是 T=16、H=1，两个 sequence 的 initial/final state 为 `[2,1,128,128]`、dtype FP32，初态填 1.001、cu `[0,0,16]`。

修复或契约选择：若支持空序列的精确恒等，对空条目直接保留/拷贝原 dtype 初态；若继续采用 BF16 state 语义，公开说明量化边界，并明确测试精度要求。长序列或 loop 扩展的验证应区分 IO dtype、状态存储 dtype、MMA accumulate dtype 三者。

## 4. [P2，上游继承，输入拒绝缺口] 非连续 cu_seqlens 被当作连续裸数组

位置：上游 `csrc/flash_kda.cpp:151-157`；应用后 `:195-201`。这段检查 CUDA、int64、1D 和元素数量，却没有检查 contiguous / stride，随后直接传 `data_ptr<int64_t>()`，GPU 以 `ptr[i]` 读取。

安全的复现数据：`backing=torch.tensor([0,0,16,16], device='cuda', dtype=torch.int64)`，`cu=backing[::2]`。逻辑 cu 值为合法的 `[0,16]`，但裸指针前两个元素是 `[0,0]`。T=16 的 output 预填 sentinel，调用会按空序列处理，output 没被写入。这里不必引入越界端点，即可复现错误。

Python docstring 已要求所有输入 contiguous，因此这是未正确拒绝不受支持输入的缺口，不能报告为“违反已承诺的任意 stride 支持”。修复可以是 C++ 明确拒绝，或 wrapper 做 `.contiguous()`；若目标是 CUDA graph/零拷贝，应优先明确拒绝并要求 caller 满足契约。

## 5. [P1，上游继承，B300 Job 19844 已复现] 合法连续 beta view 可导致进程 abort

最直接位置：上游 `assignment02-work/team/c1_flashkda/FlashKDA/csrc/flash_kda.cpp:131`，应用补丁后 `csrc/flash_kda.cpp:175`。前置契约检查在上游 `:46`，应用后 `:86`；beta 的 TMA descriptor 在上游 `csrc/smxx/fwd_launch.cu:89` / `:107`，应用后 `:94` / `:112`。

主审在 T=256、H=2 的逐 token segmentation 实验中，第二个 token 的 beta view 触发 CUTLASS `copy_traits_sm90_tma.hpp:949` 的 CPU assertion，进程 abort。beta view 形状 `[1,1,H]` 已连续，`.contiguous()` 返回原 storage；随后 `beta_2d.t().contiguous()` 因 `T_total=1` 也已连续，再次不拷贝。第二 token 的实际 `data_ptr()` 相对原始对齐 allocation 偏移 `H*2=4` 字节，不满足 TMA descriptor 的 16 字节地址对齐要求。

这不是“不支持任意 stride”的重复 finding：触发输入满足现有 CUDA、dtype、shape、contiguous 契约。`is_contiguous()` 只描述 stride，不保证 storage_offset 为 0，更不保证 TMA base 对齐。该反例落在正常 streaming/segmented 调用路径，而非人为损坏地址。主审随后完成 Job19845 的独立子进程对照：同一形状的 contiguous beta view 地址模16为2，returncode为−6；clone后地址模16为0，调用正常且输出finite。完整证据见 [followup_19845.log](followup_19845.log)，补充任务执行到 `kind=complete`。

这种隐式拷贝失效不只发生在 H=2：只要 `T_total=1` 或 `H=1`，二维转置可以继续保持连续；若实际地址不齐，就会触发。逐 token H=12 时第一步之后偏移 24 字节，奇数 token 也不齐；H 为 8 的倍数时每步偏移恰好是 16 字节的倍数，会掩盖问题。应以 `uintptr_t(data_ptr()) % 16` 检查实际地址，不能把 `storage_offset != 0` 一概视为错误。

K1 的 `fwd_kernel1.cuh:221-224` 和 K2 原始 `fwd_kernel2.cuh:362-365`（应用后 `:383-386`）确实有 `beta_aligned = beta_linear & ~7`。它只将相对 beta base 的 BF16 元素索引向 8 对齐，不会改变 base pointer，本身无法修复这个问题。失败发生在 host 构造 descriptor 时，也早于这段设备端逻辑。

### 相关指针的完整对齐审计

下表位置均为上游 `csrc/smxx/fwd_launch.cu` 的 descriptor 构造行；在应用后版本中，这些 descriptor 位置分别增加 5 行。D 固定 128，现有非最内维 global strides 都已是 16 字节的倍数；此次缺口主要是 tensor view 的 base address。

| 实际传入 descriptor 的指针 | 来源和位置 | 连续但未对齐 view 风险 | 修复注意点 |
|---|---|---|---|
| q / k | `flash_kda.cpp:120-121`；launch `:87-88` | BF16 平坦 storage_offset 不是 8 的倍数即可不齐；常规整 token 切片偏移是 `256*H` 字节，因此对齐不被破坏 | 只读，必要时分配对齐 contiguous 副本；不应仅再次 `.contiguous()` |
| g / v | `flash_kda.cpp:122-123`；launch `:92` / `:106` | 与 q/k 同样的 base 风险 | 同上 |
| beta_t | `flash_kda.cpp:131-132`；launch `:89` / `:107` | 已 GPU 复现；当转置免拷贝时保留原 beta offset | 在最终 beta_t 上检查地址；不齐才 `.clone(Contiguous)`，避免给正常多头 prefill 重复拷贝 |
| dt_bias | `flash_kda.cpp:127`；launch `:96` | FP32 flat offset 不是 4 的倍数即可不齐；普通整行 head 切片偏移 512 字节，安全 | 只读，必要时对齐副本 |
| initial_state | `flash_kda.cpp:141`；launch `:126` / `:139` | BF16 offset 非 8 倍数或 FP32 offset 非 4 倍数不齐；普通完整 state/head 切片仍对齐 | 只读，必要时对齐副本；保留与 final_state alias 的读前写后语义 |
| out | `flash_kda.cpp:125`；launch `:116` | 任意连续 offset view 仍可能不齐；即便 T=1 最后实际走标量 tail store，host 仍构造 TMA descriptor | 必须写回 caller 原 output，不能简单替换为 clone 后丢失结果 |
| final_state | `flash_kda.cpp:142`；launch `:127` / `:140` | 与 initial_state 相同 | 若走对齐 scratch，CUDA kernel 后在同 stream `copy_` 回原 buffer |
| workspace 六段 | `flash_kda.cpp:134`；launch `:64-69`、`:98-103`、`:109-114` | 六段偏移都是 128 字节倍数，因此继承 workspace base 的对齐；Python wrapper 的 fresh empty 正常对齐，raw extension 的 offset workspace 仍可能不齐 | raw 接口校验 base 和容量；不能默默换掉用于隔离 K2 重放的外部 workspace |
| A_log | `flash_kda.cpp:126`；K1 `fwd_kernel1.cuh:257` 标量读取 | 不构造 TMA descriptor，普通 FP32 自然对齐即可 | 不应因 TMA 的要求无端强制 16 字节对齐 |
| cu_seqlens | `flash_kda.cpp:157`；普通 int64 读取 | 不构造 TMA descriptor，连续 int64 view 保留 8 字节自然对齐；另有第 4 项 stride 检查缺口 | 需校验 contiguous/device，但不能把 storage_offset 为 1 自动判为 TMA 对齐错误 |

最小修复建议：先持有 device guard，在 beta 转置完成之后、取得 `beta_t_ptr` 之前，若 `(reinterpret_cast<uintptr_t>(beta_t.data_ptr()) & 15) != 0`，执行 `beta_t = beta_t.clone(at::MemoryFormat::Contiguous)`。这保留已对齐常见路径和 Python 异步 stream 语义，无需 host synchronize。

完整 API 修复应为所有实际参与 TMA 的 bases 建立明确策略：保持现有 contiguous-view 支持时，对不齐只读输入做对齐副本，对 output/final_state 使用对齐 scratch 后同 stream 写回；raw workspace 可以明确拒绝。若选择收紧契约而只允许 16 字节对齐，应在 C++ 入口用可捕获的 `TORCH_CHECK` 拒绝，并同步文档，至少消除 CUTLASS assert 杀掉 Python/服务进程的行为。不能把输入 `.contiguous()`、`reshape()`、尾块标量写回或相对索引对齐当作 base 对齐保证。

## ValueSlice 核心审查结论

没有从静态审查中发现可成立的新增 slice 算术错误。主要映射如下：

| V | slices/seq/head | compute warps | total threads | 16-wide blocks/warp |
|---:|---:|---:|---:|---:|
| 16 | 8 | 1 | 96 | 1 |
| 32 | 4 | 2 | 128 | 1 |
| 64 | 2 | 4 | 192 | 1 |
| 128 | 1 | 4 | 192 | 2 |

输出和 state 的 `value_offset` 同步出现在 TMA loads/stores、尾块标量写回和初态/终态读写中；workspace 仍保留 D=128 的 K 维中间量，新增的 `TMAWorkspaceSmemLayout` 避免错误复用缩小后的 V layout；FP32 转换循环按 Rows=V / Cols=D 覆盖所有 8×8 原子；barrier consumer 数量随 compute warps 变化。V 切片之间不合并 partial reductions，因此保持每个 value row 的 K 维运算顺序是合理的数值设计。

这只是审查结论，不替代针对尾块、空序列、单侧 state、异设备等路径的 GPU 验证；已有 shape parity 也不自动覆盖这些集成边界。

## 可实施的性能 Insight：K2 重复扫描已经存在的 prefix

应用后 `csrc/smxx/fwd_launch.cu:76` 分配 `ws_tile_prefix`，`:170` 已由 GPU kernel 生成 prefix；但 `csrc/smxx/fwd_kernel2.cuh:246-249` 对每个 `seq_idx` 重新线性扫描所有前置 `cu_seqlens`。

以 CTA 逻辑工作量计，前置扫描总迭代数为 `H × (128/V) × N(N-1)/2`。它是上游的 O(N²) 序列元数据路径，ValueSlice 又按 slices 复制。它不影响 N=1 长 prefill 的快照，却可能在大量短 packed 请求中成为边界条件。现 dispatcher 对 multi-sequence varlen 回退 V128 是合理的保守行为，但仍保留该扫描开销。

低风险优化候选：把现成 `ws_tile_prefix` 传入 K2，用 `tile_base=ws_tile_prefix[seq_idx]` 代替扫描，无需新增 kernel、CPU 拷贝或 host synchronize。需单独验证隔离 K2 的实验构建怎样初始化 prefix，不能直接假定生产 K1 在该构建中存在。收益应扫 N 与 sequence-length 分布，而不是只扫 total tokens。

## 尚未提升为 finding 的异步 store 问题

应用后 `csrc/smxx/fwd_kernel2.cuh:827` / `:860` 对 final state 调用 `tma_store_arrive()`，末尾只有 `__syncthreads()`，未显式 `tma_store_wait<0>()`。相比之下 K1 对 workspace 的 TMA stores 在原始 `fwd_kernel1.cuh:586` 有显式等待。这是上游继承的规范/同步审查点。这里尚未证明它在 CTA 生命周期规则下导致错误，须查清 TMA completion 规则并用 sanitizer/压力复现后才能列 confirmed bug；不要将单次 parity PASS 当作该同步点已获得证明。
