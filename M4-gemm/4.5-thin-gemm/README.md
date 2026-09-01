# 4.5 · Thin GEMM

状态：待 C 实验。`05_thin_gemm.cu` 提供 shape 与 benchmark 骨架。

需要覆盖 decode 小 M、过渡区和较大 chunked-prefill M，记录理论 roof、实测
吞吐、相对 cuBLAS 表现，并解释小 M 时并行度和 launch/访存成本的影响。

