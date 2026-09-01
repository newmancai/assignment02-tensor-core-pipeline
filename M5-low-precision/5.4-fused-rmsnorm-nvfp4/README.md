# 5.4 · Fused RMSNorm + NVFP4

状态：待 C 完成。融合 kernel 与公平的两步基线仍为 TODO。

最终结果要逐 shape 比较 fused、两步 baseline 与 5.3(c) ceiling probe，并把
收益分解为中间张量流量减少、launch 次数变化和额外归约/量化开销。

