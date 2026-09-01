# 5.3(b) · NVFP4 quant kernel

状态：待 C 完成。核心 kernel 与 launch 配置仍为 TODO。

目录包含量化入口、E2M1 编码、scale-factor swizzle 工具以及 FP4 GEMM 消费端
测试。需要同时验证 packed data 与 E4M3 scale-factor 布局。

