# 构建与复现

## CUDA

服务器上的推荐环境：

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
```

B300 使用显式 family target：

```bash
nvcc -std=c++17 -O3 \
  -gencode arch=compute_100f,code=sm_100f \
  source.cu -o program
```

不要把 `sm_120a` 生成物拿到 B300 上运行；它可以编译成功，但运行时没有
兼容的 kernel image。

## A 部分常用判测

每条命令都从仓库根目录执行：

```bash
mkdir -p build

g++ -std=c++17 -O2 -x c++ \
  M1-fragment-and-mma/1.1-fragment-map/01_fragment_map.cu \
  -o build/m1_1_fragment_map

g++ -std=c++17 -O2 -x c++ \
  M2-descriptor-and-swizzle/2.2-descriptor/02_descriptor.cu \
  -o build/m2_2_descriptor

g++ -std=c++17 -O2 -x c++ \
  M2-descriptor-and-swizzle/2.3-swizzle/03_swizzle.cu \
  -o build/m2_3_swizzle
```

需要 GPU 的 M0/M1 题使用上面的 `nvcc` 命令编译；相对路径中的
`../common.h` 指向各模块根目录内的公共头文件。

## Python / TileLang

```bash
uv sync --extra tilelang
uv run python M6-tilelang/6.1-lowering/m6_tilelang_lowering.py
```

生成结果归档在 `M6-tilelang/6.1-lowering/generated/`。

