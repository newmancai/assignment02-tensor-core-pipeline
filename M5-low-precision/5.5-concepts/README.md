# 5.5 · W4A16 与 NVFP4

状态：已完成。

## (a) 存储量化还是计算量化

W4A16 + Marlin 属于存储量化：权重以 INT4 保存和搬运，在 kernel 内解码/反量化
后仍以 FP16 激活和 FP16 Tensor Core 路径完成计算。NVFP4 属于计算量化：权重和
激活都按 K 向 16 元素分组，并由 block-scaled FP4 Tensor Core 直接消费 E2M1
数据与 E4M3 scale。

## (b) 节省的资源

W4A16 直接减少权重显存容量和权重读取带宽，但不压缩 FP16 激活，也没有获得
FP4 Tensor Core 的计算吞吐；它还要付出在线反量化和 scale 读取开销。NVFP4
同时减少量化后权重/激活的容量和搬运字节，并提高低精度 Tensor Core 吞吐，代价
是 activation quant、更多 block scale 以及更严格的数据布局。

## (c) 小 batch decode

4.5 的小 M decode 中，同一批 token 对大权重矩阵的复用很少，瓶颈通常先落在
权重带宽和 CTA 并行度，而不是 Tensor Core 峰值。因此 W4A16 的权重压缩收益更
直接；NVFP4 的更高计算峰值只有在 M 增大、并行度和数据复用足够时才更容易兑现，
小 M 下还可能被量化与启动开销抵消。

