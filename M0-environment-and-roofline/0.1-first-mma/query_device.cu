#include <cuda_runtime.h>
#include <cstdio>

int main() {
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return 1;
  std::printf("name=%s\n", p.name);
  std::printf("cc=%d.%d\n", p.major, p.minor);
  std::printf("sm_count=%d\n", p.multiProcessorCount);
  std::printf("memory_bus_width_bits=%d\n", p.memoryBusWidth);
  std::printf("global_memory_bytes=%zu\n", p.totalGlobalMem);
  return 0;
}
