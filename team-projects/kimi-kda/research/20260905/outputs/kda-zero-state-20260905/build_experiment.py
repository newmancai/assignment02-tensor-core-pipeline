"""Isolated same-binary zero-init ablations; not an installer."""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

src = Path(os.environ['KDA_ZERO_SOURCE']).resolve()
cutlass = Path(os.environ['KDA_CUTLASS_SOURCE']).resolve()
setup(name='kda-zero-experiment', version='0.0.1',
      ext_modules=[CUDAExtension(
          name='flash_kda_zero_C',
          sources=[str(src/'csrc/flash_kda.cpp'),str(src/'csrc/smxx/fwd_launch.cu')],
          include_dirs=[str(src/'csrc'),str(cutlass/'include'),
                        str(cutlass/'examples/common'),str(cutlass/'tools/util/include')],
          extra_compile_args={
              'cxx':['-O3','-Wno-psabi','-DKDA_ZERO_ABLATION'],
              'nvcc':['-O3','-DKDA_ZERO_ABLATION','-DFLASH_KDA_ENABLE_V16_PREFETCH4=1',
                      '-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__',
                      '-U__CUDA_NO_HALF2_OPERATORS__','-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                      '--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math',
                      '--ptxas-options=-v,--register-usage-level=10,--warn-on-spills',
                      '-lineinfo','--threads','8','-gencode','arch=compute_103a,code=sm_103a']})],
      cmdclass={'build_ext':BuildExtension})
