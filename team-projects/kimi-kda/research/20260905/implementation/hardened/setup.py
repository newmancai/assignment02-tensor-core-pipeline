import os
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME

this_dir = os.path.dirname(os.path.abspath(__file__))
subprocess.run(["git", "submodule", "update", "--init", "cutlass"])


def is_flag_set(flag: str) -> bool:
    return os.getenv(flag, "FALSE").lower() in ["true", "1", "y", "yes"]


def get_nvcc_thread_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return ["--threads", nvcc_threads]


def get_experiment_flags():
    """Opt-in flags for isolated-kernel profiling builds."""

    flags = []
    if is_flag_set("FLASH_KDA_DISABLE_K1"):
        flags.append("-DBLOCK_LEVEL_K1=-1")
    if is_flag_set("FLASH_KDA_DISABLE_K2"):
        flags.append("-DBLOCK_LEVEL_K2=-1")
    value_slice = os.getenv("FLASH_KDA_K2_VALUE_SLICE")
    if value_slice:
        flags.append(f"-DK2_VALUE_SLICE={int(value_slice)}")
    return flags


SUPPORTED_CUDA_ARCHS = ["90a", "100a", "103a", "120a"]


def detect_cuda_arch():
    import torch

    if not torch.cuda.is_available():
        return None

    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return f"{major}{minor}a"


def get_arch_flags():
    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"

    requested = os.getenv("FLASH_KDA_CUDA_ARCHS", "auto").lower()
    if requested == "auto":
        arch = detect_cuda_arch()
        if arch is None:
            raise RuntimeError(
                "FLASH_KDA_CUDA_ARCHS=auto requires a visible CUDA device. "
                "Set FLASH_KDA_CUDA_ARCHS=all to build all supported archs."
            )
        archs = [arch]
    elif requested == "all":
        archs = SUPPORTED_CUDA_ARCHS
    else:
        archs = [arch.strip() for arch in requested.split(",") if arch.strip()]

    flags = []
    for arch in archs:
        flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
    return flags


def make_cuda_extension(name, experiment_flags):
    # The default slice is read by the C++ binding, not the CUDA translation
    # unit. Keep that define consistent without forwarding CUDA-only flags.
    binding_flags = [flag for flag in experiment_flags if flag.startswith('-DK2_VALUE_SLICE=')]
    return CUDAExtension(
        name=name,
        sources=[
            'csrc/flash_kda.cpp',
            'csrc/smxx/fwd_launch.cu',
        ],
        include_dirs=[
            os.path.join(this_dir, 'cutlass', 'include'),
            os.path.join(this_dir, 'cutlass', 'examples', 'common'),
            os.path.join(this_dir, 'cutlass', 'tools', 'util', 'include'),
            os.path.join(this_dir, 'csrc'),
        ],
        extra_compile_args={
            'cxx': ['-O3', '-Wno-psabi', *binding_flags],
            'nvcc': [
                '-O3',
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__',
                '-U__CUDA_NO_HALF2_OPERATORS__',
                '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                '--expt-relaxed-constexpr',
                '--expt-extended-lambda',
                '--use_fast_math',
                '--ptxas-options=-v,--register-usage-level=10,--warn-on-spills',
                '-lineinfo',
                *experiment_flags,
                *get_nvcc_thread_args(),
                *get_arch_flags(),
            ],
        },
    )


ext_modules = [
    make_cuda_extension(
        os.getenv('FLASH_KDA_EXTENSION_NAME', 'flash_kda_C'),
        get_experiment_flags(),
    )
]

# Backward-compatible profiling target.  Production dispatch uses the single
# flash_kda_C extension; this opt-in alias keeps isolated experiment scripts
# able to select V16 by default.
if is_flag_set("FLASH_KDA_BUILD_VSPLIT16"):
    if os.getenv("FLASH_KDA_K2_VALUE_SLICE") or is_flag_set("FLASH_KDA_DISABLE_K1") or is_flag_set("FLASH_KDA_DISABLE_K2"):
        raise RuntimeError(
            "FLASH_KDA_BUILD_VSPLIT16 cannot be combined with isolated-kernel "
            "or FLASH_KDA_K2_VALUE_SLICE experiment flags"
        )
    ext_modules.append(
        make_cuda_extension("flash_kda_vsplit16_C", ["-DK2_VALUE_SLICE=16"])
    )
cmdclass = {"build_ext": BuildExtension}

rev = os.getenv("FLASH_KDA_VERSION_SUFFIX", "")
if not rev:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        rev = "+" + subprocess.check_output(cmd, cwd=this_dir).decode("ascii").rstrip()
    except Exception:
        rev = ""

setup(
    name='flash_kda',
    version='0.0.1' + rev,
    description='FlashKDA: Flash Kimi Delta Attention',
    ext_modules=ext_modules,
    packages=['flash_kda'],
    cmdclass=cmdclass,
    zip_safe=False,
)
