"""问题 5.2：block scaling 的两种乘回范围。

这里只用 fp64 模拟 scale 的代数位置，不模拟窄精度舍入。

gemm_scale_per_row_col:
    A 每行一个 scale，B 每行（即 GEMM 的每个输出列）一个
    scale。scale 乘积在整个 K 归约中不变，可以在完整点积后
    只乘回一次。

gemm_scale_along_k:
    scale 每 SEG 个 K 元素变一次。每个 K block 要先计算
    量化视角的 partial sum，乘回该段的 sA*sB，再累加到输出。

gemm_scale_along_k_one_restore:
    题面中的反例。它把不同 K block 的归一化 partial sum 先相加，
    最后只乘回第一段的 scale 乘积。因为 scale 乘积随 K block
    改变，这个因子不能从整个 K 和式中提出，结果应与参考不同。

从仓库根目录运行：
    python -m pytest M5-low-precision/5.2-block-scaling/test_block_scale.py -q
    python M5-low-precision/5.2-block-scaling/block_scale_sim.py
"""

import torch

SEG = 128


def gemm_fp64(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A.double() @ B.double().T


def gemm_scale_per_row_col(A: torch.Tensor, B: torch.Tensor,
                           sA: torch.Tensor, sB: torch.Tensor) -> torch.Tensor:
    """sA: [M]，sB: [N]，均为正数。

    先用 row/column scale 得到归一化的 A、B，完整点积后在输出
    [M, N] 上乘回 scale 外积。返回 fp64 结果。
    """
    M, N, _ = _validate_operands(A, B)
    sA64 = _validate_scales(sA, (M,), "sA", A.device)
    sB64 = _validate_scales(sB, (N,), "sB", B.device)

    qA = A.double() / sA64[:, None]
    qB = B.double() / sB64[:, None]
    normalized_gemm = qA @ qB.T
    return normalized_gemm * (sA64[:, None] * sB64[None, :])


def gemm_scale_along_k(A: torch.Tensor, B: torch.Tensor,
                       sA: torch.Tensor, sB: torch.Tensor) -> torch.Tensor:
    """sA: [M, K//SEG]，sB: [N, K//SEG]，均为正数。

    逐个 K block 计算归一化 partial sum，在段末乘回该段的
    sA*sB，再累加。返回 [M, N] 的 fp64 结果。
    """
    M, N, K = _validate_operands(A, B)
    if K % SEG != 0:
        raise ValueError(f"K must be divisible by SEG={SEG}, got K={K}")

    blocks = K // SEG
    sA64 = _validate_scales(sA, (M, blocks), "sA", A.device)
    sB64 = _validate_scales(sB, (N, blocks), "sB", B.device)
    A64 = A.double()
    B64 = B.double()
    output = torch.zeros((M, N), dtype=torch.float64, device=A.device)

    for block in range(blocks):
        sl = slice(block * SEG, (block + 1) * SEG)
        qA = A64[:, sl] / sA64[:, block, None]
        qB = B64[:, sl] / sB64[:, block, None]
        normalized_partial = qA @ qB.T
        block_scale = sA64[:, block, None] * sB64[None, :, block]
        output += normalized_partial * block_scale
    return output


def _validate_operands(
    A: torch.Tensor, B: torch.Tensor
) -> tuple[int, int, int]:
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError("A and B must be torch.Tensor objects")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"A and B must be rank-2, got {A.ndim} and {B.ndim}")
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"K mismatch: A is {tuple(A.shape)}, B is {tuple(B.shape)}")
    if A.device != B.device:
        raise ValueError(f"A and B must share a device, got {A.device} and {B.device}")
    if A.is_complex() or B.is_complex():
        raise TypeError("A and B must be real-valued tensors")
    return A.shape[0], B.shape[0], A.shape[1]


def _validate_scales(
    scale: torch.Tensor,
    expected_shape: tuple[int, ...],
    name: str,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(scale, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(scale.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(scale.shape)}"
        )
    if scale.device != device:
        raise ValueError(f"{name} must be on {device}, got {scale.device}")
    if scale.is_complex():
        raise TypeError(f"{name} must be real-valued")
    scale64 = scale.double()
    if not torch.isfinite(scale64).all().item() or not (scale64 > 0).all().item():
        raise ValueError(f"{name} must contain only finite positive values")
    return scale64


def gemm_scale_along_k_one_restore(A: torch.Tensor, B: torch.Tensor,
                                   sA: torch.Tensor,
                                   sB: torch.Tensor) -> torch.Tensor:
    """故意错误的对照：只在整个 K 归约后乘回第一段 scale。"""
    M, N, K = _validate_operands(A, B)
    if K % SEG != 0:
        raise ValueError(f"K must be divisible by SEG={SEG}, got K={K}")

    blocks = K // SEG
    sA64 = _validate_scales(sA, (M, blocks), "sA", A.device)
    sB64 = _validate_scales(sB, (N, blocks), "sB", B.device)
    A64 = A.double()
    B64 = B.double()
    normalized_sum = torch.zeros(
        (M, N), dtype=torch.float64, device=A.device
    )
    for block in range(blocks):
        sl = slice(block * SEG, (block + 1) * SEG)
        qA = A64[:, sl] / sA64[:, block, None]
        qB = B64[:, sl] / sB64[:, block, None]
        normalized_sum += qA @ qB.T
    return normalized_sum * sA64[:, 0, None] * sB64[None, :, 0]


def _demo() -> None:
    """打印正确恢复与故意错误恢复相对 fp64 GEMM 的数值误差。"""
    M, N, K = 7, 5, 512
    generator = torch.Generator().manual_seed(3)
    A = torch.randn(M, K, generator=generator, dtype=torch.float64)
    B = torch.randn(N, K, generator=generator, dtype=torch.float64)
    blocks = K // SEG
    sA_k = torch.rand(M, blocks, generator=generator, dtype=torch.float64) + 0.5
    sB_k = torch.rand(N, blocks, generator=generator, dtype=torch.float64) + 0.5
    sA_row = torch.rand(M, generator=generator, dtype=torch.float64) + 0.5
    sB_col = torch.rand(N, generator=generator, dtype=torch.float64) + 0.5

    reference = gemm_fp64(A, B)
    variants = {
        "row/col，点积后恢复一次": gemm_scale_per_row_col(
            A, B, sA_row, sB_col
        ),
        "K-block，每段恢复": gemm_scale_along_k(A, B, sA_k, sB_k),
        "K-block，错误地仅恢复一次": gemm_scale_along_k_one_restore(
            A, B, sA_k, sB_k
        ),
    }
    for label, result in variants.items():
        difference = (result - reference).abs()
        print(
            f"{label}: max_abs_error={difference.max().item():.6e}, "
            f"mean_abs_error={difference.mean().item():.6e}"
        )


if __name__ == "__main__":
    _demo()
