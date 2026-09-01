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

补全前两个函数后运行：
    uv run pytest tests/test_block_scale.py
"""

import torch

SEG = 128


def gemm_fp64(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A.double() @ B.double().T


def gemm_scale_per_row_col(A: torch.Tensor, B: torch.Tensor,
                           sA: torch.Tensor, sB: torch.Tensor) -> torch.Tensor:
    """sA: [M]，sB: [N]，均为正数。

    TODO: 先用 row/column scale 得到归一化的 A、B，完整点积后
    在输出 [M, N] 上乘回 scale 乘积。
    """
    raise NotImplementedError


def gemm_scale_along_k(A: torch.Tensor, B: torch.Tensor,
                       sA: torch.Tensor, sB: torch.Tensor) -> torch.Tensor:
    """sA: [M, K//SEG]，sB: [N, K//SEG]，均为正数。

    TODO: 逐个 K block 计算归一化 partial sum，在段末乘回
    该段的 sA*sB，再累加。返回 [M, N] 的 fp64 结果。
    """
    raise NotImplementedError


def gemm_scale_along_k_one_restore(A: torch.Tensor, B: torch.Tensor,
                                   sA: torch.Tensor,
                                   sB: torch.Tensor) -> torch.Tensor:
    """故意错误的对照：只在整个 K 归约后乘回第一段 scale。"""
    M, K = A.shape
    N = B.shape[0]
    normalized_sum = torch.zeros((M, N), dtype=torch.float64)
    for block in range(K // SEG):
        sl = slice(block * SEG, (block + 1) * SEG)
        qA = A[:, sl].double() / sA[:, block, None]
        qB = B[:, sl].double() / sB[:, block, None]
        normalized_sum += qA @ qB.T
    return normalized_sum * sA[:, 0, None] * sB[None, :, 0]
