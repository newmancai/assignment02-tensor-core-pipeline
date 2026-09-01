"""问题 5.1:per-tensor scale 与 outlier。

构造一个张量:一万个元素均匀分布在 [-1, 1],外加一个 3000 的
outlier。按 per-tensor 方式量化到 E4M3(scale = amax / 448,cast 用
torch.float8_e4m3fn),反量化后测逐点相对误差,填题面的表并回答三问。

需要动手的是下面两个 TODO;跑法:
    uv run python kernels/quant_outlier.py
输出直接用于报告,没有自动判测。
"""

import torch

E4M3_MAX = 448.0


def build_tensor(n: int = 10000, outlier: float = 3000.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n, generator=g) * 2 - 1
    return torch.cat([x, torch.tensor([outlier])])


def quant_dequant_per_tensor(x: torch.Tensor) -> torch.Tensor:
    """per-tensor E4M3 量化再反量化。

    TODO: 实现。步骤:算 scale = amax / 448;除 scale 后 cast 到
    torch.float8_e4m3fn;cast 回 float 再乘 scale。
    """
    raise NotImplementedError


def rel_err_at(x: torch.Tensor, y: torch.Tensor, value: float) -> float:
    """取 x 中最接近 value 的元素,返回该点的相对误差。

    TODO: 实现(表格的每一格都从这里来)。
    """
    raise NotImplementedError


def main() -> None:
    x = build_tensor()
    y = quant_dequant_per_tensor(x)
    print("含 outlier:")
    for v in (0.5, 0.1, 0.01, 0.005, 3000.0):
        print(f"  x≈{v:<8} rel_err={rel_err_at(x, y, v):.3e}")
    # (a) 去掉 outlier 重新量化,对比 0.5 处的误差
    # (b) 找出被量化成 0 的阈值,写出它与 scale 的关系式
    # (c) 换 1x128 的 per-block scale,对比含/不含 outlier 的 block
    # 这三问自己补代码,结果写进报告。


if __name__ == "__main__":
    main()
