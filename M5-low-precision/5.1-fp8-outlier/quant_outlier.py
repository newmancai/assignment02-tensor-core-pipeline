"""问题 5.1：per-tensor scale 与 outlier。

构造一个张量:一万个元素均匀分布在 [-1, 1],外加一个 3000 的
outlier。按 per-tensor 方式量化到 E4M3(scale = amax / 448,cast 用
torch.float8_e4m3fn),反量化后测逐点相对误差,填题面的表并回答三问。

从仓库根目录运行：
    python M5-low-precision/5.1-fp8-outlier/quant_outlier.py

程序同时给出题面五个采样点、移除 outlier 的对照，以及 1x128
per-block scale 的误差范围；输出可直接用于报告。
"""

from dataclasses import dataclass

import torch

E4M3_MAX = 448.0
E4M3_MIN_SUBNORMAL = 2.0 ** -9
DEFAULT_BLOCK_SIZE = 128


def build_tensor(n: int = 10000, outlier: float = 3000.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n, generator=g) * 2 - 1
    return torch.cat([x, torch.tensor([outlier])])


def quant_dequant_per_tensor(x: torch.Tensor) -> torch.Tensor:
    """用单一绝对最大值 scale 完成 E4M3 量化与反量化。

    计算均在 fp32 中进行，返回值也是 fp32。全零张量单独处理，避免
    ``0 / 0``；空张量保持为空。输入必须是有限的实数张量。
    """
    work = _as_finite_fp32(x)
    if work.numel() == 0:
        return work.clone()

    amax = work.abs().amax()
    if amax.item() == 0.0:
        return torch.zeros_like(work)

    scale = amax / E4M3_MAX
    quantized = (work / scale).to(torch.float8_e4m3fn)
    return quantized.float() * scale


def rel_err_at(x: torch.Tensor, y: torch.Tensor, value: float) -> float:
    """返回 ``x`` 中最接近 ``value`` 的元素对应的逐点相对误差。"""
    if x.shape != y.shape:
        raise ValueError(f"x/y shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    if x.numel() == 0:
        raise ValueError("x and y must be non-empty")

    x_flat = x.detach().float().reshape(-1)
    y_flat = y.detach().float().reshape(-1)
    idx = (x_flat - float(value)).abs().argmin()
    reference = x_flat[idx].abs()
    absolute_error = (y_flat[idx] - x_flat[idx]).abs()
    if reference.item() == 0.0:
        return 0.0 if absolute_error.item() == 0.0 else float("inf")
    return (absolute_error / reference).item()


def quant_dequant_per_block(
    x: torch.Tensor, block_size: int = DEFAULT_BLOCK_SIZE
) -> torch.Tensor:
    """沿扁平元素维按 ``block_size`` 独立进行 E4M3 量化与反量化。

    最后一个 block 可以不足 ``block_size``。每段独立调用 per-tensor
    实现，因此全零段也具有定义良好的行为。
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    work = _as_finite_fp32(x)
    flat = work.reshape(-1)
    restored = torch.empty_like(flat)
    for start in range(0, flat.numel(), block_size):
        stop = min(start + block_size, flat.numel())
        restored[start:stop] = quant_dequant_per_tensor(flat[start:stop])
    return restored.reshape(work.shape)


def _as_finite_fp32(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.is_complex():
        raise TypeError("x must be a real-valued tensor")
    work = x.detach().to(torch.float32)
    if work.numel() and not torch.isfinite(work).all().item():
        raise ValueError("x must contain only finite values")
    return work


def _scale_and_zero_boundary(x: torch.Tensor) -> tuple[float, float]:
    """返回 ``scale`` 和 round-to-nearest 下落入零的理论边界。"""
    if x.numel() == 0:
        return 0.0, 0.0
    scale = x.detach().float().abs().amax().item() / E4M3_MAX
    # E4M3 最小 subnormal 为 2^-9；与 0 的中点为 2^-10。
    return scale, scale * E4M3_MIN_SUBNORMAL / 2.0


@dataclass(frozen=True)
class ErrorSummary:
    count: int
    scale: float
    zero_boundary: float
    mean_abs_error: float
    max_abs_error: float
    zero_fraction: float


def _summarize_block(
    x: torch.Tensor, y: torch.Tensor, scale_source: torch.Tensor | None = None
) -> ErrorSummary:
    scale, zero_boundary = _scale_and_zero_boundary(
        x if scale_source is None else scale_source
    )
    abs_error = (y.float() - x.float()).abs()
    return ErrorSummary(
        count=x.numel(),
        scale=scale,
        zero_boundary=zero_boundary,
        mean_abs_error=abs_error.mean().item() if x.numel() else 0.0,
        max_abs_error=abs_error.max().item() if x.numel() else 0.0,
        zero_fraction=(y == 0).float().mean().item() if x.numel() else 0.0,
    )


def _print_summary(label: str, summary: ErrorSummary) -> None:
    print(
        f"  {label:<18} n={summary.count:<4d} "
        f"scale={summary.scale:.6e} zero_boundary={summary.zero_boundary:.6e} "
        f"MAE={summary.mean_abs_error:.6e} max_abs={summary.max_abs_error:.6e} "
        f"zero_fraction={summary.zero_fraction:.3%}"
    )


def main() -> None:
    x = build_tensor()
    y = quant_dequant_per_tensor(x)
    scale, theoretical_zero_boundary = _scale_and_zero_boundary(x)

    print("[per-tensor，含 outlier]")
    print(f"  amax={x.abs().amax().item():.1f} scale={scale:.9f}")
    for v in (0.5, 0.1, 0.01, 0.005, 3000.0):
        idx = (x - v).abs().argmin()
        print(
            f"  target≈{v:<8} x={x[idx].item():.9f} "
            f"y={y[idx].item():.9f} rel_err={rel_err_at(x, y, v):.3e}"
        )

    # (a) 去掉 outlier 重新量化，对比 0.5 处的误差。
    x_without_outlier = x[:-1]
    y_without_outlier = quant_dequant_per_tensor(x_without_outlier)
    err_with = rel_err_at(x, y, 0.5)
    err_without = rel_err_at(x_without_outlier, y_without_outlier, 0.5)
    improvement = err_with / err_without if err_without else float("inf")
    scale_without, _ = _scale_and_zero_boundary(x_without_outlier)
    print("\n[移除 outlier]")
    print(f"  scale={scale_without:.9f}")
    print(f"  x≈0.5 rel_err={err_without:.3e}")
    print(f"  含/不含 outlier 的 0.5 相对误差比={improvement:.3f}x")

    # (b) 在实际样本中找出仍被量化为 0 的最大绝对值，并同时报告
    # 理论 round-to-nearest 边界 scale * 2^-10。
    zero_mask = y == 0
    observed_max_zero = x[zero_mask].abs().max().item() if zero_mask.any() else 0.0
    print("\n[量化为零]")
    print(f"  理论边界=scale*2^-10={theoretical_zero_boundary:.9f}")
    print(f"  样本中量化为 0 的最大 |x|={observed_max_zero:.9f}")

    # (c) 1x128 per-block scale。outlier 只污染它所在的最后一个 block；
    # 统计时从该 block 排除 outlier 本身，观察普通元素受到的影响。
    block_size = DEFAULT_BLOCK_SIZE
    y_block = quant_dequant_per_block(x, block_size)
    outlier_block = (x.numel() - 1) // block_size
    outlier_start = outlier_block * block_size
    ordinary_outlier_block_x = x[outlier_start:-1]
    ordinary_outlier_block_y = y_block[outlier_start:-1]
    full_outlier_block_x = x[outlier_start:]
    same_values_without_outlier_y = quant_dequant_per_tensor(
        ordinary_outlier_block_x
    )
    clean_x = x[:block_size]
    clean_y = y_block[:block_size]
    n_blocks = (x.numel() + block_size - 1) // block_size
    print("\n[1x128 per-block]")
    print(
        f"  blocks={n_blocks}，outlier 位于 block {outlier_block}，"
        f"未受其 scale 影响的 blocks={n_blocks - 1}"
    )
    _print_summary("clean block 0", _summarize_block(clean_x, clean_y))
    _print_summary(
        "同组普通值(无outlier)",
        _summarize_block(ordinary_outlier_block_x, same_values_without_outlier_y),
    )
    _print_summary(
        "同组普通值(含outlier)",
        _summarize_block(
            ordinary_outlier_block_x,
            ordinary_outlier_block_y,
            scale_source=full_outlier_block_x,
        ),
    )


if __name__ == "__main__":
    main()
