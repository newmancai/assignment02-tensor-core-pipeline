"""5.1 FP8 outlier 的 host 回归测试。"""

import torch

from quant_outlier import (
    E4M3_MAX,
    quant_dequant_per_block,
    quant_dequant_per_tensor,
    rel_err_at,
)


def test_per_tensor_matches_e4m3_definition():
    x = torch.tensor([-1.0, -0.1, 0.0, 0.25, 1.0])
    scale = x.abs().amax() / E4M3_MAX
    expected = (x / scale).to(torch.float8_e4m3fn).float() * scale
    torch.testing.assert_close(quant_dequant_per_tensor(x), expected)


def test_zero_and_empty_inputs_are_well_defined():
    zeros = torch.zeros(9, dtype=torch.float64)
    restored = quant_dequant_per_tensor(zeros)
    assert restored.dtype == torch.float32
    torch.testing.assert_close(restored, torch.zeros(9))

    empty = quant_dequant_per_tensor(torch.empty(0))
    assert empty.shape == (0,)
    assert empty.dtype == torch.float32


def test_rel_err_uses_the_closest_input_value():
    x = torch.tensor([0.49, 0.8, 1.2])
    y = torch.tensor([0.50, 0.7, 1.2])
    expected = abs(0.50 - 0.49) / 0.49
    assert abs(rel_err_at(x, y, 0.5) - expected) < 1e-7


def test_block_scaling_confines_outlier_to_its_block():
    x = torch.full((256,), 0.005)
    x[-1] = 3000.0
    restored = quant_dequant_per_block(x, block_size=128)

    # 第一个 block 有自己的小 scale，0.005 不会被 outlier 冲成 0；
    # outlier 所在 block 的其余 127 个相同值则低于零阈值。
    assert (restored[:128] != 0).all().item()
    assert (restored[128:-1] == 0).all().item()
    assert restored[-1].item() == 3000.0
