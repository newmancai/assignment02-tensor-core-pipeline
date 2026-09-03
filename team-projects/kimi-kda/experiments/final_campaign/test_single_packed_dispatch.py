"""Guardrails for the B300 packed-single-sequence K2 dispatch policy."""

from __future__ import annotations

import os
from typing import Optional

import torch

import flash_kda


def decision(q: torch.Tensor, cu_seqlens: Optional[torch.Tensor]) -> dict:
    result = flash_kda.explain_k2_dispatch(q, cu_seqlens=cu_seqlens)
    assert isinstance(result, dict)
    return result


@torch.inference_mode()
def main() -> None:
    os.environ.pop("FLASH_KDA_K2_VALUE_SLICE", None)
    os.environ.pop("FLASH_KDA_K2_DISPATCH", None)

    q = torch.empty((1, 8192, 12, 128), device="cuda", dtype=torch.bfloat16)
    fixed = decision(q, None)
    packed_one = decision(
        q,
        torch.tensor([0, 8192], device="cuda", dtype=torch.int64),
    )
    packed_many = decision(
        q,
        torch.tensor(
            [0, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192],
            device="cuda",
            dtype=torch.int64,
        ),
    )

    assert fixed["value_slice"] == 16, fixed
    assert packed_one == fixed, (packed_one, fixed)
    assert packed_many["value_slice"] == 128, packed_many
    assert packed_many["reason"] == "varlen_not_calibrated", packed_many

    print(f"fixed={fixed}")
    print(f"packed_one={packed_one}")
    print(f"packed_many={packed_many}")
    print("SUMMARY,PASS")


if __name__ == "__main__":
    main()
