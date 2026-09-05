import os
from functools import lru_cache

import torch
from flash_kda_C import fwd as _fwd_raw, get_device_characteristics, get_workspace_size

from .dispatch import select_k2_value_slice


@lru_cache(maxsize=None)
def _device_characteristics(device_index):
    with torch.cuda.device(device_index):
        return dict(get_device_characteristics())


@lru_cache(maxsize=256)
def _cached_auto_decision(device_index, batch, tokens, heads, state_fp32, is_varlen):
    return select_k2_value_slice(
        batch=batch,
        tokens_per_sequence=tokens,
        heads=heads,
        state_fp32=state_fp32,
        is_varlen=is_varlen,
        device=_device_characteristics(device_index),
    )


def _dispatch_decision(q, initial_state, final_state, cu_seqlens):
    forced = os.getenv("FLASH_KDA_K2_VALUE_SLICE")
    if forced is not None:
        value_slice = int(forced)
        if value_slice not in (16, 32, 64, 128):
            raise ValueError("FLASH_KDA_K2_VALUE_SLICE must be 16, 32, 64, or 128")
        return value_slice, {"value_slice": value_slice, "reason": "environment_override"}
    if os.getenv("FLASH_KDA_K2_DISPATCH", "auto").lower() in ("0", "false", "off"):
        return 128, {"value_slice": 128, "reason": "dispatch_disabled"}

    state_fp32 = any(
        tensor is not None and tensor.dtype == torch.float32
        for tensor in (initial_state, final_state)
    )

    # A packed batch containing exactly one sequence executes the same K2 grid
    # and recurrence length as fixed B=1.  Treating it as generic varlen made
    # the dispatcher unnecessarily fall back to V128 in the common
    # single-request prefill path.  numel() is tensor metadata, so this policy
    # does not copy cu_seqlens to the host or introduce a device synchronize.
    packed_sequence_count = (
        cu_seqlens.numel() - 1 if cu_seqlens is not None else None
    )
    is_unmodelled_varlen = (
        packed_sequence_count is not None and packed_sequence_count != 1
    )
    decision = _cached_auto_decision(
        q.device.index,
        1 if packed_sequence_count == 1 else q.shape[0],
        q.shape[1],
        q.shape[2],
        state_fp32,
        is_unmodelled_varlen,
    )
    return decision.value_slice, decision


def explain_k2_dispatch(q, initial_state=None, final_state=None, cu_seqlens=None):
    """Return the selected K2 slice and the model/guard-band diagnostics."""

    _, decision = _dispatch_decision(q, initial_state, final_state, cu_seqlens)
    return decision if isinstance(decision, dict) else decision.as_dict()


def fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, initial_state=None, final_state=None, cu_seqlens=None):
    """FlashKDA forward (Flash Kimi Delta Attention).

    Args:
        q (torch.Tensor): Query, bf16, shape ``[B, T, H, K]``.
        k (torch.Tensor): Key, bf16, shape ``[B, T, H, K]``.
        v (torch.Tensor): Value, bf16, shape ``[B, T, H, V]``.
        g (torch.Tensor): Gate before activation, bf16, shape ``[B, T, H, K]``.
        beta (torch.Tensor): Beta logits (pre-activation; sigmoid is applied
            internally), bf16, shape ``[B, T, H]``.
        scale (float): Scaling factor.
        out (torch.Tensor): Output buffer, bf16, shape ``[B, T, H, V]``. Written
            in place.
        A_log (torch.Tensor): Log-gate parameter, fp32, shape ``[H]``.
        dt_bias (torch.Tensor): Gate bias, fp32, shape ``[H, K]``.
        lower_bound (float): Gate lower bound, expected in ``[-5.0, 0]``.
        initial_state (torch.Tensor, optional): Initial recurrent state, bf16
            or fp32. Shape ``[B, H, V, K]`` for batched mode, or ``[N, H, V, K]``
            for varlen mode. ``None`` means start from zero.
        final_state (torch.Tensor, optional): Output buffer for the final
            recurrent state. Same dtype/shape rules as ``initial_state``.
        cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths, int64,
            shape ``[N+1]``. When provided, ``B`` must be 1.

    Notes:
        * Currently requires ``K = V = 128``.
        * All input tensors must be CUDA, contiguous, and have the dtypes
          listed above.
    """
    B, T_seq, H = q.shape[0], q.shape[1], q.shape[2]
    T_total = B * T_seq
    N = cu_seqlens.numel() - 1 if cu_seqlens is not None else B

    workspace = torch.empty(get_workspace_size(T_total, H, N), dtype=torch.uint8, device=q.device)

    k2_value_slice, _ = _dispatch_decision(q, initial_state, final_state, cu_seqlens)
    _fwd_raw(q, k, v, g, beta, float(scale), out, workspace, A_log, dt_bias, lower_bound,
             initial_state=initial_state, final_state=final_state, cu_seqlens=cu_seqlens,
             k2_value_slice=k2_value_slice)
