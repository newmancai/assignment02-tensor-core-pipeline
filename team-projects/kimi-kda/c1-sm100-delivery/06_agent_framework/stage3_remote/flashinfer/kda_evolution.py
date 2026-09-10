"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass
from typing import cast

import torch

from .jit.flash_kda_evolution import (
    FLASH_KDA_EVOLUTION_VARIANTS,
    FlashKDAEvolutionVariant,
    load_flash_kda_evolution_module,
)
from .kda_prefill import (
    _FLASH_KDA_ROUTE_BT16_M64,
    _FLASH_KDA_ROUTE_DIRECT_M128_N16,
    _check_output_does_not_overlap_inputs,
    _get_stream_workspace,
    _run_flash_kda_prefill,
    _select_flash_kda_bf16_route,
    _select_flash_kda_prefill_target,
)

_HEAD_DIM = 128
_PERSISTENT_SCALAR_CTAS = 152
_DESCRIPTOR_STORAGE_BYTES = 6 * 128

# Keep the exported specialization manifest shape-exact. Nearby shapes retain
# the production dispatcher rather than inheriting a frozen schedule selected
# for a different workload.
_EVOLUTION_WINNER_SHAPES = frozenset(
    {
        (False, 32, (1300, 547, 2048, 963, 271, 3063)),
        (True, 64, (8192,)),
        (False, 64, (1300, 547, 2048, 963, 271, 3063)),
        (False, 64, (1024,) * 8),
        (True, 96, (37,)),
        (True, 96, (97,)),
        (True, 96, (8192,)),
        (False, 96, (64, 128, 256)),
        (False, 96, (17, 33, 65)),
        (False, 96, (1300, 547, 2048, 963, 271, 3063)),
        (False, 96, (1024,) * 8),
        (False, 96, (1024,) * 16),
        (False, 96, (1024,) * 32),
        (False, 96, (1024,) * 64),
        (False, 96, (1024,) * 128),
        (False, 96, (1024,) * 256),
    }
)

# Stage-3 production activation is intentionally narrower than the frozen
# exploration corpus. These are the five legacy shapes whose Stage-2 bootstrap
# 95% speedup lower bound exceeded one. H96 mixed remains on Cake; all other
# shapes also fall back until they receive scope-identical public evidence.
_PUBLIC_EVOLUTION_ACTIVATION_SHAPES = frozenset(
    {
        (True, 96, (8192,)),
        (False, 96, (1024,) * 8),
        (True, 64, (8192,)),
        (False, 64, (1300, 547, 2048, 963, 271, 3063)),
        (False, 64, (1024,) * 8),
    }
)


def _use_evolution_route(
    sequence_lengths: tuple[int, ...],
    num_heads: int,
    fixed_layout: bool,
) -> bool:
    """Whether this measured shape should use its generated specialization."""

    return (fixed_layout, num_heads, sequence_lengths) in _EVOLUTION_WINNER_SHAPES


def _uses_production_general(
    sequence_lengths: tuple[int, ...],
    num_heads: int,
    fixed_layout: bool,
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
    use_initial_state: bool,
    store_final_state: bool,
) -> bool:
    """Whether the shared planner selects a materially different schedule."""

    route = _select_flash_kda_bf16_route(
        compute_capability=compute_capability,
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_sequences=len(sequence_lengths),
        num_heads=num_heads,
        uniform_sequences=len(set(sequence_lengths)) == 1,
        max_sequence_length=max(sequence_lengths),
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
    )
    independent_dvsplit = (
        fixed_layout and len(sequence_lengths) == 1 and num_heads == 64
    )
    return (
        route
        in {
            _FLASH_KDA_ROUTE_BT16_M64,
            _FLASH_KDA_ROUTE_DIRECT_M128_N16,
        }
        and not independent_dvsplit
    )


def _build_persistent_scalar_schedule(
    sequence_lengths: tuple[int, ...],
    num_heads: int,
    num_ctas: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    bins: list[tuple[int, int, list[tuple[int, int]]]] = [
        (0, cta, []) for cta in range(num_ctas)
    ]
    heapq.heapify(bins)
    ordered_sequences = sorted(
        range(len(sequence_lengths)),
        key=lambda sequence: (sequence_lengths[sequence] + 31) // 32,
        reverse=True,
    )
    for sequence in ordered_sequences:
        chunks = (sequence_lengths[sequence] + 31) // 32
        for head in range(num_heads):
            load, cta, tasks = heapq.heappop(bins)
            tasks.append((sequence * num_heads + head, chunks))
            heapq.heappush(bins, (load + chunks, cta, tasks))

    while True:
        light_index = min(
            range(num_ctas), key=lambda index: (bins[index][0], bins[index][1])
        )
        heavy_index = max(
            range(num_ctas), key=lambda index: (bins[index][0], -bins[index][1])
        )
        light_load, light_cta, light_tasks = bins[light_index]
        heavy_load, heavy_cta, heavy_tasks = bins[heavy_index]
        pair_tasks = light_tasks + heavy_tasks
        pair_load = light_load + heavy_load
        reachable = {0: 0}
        for task_index, (_task, chunks) in enumerate(pair_tasks):
            for load, mask in list(reachable.items()):
                reachable.setdefault(load + chunks, mask | (1 << task_index))
        split_load = min(
            reachable,
            key=lambda load: (
                max(load, pair_load - load),
                abs(pair_load - 2 * load),
            ),
        )
        if max(split_load, pair_load - split_load) >= heavy_load:
            break
        split_mask = reachable[split_load]
        light_tasks = [
            task for index, task in enumerate(pair_tasks) if split_mask & (1 << index)
        ]
        heavy_tasks = [
            task
            for index, task in enumerate(pair_tasks)
            if not split_mask & (1 << index)
        ]
        bins[light_index] = (split_load, light_cta, light_tasks)
        bins[heavy_index] = (pair_load - split_load, heavy_cta, heavy_tasks)

    while num_ctas >= 3:
        ordered_bins = sorted(
            range(num_ctas), key=lambda index: (bins[index][0], bins[index][1])
        )
        light_index = ordered_bins[0]
        heavy_index = ordered_bins[-1]
        average_load = sum(load for load, _cta, _tasks in bins) / num_ctas
        middle_index = min(
            ordered_bins[1:-1],
            key=lambda index: (
                abs(bins[index][0] - average_load),
                -max(chunks for _task, chunks in bins[index][2]),
                bins[index][1],
            ),
        )
        selected_indices = (light_index, middle_index, heavy_index)
        selected_tasks = [task for index in selected_indices for task in bins[index][2]]
        selected_load = sum(chunks for _task, chunks in selected_tasks)
        heavy_load = bins[heavy_index][0]
        if selected_load > 1024:
            break
        reachable_pairs = {(0, 0): 0}
        processed_load = 0
        for task_index, (_task, chunks) in enumerate(selected_tasks):
            next_pairs = dict(reachable_pairs)
            for (first_load, second_load), assignment in reachable_pairs.items():
                third_load = processed_load - first_load - second_load
                if first_load + chunks < heavy_load:
                    next_pairs.setdefault(
                        (first_load + chunks, second_load),
                        assignment | (1 << (2 * task_index)),
                    )
                if second_load + chunks < heavy_load:
                    next_pairs.setdefault(
                        (first_load, second_load + chunks),
                        assignment | (2 << (2 * task_index)),
                    )
                if third_load + chunks >= heavy_load:
                    next_pairs.pop((first_load, second_load), None)
            reachable_pairs = next_pairs
            processed_load += chunks
        if not reachable_pairs:
            break
        (first_load, second_load), assignment = min(
            reachable_pairs.items(),
            key=lambda item: max(item[0][0], item[0][1], selected_load - sum(item[0])),
        )
        split_loads = (
            first_load,
            second_load,
            selected_load - first_load - second_load,
        )
        if max(split_loads) >= heavy_load:
            break
        split_tasks: list[list[tuple[int, int]]] = [[], [], []]
        for task_index, task in enumerate(selected_tasks):
            encoded_group = (assignment >> (2 * task_index)) & 3
            group = 0 if encoded_group == 1 else 1 if encoded_group == 2 else 2
            split_tasks[group].append(task)
        for index, load, tasks in zip(
            selected_indices, split_loads, split_tasks, strict=True
        ):
            _old_load, cta, _old_tasks = bins[index]
            bins[index] = (load, cta, tasks)

    bins.sort(key=lambda item: item[1])
    counts = [load for load, _cta, _tasks in bins]
    stride = max(counts)
    schedule = [0] * (num_ctas * stride)
    for _load, cta, tasks in bins:
        slot = 0
        for encoded_task, chunks in tasks:
            for local_chunk in range(chunks):
                schedule[cta * stride + slot] = (
                    encoded_task | (local_chunk << 10) | (chunks << 18)
                )
                slot += 1
    return (
        torch.tensor(schedule, dtype=torch.int32, device=device),
        torch.tensor(counts, dtype=torch.int32, device=device),
        stride,
    )


@dataclass(frozen=True)
class _Route:
    variant: FlashKDAEvolutionVariant
    grid_x: int
    tile_schedule: torch.Tensor
    tile_schedule_counts: torch.Tensor


def _route(
    sequence_lengths: tuple[int, ...],
    num_heads: int,
    fixed_layout: bool,
    device: torch.device,
) -> _Route:
    num_sequences = len(sequence_lengths)
    full_chunks = all(length % 32 == 0 for length in sequence_lengths)
    use_m64 = fixed_layout and num_sequences == 1 and num_heads == 64
    use_vtile = not use_m64 and (fixed_layout or len(set(sequence_lengths)) == 1)
    dummy = torch.empty((1,), dtype=torch.int32, device=device)

    if use_m64:
        variant = f"m64_f{int(full_chunks)}_t{sequence_lengths[0]}_h{num_heads}"
        grid_x = 2 * num_sequences * num_heads
        tile_schedule = dummy
        tile_schedule_counts = dummy
    elif use_vtile:
        persistent_tasks = 1
        persistent_stride = num_sequences * num_heads
        if (
            full_chunks
            and num_sequences == 8
            and sequence_lengths[0] == 1024
            and num_heads in (64, 96)
        ):
            persistent_stride = 128
            persistent_tasks = num_sequences * num_heads // persistent_stride
        variant = (
            f"vtile_f{int(full_chunks)}_t{sequence_lengths[0]}_h{num_heads}"
            f"_p{persistent_tasks}_s{persistent_stride}"
        )
        grid_x = persistent_stride
        tile_schedule = dummy
        tile_schedule_counts = dummy
    else:
        persistent_scalar_ctas = min(
            _PERSISTENT_SCALAR_CTAS,
            torch.cuda.get_device_properties(device).multi_processor_count,
        )
        use_persistent_scalar = (
            num_heads in (64, 96)
            and num_sequences * num_heads >= 2 * persistent_scalar_ctas
            and num_sequences * num_heads < 1024
            and max((length + 31) // 32 for length in sequence_lengths) < 256
        )
        if use_persistent_scalar:
            tile_schedule, tile_schedule_counts, stride = (
                _build_persistent_scalar_schedule(
                    sequence_lengths,
                    num_heads,
                    persistent_scalar_ctas,
                    device,
                )
            )
            grid_x = persistent_scalar_ctas
        else:
            stride = 1
            grid_x = num_sequences * num_heads
            tile_schedule = dummy
            tile_schedule_counts = dummy
        variant = f"m128_h{num_heads}_p{int(use_persistent_scalar)}_s{stride}"

    if variant not in FLASH_KDA_EVOLUTION_VARIANTS:
        raise ValueError(
            "no generated Blackwell specialization for "
            f"heads={num_heads}, sequence_lengths={sequence_lengths}, "
            f"fixed_layout={fixed_layout}"
        )
    return _Route(
        variant=cast(FlashKDAEvolutionVariant, variant),
        grid_x=grid_x,
        tile_schedule=tile_schedule,
        tile_schedule_counts=tile_schedule_counts,
    )


class PreparedFlashKDAEvolution:
    """Prepared launch of the measured-best Blackwell recurrent-KDA route.

    Stable generated winners retain their exported specialization. Other
    shapes use the production dispatcher while preserving this adapter's
    independent initial-state and final-state buffers.

    A prepared evolution route is bound to the CUDA stream used by its first
    launch. Create a separate prepared instance for another stream.
    """

    def __init__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        initial_state: torch.Tensor,
        output: torch.Tensor,
        final_state: torch.Tensor,
        *,
        scale: float,
        lower_bound: float,
        cu_seqlens: torch.Tensor | None = None,
    ) -> None:
        if q.shape != k.shape or q.shape != v.shape or q.shape != g.shape:
            raise ValueError("q, k, v, and g must have identical shapes")
        if q.shape != output.shape or q.ndim != 4 or q.shape[-1] != _HEAD_DIM:
            raise ValueError("q and output must have shape [B, T, H, 128]")
        if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v, g, beta)):
            raise TypeError("q, k, v, g, and beta must use torch.bfloat16")
        batch, tokens_per_batch, num_heads, _head_dim = q.shape
        total_tokens = batch * tokens_per_batch
        fixed_layout = cu_seqlens is None
        if fixed_layout:
            offsets = tuple(range(0, total_tokens + 1, tokens_per_batch))
            cu_seqlens = torch.tensor(offsets, dtype=torch.int64, device=q.device)
        else:
            offsets = tuple(int(value) for value in cu_seqlens.tolist())
        sequence_lengths = tuple(
            end - start for start, end in itertools.pairwise(offsets)
        )
        target = _select_flash_kda_prefill_target(q.device)
        properties = torch.cuda.get_device_properties(q.device)
        use_evolution = _use_evolution_route(
            sequence_lengths, num_heads, fixed_layout
        ) and not _uses_production_general(
            sequence_lengths,
            num_heads,
            fixed_layout,
            compute_capability=(properties.major, properties.minor),
            sm_count=properties.multi_processor_count,
            use_initial_state=initial_state is not None,
            store_final_state=final_state is not None,
        )
        if not use_evolution:
            self.module = None
            self.variant = "cake_dispatcher"
            self.target = target
            self.grid_x = None
            self.route = "cake"
            self._cake_args = dict(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                initial_state=initial_state,
                output_final_state=True,
                lower_bound=lower_bound,
                cu_seqlens=None if fixed_layout else cu_seqlens,
                output=output,
                seq_order=None,
                prefill_workspace=None,
                state_indices=None,
                state_checkpoints=None,
                checkpoint_cu_starts=None,
                checkpoint_every_n_tokens=0,
                backend="cake",
                final_state=final_state,
            )
            return

        route = _route(sequence_lengths, num_heads, fixed_layout, q.device)
        seq_order = torch.tensor(
            sorted(
                range(len(sequence_lengths)),
                key=sequence_lengths.__getitem__,
                reverse=True,
            ),
            dtype=torch.int32,
            device=q.device,
        )
        beta_flat = beta.reshape(total_tokens, num_heads)
        padded_heads = ((num_heads + 7) // 8) * 8
        if padded_heads == num_heads and total_tokens >= 32:
            beta_tma = beta_flat
        else:
            beta_tma = torch.empty(
                (max(total_tokens, 32), padded_heads),
                dtype=beta.dtype,
                device=beta.device,
            )

        self.module = load_flash_kda_evolution_module(route.variant, target)
        self.variant = route.variant
        self.target = target
        self.grid_x = route.grid_x
        self.route = "evolution"
        self._device = q.device
        self._launch_stream_ptr: int | None = None
        self._prepare_descriptors = True
        self._args = (
            q.reshape(total_tokens, num_heads, _HEAD_DIM),
            k.reshape(total_tokens, num_heads, _HEAD_DIM),
            v.reshape(total_tokens, num_heads, _HEAD_DIM),
            g.reshape(total_tokens, num_heads, _HEAD_DIM),
            beta_flat,
            beta_tma,
            A_log,
            dt_bias,
            cu_seqlens,
            seq_order,
            route.tile_schedule,
            route.tile_schedule_counts,
            initial_state,
            output.reshape(total_tokens, num_heads, _HEAD_DIM),
            final_state,
            torch.empty(
                (_DESCRIPTOR_STORAGE_BYTES,),
                dtype=torch.uint8,
                device=q.device,
            ),
        )
        self._launch_scalars = (
            route.grid_x,
            num_heads,
            1,
            1,
            float(scale),
            float(lower_bound),
        )

    def launch(self) -> None:
        if self.route == "cake":
            _run_flash_kda_prefill(**self._cake_args)
            return
        stream_ptr = int(torch.cuda.current_stream(self._device).cuda_stream)
        if self._launch_stream_ptr is None:
            # Bind before entering FFI: if a launch reports an error after
            # enqueueing descriptor publication, a retry cannot race it from
            # another stream.
            self._launch_stream_ptr = stream_ptr
        elif stream_ptr != self._launch_stream_ptr:
            raise RuntimeError(
                "PreparedFlashKDAEvolution launches must remain on the CUDA "
                "stream used by the first launch"
            )
        self.module.run(
            *self._args,
            int(self._prepare_descriptors),
            *self._launch_scalars,
            stream_ptr,
        )
        self._prepare_descriptors = False

    def rebind_state(
        self, initial_state: torch.Tensor, final_state: torch.Tensor
    ) -> None:
        """Rebind dynamic state pointers while retaining the prepared route."""

        if self.route != "evolution":
            raise RuntimeError("only a generated evolution route can rebind state")
        expected_initial = self._args[12]
        expected_final = self._args[14]
        if (
            initial_state.shape != expected_initial.shape
            or final_state.shape != expected_final.shape
            or initial_state.dtype != expected_initial.dtype
            or final_state.dtype != expected_final.dtype
            or initial_state.device != self._device
            or final_state.device != self._device
        ):
            raise ValueError("rebound state tensors must match the prepared state ABI")
        args = list(self._args)
        args[12] = initial_state
        args[14] = final_state
        self._args = tuple(args)


def prepare_flash_kda_evolution(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    output: torch.Tensor,
    final_state: torch.Tensor,
    *,
    scale: float,
    lower_bound: float = -5.0,
    cu_seqlens: torch.Tensor | None = None,
) -> PreparedFlashKDAEvolution:
    """Prepare the measured-best generated or production route for launch."""
    return PreparedFlashKDAEvolution(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        initial_state,
        output,
        final_state,
        scale=scale,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
    )


def public_flash_kda_evolution_rejection_reason(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    scale: float | None,
    initial_state: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    lower_bound: float | None,
    cu_seqlens: torch.Tensor | None,
    ssm_state_indices: torch.Tensor | None,
    num_spec_tokens: int | None,
    num_accepted_tokens: torch.Tensor | None,
    output: torch.Tensor | None,
    initial_state_source: torch.Tensor | None,
    initial_state_indices: torch.Tensor | None,
    beta_is_logit: bool,
    seq_order: torch.Tensor | None,
    prefill_workspace: object | None,
    state_checkpoints: torch.Tensor | None,
    checkpoint_cu_starts: torch.Tensor | None,
    checkpoint_every_n_tokens: int,
) -> str | None:
    """Return why the proof-carrying public adapter cannot accept a call."""

    if not q.is_cuda or torch.cuda.get_device_capability(q.device) not in (
        (10, 0),
        (10, 3),
    ):
        return "requires an SM100-family or SM103 GPU"
    if q.ndim != 4 or q.shape[-1] != _HEAD_DIM or q.shape[1] <= 1:
        return "requires ordinary multi-token [B, T, H, 128] prefill"
    if q.shape != k.shape or q.shape != v.shape or q.shape != g.shape:
        return "q, k, v, and g must have identical shapes"
    if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v, g, beta)):
        return "q, k, v, g, and beta must use torch.bfloat16"
    if any(tensor.device != q.device for tensor in (k, v, g, beta)):
        return "q, k, v, g, and beta must be on the same CUDA device"
    if any(not tensor.is_contiguous() for tensor in (q, k, v, g, beta)):
        return "q, k, v, g, and beta must be contiguous"
    batch, tokens_per_batch, num_heads, _ = q.shape
    total_tokens = batch * tokens_per_batch
    if beta.shape != (batch, tokens_per_batch, num_heads):
        return "beta must have shape [B, T, H]"
    if A_log is None or A_log.dtype != torch.float32 or A_log.shape != (num_heads,):
        return "A_log must be float32 with shape [H]"
    if (
        dt_bias is None
        or dt_bias.dtype != torch.float32
        or dt_bias.shape not in ((num_heads, _HEAD_DIM), (num_heads * _HEAD_DIM,))
    ):
        return "dt_bias must be float32 with shape [H, 128] or [H*128]"
    if (
        A_log.device != q.device
        or dt_bias.device != q.device
        or not A_log.is_contiguous()
        or not dt_bias.is_contiguous()
    ):
        return "A_log and dt_bias must be contiguous on the input CUDA device"
    if initial_state is None:
        return "requires an explicit initial_state for in-place state update"
    num_sequences = batch if cu_seqlens is None else cu_seqlens.numel() - 1
    if initial_state.dtype != torch.bfloat16 or initial_state.shape != (
        num_sequences,
        num_heads,
        _HEAD_DIM,
        _HEAD_DIM,
    ):
        return "initial_state must be bfloat16 with shape [N, H, 128, 128]"
    if not initial_state.is_contiguous():
        return "initial_state must be contiguous"
    if initial_state.device != q.device:
        return "initial_state must be on the input CUDA device"
    if output is not None and (
        output.shape != q.shape
        or output.dtype != q.dtype
        or output.device != q.device
        or not output.is_contiguous()
    ):
        return "output must be contiguous and match q shape, dtype, and device"
    if not use_qk_l2norm_in_kernel or not use_gate_in_kernel or not beta_is_logit:
        return "requires fused QK L2 norm, gate, and beta sigmoid"
    if lower_bound is None or not math.isfinite(lower_bound) or lower_bound >= 0:
        return "lower_bound must be a finite negative float"
    if scale is not None and (not math.isfinite(scale) or scale <= 0):
        return "scale must be a finite positive float"
    if cu_seqlens is not None and (
        cu_seqlens.dtype != torch.int64
        or not cu_seqlens.is_cuda
        or cu_seqlens.device != q.device
        or not cu_seqlens.is_contiguous()
    ):
        return "cu_seqlens must be contiguous CUDA int64"
    if cu_seqlens is not None and (
        q.shape[0] != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2
    ):
        return "packed prefill requires q batch 1 and cu_seqlens shape [N+1]"
    unsupported = {
        "ssm_state_indices": ssm_state_indices,
        "num_spec_tokens": num_spec_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "initial_state_source": initial_state_source,
        "initial_state_indices": initial_state_indices,
        "seq_order": seq_order,
        "prefill_workspace": prefill_workspace,
        "state_checkpoints": state_checkpoints,
        "checkpoint_cu_starts": checkpoint_cu_starts,
    }
    present = [name for name, value in unsupported.items() if value is not None]
    if checkpoint_every_n_tokens != 0:
        present.append("checkpoint_every_n_tokens")
    if present:
        return "unsupported arguments: " + ", ".join(present)
    if torch.cuda.is_current_stream_capturing():
        return "CUDA graph capture requires a future explicit evolution workspace"
    if cu_seqlens is None and total_tokens == 0:
        return "empty prefill is unsupported"
    return None


def run_public_flash_kda_evolution(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float | None,
    initial_state: torch.Tensor,
    output_final_state: bool,
    lower_bound: float | None,
    cu_seqlens: torch.Tensor | None,
    output: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the guarded evolution route with public in-place state semantics."""

    batch, tokens_per_batch, num_heads, _ = q.shape
    workspace = _get_stream_workspace(q.device)
    if cu_seqlens is None:
        sequence_lengths = (tokens_per_batch,) * batch
        fixed_layout = True
    else:
        metadata_key = (
            cu_seqlens.data_ptr(),
            cu_seqlens._version,
            cu_seqlens.numel(),
            q.shape[1],
        )
        metadata_cache = getattr(workspace, "_evolution_sequence_lengths", {})
        sequence_lengths = metadata_cache.get(metadata_key)
        if sequence_lengths is None:
            offsets = tuple(int(value) for value in cu_seqlens.tolist())
            if (
                offsets[0] != 0
                or offsets[-1] != q.shape[1]
                or any(end <= start for start, end in itertools.pairwise(offsets))
            ):
                raise ValueError(
                    "cu_seqlens must describe positive sequences covering all tokens"
                )
            sequence_lengths = tuple(
                end - start for start, end in itertools.pairwise(offsets)
            )
            metadata_cache = {metadata_key: sequence_lengths}
            setattr(workspace, "_evolution_sequence_lengths", metadata_cache)
        fixed_layout = False
    out = torch.empty_like(q) if output is None else output
    _check_output_does_not_overlap_inputs(
        out,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
    )
    scale_value = _HEAD_DIM**-0.5 if scale is None else float(scale)
    lower_bound_value = cast(float, lower_bound)
    shape_key = (fixed_layout, num_heads, sequence_lengths)
    if shape_key not in _PUBLIC_EVOLUTION_ACTIVATION_SHAPES:
        return _run_flash_kda_prefill(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale_value,
            initial_state=initial_state,
            output_final_state=output_final_state,
            lower_bound=lower_bound_value,
            cu_seqlens=cu_seqlens,
            output=out,
            seq_order=None,
            prefill_workspace=None,
            state_indices=None,
            state_checkpoints=None,
            checkpoint_cu_starts=None,
            checkpoint_every_n_tokens=0,
            backend="cake",
        )
    # The generated ABI has distinct initial/final state pointers. Preserve
    # public in-place semantics with a same-stream copy-back; aliasing those
    # pointers is outside the winner's proof and can serialize its schedule.
    final_storage = getattr(workspace, "_evolution_final_state", None)
    if (
        final_storage is None
        or final_storage.dtype != initial_state.dtype
        or final_storage.numel() < initial_state.numel()
    ):
        final_storage = torch.empty_like(initial_state).reshape(-1)
        setattr(workspace, "_evolution_final_state", final_storage)
    final_state = final_storage[: initial_state.numel()].view_as(initial_state)
    prepared_key = (
        *(tensor.data_ptr() for tensor in (q, k, v, g, beta, A_log, dt_bias, out)),
        None if cu_seqlens is None else cu_seqlens.data_ptr(),
        scale_value,
        lower_bound_value,
        shape_key,
    )
    prepared_cache = getattr(workspace, "_evolution_prepared_routes", {})
    prepared = prepared_cache.get(prepared_key)
    if prepared is None:
        prepared = prepare_flash_kda_evolution(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            initial_state,
            out,
            final_state,
            scale=scale_value,
            lower_bound=lower_bound_value,
            cu_seqlens=cu_seqlens,
        )
        if len(prepared_cache) >= 32:
            prepared_cache.pop(next(iter(prepared_cache)))
        prepared_cache[prepared_key] = prepared
        setattr(workspace, "_evolution_prepared_routes", prepared_cache)
    else:
        prepared.rebind_state(initial_state, final_state)
    if prepared.route != "evolution":
        raise RuntimeError(
            "public evolution activation unexpectedly resolved to Cake for "
            f"shape {shape_key}"
        )
    prepared.launch()
    initial_state.copy_(final_state)
    return out, initial_state if output_final_state else None


__all__ = [
    "PreparedFlashKDAEvolution",
    "prepare_flash_kda_evolution",
    "public_flash_kda_evolution_rejection_reason",
    "run_public_flash_kda_evolution",
]
