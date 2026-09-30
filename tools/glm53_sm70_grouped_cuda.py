# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-only dispatch helpers for the bounded standard grouped probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from tools.glm53_sm70_expert import PreparedProjection


@dataclass(frozen=True)
class GroupedProjectionPointers:
    """Prepared pointer arrays for one compact local slot count."""

    w13_ptrs: tuple[torch.Tensor, torch.Tensor]
    w2_ptrs: tuple[torch.Tensor, torch.Tensor]
    w13_storage: tuple[torch.Tensor, torch.Tensor]
    w2_storage: tuple[torch.Tensor, torch.Tensor]
    slots: int
    local_intermediate: int


def _stack_prepared(
    prepared: dict[int, PreparedProjection],
    selected_experts: tuple[int, int],
    slots: int,
) -> tuple[PreparedProjection, PreparedProjection]:
    if slots < len(selected_experts):
        raise ValueError("compact slots must contain both selected experts")
    ordered = [prepared[expert] for expert in selected_experts]
    base_weight = torch.stack([item[0][0] for item in ordered], dim=0)
    base_scales = torch.stack([item[0][1] for item in ordered], dim=0)
    meta = ordered[0][0][2]
    if any(tuple(item[0][2].shape) != tuple(meta.shape) for item in ordered):
        raise ValueError("selected experts have different W13 metadata geometry")
    indices = [index % len(selected_experts) for index in range(slots)]
    index_tensor = torch.tensor(indices, dtype=torch.long, device=base_weight.device)
    w13 = base_weight.index_select(0, index_tensor).contiguous()
    s13 = base_scales.index_select(0, index_tensor).contiguous()

    base_weight_2 = torch.stack([item[1][0] for item in ordered], dim=0)
    base_scales_2 = torch.stack([item[1][1] for item in ordered], dim=0)
    meta_2 = ordered[0][1][2]
    if any(tuple(item[1][2].shape) != tuple(meta_2.shape) for item in ordered):
        raise ValueError("selected experts have different W2 metadata geometry")
    w2 = base_weight_2.index_select(0, index_tensor).contiguous()
    s2 = base_scales_2.index_select(0, index_tensor).contiguous()
    return (w13, s13, meta), (w2, s2, meta_2)


def pointer_bundle(
    prepared: dict[int, PreparedProjection],
    selected_experts: tuple[int, int],
    slots: int,
    local_intermediate: int,
) -> GroupedProjectionPointers:
    """Build pointer rows, duplicating selected rows for empty compact slots."""

    from vllm import _sm70_ops as sm70_ops

    w13, w2 = _stack_prepared(prepared, selected_experts, slots)
    w13_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
        w13[0], w13[1], int(w13[2][0]), int(w13[2][1]), slots
    )
    w2_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
        w2[0], w2[1], int(w2[2][0]), int(w2[2][1]), slots
    )
    return GroupedProjectionPointers(
        (w13_ptrs[0], w13_ptrs[1]),
        (w2_ptrs[0], w2_ptrs[1]),
        (w13[0], w13[1]),
        (w2[0], w2[1]),
        slots,
        local_intermediate,
    )


def run_grouped_once(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    pointers: GroupedProjectionPointers,
    device: torch.device,
    *,
    expert_map: torch.Tensor,
    global_experts: int,
    top_k: int,
    hidden_size: int,
    swiglu_limit: float,
    group_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run upstream permutation, compact grouped stages, and unpermute."""

    from vllm import _sm70_ops as sm70_ops

    tokens = hidden.shape[0]
    slots = tokens * top_k
    if slots != pointers.slots:
        raise ValueError("pointer slot count does not match token route")
    hidden_device = hidden.to(device)
    topk_ids_device = topk_ids.to(device)
    topk_weights_device = topk_weights.to(device)
    token_indices = torch.arange(slots, dtype=torch.int32, device=device).view(
        tokens, top_k
    )
    if hidden.ndim != 2 or hidden.shape[1] != hidden_size:
        raise ValueError("hidden must have the declared model width")
    permuted = torch.empty((slots, hidden_size), dtype=torch.float16, device=device)
    offsets64 = torch.empty(slots + 1, dtype=torch.int64, device=device)
    inverse = torch.empty((tokens, top_k), dtype=torch.int32, device=device)
    permuted_idx = torch.empty(slots, dtype=torch.int32, device=device)
    sorted_experts = torch.empty(slots, dtype=torch.int32, device=device)
    sorted_rows = torch.empty(slots, dtype=torch.int32, device=device)
    topk_for_sort = torch.empty(slots, dtype=torch.int32, device=device)
    workspace_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
        slots, global_experts
    )
    workspace = torch.empty(workspace_size, dtype=torch.int8, device=device)
    torch.ops._moe_C.moe_permute_with_scratch(
        hidden_device,
        topk_ids_device,
        token_indices,
        expert_map,
        global_experts,
        slots,
        top_k,
        permuted,
        offsets64,
        inverse,
        permuted_idx,
        workspace,
        sorted_experts,
        sorted_rows,
        topk_for_sort,
    )
    compact_offsets = torch.arange(slots + 1, dtype=torch.int32, device=device)
    gate_up = torch.empty(
        slots,
        pointers.local_intermediate * 2,
        dtype=torch.float16,
        device=device,
    )
    sm70_ops.nvfp4_moe_dense_stage_sm70_out(
        gate_up,
        permuted,
        compact_offsets,
        sorted_experts,
        pointers.w13_ptrs[0],
        pointers.w13_ptrs[1],
        slots,
        hidden_size,
        pointers.local_intermediate * 2,
        group_size,
    )
    intermediate = torch.empty(
        slots, pointers.local_intermediate, dtype=torch.float16, device=device
    )
    torch.ops._C.silu_and_mul_with_clamp(intermediate, gate_up, swiglu_limit)
    routed = torch.empty(slots, hidden_size, dtype=torch.float16, device=device)
    sm70_ops.nvfp4_moe_dense_stage_sm70_out(
        routed,
        intermediate,
        compact_offsets,
        sorted_experts,
        pointers.w2_ptrs[0],
        pointers.w2_ptrs[1],
        slots,
        pointers.local_intermediate,
        hidden_size,
        group_size,
    )
    output = torch.empty(tokens, hidden_size, dtype=torch.float16, device=device)
    torch.ops._moe_C.moe_unpermute(
        routed, topk_weights_device, inverse, offsets64, top_k, output
    )
    torch.accelerator.synchronize(device)
    return output.detach().cpu(), {
        "expanded_rows": slots,
        "active_local_expert_ids": sorted_experts.detach().cpu().tolist(),
        "expert_offsets": offsets64.detach().cpu().tolist(),
    }


__all__ = ["GroupedProjectionPointers", "pointer_bundle", "run_grouped_once"]
