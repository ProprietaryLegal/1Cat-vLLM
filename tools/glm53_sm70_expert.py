"""Standard SM70 NVFP4 single-expert dense-stage adapter."""

from __future__ import annotations

import os
from typing import Any

import torch

from tools.glm53_sm70_reference import SWIGLU_LIMIT

EXPERIMENTAL_GROUPED_ENV = "VLLM_SM70_NVFP4_MOE_GROUPED_EXPERT_ROWS"
PreparedProjection = tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


def _decode_scale_bytes(raw: torch.Tensor) -> torch.Tensor:
    if raw.dtype != torch.uint8 or not raw.is_contiguous():
        raise TypeError("native E4M3 scales must be contiguous uint8 bytes")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("installed Torch has no float8_e4m3fn dtype")
    return raw.view(torch.float8_e4m3fn).float()


def _prepare_expert(
    gate: Any,
    up: Any,
    down: Any,
    device: torch.device,
) -> PreparedProjection:
    from vllm import _sm70_ops as sm70_ops
    from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm

    if gate.codes.shape != up.codes.shape:
        raise ValueError("gate/up native geometry differs")
    w13_codes = torch.cat([gate.codes, up.codes], dim=0).to(device)
    gate_scale = _decode_scale_bytes(gate.scales.to(device)) * (
        gate.global_scale.to(device).float()
    )
    up_scale = _decode_scale_bytes(up.scales.to(device)) * (
        up.global_scale.to(device).float()
    )
    w13_scales = torch.cat([gate_scale, up_scale], dim=0).half().t().contiguous()
    prepared_w13 = sm70_ops.nvfp4_sm70_prepare(
        sm70_tm.unpack_mxfp4_weight(w13_codes),
        w13_scales,
        sm70_tm.NVFP4_GROUP_SIZE,
        False,
    )

    w2_codes = down.codes.to(device)
    w2_scales = (
        _decode_scale_bytes(down.scales.to(device))
        * down.global_scale.to(device).float()
    ).half().t().contiguous()
    prepared_w2 = sm70_ops.nvfp4_sm70_prepare(
        sm70_tm.unpack_mxfp4_weight(w2_codes),
        w2_scales,
        sm70_tm.NVFP4_GROUP_SIZE,
        False,
    )
    return prepared_w13, prepared_w2


def run_standard_expert(
    projections: dict[str, Any],
    hidden: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run one expert with the base TurboMind dense-stage path.

    This exercises one expert through the standard TurboMind dense stage.  It
    does not prove multi-expert grouped dispatch; the experimental
    grouped-expert-rows switch is refused before any CUDA work.
    """

    if os.getenv(EXPERIMENTAL_GROUPED_ENV, "0").lower() in {"1", "true", "yes"}:
        raise RuntimeError(
            f"{EXPERIMENTAL_GROUPED_ENV}=1 is experimental and refused by this probe"
        )
    from vllm import _sm70_ops as sm70_ops
    from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm

    gate = projections["gate_proj"]
    prepared_w13, prepared_w2 = _prepare_expert(
        gate, projections["up_proj"], projections["down_proj"], device
    )
    w13_weight, w13_scales, w13_meta = prepared_w13
    w2_weight, w2_scales, w2_meta = prepared_w2
    w13_weight = w13_weight.unsqueeze(0).contiguous()
    w13_scales = w13_scales.unsqueeze(0).contiguous()
    w2_weight = w2_weight.unsqueeze(0).contiguous()
    w2_scales = w2_scales.unsqueeze(0).contiguous()
    w13_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
        w13_weight, w13_scales, int(w13_meta[0]), int(w13_meta[1]), 1
    )
    w2_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
        w2_weight, w2_scales, int(w2_meta[0]), int(w2_meta[1]), 1
    )
    intermediate_size = gate.codes.shape[0] * 2
    hidden_size = projections["down_proj"].codes.shape[0]
    hidden_device = hidden.to(device)
    offsets = torch.tensor(
        [0, hidden_device.shape[0]], dtype=torch.int32, device=device
    )
    expert_ids = torch.zeros(1, dtype=torch.int32, device=device)
    gate_up = torch.empty(
        hidden_device.shape[0], intermediate_size, dtype=torch.float16, device=device
    )
    sm70_ops.nvfp4_moe_dense_stage_sm70_out(
        gate_up, hidden_device, offsets, expert_ids, w13_ptrs[0], w13_ptrs[1],
        1, hidden_size, intermediate_size, sm70_tm.NVFP4_GROUP_SIZE,
    )
    intermediate = torch.empty(
        hidden_device.shape[0], gate.codes.shape[0], dtype=torch.float16, device=device
    )
    torch.ops._C.silu_and_mul_with_clamp(intermediate, gate_up, SWIGLU_LIMIT)
    output = torch.empty(
        hidden_device.shape[0], hidden_size, dtype=torch.float16, device=device
    )
    sm70_ops.nvfp4_moe_dense_stage_sm70_out(
        output, intermediate, offsets, expert_ids, w2_ptrs[0], w2_ptrs[1],
        1, gate.codes.shape[0], hidden_size, sm70_tm.NVFP4_GROUP_SIZE,
    )
    torch.cuda.synchronize(device)
    return output.detach().cpu()
