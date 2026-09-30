"""Small source-contract checks shared by the SM70 primitive probe."""

from __future__ import annotations

from typing import Any, Literal

import torch

ProjectionRole = Literal["gate_proj", "up_proj", "down_proj"]


def expected_native_shape(
    layer: int, role: ProjectionRole, tp_size: int
) -> tuple[int, int]:
    """Return the packed native [N, K/2] shape for a TP-local projection."""

    if layer < 3:
        full = (12288, 2048) if role != "down_proj" else (4096, 6144)
    else:
        full = (2048, 2048) if role != "down_proj" else (4096, 1024)
    axis = 0 if role != "down_proj" else 1
    if full[axis] % tp_size:
        raise ValueError(f"native shape is not divisible by tp_size={tp_size}: {full}")
    local = list(full)
    local[axis] //= tp_size
    return tuple(local)


def validate_nvfp4_scalars(
    global_scale: torch.Tensor,
    input_scale: torch.Tensor,
    provenance: dict[str, Any],
) -> None:
    """Require finite FP32 scalars and an explicit W4A16 input-scale policy."""

    for name, value in (("global", global_scale), ("input", input_scale)):
        if value.dtype != torch.float32 or value.numel() != 1:
            raise TypeError(
                f"unexpected {name}-scale tensor: {value.dtype}, {value.shape}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"native {name} scale must be finite")
    contract = provenance.get("adapter_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("input_scale_applied") is not False
    ):
        raise ValueError(
            "loader did not prove that W4A16 input_scale is intentionally ignored"
        )
    if not contract.get("input_scale_ignored_reason"):
        raise ValueError("loader omitted the W4A16 input_scale ignored reason")
