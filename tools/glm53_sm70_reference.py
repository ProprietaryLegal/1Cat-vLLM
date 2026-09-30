"""Independent CPU reference for the GLM-5.3 ModelOpt NVFP4 probe.

This module intentionally does not import vLLM or any CUDA extension.  It
decodes the checkpoint's low-first E2M1 nibbles, finite E4M3 block scales, and
FP32 global scale before each FP16 projection boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

E2M1_MAGNITUDES = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32
)
SWIGLU_LIMIT = 10.0
RMS_LIMIT = 0.01
NORMALIZED_MAX_LIMIT = 0.02


def decode_e2m1(codes: np.ndarray) -> np.ndarray:
    """Decode low-first signed E2M1 nibbles to FP32."""

    packed = np.asarray(codes, dtype=np.uint8)
    if packed.ndim != 2:
        raise ValueError(f"E2M1 codes must be rank-2, got {packed.shape}")
    nibbles = np.empty(packed.size * 2, dtype=np.uint8)
    flat = packed.reshape(-1)
    nibbles[0::2] = flat & 0x0F
    nibbles[1::2] = flat >> 4
    signs = np.where((nibbles & 0x08) != 0, -1.0, 1.0).astype(np.float32)
    values = signs * E2M1_MAGNITUDES[nibbles & 0x07]
    return values.reshape(packed.shape[0], -1)


def decode_e4m3(scales: np.ndarray) -> np.ndarray:
    """Decode all finite E4M3FN scale bytes; reject NaN encodings."""

    raw = np.asarray(scales, dtype=np.uint8)
    exponent = ((raw >> 3) & 0x0F).astype(np.int32)
    mantissa = (raw & 0x07).astype(np.int32)
    sign = np.where((raw & 0x80) != 0, -1.0, 1.0).astype(np.float32)
    output = np.empty(raw.shape, dtype=np.float32)
    subnormal = exponent == 0
    normal = (exponent > 0) & (exponent < 15)
    extended = (exponent == 15) & (mantissa < 7)
    invalid = (exponent == 15) & (mantissa == 7)
    output[subnormal] = mantissa[subnormal] * 2.0**-9
    output[normal] = (1.0 + mantissa[normal] / 8.0) * np.exp2(
        exponent[normal] - 7
    )
    output[extended] = (1.0 + mantissa[extended] / 8.0) * 2.0**8
    output[invalid] = np.nan
    decoded = sign * output
    if not np.isfinite(decoded).all():
        raise ValueError("E4M3 scale payload contains a nonfinite encoding")
    return decoded.astype(np.float32, copy=False)


def dequantize(
    codes: np.ndarray, scales: np.ndarray, global_scale: float
) -> np.ndarray:
    """Return ``[N,K]`` FP32 weights using ModelOpt's W4A16 formula."""

    packed = np.asarray(codes, dtype=np.uint8)
    group_scales = np.asarray(scales, dtype=np.uint8)
    expected_shape = (packed.shape[0], packed.shape[1] * 2 // 16)
    if packed.ndim != 2 or group_scales.shape != expected_shape:
        raise ValueError(
            f"invalid NVFP4 geometry codes={packed.shape}, scales={group_scales.shape}"
        )
    values = decode_e2m1(packed).reshape(packed.shape[0], -1, 16)
    effective = decode_e4m3(group_scales)[..., None] * np.float32(global_scale)
    return (values * effective).reshape(packed.shape[0], -1).astype(np.float32)


def projection_reference(
    inputs: torch.Tensor,
    codes: np.ndarray,
    scales: np.ndarray,
    global_scale: float,
) -> torch.Tensor:
    """Compute an FP32 CPU matmul after explicitly rounding inputs to FP16."""

    if inputs.device.type != "cpu" or inputs.dtype != torch.float16 or inputs.ndim != 2:
        raise ValueError("reference inputs must be CPU FP16 [M,K]")
    weights = torch.from_numpy(dequantize(codes, scales, global_scale))
    result = inputs.float() @ weights.t()
    rounded = result.half()
    if not bool(torch.isfinite(rounded).all()):
        raise ValueError("reference projection overflowed FP16")
    return rounded


def swiglu_reference(
    gate: torch.Tensor, up: torch.Tensor, limit: float = SWIGLU_LIMIT
) -> torch.Tensor:
    """Independent FP16-boundary GLM SwiGLU with the model's input clamps."""

    if (
        gate.shape != up.shape
        or gate.dtype != torch.float16
        or up.dtype != torch.float16
    ):
        raise ValueError("SwiGLU inputs must be equal-shape FP16 tensors")
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError(f"SwiGLU limit must be positive and finite, got {limit!r}")
    gate_f = gate.float().numpy()
    gate_f = np.minimum(gate_f, np.float32(limit))
    up_f = np.clip(up.float().numpy(), -np.float32(limit), np.float32(limit))
    exp_term = np.exp(-np.abs(gate_f))
    sigmoid = np.where(gate_f >= 0, 1.0 / (1.0 + exp_term), exp_term / (1.0 + exp_term))
    result = gate_f * sigmoid * up_f
    rounded = torch.from_numpy(result.astype(np.float32)).half()
    if not bool(torch.isfinite(rounded).all()):
        raise ValueError("reference SwiGLU overflowed FP16")
    return rounded


def mlp_reference(
    hidden: torch.Tensor,
    projections: dict[str, tuple[np.ndarray, np.ndarray, float]],
) -> torch.Tensor:
    """Evaluate independent gate/up/SwiGLU/down FP16 projection boundaries."""

    gate = projection_reference(hidden, *projections["gate_proj"])
    up = projection_reference(hidden, *projections["up_proj"])
    activated = swiglu_reference(gate, up)
    return projection_reference(activated, *projections["down_proj"])


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    """Return fixed probe metrics and admission decision."""

    if actual.shape != expected.shape:
        raise ValueError(
            f"output shape mismatch: {actual.shape} versus {expected.shape}"
        )
    actual_f = actual.detach().float().cpu()
    expected_f = expected.detach().float().cpu()
    if not bool(torch.isfinite(actual_f).all() and torch.isfinite(expected_f).all()):
        raise ValueError("comparison contains nonfinite values")
    delta = actual_f - expected_f
    absolute_rms = float(delta.square().mean().sqrt())
    expected_rms = max(float(expected_f.square().mean().sqrt()), 1e-8)
    expected_absmax = max(float(expected_f.abs().max()), 1e-8)
    absolute_max = float(delta.abs().max())
    relative_rms = absolute_rms / expected_rms
    normalized_max = absolute_max / expected_absmax
    return {
        "absolute_rms": absolute_rms,
        "relative_rms": relative_rms,
        "absolute_max": absolute_max,
        "normalized_max_abs": normalized_max,
        "relative_rms_limit": RMS_LIMIT,
        "normalized_max_abs_limit": NORMALIZED_MAX_LIMIT,
        "passed": relative_rms <= RMS_LIMIT and normalized_max <= NORMALIZED_MAX_LIMIT,
    }


__all__ = [
    "NORMALIZED_MAX_LIMIT",
    "RMS_LIMIT",
    "SWIGLU_LIMIT",
    "decode_e2m1",
    "decode_e4m3",
    "dequantize",
    "metrics",
    "mlp_reference",
    "projection_reference",
    "swiglu_reference",
]
