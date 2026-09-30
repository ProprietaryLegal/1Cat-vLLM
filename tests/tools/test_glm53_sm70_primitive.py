"""CPU contract tests for the bounded GLM-5.3 SM70 primitive probe."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.glm53_sm70_contract import expected_native_shape, validate_nvfp4_scalars
from tools.glm53_sm70_expert import _decode_scale_bytes
from tools.glm53_sm70_primitive import _git_head
from tools.glm53_sm70_reference import (
    SWIGLU_LIMIT,
    decode_e2m1,
    decode_e4m3,
    dequantize,
    metrics,
    projection_reference,
    swiglu_reference,
)


def test_e2m1_decodes_low_first_signed_nibbles() -> None:
    decoded = decode_e2m1(np.asarray([[0x10, 0xF8, 0x76, 0xED]], dtype=np.uint8))
    expected = np.asarray(
        [[0.0, 0.5, -0.0, -6.0, 4.0, 6.0, -3.0, -4.0]], dtype=np.float32
    )
    np.testing.assert_array_equal(decoded, expected)


def test_e4m3_matches_torch_for_all_finite_encodings() -> None:
    raw = torch.arange(256, dtype=torch.uint8)
    expected = raw.view(torch.float8_e4m3fn).float().numpy()
    finite = np.isfinite(expected)
    actual = decode_e4m3(np.arange(256, dtype=np.uint8)[finite])
    np.testing.assert_array_equal(actual, expected[finite])


def test_e4m3_rejects_nan_encoding() -> None:
    with pytest.raises(ValueError, match="nonfinite"):
        decode_e4m3(np.asarray([[0x7F]], dtype=np.uint8))


def test_scale_bytes_are_presented_as_float8_without_changing_bytes() -> None:
    raw = torch.tensor([[0x38, 0x40]], dtype=torch.uint8)
    decoded = _decode_scale_bytes(raw)
    assert decoded.dtype == torch.float32
    assert decoded.tolist() == [[1.0, 2.0]]
    assert raw.tolist() == [[0x38, 0x40]]


def test_dequantize_applies_independent_global_scale() -> None:
    codes = np.asarray(
        [[0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0]],
        dtype=np.uint8,
    )
    scales = np.asarray([[0x38]], dtype=np.uint8)
    decoded = dequantize(codes, scales, 3.0)
    assert decoded.shape == (1, 16)
    np.testing.assert_array_equal(decoded, decode_e2m1(codes) * 3.0)


def test_projection_reference_rounds_output_to_fp16() -> None:
    codes = np.full((4, 8), 0x11, dtype=np.uint8)
    scales = np.full((4, 1), 0x38, dtype=np.uint8)
    hidden = torch.ones((2, 16), dtype=torch.float16)
    result = projection_reference(hidden, codes, scales, 2.0)
    assert result.dtype == torch.float16
    assert result.shape == (2, 4)
    assert torch.equal(result, torch.full((2, 4), 16.0, dtype=torch.float16))


def test_swiglu_reference_has_explicit_fp16_boundary() -> None:
    gate = torch.tensor([[0.5, -1.0]], dtype=torch.float16)
    up = torch.tensor([[2.0, 3.0]], dtype=torch.float16)
    result = swiglu_reference(gate, up)
    expected = (gate.float() * torch.sigmoid(gate.float()) * up.float()).half()
    assert torch.equal(result, expected)


def test_swiglu_reference_applies_glm_gate_and_up_clamps() -> None:
    gate = torch.tensor([[20.0, -20.0]], dtype=torch.float16)
    up = torch.tensor([[-20.0, 20.0]], dtype=torch.float16)
    result = swiglu_reference(gate, up)
    clipped_gate = gate.float().clamp(max=SWIGLU_LIMIT)
    clipped_up = up.float().clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    expected = (torch.nn.functional.silu(clipped_gate) * clipped_up).half()
    assert torch.equal(result, expected)


def test_expected_native_shapes_cover_tp2_and_tp4_axes() -> None:
    assert expected_native_shape(0, "gate_proj", 2) == (6144, 2048)
    assert expected_native_shape(0, "gate_proj", 4) == (3072, 2048)
    assert expected_native_shape(10, "down_proj", 2) == (4096, 512)
    assert expected_native_shape(10, "down_proj", 4) == (4096, 256)


def test_input_scale_contract_requires_explicit_ignored_reason() -> None:
    provenance = {
        "adapter_contract": {
            "input_scale_applied": False,
            "input_scale_ignored_reason": (
                "W4A16 route validates but does not apply input_scale."
            ),
        }
    }
    validate_nvfp4_scalars(
        torch.tensor(2.0, dtype=torch.float32),
        torch.tensor(3.0, dtype=torch.float32),
        provenance,
    )
    with pytest.raises(ValueError, match="ignored reason"):
        validate_nvfp4_scalars(
            torch.tensor(2.0, dtype=torch.float32),
            torch.tensor(3.0, dtype=torch.float32),
            {"adapter_contract": {"input_scale_applied": False}},
        )


def test_metrics_fixed_limits_and_shape_guard() -> None:
    expected = torch.ones((2, 3), dtype=torch.float16)
    result = metrics(expected.clone(), expected)
    assert result["passed"] is True
    assert result["relative_rms_limit"] == 0.01
    assert result["normalized_max_abs_limit"] == 0.02
    with pytest.raises(ValueError, match="shape mismatch"):
        metrics(expected[:1], expected)


def test_probe_source_head_is_bound_to_probe_checkout() -> None:
    expected = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    assert _git_head() == expected
