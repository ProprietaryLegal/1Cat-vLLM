# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the current-main GLM router dtype probe."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

from tools.glm53_gate_probe import (
    EXPECTED_INDEX_SHA256,
    EXPECTED_REVISION,
    GateProbeError,
    _route,
    load_router_input,
    load_router_sources,
)

CHECKPOINT = Path("/opt/hf-models/GLM-5.3-Flash-NVFP4")
CAPTURED_INPUT = Path(
    "/opt/glm53-traces/full8-tp4-moe-diagnostic-20260917T030817Z/"
    "moe/rank-04/layer-33/router_input.pt"
)


def test_real_sources_have_checkpoint_contract() -> None:
    if not CHECKPOINT.is_dir():
        pytest.skip("local NVIDIA checkpoint is unavailable")
    weight, correction, index_sha = load_router_sources(CHECKPOINT, 33)
    assert index_sha == EXPECTED_INDEX_SHA256
    assert weight.dtype == torch.bfloat16
    assert weight.shape == (288, 4096)
    assert correction.dtype == torch.float32
    assert correction.shape == (288,)
    assert bool(torch.isfinite(weight.float()).all())
    assert bool(torch.isfinite(correction).all())


def test_route_is_stable_and_normalizes_to_glm_sum() -> None:
    scores = torch.tensor([[0.1, 0.1, -1.0, 4.0]], dtype=torch.float32)
    correction = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    ids, values = _route(scores, correction)
    assert ids.tolist() == [[3, 0, 1, 2]]
    assert torch.allclose(values.sum(dim=1), torch.tensor([2.5]))


def test_input_rejects_wrong_dtype_and_rank(tmp_path: Path) -> None:
    wrong_dtype = tmp_path / "wrong_dtype.pt"
    torch.save(torch.zeros((1, 4096), dtype=torch.float32), wrong_dtype)
    with pytest.raises(GateProbeError, match="FP16"):
        load_router_input(wrong_dtype)

    wrong_rank = tmp_path / "wrong_rank.pt"
    torch.save(torch.zeros((4096,), dtype=torch.float16), wrong_rank)
    with pytest.raises(GateProbeError, match="shape"):
        load_router_input(wrong_rank)

    empty = tmp_path / "empty.pt"
    torch.save(torch.zeros((0, 4096), dtype=torch.float16), empty)
    with pytest.raises(GateProbeError, match="shape"):
        load_router_input(empty)


@pytest.mark.skipif(
    not CHECKPOINT.is_dir() or not CAPTURED_INPUT.is_file(),
    reason="real local checkpoint/captured input unavailable",
)
def test_real_captured_router_venv_integration_has_bound_receipt() -> None:
    """Exercise the actual .venv constructor path without a cbor2 skip."""
    repo = Path(__file__).resolve().parents[2]
    python = repo / ".venv/bin/python"
    script = repo / "tools/glm53_gate_probe.py"
    assert python.is_file()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [
            str(python),
            str(script),
            str(CHECKPOINT),
            str(CAPTURED_INPUT),
            "--layer",
            "33",
        ],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    start = completed.stdout.find('{\n  "checkpoint"')
    assert start >= 0, completed.stdout
    result, end = json.JSONDecoder().raw_decode(completed.stdout[start:])
    assert not completed.stdout[start + end :].strip()
    assert result["revision"] == EXPECTED_REVISION
    assert result["source_weight_dtype"] == "torch.bfloat16"
    assert result["model_weight_dtype"] == "torch.float16"
    assert result["observed_linear_input_dtype"] == "torch.float16"
    assert result["observed_linear_weight_dtype"] == "torch.float16"
    assert result["observed_linear_output_dtype"] == "torch.float16"
    assert result["force_observed_linear_input_dtype"] == "torch.float32"
    assert result["force_observed_linear_weight_dtype"] == "torch.float32"
    assert result["force_observed_linear_output_dtype"] == "torch.float32"
    assert result["input_shape"] == [19, 4096]
    assert result["routes_equal"] == (result["route_mismatch_count"] == 0)
    assert result["force_fp32_routes_equal"] == (
        result["force_fp32_route_mismatch_count"] == 0
    )
    assert result["input_manifest_sha256"]
    assert result["input_ids_sha256"]
