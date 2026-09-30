# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for the bounded standard grouped SM70 probe."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import weakref
from pathlib import Path

import pytest
import torch

from tools.glm53_sm70_grouped import (
    GLOBAL_EXPERTS,
    SELECTED_EXPERTS,
    _record_scales,
    _require_asymmetric_globals,
    build_expert_map,
    build_route_plan,
    validate_contract,
)
from tools.glm53_sm70_grouped_cuda import (
    _stack_prepared,
    pointer_bundle,
)
from tools.glm53_sm70_grouped_reference import (
    merged_scale_dequantize,
    route_reference,
)
from tools.glm53_sm70_primitive import NativeProjection
from tools.glm53_sm70_reference import dequantize


def _native_projection(global_scale: float) -> NativeProjection:
    return NativeProjection(
        "gate_proj",
        torch.tensor(
            [[0x11, 0x21, 0x31, 0x41, 0x51, 0x61, 0x71, 0x81]],
            dtype=torch.uint8,
        ),
        torch.tensor([[0x38]], dtype=torch.uint8),
        torch.tensor(global_scale, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
        {"adapter_contract": {"input_scale_applied": False}},
    )


def _prepared(width: int) -> tuple[tuple[torch.Tensor, ...], ...]:
    meta = torch.tensor([16, width], dtype=torch.int32)
    w13 = torch.arange(width, dtype=torch.uint8).view(1, width)
    s13 = torch.arange(width, dtype=torch.float16).view(1, width)
    w2 = torch.arange(width, dtype=torch.uint8).view(1, width)
    s2 = torch.arange(width, dtype=torch.float16).view(1, width)
    return ((w13, s13, meta), (w2, s2, meta.clone()))


def test_contract_rejects_bool_and_wrong_token_counts() -> None:
    assert validate_contract(2, 1) == 2
    assert validate_contract(4, 8) == 16
    with pytest.raises(ValueError, match="tp_size"):
        validate_contract(True, 1)
    with pytest.raises(ValueError, match="token_count"):
        validate_contract(2, 3)
    with pytest.raises(ValueError, match="route_weight_sum"):
        validate_contract(2, 1, 3.0)


def test_route_plan_repeats_two_real_experts_with_unequal_weights() -> None:
    for route_weight_sum in (1.0, 2.5):
        plan = build_route_plan(8, route_weight_sum)
        assert plan.topk_ids.dtype == torch.int32
        assert plan.topk_ids.shape == (8, 2)
        assert plan.topk_weights.dtype == torch.float32
        assert torch.equal(plan.topk_ids.unique(), torch.tensor(SELECTED_EXPERTS))
        assert torch.allclose(
            plan.topk_weights.sum(dim=1), torch.full((8,), route_weight_sum)
        )
        assert not torch.allclose(plan.topk_weights[:, 0], plan.topk_weights[:, 1])


def test_expert_map_only_exposes_selected_global_ids() -> None:
    mapping = build_expert_map(torch.device("cpu"))
    assert mapping.shape == (GLOBAL_EXPERTS,)
    assert mapping.dtype == torch.int32
    assert mapping[36].item() == 0
    assert mapping[79].item() == 1
    assert mapping[0].item() == -1
    assert mapping[287].item() == -1


def test_secondary_oracle_rounds_merged_scale_before_e2m1() -> None:
    projection = _native_projection(1.0003)
    merged = merged_scale_dequantize(projection)
    primary = torch.from_numpy(
        dequantize(
            projection.codes.numpy(),
            projection.scales.numpy(),
            projection.global_scale.item(),
        )
    )
    assert merged.shape == primary.shape
    assert not torch.equal(merged, primary)


def test_cpu_route_reference_preserves_weighted_combine_contract() -> None:
    hidden = torch.ones((2, 4), dtype=torch.float16)
    weights = {
        expert: {
            "gate_proj": torch.full((3, 4), float(expert == 36)),
            "up_proj": torch.ones((3, 4)),
            "down_proj": torch.full((4, 3), 1.0 + 0.1 * index),
        }
        for index, expert in enumerate(SELECTED_EXPERTS)
    }
    outputs = {}
    for route_weight_sum in (1.0, 2.5):
        plan = build_route_plan(2, route_weight_sum)
        output = route_reference(
            hidden,
            plan.topk_ids,
            plan.topk_weights,
            weights,
            SELECTED_EXPERTS,
            hidden_size=4,
        )
        outputs[route_weight_sum] = output
        assert output.shape == (2, 4)
        assert output.dtype == torch.float16
        assert bool(torch.isfinite(output).all())
        assert not torch.equal(output[0], output[1])
    torch.testing.assert_close(
        outputs[2.5].float(), outputs[1.0].float() * 2.5, rtol=1e-3, atol=1e-2
    )


def test_compact_pointer_rows_duplicate_only_for_empty_slots() -> None:
    prepared = {36: _prepared(4), 79: _prepared(4)}
    w13, w2 = _stack_prepared(prepared, SELECTED_EXPERTS, 8)
    assert w13[0].shape[0] == 8
    assert w2[0].shape[0] == 8
    assert torch.equal(w13[0][0], prepared[36][0][0])
    assert torch.equal(w13[0][1], prepared[79][0][0])
    assert torch.equal(w13[0][2], prepared[36][0][0])
    assert torch.equal(w2[0][3], prepared[79][1][0])


def test_pointer_bundle_owns_storage_backing_raw_pointer_arrays() -> None:
    import vllm._sm70_ops as sm70_ops

    prepared = {36: _prepared(4), 79: _prepared(4)}
    weak_buffers: list[weakref.ReferenceType[torch.Tensor]] = []

    def fake_pointer_builder(
        weights: torch.Tensor,
        scales: torch.Tensor,
        _meta0: int,
        _meta1: int,
        _slots: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weak_buffers.extend((weakref.ref(weights), weakref.ref(scales)))
        return torch.tensor([1]), torch.tensor([2])

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(sm70_ops, "awq_moe_build_strided_ptrs", fake_pointer_builder)
    try:
        bundle = pointer_bundle(prepared, SELECTED_EXPERTS, 8, local_intermediate=2)
    finally:
        monkeypatch.undo()
    gc.collect()
    assert len(weak_buffers) == 4
    assert all(reference() is not None for reference in weak_buffers)
    assert bundle.w13_storage[0] is weak_buffers[0]()
    assert bundle.w13_storage[1] is weak_buffers[1]()
    assert bundle.w2_storage[0] is weak_buffers[2]()
    assert bundle.w2_storage[1] is weak_buffers[3]()


def test_asymmetric_global_requirement_is_fail_closed() -> None:
    gate = _native_projection(1.0)
    up_equal = NativeProjection(
        "up_proj",
        gate.codes,
        gate.scales,
        gate.global_scale,
        gate.input_scale,
        gate.provenance,
    )
    down = NativeProjection(
        "down_proj",
        gate.codes,
        gate.scales,
        gate.global_scale,
        gate.input_scale,
        gate.provenance,
    )
    with pytest.raises(RuntimeError, match="synthetic"):
        _require_asymmetric_globals(
            {36: {"gate_proj": gate, "up_proj": up_equal, "down_proj": down}}
        )
    up_asymmetric = NativeProjection(
        "up_proj",
        gate.codes,
        gate.scales,
        torch.tensor(2.0),
        gate.input_scale,
        gate.provenance,
    )
    _require_asymmetric_globals(
        {36: {"gate_proj": gate, "up_proj": up_asymmetric, "down_proj": down}}
    )


def test_scale_coverage_labels_keep_equal_real_and_synthetic_paths_separate() -> None:
    gate = _native_projection(1.0)
    up = NativeProjection(
        "up_proj",
        gate.codes,
        gate.scales,
        gate.global_scale,
        gate.input_scale,
        gate.provenance,
    )
    down = NativeProjection(
        "down_proj",
        gate.codes,
        gate.scales,
        gate.global_scale,
        gate.input_scale,
        gate.provenance,
    )
    equal = _record_scales({36: {"gate_proj": gate, "up_proj": up, "down_proj": down}})
    assert equal["scale_coverage_label"] == "ordinary_real_equal_gate_up_globals"
    assert equal["synthetic_asymmetric_coverage_required"] is True
    asymmetric = NativeProjection(
        "up_proj",
        gate.codes,
        gate.scales,
        torch.tensor(2.0),
        gate.input_scale,
        gate.provenance,
    )
    synthetic = _record_scales(
        {36: {"gate_proj": gate, "up_proj": asymmetric, "down_proj": down}}
    )
    assert synthetic["scale_coverage_label"] == "synthetic_asymmetric_gate_up_globals"
    assert synthetic["synthetic_asymmetric_coverage_required"] is False


def test_cli_emits_real_grouped_contract_report() -> None:
    repo = Path(__file__).resolve().parents[2]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": str(repo),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_glm53_sm70_grouped.py",
            "--tokens",
            "1,2",
            "--tp-size",
            "4",
            "--rank",
            "0",
        ],
        cwd=repo,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["status"] == "DRY_RUN"
    assert report["standard_grouped_dispatch"] is True
    assert report["multi_expert_grouped_execution"] is True
    assert report["experimental_grouped_expert_rows"] is False
    assert report["fused_q8_dispatch"] is False
    assert report["selected_global_experts"] == [36, 79]
    assert report["component_top_k"] == 2
    assert report["model_top_k"] == 8
    assert report["component_route_weight_sum"] == 1.0
    assert report["component_route_weight_sum_variants"] == [1.0, 2.5]
    assert report["model_route_weight_sum"] == 2.5
    assert report["limits"] == {"relative_rms": 0.01, "normalized_max_abs": 0.02}
    closure = report["grouped_source_closure"]
    assert closure["csrc/moe/moe_permute_unpermute_op.cu"]
    assert all(len(digest) == 64 for digest in closure.values())
