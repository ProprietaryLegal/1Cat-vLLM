# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded multi-expert SM70 NVFP4 component probe.

The probe uses the standard TurboMind compact grouped route: CUDA MoE
permutation, one-row compact expert groups, two NVFP4 dense stages, and the
CUDA unpermute/reduction.  It deliberately leaves
``VLLM_SM70_NVFP4_MOE_GROUPED_EXPERT_ROWS`` disabled; that switch is a
different experimental dispatch policy.

Only two real layer-10 experts are loaded.  Their global expert ids are mapped
to two local slots and repeated route rows exercise T=1, 2, and 8 without
loading a model or pretending to validate the full router.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tools.glm53_sm70_expert import (
    EXPERIMENTAL_GROUPED_ENV,
    _prepare_expert,
)
from tools.glm53_sm70_grouped_cuda import (
    pointer_bundle,
    run_grouped_once,
)
from tools.glm53_sm70_grouped_reference import (
    build_weight_cache,
    route_reference,
)
from tools.glm53_sm70_primitive import (
    DEFAULT_CHECKPOINT,
    DEFAULT_REFERENCE_ROOT,
    DEFAULT_REVISION,
    NativeProjection,
    _device_check,
    _file_sha256,
    _hidden,
    _source_provenance,
    load_projection,
)
from tools.glm53_sm70_reference import (
    NORMALIZED_MAX_LIMIT,
    RMS_LIMIT,
    SWIGLU_LIMIT,
    metrics,
)

GLOBAL_EXPERTS = 288
LAYER = 10
TOP_K = 2
MODEL_TOP_K = 8
MODEL_ROUTE_WEIGHT_SUM = 2.5
ROUTE_WEIGHT_SUMS = (1.0, 2.5)
SELECTED_EXPERTS = (36, 79)
TOKEN_COUNTS = (1, 2, 8)
HIDDEN_SIZE = 4096
GROUP_SIZE = 16
GROUPED_SOURCE_FILES = (
    "tools/glm53_sm70_grouped.py",
    "tools/glm53_sm70_grouped_cuda.py",
    "tools/glm53_sm70_grouped_reference.py",
    "tools/glm53_sm70_expert.py",
    "tools/glm53_sm70_primitive.py",
    "tools/glm53_sm70_reference.py",
    "tools/run_glm53_sm70_grouped.py",
    "vllm/_sm70_ops.py",
    "vllm/model_executor/layers/quantization/sm70_turbomind.py",
    "csrc/torch_bindings.cpp",
    "csrc/moe/torch_bindings.cpp",
    "csrc/moe/moe_permute_unpermute_op.cu",
    "csrc/sm70_turbomind/ops/awq_sm70_gemm.cu",
)


@dataclass(frozen=True)
class RoutePlan:
    """CPU route tensors before the explicit CUDA transfer."""

    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


def validate_contract(
    tp_size: int, token_count: int, route_weight_sum: float = 1.0
) -> int:
    """Validate the bounded probe contract and return expanded route rows."""

    if type(tp_size) is not int or tp_size not in (2, 4):
        raise ValueError(f"tp_size must be exactly 2 or 4, got {tp_size!r}")
    if type(token_count) is not int or token_count not in TOKEN_COUNTS:
        raise ValueError(
            f"token_count must be one of {TOKEN_COUNTS}, got {token_count!r}"
        )
    if type(route_weight_sum) is not float or route_weight_sum not in ROUTE_WEIGHT_SUMS:
        raise ValueError(
            f"route_weight_sum must be exactly 1.0 or 2.5, got {route_weight_sum!r}"
        )
    return token_count * TOP_K


def build_route_plan(token_count: int, route_weight_sum: float = 1.0) -> RoutePlan:
    """Create repeated two-expert routes at the requested weight magnitude."""

    validate_contract(2, token_count, route_weight_sum)
    rows = ((0.73, 0.27), (0.61, 0.39), (0.82, 0.18), (0.57, 0.43))
    weights = (
        torch.tensor(
            [rows[index % len(rows)] for index in range(token_count)],
            dtype=torch.float32,
        )
        * route_weight_sum
    )
    ids = torch.tensor([SELECTED_EXPERTS] * token_count, dtype=torch.int32)
    if bool(torch.allclose(weights[:, 0], weights[:, 1])):
        raise AssertionError("route weights must be unequal")
    if not bool(
        torch.allclose(weights.sum(dim=1), torch.full((token_count,), route_weight_sum))
    ):
        raise AssertionError("route weights have the wrong requested magnitude")
    return RoutePlan(ids, weights)


def build_expert_map(device: torch.device) -> torch.Tensor:
    """Map the two real global ids to compact local ids; reject other ids."""

    mapping = torch.full((GLOBAL_EXPERTS,), -1, dtype=torch.int32, device=device)
    for local_id, global_id in enumerate(SELECTED_EXPERTS):
        mapping[global_id] = local_id
    return mapping


def _require_asymmetric_globals(
    projections: dict[int, dict[str, NativeProjection]],
) -> None:
    asymmetric = [
        expert
        for expert, roles in projections.items()
        if roles["gate_proj"].global_scale.item()
        != roles["up_proj"].global_scale.item()
    ]
    if not asymmetric:
        raise RuntimeError(
            "synthetic asymmetric-global coverage requires unequal gate/up "
            "weight_scale_2 values"
        )


def _scope_metadata() -> dict[str, Any]:
    """Describe the component route separately from the full GLM router."""

    return {
        "component_top_k": TOP_K,
        "component_route_weight_sum": 1.0,
        "component_route_weight_sum_variants": list(ROUTE_WEIGHT_SUMS),
        "model_top_k": MODEL_TOP_K,
        "model_route_weight_sum": MODEL_ROUTE_WEIGHT_SUM,
        "component_scope": (
            "K=2 routes test sums 1.0 and 2.5; GLM runtime K=8 remains unexercised"
        ),
    }


def grouped_source_closure() -> dict[str, str]:
    """Hash the probe and every direct runtime source it calls."""

    root = Path(__file__).resolve().parents[1]
    closure: dict[str, str] = {}
    for relative in GROUPED_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"grouped source closure file is missing: {path}")
        closure[relative] = _file_sha256(path)
    return closure


def _load_projections(
    checkpoint: str | os.PathLike[str],
    revision: str,
    reference_root: str | os.PathLike[str],
    rank: int,
    tp_size: int,
) -> dict[int, dict[str, NativeProjection]]:
    checkpoint_path = Path(checkpoint)
    reference_path = Path(reference_root)
    result: dict[int, dict[str, NativeProjection]] = {}
    for expert in SELECTED_EXPERTS:
        result[expert] = {}
        for role in ("gate_proj", "up_proj", "down_proj"):
            result[expert][role] = load_projection(
                checkpoint=checkpoint_path,
                revision=revision,
                layer=LAYER,
                role=role,
                rank=rank,
                tp_size=tp_size,
                expert=expert,
                reference_root=reference_path,
            )
    return result


def _record_scales(
    projections: dict[int, dict[str, NativeProjection]],
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for expert, roles in projections.items():
        values[str(expert)] = {
            role: float(projection.global_scale.item())
            for role, projection in roles.items()
        }
    asymmetric = any(
        scales["gate_proj"] != scales["up_proj"] for scales in values.values()
    )
    return {
        "global_scales": values,
        "asymmetric_gate_up_globals": asymmetric,
        "scale_coverage_label": (
            "synthetic_asymmetric_gate_up_globals"
            if asymmetric
            else "ordinary_real_equal_gate_up_globals"
        ),
        "synthetic_asymmetric_coverage_required": not asymmetric,
        "projection_provenance": {
            str(expert): {
                role: projection.provenance for role, projection in roles.items()
            }
            for expert, roles in projections.items()
        },
        "input_scale_policy": "validated_and_ignored_by_W4A16",
    }


def run_gpu(
    checkpoint: str | os.PathLike[str] = DEFAULT_CHECKPOINT,
    revision: str = DEFAULT_REVISION,
    reference_root: str | os.PathLike[str] = DEFAULT_REFERENCE_ROOT,
    *,
    tp_size: int,
    rank: int,
    device: torch.device,
    token_counts: tuple[int, ...] = TOKEN_COUNTS,
    route_weight_sums: tuple[float, ...] = ROUTE_WEIGHT_SUMS,
) -> dict[str, Any]:
    """Run the guarded two-expert standard grouped route on one TP rank."""

    if os.getenv(EXPERIMENTAL_GROUPED_ENV, "0").lower() in {"1", "true", "yes"}:
        raise RuntimeError(
            f"{EXPERIMENTAL_GROUPED_ENV}=1 is experimental and refused by this probe"
        )
    if type(rank) is not int or rank not in range(tp_size):
        raise ValueError(f"rank must be in [0,{tp_size}), got {rank!r}")
    for token_count in token_counts:
        for route_weight_sum in route_weight_sums:
            validate_contract(tp_size, token_count, route_weight_sum)
    source = _source_provenance(Path(reference_root))
    if source["source_dirty"]:
        raise RuntimeError("GPU probe requires a clean, committed probe checkout")
    projections = _load_projections(checkpoint, revision, reference_root, rank, tp_size)
    _device_check(device)
    native_weights = build_weight_cache(projections, merged=False)
    merged_weights = build_weight_cache(projections, merged=True)
    prepared = {
        expert: _prepare_expert(
            roles["gate_proj"], roles["up_proj"], roles["down_proj"], device
        )
        for expert, roles in projections.items()
    }
    records: list[dict[str, Any]] = []
    for token_count in token_counts:
        hidden = _hidden(token_count, HIDDEN_SIZE, 20260917 + token_count)
        pointers = pointer_bundle(
            prepared,
            SELECTED_EXPERTS,
            token_count * TOP_K,
            local_intermediate=int(
                projections[SELECTED_EXPERTS[0]]["gate_proj"].codes.shape[0]
            ),
        )
        for route_weight_sum in route_weight_sums:
            plan = build_route_plan(token_count, route_weight_sum)
            actual, route = run_grouped_once(
                hidden,
                plan.topk_ids,
                plan.topk_weights,
                pointers,
                device,
                expert_map=build_expert_map(device),
                global_experts=GLOBAL_EXPERTS,
                top_k=TOP_K,
                hidden_size=HIDDEN_SIZE,
                swiglu_limit=SWIGLU_LIMIT,
                group_size=GROUP_SIZE,
            )
            native = metrics(
                actual,
                route_reference(
                    hidden,
                    plan.topk_ids,
                    plan.topk_weights,
                    native_weights,
                    SELECTED_EXPERTS,
                    HIDDEN_SIZE,
                ),
            )
            merged = metrics(
                actual,
                route_reference(
                    hidden,
                    plan.topk_ids,
                    plan.topk_weights,
                    merged_weights,
                    SELECTED_EXPERTS,
                    HIDDEN_SIZE,
                ),
            )
            records.append(
                {
                    "case": (
                        f"layer{LAYER}/experts{SELECTED_EXPERTS}/T{token_count}"
                        f"/sum{route_weight_sum}"
                    ),
                    "token_count": token_count,
                    "route_weight_sum": route_weight_sum,
                    "tp_size": tp_size,
                    "rank": rank,
                    "local_intermediate": pointers.local_intermediate,
                    "route_weights": plan.topk_weights.tolist(),
                    "native_oracle": native,
                    "merged_scale_oracle": merged,
                    "passed": bool(native["passed"]),
                    "secondary_merged_scale_pass": bool(merged["passed"]),
                    "routing": route,
                }
            )
    return {
        "status": "PASS" if all(item["passed"] for item in records) else "FAIL",
        "backend": "standard_turbomind_compact_grouped_moe",
        "standard_grouped_dispatch": True,
        "multi_expert_grouped_execution": True,
        "experimental_grouped_expert_rows": False,
        "claim_scope": (
            "two real layer-10 experts and one TP-local rank; "
            "K=2 routes at sums 1.0 and 2.5; no full model/router proof"
        ),
        **_scope_metadata(),
        "selected_global_experts": list(SELECTED_EXPERTS),
        "global_expert_count": GLOBAL_EXPERTS,
        "token_counts": list(token_counts),
        "route_weight_sum_variants": list(route_weight_sums),
        "limits": {
            "relative_rms": RMS_LIMIT,
            "normalized_max_abs": NORMALIZED_MAX_LIMIT,
        },
        "admission_oracle": "native_e2m1_e4m3_fp32_global",
        "secondary_merged_scale_oracle": "diagnostic_only",
        "scale_metadata": _record_scales(projections),
        "source_head": source["source_head"],
        "source_provenance": source,
        "grouped_source_closure": grouped_source_closure(),
        "checkpoint": os.fspath(checkpoint),
        "revision": revision,
        "records": records,
    }


__all__ = [
    "GLOBAL_EXPERTS",
    "GROUP_SIZE",
    "HIDDEN_SIZE",
    "LAYER",
    "RoutePlan",
    "ROUTE_WEIGHT_SUMS",
    "SELECTED_EXPERTS",
    "TOKEN_COUNTS",
    "TOP_K",
    "build_expert_map",
    "build_route_plan",
    "run_gpu",
    "validate_contract",
]
