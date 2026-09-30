# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only references for the bounded grouped GLM-5.3 probe."""

from __future__ import annotations

import hashlib
import importlib
import math
import struct
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from tools.glm53_sm70_primitive import NativeProjection
from tools.glm53_sm70_reference import (
    decode_e2m1,
    decode_e4m3,
    dequantize,
    swiglu_reference,
)

ProjectionWeights = dict[str, torch.Tensor]
WeightCache = dict[int, ProjectionWeights]


def merged_scale_dequantize(projection: NativeProjection) -> torch.Tensor:
    """Decode after FP16 rounding of merged block/global scales.

    The primary oracle keeps E4M3 block scales and the FP32 checkpoint global
    scalar separate.  This diagnostic oracle models the SM70 preparation
    boundary by rounding their product to FP16 before multiplying E2M1 values.
    """

    codes = projection.codes.numpy()
    values = torch.from_numpy(decode_e2m1(codes)).view(codes.shape[0], -1, 16)
    scales = torch.from_numpy(decode_e4m3(projection.scales.numpy()))
    merged = (scales * projection.global_scale.item()).half().float()
    return (values * merged.unsqueeze(-1)).reshape(codes.shape[0], -1).float()


def _projection_from_weight(inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if inputs.dtype != torch.float16 or inputs.device.type != "cpu":
        raise ValueError("reference inputs must be CPU FP16")
    output = (inputs.float() @ weight.t()).half()
    if not bool(torch.isfinite(output).all()):
        raise ValueError("reference projection overflowed FP16")
    return output


def _mlp_from_weights(hidden: torch.Tensor, weights: ProjectionWeights) -> torch.Tensor:
    gate = _projection_from_weight(hidden, weights["gate_proj"])
    up = _projection_from_weight(hidden, weights["up_proj"])
    activated = swiglu_reference(gate, up)
    return _projection_from_weight(activated, weights["down_proj"])


def build_weight_cache(
    projections: dict[int, dict[str, NativeProjection]], merged: bool
) -> WeightCache:
    """Materialize only the selected expert union for a CPU oracle."""

    result: WeightCache = {}
    decoder: Callable[[NativeProjection], torch.Tensor]
    if merged:
        decoder = merged_scale_dequantize
    else:
        decoder = lambda projection: torch.from_numpy(
            dequantize(
                projection.codes.numpy(),
                projection.scales.numpy(),
                float(projection.global_scale.item()),
            )
        )
    for expert, roles in projections.items():
        result[expert] = {
            role: decoder(projection) for role, projection in roles.items()
        }
    return result


def route_reference(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: WeightCache,
    selected_experts: tuple[int, int],
    hidden_size: int,
) -> torch.Tensor:
    """Reference route/MLP/combine with FP32 weighted reduction."""

    if hidden.dtype != torch.float16 or hidden.device.type != "cpu":
        raise ValueError("reference hidden must be CPU FP16")
    if topk_ids.dtype != torch.int32 or topk_ids.shape[1] != 2:
        raise ValueError("reference topk ids must be int32 [tokens,2]")
    if topk_weights.dtype != torch.float32 or topk_weights.shape != topk_ids.shape:
        raise ValueError("reference route weights must be FP32 and match ids")
    output = torch.zeros(hidden.shape[0], hidden_size, dtype=torch.float32)
    for slot in range(2):
        for expert in selected_experts:
            rows = topk_ids[:, slot] == expert
            if bool(rows.any()):
                expert_output = _mlp_from_weights(hidden[rows], weights[expert])
                output[rows] += expert_output.float() * topk_weights[rows, slot, None]
    rounded = output.half()
    if not bool(torch.isfinite(rounded).all()):
        raise ValueError("reference MoE combine overflowed FP16")
    return rounded


def scan_scale2_metadata(
    checkpoint: str | Path,
    revision: str,
    reference_root: str | Path,
    *,
    first_layer: int = 3,
    last_layer: int = 44,
    expert_count: int = 288,
) -> dict[str, Any]:
    """Scan routed gate/up FP32 globals without reading packed weight payloads."""

    root = Path(checkpoint).resolve()
    source_root = str(Path(reference_root).resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    catalog_module = importlib.import_module("nvfp4_catalog")
    io_module = importlib.import_module("checkpoint_io")
    catalog = catalog_module.load_nvfp4_catalog(root, revision)
    source_files = {}
    for filename in ("nvfp4_catalog.py", "checkpoint_io.py"):
        source_path = Path(source_root) / filename
        if not source_path.is_file():
            raise RuntimeError(f"metadata scanner source is missing: {source_path}")
        source_files[filename] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    observations: list[dict[str, Any]] = []
    asymmetric: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for layer in range(first_layer, last_layer + 1):
        for expert in range(expert_count):
            roles: dict[str, dict[str, Any]] = {}
            values: dict[str, float] = {}
            for role in ("gate_proj", "up_proj"):
                name = (
                    f"model.language_model.layers.{layer}.mlp.experts."
                    f"{expert}.{role}.weight_scale_2"
                )
                record = catalog.records.get(name)
                if record is None or record.dtype != "F32" or record.shape != ():
                    raise RuntimeError(f"invalid or missing scalar record: {name}")
                payload = io_module.read_tensor(record, max_bytes=4)
                if len(payload) != 4:
                    raise RuntimeError(f"short scalar payload: {name}")
                value = float(struct.unpack("<f", payload)[0])
                if not math.isfinite(value) or value <= 0:
                    raise RuntimeError(f"nonfinite/nonpositive scalar: {name}")
                values[role] = value
                roles[role] = {
                    "name": name,
                    "shard": record.shard.name,
                    "data_offsets": list(record.data_offsets),
                    "selected_bytes": len(payload),
                    "selected_sha256": hashlib.sha256(payload).hexdigest(),
                }
            digest.update(
                f"{layer}/{expert}/{values['gate_proj']:.17g}/"
                f"{values['up_proj']:.17g}\n".encode()
            )
            observation = {
                "layer": layer,
                "expert": expert,
                "values": values,
                "provenance": roles,
            }
            observations.append(observation)
            if values["gate_proj"] != values["up_proj"]:
                asymmetric.append(observation)
    return {
        "status": "PASS",
        "checkpoint": str(root),
        "revision": revision,
        "reference_source_files_sha256": source_files,
        "revision_metadata": dict(catalog.revision_metadata),
        "snapshot": dict(catalog.snapshot),
        "index_sha256": catalog.index_sha256,
        "layers": [first_layer, last_layer],
        "expert_count": expert_count,
        "records_scanned": len(observations) * 2,
        "all_finite_positive": True,
        "asymmetric_gate_up_pairs": asymmetric,
        "asymmetric_pair_count": len(asymmetric),
        "scan_digest_sha256": digest.hexdigest(),
        "observations": observations,
    }


__all__ = [
    "build_weight_cache",
    "merged_scale_dequantize",
    "route_reference",
    "scan_scale2_metadata",
]
