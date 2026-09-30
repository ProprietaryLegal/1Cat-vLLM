"""Bounded real-checkpoint SM70 NVFP4 primitive probe.

The probe reads one dense or routed projection triplet from the existing strict
NVIDIA-checkpoint loader, prepares it through the current 1Cat SM70 path, and
compares either the individual linear or standard TurboMind expert stage with
an independent CPU E2M1/E4M3/FP32-global oracle.  It is intentionally not a
model, router, distributed, or performance test.

GPU execution requires the explicit ``--execute-gpu`` flag.  The default is a
dry-run contract report so source review and CPU tests cannot launch CUDA.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from tools.glm53_sm70_contract import expected_native_shape, validate_nvfp4_scalars
from tools.glm53_sm70_expert import run_standard_expert
from tools.glm53_sm70_reference import (
    metrics,
    mlp_reference,
    projection_reference,
)

ProjectionRole = Literal["gate_proj", "up_proj", "down_proj"]
ROLES: tuple[ProjectionRole, ...] = ("gate_proj", "up_proj", "down_proj")
DEFAULT_CHECKPOINT = Path("/opt/hf-models/GLM-5.3-Flash-NVFP4")
DEFAULT_REVISION = "09b04e5e74bca08ca8549fc736d4cdd8624bfde3"
DEFAULT_REFERENCE_ROOT = Path("/opt/data/v100-research/scripts/glm-5.3-flash")
REFERENCE_SOURCE_FILES = (
    "nvfp4_checkpoint.py",
    "nvfp4_checkpoint_io.py",
    "nvfp4_catalog.py",
)


@dataclass(frozen=True)
class NativeProjection:
    """One bounded native source projection and its loader provenance."""

    role: ProjectionRole
    codes: torch.Tensor
    scales: torch.Tensor
    global_scale: torch.Tensor
    input_scale: torch.Tensor
    provenance: dict[str, Any]


def _probe_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_probe_repo_root()), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_status(repo: Path) -> tuple[str, bool]:
    try:
        status = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    return hashlib.sha256(status.encode()).hexdigest(), bool(status)


def _source_provenance(reference_root: Path) -> dict[str, Any]:
    probe_root = _probe_repo_root()
    status_sha256, source_dirty = _git_status(probe_root)
    reference_status_sha256, reference_dirty = _git_status(reference_root)
    reference_hashes = {
        name: _file_sha256(reference_root / name)
        for name in REFERENCE_SOURCE_FILES
        if (reference_root / name).is_file()
    }
    return {
        "probe_repo_root": str(probe_root),
        "source_head": _git_head(),
        "source_dirty": source_dirty,
        "source_status_sha256": status_sha256,
        "reference_root": str(reference_root.resolve()),
        "reference_dirty": reference_dirty,
        "reference_status_sha256": reference_status_sha256,
        "reference_files_sha256": reference_hashes,
    }


def _load_checkpoint_api(reference_root: Path) -> Any:
    root = str(reference_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        return importlib.import_module("nvfp4_checkpoint")
    except ImportError as exc:
        raise RuntimeError(
            "the strict NVIDIA checkpoint loader is unavailable; pass "
            f"--reference-root containing nvfp4_checkpoint.py (got {root})"
        ) from exc


def load_projection(
    checkpoint: Path,
    revision: str,
    layer: int,
    role: ProjectionRole,
    rank: int,
    tp_size: int,
    *,
    expert: int | None,
    reference_root: Path,
) -> NativeProjection:
    """Load and validate one TP-local native projection only."""

    if type(tp_size) is not int or tp_size not in (2, 4):
        raise ValueError(f"tp_size must be exactly 2 or 4, got {tp_size!r}")
    if type(rank) is not int or rank not in range(tp_size):
        raise ValueError(f"rank must be in [0,{tp_size}), got {rank!r}")
    loader = _load_checkpoint_api(reference_root)
    values, provenance = loader.load_tp2_nvfp4(
        checkpoint,
        revision,
        layer,
        role,
        rank,
        expert,
        tp_size=tp_size,
    )
    required = {"weight", "weight_scale", "weight_scale_2", "input_scale"}
    if set(values) != required:
        raise ValueError(
            f"loader contract changed: expected {required}, got {set(values)}"
        )
    codes = values["weight"].detach().cpu().contiguous()
    scales = values["weight_scale"].detach().cpu().contiguous()
    global_scale = values["weight_scale_2"].detach().cpu().contiguous()
    input_scale = values["input_scale"].detach().cpu().contiguous()
    if codes.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise TypeError(f"unexpected native dtypes: {codes.dtype}, {scales.dtype}")
    if global_scale.dtype != torch.float32 or global_scale.numel() != 1:
        raise TypeError(
            "unexpected global-scale tensor: "
            f"{global_scale.dtype}, {global_scale.shape}"
        )
    validate_nvfp4_scalars(global_scale, input_scale, dict(provenance))
    expected_shape = expected_native_shape(layer, role, tp_size)
    if tuple(codes.shape) != expected_shape:
        raise ValueError(
            f"unexpected {role} native shape for layer={layer}, tp{tp_size}: "
            f"expected {expected_shape}, got {tuple(codes.shape)}"
        )
    if tuple(scales.shape) != (codes.shape[0], codes.shape[1] * 2 // 16):
        raise ValueError(
            f"invalid native scale geometry: {codes.shape}, {scales.shape}"
        )
    return NativeProjection(
        role, codes, scales, global_scale, input_scale, dict(provenance)
    )


def _reference_tuple(projection: NativeProjection) -> tuple[Any, Any, float]:
    return (
        projection.codes.numpy(),
        projection.scales.numpy(),
        float(projection.global_scale.item()),
    )


def _scale_metadata(projection: NativeProjection) -> dict[str, Any]:
    contract = projection.provenance["adapter_contract"]
    return {
        "weight_scale_2": float(projection.global_scale.item()),
        "input_scale": float(projection.input_scale.item()),
        "input_scale_applied": contract["input_scale_applied"],
        "input_scale_ignored_reason": contract["input_scale_ignored_reason"],
    }


def _scale_as_float8(raw: torch.Tensor) -> torch.Tensor:
    """Present native scale bytes as the FP8 dtype expected by TurboMind."""

    if raw.dtype != torch.uint8 or not raw.is_contiguous():
        raise TypeError("native E4M3 scales must be contiguous uint8 bytes")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("installed Torch has no float8_e4m3fn dtype")
    return raw.view(torch.float8_e4m3fn)


def _device_check(device: torch.device) -> None:
    if device.type != "cuda" or device.index is None:
        raise ValueError("GPU probe requires an explicit cuda:<index> device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 0):
        raise RuntimeError(f"SM70 probe requires capability (7, 0), got {capability}")


def _prepare_linear(
    projection: NativeProjection, device: torch.device
) -> torch.nn.Module:
    from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(projection.codes.to(device), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        _scale_as_float8(projection.scales).to(device), requires_grad=False
    )
    layer.weight_global_scale = projection.global_scale.to(device)
    sm70_tm.prepare_nvfp4_linear(layer)
    return layer


def _run_linear(
    projection: NativeProjection, hidden: torch.Tensor, device: torch.device
) -> torch.Tensor:
    from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm

    layer = _prepare_linear(projection, device)
    actual = sm70_tm.apply_prepared_linear(layer, hidden.to(device), None)
    torch.cuda.synchronize(device)
    return actual.detach().cpu()


def _hidden(length: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(length, width, generator=generator, dtype=torch.float32).half()


def _run_case(
    *,
    checkpoint: Path,
    revision: str,
    reference_root: Path,
    tp_size: int,
    rank: int,
    kind: Literal["dense", "routed"],
    device: torch.device,
    token_counts: tuple[int, ...],
) -> list[dict[str, Any]]:
    layer = 0 if kind == "dense" else 10
    expert = None if kind == "dense" else 36
    projections = {
        role: load_projection(
            checkpoint,
            revision,
            layer,
            role,
            rank,
            tp_size,
            expert=expert,
            reference_root=reference_root,
        )
        for role in ROLES
    }
    globals_ = {
        role: float(projections[role].global_scale.item()) for role in ROLES
    }
    print(
        json.dumps(
            {"kind": kind, "tp_size": tp_size, "rank": rank, "globals": globals_}
        )
    )
    records: list[dict[str, Any]] = []
    for token_count in token_counts:
        if kind == "dense":
            for role in ROLES:
                projection = projections[role]
                input_width = projection.codes.shape[1] * 2
                hidden = _hidden(token_count, input_width, 20260917 + token_count)
                expected = projection_reference(hidden, *_reference_tuple(projection))
                actual = _run_linear(projection, hidden, device)
                result = metrics(actual, expected)
                result.update(
                    {
                        "case": f"dense/layer{layer}/{role}/T{token_count}",
                        "tp_size": tp_size,
                        "rank": rank,
                        "native_shape": list(projection.codes.shape),
                        "scale_metadata": _scale_metadata(projection),
                    }
                )
                records.append(result)
        else:
            hidden = _hidden(token_count, 4096, 20260917 + token_count)
            expected = mlp_reference(
                hidden,
                {role: _reference_tuple(projections[role]) for role in ROLES},
            )
            actual = run_standard_expert(projections, hidden, device)
            result = metrics(actual, expected)
            result.update(
                {
                    "case": f"routed/layer{layer}/expert{expert}/T{token_count}",
                    "tp_size": tp_size,
                    "rank": rank,
                    "native_shapes": {
                        role: list(projections[role].codes.shape) for role in ROLES
                    },
                    "scale_metadata": {
                        role: _scale_metadata(projections[role]) for role in ROLES
                    },
                    "experimental_grouped_expert_rows": False,
                    "multi_expert_grouped_execution": False,
                }
            )
            records.append(result)
    return records


def run_gpu(
    checkpoint: Path,
    revision: str,
    reference_root: Path,
    tp_size: int,
    rank: int,
    cases: tuple[Literal["dense", "routed"], ...],
    device: torch.device,
    token_counts: tuple[int, ...],
) -> dict[str, Any]:
    source = _source_provenance(reference_root)
    if source["source_dirty"]:
        raise RuntimeError(
            "GPU probe requires a clean probe checkout; freeze and commit the "
            f"source first ({source['probe_repo_root']})"
        )
    _device_check(device)
    records: list[dict[str, Any]] = []
    for kind in cases:
        records.extend(
            _run_case(
                checkpoint=checkpoint,
                revision=revision,
                reference_root=reference_root,
                tp_size=tp_size,
                rank=rank,
                kind=kind,
                device=device,
                token_counts=token_counts,
            )
        )
    failed = [record for record in records if not record["passed"]]
    return {
        "status": "FAIL" if failed else "PASS",
        "backend": "standard_turbomind_single_expert_stage",
        "experimental_grouped_expert_rows": False,
        "multi_expert_grouped_execution": False,
        "claim_scope": "single-expert stage only; no grouped multi-expert proof",
        "source_head": source["source_head"],
        "source_provenance": source,
        "checkpoint": str(checkpoint.resolve()),
        "revision": revision,
        "records": records,
    }
