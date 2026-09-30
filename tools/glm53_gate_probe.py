# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded current-main GLM router dtype and route probe.

This intentionally constructs only ``GateLinear``.  It reads one real router
weight, one correction vector, and an optional captured router input; it never
initializes the full model.  CUDA execution is fail-closed behind the explicit
``V100_ALLOW_GATE_CUDA=1`` guard and is not part of the CPU test contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors import safe_open

try:
    from tools.glm53_gate_forward import observe_replicated_forwards
except ModuleNotFoundError:  # direct ``python tools/glm53_gate_probe.py``
    from glm53_gate_forward import observe_replicated_forwards

try:
    from tools.glm53_gate_manifest import (
        EXPECTED_INDEX_SHA256,
        EXPECTED_REVISION,
        ManifestError,
        _unique_pairs,
        validate_capture_manifest,
    )
except ModuleNotFoundError:  # direct ``python tools/glm53_gate_probe.py``
    from glm53_gate_manifest import (
        EXPECTED_INDEX_SHA256,
        EXPECTED_REVISION,
        ManifestError,
        _unique_pairs,
        validate_capture_manifest,
    )
WEIGHT_SHAPE = (288, 4096)
INPUT_SHAPE = (4096,)
TOP_K = 8


class GateProbeError(RuntimeError):
    """The bounded router probe cannot establish its contract."""


@dataclass(frozen=True)
class GateProbeResult:
    checkpoint: str
    revision: str
    index_sha256: str
    source_weight_dtype: str
    model_weight_dtype: str
    output_dtype: str
    input_dtype: str
    input_shape: tuple[int, int]
    input_sha256: str
    input_manifest_sha256: str
    input_manifest_path: str
    input_ids_sha256: str
    observed_linear_input_dtype: str
    observed_linear_weight_dtype: str
    observed_linear_output_dtype: str
    force_observed_linear_input_dtype: str
    force_observed_linear_weight_dtype: str
    force_observed_linear_output_dtype: str
    score_rms_abs: float
    score_max_abs: float
    score_normalized_max_abs: float
    routes_equal: bool
    route_mismatch_count: int
    route_weight_rms_abs: float
    route_weight_max_abs: float
    force_fp32_score_rms_abs: float
    force_fp32_score_normalized_max_abs: float
    force_fp32_routes_equal: bool
    force_fp32_route_mismatch_count: int
    device: str
    source_head: str
    source_dirty: bool


def _index(root: Path) -> tuple[dict[str, str], str]:
    path = root / "model.safetensors.index.json"
    try:
        raw = path.read_bytes()
        parsed = json.loads(raw, object_pairs_hook=_unique_pairs)
    except (OSError, json.JSONDecodeError) as exc:
        raise GateProbeError(f"cannot read index {path}: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_INDEX_SHA256:
        raise GateProbeError(
            f"unexpected index sha256 {digest}; expected {EXPECTED_INDEX_SHA256}"
        )
    weight_map = parsed.get("weight_map")
    if not isinstance(weight_map, dict):
        raise GateProbeError("index has no object weight_map")
    return weight_map, digest


def _read_tensor(
    root: Path,
    weight_map: dict[str, str],
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    shard = weight_map.get(name)
    if not isinstance(shard, str) or not shard.endswith(".safetensors"):
        raise GateProbeError(f"missing or invalid shard for {name}")
    path = root / shard
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            value = handle.get_tensor(name)
    except (OSError, KeyError, RuntimeError) as exc:
        raise GateProbeError(f"cannot read {name} from {path}: {exc}") from exc
    if value.dtype != dtype or tuple(value.shape) != shape:
        raise GateProbeError(
            f"{name}: expected {dtype} {shape}, got {value.dtype} {tuple(value.shape)}"
        )
    value = value.contiguous()
    if not bool(torch.isfinite(value.float()).all()):
        raise GateProbeError(f"{name}: nonfinite source tensor")
    return value


def load_router_sources(
    checkpoint: Path | str, layer: int
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if isinstance(layer, bool) or not isinstance(layer, int) or not 3 <= layer < 45:
        raise GateProbeError(f"invalid MoE layer: {layer!r}")
    root = Path(checkpoint).resolve()
    weight_map, index_sha = _index(root)
    base = f"model.language_model.layers.{layer}.mlp.gate"
    weight = _read_tensor(
        root, weight_map, f"{base}.weight", torch.bfloat16, WEIGHT_SHAPE
    )
    correction = _read_tensor(
        root,
        weight_map,
        f"{base}.e_score_correction_bias",
        torch.float32,
        (WEIGHT_SHAPE[0],),
    )
    return weight, correction, index_sha


def load_router_input(path: Path | str) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise GateProbeError(f"cannot load captured input {path}: {exc}") from exc
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float16:
        raise GateProbeError("captured input must be a CPU FP16 tensor")
    if (
        value.ndim != 2
        or value.shape[0] == 0
        or value.shape[1] != INPUT_SHAPE[0]
        or value.shape[0] > 128
    ):
        raise GateProbeError(
            f"captured input must have shape [T,4096], got {value.shape}"
        )
    if not bool(torch.isfinite(value).all()):
        raise GateProbeError("captured input is nonfinite")
    return value.contiguous()


def _route(
    scores: torch.Tensor, correction: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.sigmoid(scores.float())
    order = torch.argsort(
        probabilities + correction.float(), dim=-1, descending=True, stable=True
    )
    ids = order[:, :TOP_K]
    values = torch.gather(probabilities, 1, ids)
    values = values / values.sum(dim=1, keepdim=True) * 2.5
    return ids, values


def _source_provenance() -> tuple[str, bool]:
    repo = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain"], text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GateProbeError(f"cannot establish source provenance: {exc}") from exc
    return head, dirty


@contextmanager
def _world_one(device: torch.device) -> Iterable[None]:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    if torch.distributed.is_initialized():
        raise GateProbeError("probe requires a fresh process-group")
    if device.type == "cuda":
        if os.getenv("V100_ALLOW_GATE_CUDA") != "1":
            raise GateProbeError("CUDA probe requires V100_ALLOW_GATE_CUDA=1")
        if not torch.accelerator.is_available():
            raise GateProbeError("CUDA requested but unavailable")
        torch.accelerator.set_device_idx(device)
    fd, rendezvous = tempfile.mkstemp(prefix="glm53-gate-")
    os.close(fd)
    try:
        backend = "nccl" if device.type == "cuda" else "gloo"
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{rendezvous}",
                local_rank=0,
                backend=backend,
            )
            initialize_model_parallel(1, 1, backend=backend)
            yield
    finally:
        cleanup_dist_env_and_memory()
        with suppress(FileNotFoundError):
            os.unlink(rendezvous)


def run_probe(
    checkpoint: Path | str,
    captured_input: Path | str,
    *,
    layer: int = 33,
    device: str = "cpu",
    revision: str = EXPECTED_REVISION,
    manifest_path: Path | str | None = None,
) -> GateProbeResult:
    if revision != EXPECTED_REVISION:
        raise GateProbeError("revision is not the pinned NVIDIA checkpoint revision")
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise GateProbeError(f"unsupported probe device: {target}")
    source, correction, index_sha = load_router_sources(checkpoint, layer)
    input_path = Path(captured_input).resolve()
    resolved_manifest = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else input_path.with_name("manifest.json")
    )
    try:
        manifest_sha, input_sha, input_ids_sha = validate_capture_manifest(
            resolved_manifest,
            input_path,
            layer=layer,
            revision=revision,
            index_sha=index_sha,
        )
    except ManifestError as exc:
        raise GateProbeError(str(exc)) from exc
    inputs = load_router_input(input_path)
    source_head, source_dirty = _source_provenance()
    with _world_one(target):
        from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
        from vllm.utils.torch_utils import set_default_torch_dtype

        def make_gate(force_fp32_compute: bool) -> GateLinear:
            with set_default_torch_dtype(torch.float16):
                result = GateLinear(
                    4096,
                    288,
                    out_dtype=torch.float32,
                    force_fp32_compute=force_fp32_compute,
                    prefix=f"model.language_model.layers.{layer}.mlp.gate",
                )
            result = result.to(target)
            result.weight_loader(result.weight, source.to(target))
            return result

        gate = make_gate(False)
        force_gate = make_gate(True)
        x = inputs.to(target)
        with observe_replicated_forwards((gate, force_gate)) as observed:
            actual, _ = gate(x)
            force_actual, _ = force_gate(x)
        try:
            normal_dtypes = observed[id(gate)]
            force_dtypes = observed[id(force_gate)]
        except KeyError as exc:
            raise GateProbeError(
                "GateLinear did not take the observed ReplicatedLinear path"
            ) from exc
        oracle = torch.nn.functional.linear(x.float(), source.to(target).float())
        actual_ids, actual_values = _route(actual, correction.to(target))
        oracle_ids, oracle_values = _route(oracle, correction.to(target))
        force_ids, _ = _route(force_actual, correction.to(target))
        if target.type == "cuda":
            torch.accelerator.synchronize(target)
        delta = (actual.float() - oracle.float()).cpu()
        force_delta = (force_actual.float() - oracle.float()).cpu()
        reference = oracle.float().cpu()
        values_delta = (actual_values.float() - oracle_values.float()).cpu()
        return GateProbeResult(
            checkpoint=str(Path(checkpoint).resolve()),
            revision=revision,
            index_sha256=index_sha,
            source_weight_dtype=str(source.dtype),
            model_weight_dtype=str(gate.weight.dtype),
            output_dtype=str(actual.dtype),
            input_dtype=str(inputs.dtype),
            input_shape=(int(inputs.shape[0]), int(inputs.shape[1])),
            input_sha256=input_sha,
            input_manifest_sha256=manifest_sha,
            input_manifest_path=str(resolved_manifest),
            input_ids_sha256=input_ids_sha,
            observed_linear_input_dtype=normal_dtypes["input"],
            observed_linear_weight_dtype=normal_dtypes["weight"],
            observed_linear_output_dtype=normal_dtypes["output"],
            force_observed_linear_input_dtype=force_dtypes["input"],
            force_observed_linear_weight_dtype=force_dtypes["weight"],
            force_observed_linear_output_dtype=force_dtypes["output"],
            score_rms_abs=float(torch.sqrt(torch.mean(delta.square())).item()),
            score_max_abs=float(delta.abs().max().item()),
            score_normalized_max_abs=float(
                delta.abs().max().item() / max(float(reference.abs().max()), 1e-8)
            ),
            routes_equal=bool(torch.equal(actual_ids.cpu(), oracle_ids.cpu())),
            route_mismatch_count=int((actual_ids.cpu() != oracle_ids.cpu()).sum()),
            route_weight_rms_abs=float(
                torch.sqrt(torch.mean(values_delta.square())).item()
            ),
            route_weight_max_abs=float(values_delta.abs().max().item()),
            force_fp32_score_rms_abs=float(
                torch.sqrt(torch.mean(force_delta.square())).item()
            ),
            force_fp32_score_normalized_max_abs=float(
                force_delta.abs().max().item() / max(float(reference.abs().max()), 1e-8)
            ),
            force_fp32_routes_equal=bool(
                torch.equal(force_ids.cpu(), oracle_ids.cpu())
            ),
            force_fp32_route_mismatch_count=int(
                (force_ids.cpu() != oracle_ids.cpu()).sum()
            ),
            device=str(target),
            source_head=source_head,
            source_dirty=source_dirty,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("captured_input", type=Path)
    parser.add_argument("--layer", type=int, default=33)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--revision", default=EXPECTED_REVISION)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = run_probe(
        args.checkpoint,
        args.captured_input,
        layer=args.layer,
        device=args.device,
        revision=args.revision,
        manifest_path=args.manifest,
    )
    print(json.dumps(asdict(result), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
