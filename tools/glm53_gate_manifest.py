# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict provenance checks for the pinned GLM router-input capture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_REVISION = "09b04e5e74bca08ca8549fc736d4cdd8624bfde3"
EXPECTED_INDEX_SHA256 = (
    "26765b2601fd246ef361cfb9f5e10f9fb291a59e05ad0a109062f3a4747c7fd1"
)
EXPECTED_CONFIG_SHA256 = (
    "e23c5d98f53e861d49a51bd3c68591621c5482ce829e42c31724152322fba03d"
)
EXPECTED_INPUT_SHA256 = (
    "2d781a9e04d556d1673a562891b4676f784dc2085fa7740924fd36513f231ad4"
)
EXPECTED_MANIFEST_SHA256 = (
    "da4b257861202bbc971faf88be2e2d5b71024dd1eaab1827cc66306b0638c346"
)
EXPECTED_INPUT_IDS_SHA256 = (
    "9a0865b36830428082fd28c7f4f32fca7806af21fcb4c2210a80e6a1db32d008"
)


class ManifestError(RuntimeError):
    """The capture manifest or tensor does not match the pinned evidence."""


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ManifestError(f"cannot hash {path}: {exc}") from exc


def validate_capture_manifest(
    manifest_path: Path,
    input_path: Path,
    *,
    layer: int,
    revision: str,
    index_sha: str,
) -> tuple[str, str, str]:
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw, object_pairs_hook=_unique_pairs)
    except (OSError, json.JSONDecodeError, ManifestError) as exc:
        raise ManifestError(
            f"cannot read input manifest {manifest_path}: {exc}"
        ) from exc
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ManifestError(
            f"unexpected input manifest sha256 {manifest_sha}; "
            f"expected {EXPECTED_MANIFEST_SHA256}"
        )
    provenance = manifest.get("checkpoint_provenance", {})
    config = manifest.get("config_provenance", {})
    expected = (
        manifest.get("format") == "glm53-moe-diagnostic-v1"
        and provenance.get("revision") == revision
        and provenance.get("index_sha256") == index_sha
        and config.get("sha256") == EXPECTED_CONFIG_SHA256
        and manifest.get("layer_index") == layer
        and manifest.get("rank") == 4
        and manifest.get("physical_gpu") == 8
        and manifest.get("tp_size") == 4
        and manifest.get("layout_id") == "eight_tp4"
        and manifest.get("phase") == "prefill"
        and manifest.get("max_new_tokens") == 1
        and manifest.get("input_ids_sha256") == EXPECTED_INPUT_IDS_SHA256
    )
    if not expected:
        raise ManifestError("input manifest does not match the pinned layer-33 capture")
    entry = manifest.get("files", {}).get(input_path.name)
    if not isinstance(entry, dict) or entry.get("semantic") != (
        "normalized_router_input"
    ):
        raise ManifestError("input manifest has no normalized router-input entry")
    input_sha = _sha256(input_path)
    if (
        input_sha != EXPECTED_INPUT_SHA256
        or input_sha != entry.get("sha256")
        or entry.get("bytes") != input_path.stat().st_size
        or entry.get("dtype") != "torch.float16"
        or entry.get("shape") != [19, 4096]
        or manifest.get("full_token_count") != 19
    ):
        raise ManifestError("captured router input does not match its manifest")
    return manifest_sha, input_sha, str(manifest.get("input_ids_sha256"))
