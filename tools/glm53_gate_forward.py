# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small, scoped observer for the fallback linear call inside GateLinear."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def observe_replicated_forwards(
    targets: tuple[object, ...],
) -> Iterator[dict[int, dict[str, str]]]:
    """Record the actual dtypes entering and leaving selected ReplicatedLinear calls."""
    from vllm.model_executor.layers.linear import ReplicatedLinear

    target_ids = {id(target) for target in targets}
    if not target_ids:
        raise ValueError("at least one GateLinear target is required")
    observed: dict[int, dict[str, str]] = {}
    original = ReplicatedLinear.forward

    def wrapped(layer: object, value: torch.Tensor) -> object:
        result = original(layer, value)  # type: ignore[arg-type]
        if id(layer) in target_ids:
            output = result[0] if isinstance(result, tuple) else result
            if not isinstance(output, torch.Tensor):
                raise TypeError("ReplicatedLinear returned a non-tensor output")
            observed[id(layer)] = {
                "input": str(value.dtype),
                "weight": str(layer.weight.dtype),  # type: ignore[attr-defined]
                "output": str(output.dtype),
            }
        return result

    ReplicatedLinear.forward = wrapped  # type: ignore[method-assign]
    try:
        yield observed
    finally:
        ReplicatedLinear.forward = original  # type: ignore[method-assign]
