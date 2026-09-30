# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI for the guarded two-expert standard grouped SM70 probe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from tools.glm53_sm70_expert import EXPERIMENTAL_GROUPED_ENV
from tools.glm53_sm70_grouped import (
    DEFAULT_CHECKPOINT,
    DEFAULT_REFERENCE_ROOT,
    DEFAULT_REVISION,
    ROUTE_WEIGHT_SUMS,
    SELECTED_EXPERTS,
    TOKEN_COUNTS,
    _scope_metadata,
    _source_provenance,
    grouped_source_closure,
    run_gpu,
)
from tools.glm53_sm70_primitive import _git_head


def _parse_tokens(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tokens must be integers") from exc
    if not values or any(item not in TOKEN_COUNTS for item in values):
        raise argparse.ArgumentTypeError(f"tokens must be a subset of {TOKEN_COUNTS}")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--tp-size", type=int, choices=(2, 4), default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokens", type=_parse_tokens, default=TOKEN_COUNTS)
    parser.add_argument(
        "--execute-gpu",
        action="store_true",
        help="perform the guarded CUDA probe; omitted means dry-run only",
    )
    args = parser.parse_args(argv)
    if args.rank not in range(args.tp_size):
        parser.error(f"rank must be in [0,{args.tp_size})")
    if os.getenv(EXPERIMENTAL_GROUPED_ENV, "0").lower() in {"1", "true", "yes"}:
        parser.error(f"unset experimental {EXPERIMENTAL_GROUPED_ENV} for this probe")
    contract = {
        "status": "DRY_RUN" if not args.execute_gpu else "RUNNING",
        "source_head": _git_head(),
        "source_provenance": _source_provenance(args.reference_root),
        "tp_size": args.tp_size,
        "rank": args.rank,
        "tokens": args.tokens,
        "route_weight_sum_variants": list(ROUTE_WEIGHT_SUMS),
        "selected_global_experts": list(SELECTED_EXPERTS),
        "standard_grouped_dispatch": True,
        "multi_expert_grouped_execution": True,
        "experimental_grouped_expert_rows": False,
        "fused_q8_dispatch": False,
        **_scope_metadata(),
        "admission_oracle": "native_e2m1_e4m3_fp32_global",
        "secondary_merged_scale_oracle": "diagnostic_only",
        "limits": {"relative_rms": 0.01, "normalized_max_abs": 0.02},
        "grouped_source_closure": grouped_source_closure(),
    }
    if not args.execute_gpu:
        print(json.dumps(contract, sort_keys=True, default=str))
        return 0
    result = run_gpu(
        args.checkpoint,
        args.revision,
        args.reference_root,
        tp_size=args.tp_size,
        rank=args.rank,
        device=torch.device(args.device),
        token_counts=args.tokens,
    )
    print(json.dumps(result, sort_keys=True, default=str))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
