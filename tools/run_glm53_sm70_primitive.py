"""CLI wrapper for the guarded GLM-5.3 SM70 primitive probe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from tools.glm53_sm70_expert import EXPERIMENTAL_GROUPED_ENV
from tools.glm53_sm70_primitive import (
    DEFAULT_CHECKPOINT,
    DEFAULT_REFERENCE_ROOT,
    DEFAULT_REVISION,
    _git_head,
    _source_provenance,
    run_gpu,
)


def _parse_tokens(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tokens must be integers") from exc
    if not values or any(item < 1 or item > 128 for item in values):
        raise argparse.ArgumentTypeError(
            "tokens must be comma-separated integers 1..128"
        )
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--tp-size", type=int, choices=(2, 4), default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--case", choices=("dense", "routed", "both"), default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokens", type=_parse_tokens, default=(1, 2, 8))
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
    cases = ("dense", "routed") if args.case == "both" else (args.case,)
    contract = {
        "status": "DRY_RUN" if not args.execute_gpu else "RUNNING",
        "source_head": _git_head(),
        "source_provenance": _source_provenance(args.reference_root),
        "tp_size": args.tp_size,
        "rank": args.rank,
        "cases": cases,
        "tokens": args.tokens,
        "standard_turbomind_single_expert_stage": True,
        "multi_expert_grouped_execution": False,
        "experimental_grouped_expert_rows": False,
        "limits": {"relative_rms": 0.01, "normalized_max_abs": 0.02},
    }
    if not args.execute_gpu:
        print(json.dumps(contract, sort_keys=True))
        return 0
    result = run_gpu(
        args.checkpoint,
        args.revision,
        args.reference_root,
        args.tp_size,
        args.rank,
        cases,
        torch.device(args.device),
        args.tokens,
    )
    print(json.dumps(result, sort_keys=True, default=str))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
