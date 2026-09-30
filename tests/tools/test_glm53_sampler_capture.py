# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for the existing upstream pre-process logit dump hook.

This is instrumentation validation, not model correctness or GPU validation.
The enable-file gate permits startup to finish before one isolated request is
captured. Request input IDs must be bound separately by the launch receipt.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import torch

import vllm.v1.sample.sampler as sampler
from vllm.v1.sample.metadata import SamplingMetadata


class SamplerCaptureTests(unittest.TestCase):
    def test_gated_greedy_capture_preserves_full_logits_and_limits_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "logits"
            enable = root / "enabled"
            environment = {
                "VLLM_SM70_DUMP_SAMPLER_LOGITS_DIR": str(target),
                "VLLM_SM70_DUMP_SAMPLER_LOGITS_ENABLE_FILE": str(enable),
                "VLLM_SM70_DUMP_SAMPLER_LOGITS_MAX_STEPS": "1",
            }
            metadata = cast(
                SamplingMetadata,
                SimpleNamespace(
                    temperature=torch.zeros(1),
                    top_k=None,
                    top_p=None,
                    all_greedy=True,
                    all_random=False,
                    max_num_logprobs=None,
                    output_token_ids=[[]],
                ),
            )
            logits = torch.arange(154880, dtype=torch.float32).reshape(1, -1)
            original = logits.clone()
            with (
                patch.dict(os.environ, environment),
                patch.object(sampler, "_SM70_LOGITS_DUMP_COUNTER", 0),
                patch.object(
                    sampler, "_sm70_cuda_graph_capture_active", return_value=False
                ),
            ):
                sampler._maybe_dump_sm70_sampler_logits(logits, metadata, "pre_process")
                self.assertFalse(target.exists())
                self.assertEqual(sampler._SM70_LOGITS_DUMP_COUNTER, 0)
                enable.touch(exist_ok=False)
                sampler._maybe_dump_sm70_sampler_logits(logits, metadata, "pre_process")
                files = list(target.glob("*.pt"))
                self.assertEqual(len(files), 1)
                payload = torch.load(files[0], map_location="cpu", weights_only=True)
                self.assertEqual(payload["shape"], (1, 154880))
                self.assertEqual(payload["dtype"], "torch.float32")
                self.assertEqual(payload["stage"], "pre_process")
                self.assertTrue(payload["all_greedy"])
                self.assertEqual(payload["output_token_ids"], [[]])
                torch.testing.assert_close(payload["logits"], original, rtol=0, atol=0)
                torch.testing.assert_close(logits, original, rtol=0, atol=0)
                first_bytes = files[0].read_bytes()
                sampler._maybe_dump_sm70_sampler_logits(
                    logits + 1, metadata, "pre_process"
                )
                self.assertEqual(list(target.glob("*.pt")), files)
                self.assertEqual(files[0].read_bytes(), first_bytes)

    def test_disabled_hook_does_not_capture(self) -> None:
        with patch.dict(os.environ, {"VLLM_SM70_DUMP_SAMPLER_LOGITS_DIR": ""}):
            # Metadata is deliberately absent: the disabled hook must not read it.
            sampler._maybe_dump_sm70_sampler_logits(
                torch.zeros(1), cast(SamplingMetadata, None), "pre_process"
            )


if __name__ == "__main__":
    unittest.main()
