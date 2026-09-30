# PLI SM70 source snapshot

This is a scrubbed source release of the Apache-2.0 1Cat-vLLM fork at
`00193c049ca91eea0883bf7edd35c184cd6a35fd`, based on upstream
`2c8b9baa7c4803d84cb8790e41153d47867c08a9`.
The archive intentionally contains no Git database, deployment credentials,
private operational notes, virtual environment, model weights, or native binaries.
Original upstream history also contained private contributor identity metadata;
the publication ref is therefore a clean parentless snapshot. Original source
license headers and upstream attribution remain intact.

## Build in a dedicated environment

Use Linux x86_64, Python 3.12, PyTorch 2.10, and CUDA toolkit **12.8.93**.
Volta V100 requires architecture **SM70**; do not build with a toolkit that has
removed Volta support. Install `uv`, then preview the included build script:

```bash
bash tools/bootstrap_glm53_sm70.sh --cuda-home /opt/toolchains/cuda-12.8.93
```

Build only in a new checkout and virtual environment:

```bash
bash tools/bootstrap_glm53_sm70.sh --execute \
  --cuda-home /opt/toolchains/cuda-12.8.93 --venv "$PWD/.venv"
```

The bootstrap pins SM70 architecture flags, uses `uv` for dependencies, and
limits compiler parallelism. The build was not repeated during publication.
No wheel is supplied: existing local binaries retained private build-path strings.

## Serving profile

Use four peer-connected V100-SXM2 32GB GPUs on one NVLink board. Check GPU UUIDs
with `nvidia-smi -L` and set `CUDA_VISIBLE_DEVICES` to the chosen UUIDs.
The observed reference profile uses:

```bash
.venv/bin/vllm serve /models/Qwen3.8-27B-NVFP4 \
  --host 127.0.0.1 --port 8000 --trust-remote-code \
  --dtype half --tensor-parallel-size 4 --attention-backend FLASH_ATTN_V100 \
  --kv-cache-dtype fp8_e5m2 --max-model-len 262144 \
  --gpu-memory-utilization 0.80 --max-num-batched-tokens 8192 --max-num-seqs 4 \
  --enable-prefix-caching --mamba-cache-mode align \
  --block-size 2048 --mamba-block-size 8192 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}' --seed 0 \
  --speculative-config '{"method":"dflash","model":"/models/Qwen3.8-27B-DFlash2","num_speculative_tokens":7,"kv_cache_dtype":"auto","attention_backend":"FLASH_ATTN_V100","draft_sample_method":"probabilistic","enforce_eager":false}'
```

Checkpoint paths are placeholders. The included
`scripts/serve_qwen38_27b_nvfp4_v100.sh` provides the upstream portable launcher
and pinned public draft checkpoint revision. Capacity and performance require
validation on the deployment hardware; this publication does not certify them.

## Regression checks

After building the dedicated environment:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  --noconftest -p no:cacheprovider \
  tests/v1/attention/test_sm70_flash_v100_policy.py \
  tests/v1/core/test_engine_core_structured_drafts.py \
  tests/tools/test_glm53_sm70_primitive.py \
  tests/tools/test_glm53_sm70_grouped.py \
  tests/tools/test_glm53_gate_probe.py \
  tests/tools/test_glm53_sampler_capture.py \
  tests/tools/test_sm70_release_artifact.py -q
```

The policy suite covers Q8001–Q8191 prefill with KV below 8192: those shapes
must take the dense route rather than the padded Q8192 specialization.
CPU contract tests do not establish numerical GPU or model-quality parity.
Optional real-checkpoint probes require separately provided model artifacts
and reference-loader sources; these are not part of the release.

## Attribution and modifications

Upstream: https://github.com/1CatAI/1Cat-vLLM and https://github.com/vllm-project/vllm.
Keep `LICENSE` and all source copyright/license headers when redistributing.
No upstream NOTICE file was present in the audited source tree.
Publication modifications omit internal engineering notes and replace machine
addresses and absolute deployment paths with neutral examples. Serving kernel
and engine fixes are preserved. AI assistance was used for publication preparation.
