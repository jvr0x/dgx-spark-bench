# GLM-5.3-Flash EXL3-K2 · why there is no lmswitch profile (yet)

Recipe: https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe
Weights (downloading, tmux `glm53exl3-dl`): `~/models/GLM-5.3-Flash-EXL3-K2`
(120 shards, 91.017 GiB).

## Verdict

**Not expressible as an lmswitch recipe today.** lmswitch has four runtimes:
`llama` (host llama.cpp), `vllm` (Docker), `sglang` (Docker), and the `-dual`
cluster variants. This lane serves from a **host venv**
(`~/venvs/glm53-exl3-local`, prebuilt wheels per `scripts/install_prebuilt.sh`):
stock vLLM cannot load the pack (no `exl3` quantization method, no `Glm5Next`
arch — both come from the venv's out-of-tree plugin + fork patches), and there
is **no container image** for it. A `runtime: vllm` yaml would need an `image:`
that does not exist; writing one against stock vLLM would fail at load, hours
into a 91 GiB boot. So no yaml — this file captures the exact serve spec so
containerizing later is mechanical.

## Bench serve spec (what the harness run uses)

```bash
cd ~/dev/community-recipes/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe
export PATH=/usr/local/cuda-13.0/bin:$HOME/venvs/glm53-exl3-local/bin:$PATH  # nvcc + ninja
SPEC_METHOD=mtp MTP_TOKENS=2 MAX_MODEL_LEN=65536 GPU_MEM_UTIL=0.91 PORT=8891 \
  bash scripts/serve_one_spark.sh
```

which execs (from `scripts/serve_one_spark.sh`):

```
vllm serve ~/models/GLM-5.3-Flash-EXL3-K2
  --served-model-name GLM-5.3-Flash-EXL3 --host 0.0.0.0 --port 8891
  --tensor-parallel-size 1 --quantization exl3 --load-format auto
  --max-model-len 65536 --max-num-seqs 1 --max-num-batched-tokens 2048
  --kv-cache-dtype fp8 --no-enable-flashinfer-autotune --skip-mm-profiling
  --limit-mm-per-prompt '{"image":4,"video":1}'
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45
  --chat-template $MODEL_DIR/chat_template.jinja   # after scripts/patch_chat_template_thinking.py
  --enable-prefix-caching --gpu-memory-utilization 0.91
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
  --cudagraph-capture-sizes 1 2 3 4 5 6 8 12
  --long-prefill-token-threshold 1024
```

with env `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(required past ~200k-token prefills on unified memory),
`VLLM_EXL3_MOE_KERNEL=native`, `EXL3_FUSED_MOE=1`, `VLLM_NO_USAGE_STATS=1`,
`DO_NOT_TRACK=1`. Never `--moe-backend marlin`; never MTP + DFlash together.
Prereqs: `python scripts/preflight.py` → `bash scripts/install_prebuilt.sh`
→ weights → `python scripts/patch_chat_template_thinking.py
~/models/GLM-5.3-Flash-EXL3-K2/chat_template.jinja`. Cold start ~12 min.

Context notes (upstream AGENTS.md, measured): serving ctx 65536 (131072
allocates but prompts ≥98304 fault the engine); 262144 boots for needle
recall only (single request, pinned 3 GiB KV, spec off) — NOT the bench shape.

## Path to an lmswitch profile

Build one image `FROM` a CUDA 13 aarch64 base: install the prebuilt wheels
(`scripts/install_prebuilt.sh` steps, minus the venv), bake in
`runtime/exl3_plugin` (canonical: vcruz305/vllm-exl3), the kpool-tail fix
(ships in the 2026-08-30+ wheels — verify, do not assume), and the patched
chat template. Then the yaml is a plain `runtime: vllm` profile: `image:` the
new tag, `--quantization exl3` + the flags above as `extra_args`,
`gpu_memory_utilization: 0.91`, `max_num_seqs: 1`, `enforce_eager: false`,
`model: "GLM-5.3-Flash-EXL3-K2"`, `port: 8891`, `ctx: 65536`,
`ready_timeout: 1200`. Until that image exists and boots, serve via the
command above.
