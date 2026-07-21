# Recipe: MiMo-V2.5 Omni · NVFP4+MTP1 · vLLM (Ray, dual DGX Spark)

Concurrency-throughput sweep of [`lukealonso/MiMo-V2.5-NVFP4`](https://huggingface.co/lukealonso/MiMo-V2.5-NVFP4)
across **two DGX Sparks (GB10)**: tensor-parallel TP=2 over CX7 RoCE via Ray, NVFP4 weights +
NVFP4 KV cache, MTP×1 speculative decoding, 1M context. As-served config — not a raised
bench-profile (see [Caveats](#caveats)).

## Files

| File | Role |
|---|---|
| `mimo-v25-dual.yaml` | lmswitch dual bench profile → copy into `ai-models/` |
| `harness.yaml` | harness target + series metadata for `bench.py` |

## As-served stack (measured 2026-07-21)

| Setting | Value |
|---|---|
| Runtime | `vllm-dual-ray` — Ray executor, TP=2 across master (spark) + worker (gigabyte) over CX7 |
| Image | `vllm-mimo-v25-lmswitch` (`ghcr.io/miaai-lab/mimo-v2.5-vllm-dual-dgx-sparks:20260704` + launch patches baked in) |
| Attention backend | `triton_attn_diffkv` |
| MoE backend | `flashinfer_cutlass` |
| KV cache | `nvfp4` |
| Spec decode | MTP, `num_speculative_tokens=1` |
| `gpu_memory_utilization` | **0.88** |
| `max_num_seqs` | **3** (genuine KV-capacity ceiling at 1M ctx on this pair) |
| `max_model_len` | **1,000,000** |
| Port / served id | **8888** / `mimo-v25-dual` |

## Measured results (`results/mimo-v25-dual.json`)

Workload: chat, 1024 prompt / 256 output tokens, closed-loop, unique prefixes (prefix cache defeated).

| N | Agg tok/s | Per-session tok/s | TTFT p95 |
|---:|---:|---:|---:|
| 1 | 21.3 | 21.5 | 287 ms |
| 2 | 27.0 | 21.2 | 11,305 ms |
| 3 | **55.5** | 21.8 | 12,949 ms |

Peak aggregate **55.5 tok/s @ N=3** (the sweep ceiling). Per-session tok/s stays flat (~21-22)
across N — this pair is KV-capacity bound at 1M ctx well before it's compute bound; the TTFT
jump at N≥2 is queueing against `max_num_seqs=3`, not compute saturation.

## Run it (on the master Spark)

```bash
cd ~/dev/dgx-spark-bench/harness
uv venv .venv && uv pip install --python .venv/bin/python httpx pyyaml jsonschema

# mimo-v25-dual was already served live on this pair; if starting fresh:
cp ~/dev/dgx-spark-bench/recipes/mimo-v25-dual/mimo-v25-dual.yaml \
   ~/utils/lmswitch/ai-models/
lmswitch on mimo-v25-dual   # wait for Ready on port 8888 (both nodes)

taskset -c 0,1 .venv/bin/python bench.py \
  ../recipes/mimo-v25-dual/harness.yaml \
  -o ../results/mimo-v25-dual.json

lmswitch off mimo-v25-dual
```

## Caveats

- **As-served, not a bench-profile:** unlike single-node recipes in this repo, dual recipes are
  benched at the exact config used for daily serving — `max_num_seqs` is a genuine KV-capacity
  ceiling on 2×128 GiB unified memory at this checkpoint's context length, not an artificially
  low daily-driver throttle. Numbers below reflect the real dual-Spark ceiling.
- **Dual-node telemetry:** GPU temperature/power in `results/*.json` come from `nvidia-smi` on
  the master node only — the worker node's GPU isn't captured.
- **Filename id** must stay `mimo-v25-dual` — it is the OpenAI `model` id the harness calls.
