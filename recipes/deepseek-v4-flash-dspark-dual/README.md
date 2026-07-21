# Recipe: DeepSeek-V4-Flash DSpark 1M · vLLM 0.25 (dual DGX Spark)

Concurrency-throughput sweep of [`deepseek-ai/DeepSeek-V4-Flash-DSpark`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark)
across **two DGX Sparks (GB10)**: tensor-parallel TP=2 over CX7 RoCE, NVFP4-DS-MLA KV cache,
b12x MoE backend, DSpark speculative decoding, 1M context. As-served config — not a raised
bench-profile (see [Caveats](#caveats)).

## Files

| File | Role |
|---|---|
| `deepseek-v4-flash-dspark-dual.yaml` | lmswitch dual bench profile → copy into `ai-models/` |
| `harness.yaml` | harness target + series metadata for `bench.py` |

## As-served stack (measured 2026-07-21)

| Setting | Value |
|---|---|
| Runtime | `vllm-dual` — TP=2 across master (spark) + worker (gigabyte) over CX7 |
| Image | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` |
| MoE backend | `flashinfer_b12x` |
| KV cache | `nvfp4_ds_mla` |
| Spec decode | dspark, 3 tokens, probabilistic draft sampling |
| `gpu_memory_utilization` | **0.85** |
| `max_num_seqs` | **12** (validated ceiling on this pair) |
| `max_model_len` | **1,048,576** |
| Port / served id | **8888** / `deepseek-v4-flash-dspark-dual` |

## Measured results (`results/deepseek-v4-flash-dspark-dual.json`)

Workload: chat, 1024 prompt / 256 output tokens, closed-loop, unique prefixes (prefix cache defeated).

| N | Agg tok/s | Per-session tok/s | TTFT p95 |
|---:|---:|---:|---:|
| 1 | 46.9 | 52.6 | 525 ms |
| 2 | 71.1 | 38.7 | 603 ms |
| 4 | 95.3 | 26.0 | 3,940 ms |
| 8 | 140.8 | 18.6 | 1,086 ms |
| 12 | **177.8** | 15.6 | 1,404 ms |

Peak aggregate **177.8 tok/s @ N=12** (the sweep ceiling); scaling is smooth and monotonic
through the ceiling with no early knee.

## Run it (on the master Spark)

```bash
cd ~/dev/dgx-spark-bench/harness
uv venv .venv && uv pip install --python .venv/bin/python httpx pyyaml jsonschema

cp ~/dev/dgx-spark-bench/recipes/deepseek-v4-flash-dspark-dual/deepseek-v4-flash-dspark-dual.yaml \
   ~/utils/lmswitch/ai-models/
lmswitch on deepseek-v4-flash-dspark-dual   # wait for Ready on port 8888 (both nodes)

taskset -c 0,1 .venv/bin/python bench.py \
  ../recipes/deepseek-v4-flash-dspark-dual/harness.yaml \
  -o ../results/deepseek-v4-flash-dspark-dual.json

lmswitch off deepseek-v4-flash-dspark-dual
```

## Caveats

- **As-served, not a bench-profile:** `max_num_seqs=12` is this pair's validated concurrency
  ceiling at 1M ctx — not an artificially low daily-driver throttle.
- **1M-ctx checkpoint benched at 1024/256 tokens:** this is the short-context roofline number,
  not a long-context test.
- **Dual-node telemetry:** GPU temperature/power in `results/*.json` come from `nvidia-smi` on
  the master node only — the worker node's GPU isn't captured.
- **Filename id** must stay `deepseek-v4-flash-dspark-dual` — it is the OpenAI `model` id the
  harness calls.
