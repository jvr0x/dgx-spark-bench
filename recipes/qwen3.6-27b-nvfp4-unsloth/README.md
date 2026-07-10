# Recipe: Qwen3.6-27B-NVFP4 · NVFP4 · vLLM (AEON) — local unsloth build

Reproducible concurrency-throughput sweep of the local unsloth-hosted `Qwen3.6-27B-NVFP4`
(dense hybrid NVFP4 VLM, ~27B params) served on a DGX Spark. **Text-only mode** (vision
tower disabled) — MTP speculative decoding enabled (2 tokens, per unsloth HF docs).

## Files
| File | Role |
|---|---|
| `qwen3.6-27b-nvfp4-unsloth-bench.yaml` | lmswitch bench profile (serving layer, served-model-name `qwen3.6-27b-nvfp4-unsloth-bench`, port 8221). Goes in your lmswitch `ai-models/`. **Suffixed `-bench`** — the daily-driver config elsewhere in your `ai-models/` uses the bare `qwen3.6-27b-nvfp4-unsloth` id (port 8128); without the suffix the two profiles collide on one filename/id. |
| `harness.yaml` | what the harness drives (measurement layer). |

## Why a separate "bench" profile (not the daily-driver config)
The interactive config is tuned for low-latency agent use; benchmarking it as-is measures the
wrong thing. This profile changes exactly three things and **labels them**:

| Knob | Daily | Bench | Why |
|---|---|---|---|
| `max_num_seqs` | 4–16 | 64 | otherwise everything past N=4 just queues — the chart's knee would be a config artifact |
| `gpu_memory_utilization` | 0.85 | 0.3 | bounds non-KV allocations; KV is sized by explicit byte cap |
| `ctx` | 262144 | 32768 | 256K pre-reserves huge KV/seq and throttles batching; the 1024/256 workload needs ~1.3K |

Prefix caching stays on (as-served) — the harness sends a **unique prefix per request**, so it
never hits and we measure real prefill, not cache replays.

## Run it (on the Spark)
```bash
# 0. one-time: create the Linux venv + deps on the Spark
cd ~/dev/dgx-spark-bench/harness
uv venv .venv && uv pip install --python .venv/bin/python httpx pyyaml jsonschema

# 1. serving layer — copy the bench profile into lmswitch, start it
cp ~/dev/dgx-spark-bench/recipes/qwen3.6-27b-nvfp4-unsloth/qwen3.6-27b-nvfp4-unsloth-bench.yaml \
   ~/utils/lmswitch/ai-models/
lmswitch on qwen3.6-27b-nvfp4-unsloth-bench      # blocks until "Ready on port 8221"

# 2. measurement layer — run the sweep, pinned to a couple cores so the client
#    doesn't steal cycles from the engine on the 20-core Grace CPU
taskset -c 0,1 .venv/bin/python bench.py \
  ../recipes/qwen3.6-27b-nvfp4-unsloth/harness.yaml \
  -o ../results/qwen3.6-27b-nvfp4-unsloth.json

# 3. stop
lmswitch off qwen3.6-27b-nvfp4-unsloth-bench
```

The emitted `results/qwen3.6-27b-nvfp4-unsloth.json` is schema-valid and drops straight into the
dashboard.

## Caveats / what the first run calibrates
- **Dense model + MTP spec-decode (2 tokens)** — this is a "hardware truth" run with MTP.
  The curve should be smooth and predictable; any anomalies likely indicate memory pressure
  or KV cache issues.
- **Unsloth vs NVIDIA accuracy**: Unsloth NVFP4 scored 86.25 MMLU-Pro / 86.34 GPQA vs NVIDIA's
  85.96 / 86.87 (per HF docs). Performance should be comparable or slightly better.
- **`weights_gb` is approximate** — refine from `docker logs` or `nvidia-smi` after the first run.
- **Vision tower disabled** — the model is a VLM checkpoint but `limit_mm_per_prompt: {image:0, video:0}`
  ensures only text is processed. If you want to benchmark vision + text, remove that flag.

## Measured results
_Pending first run on the Spark._ Once `results/qwen3.6-27b-nvfp4-unsloth.json` exists, summarise the
knee (N, agg tok/s) and the single-stream tok/s here.
