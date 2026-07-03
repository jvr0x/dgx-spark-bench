# Recipes — DGX Spark Benchmark Suite

Each subdirectory is one **reproducible benchmark recipe**. A recipe consists of:

| File | Role |
|---|---|
| `*.yaml` | lmswitch bench profile (serving layer). Goes in your lmswitch `ai-models/`. |
| `harness.yaml` | what the harness drives (measurement layer). |

## How to reproduce any recipe on a DGX Spark

### 0. Prerequisites

```bash
# Clone this repo + lmswitch
cd ~/dev/dgx-spark-bench
uv venv harness/.venv && uv pip install --python harness/.venv/bin/python httpx pyyaml jsonschema
```

### 1. Start the model via lmswitch

Copy the bench profile into your lmswitch `ai-models/` directory, then start it.

```bash
cp ~/dev/dgx-spark-bench/recipes/<recipe-id>/<recipe-id>.yaml ~/utils/lmswitch/ai-models/
lmswitch on <recipe-id>            # blocks until "Ready on port <port>"
```

### 2. Run the sweep

The harness drives a concurrency-throughput sweep against the served endpoint.

```bash
cd ~/dev/dgx-spark-bench/harness
taskset -c 0,1 .venv/bin/python bench.py \
  ../recipes/<recipe-id>/harness.yaml \
  -o ../results/<recipe-id>.json
```

`taskset -c 0,1` pins the harness to cores 0–1 so the client pool doesn't steal cycles
from the model inference on the 20-core Grace CPU.

### 3. Stop the model

```bash
lmswitch off <recipe-id>
```

The emitted `<recipe-id>.json` is schema-valid and drops straight into the dashboard.

## Recipes

| Recipe | Model | Backend | Quant | Weights | Key details |
|---|---|---|---|---|---|
| [qwen3.6-35b-nvfp4-nvidia](qwen3.6-35b-nvfp4-nvidia/) | Qwen3.6-35B-A3B | vLLM AEON | NVFP4 mixed | 21.9 GB | Flagship — MTP spec-decode on, MoE |
| [qwen3.6-35b-nvfp4-specoff](qwen3.6-35b-nvfp4-specoff/) | Qwen3.6-35B-A3B | vLLM AEON | NVFP4 mixed | 21.9 GB | Same as flagship, MTP disabled |
| [qwen3.6-27b-nvfp4-nvidia](qwen3.6-27b-nvfp4-nvidia/) | Qwen3.6-27B-NVFP4 | vLLM AEON | NVFP4 mixed | 20.4 GB | Dense hybrid VLM, no spec-decode |
| [gpt-oss-20b-llamacpp](gpt-oss-20b-llamacpp/) | GPT-OSS-20B | llama.cpp | GGUF Q4_K_XL | 11.1 GB | First GGUF recipe, cross-engine |
| [gemma-4-12b-it-llamacpp](gemma-4-12b-it-llamacpp/) | Gemma-4-12B-IT | llama.cpp | GGUF Q4_K_M | 6.6 GB | Small contrast model |
| [step-3.7-flash-llamacpp](step-3.7-flash-llamacpp/) | Step-3.7-Flash | llama.cpp | GGUF IQ4_XS | 88.8 GB | "Barely fits" — 88.8G in 128GB pool |
| [ornith-35b-nvfp4-aeon7](ornith-35b-nvfp4-aeon7/) | Ornith-1.0-35B | vLLM AEON | NVFP4 mixed | 22.1 GB | Hybrid Mamba/Attention, DFlash spec |
| [ornith-35b-q8](ornith-35b-q8/) | Ornith-1.0-35B | llama.cpp | GGUF Q8_0 | 34.4 GB | Same model as above, quant comparison |
| [qwen3.6-35b-q8-llamacpp](qwen3.6-35b-q8-llamacpp/) | Qwen3.6-35B-A3B | llama.cpp | GGUF Q8_K_XL | 35.8 GB | Same as flagship, quant comparison |

## Bench profile design

All bench profiles change exactly three knobs from the daily-driver config and **label them**:

| Knob | Daily driver | Bench | Why |
|---|---|---|---|
| `max_num_seqs` | 4–16 | ≥ sweep ceiling (64) | Without it, everything past N=max_num_seqs just queues — the chart's knee would be a config artifact |
| `gpu_memory_utilization` | 0.4–0.85 | 0.3–0.5 | Bounds non-KV allocations; KV is sized by explicit byte cap |
| `ctx` | 262144 | 32768 (or 65536 for llama.cpp) | 256K pre-reserves huge KV/seq and throttles batching; the 1024/256 workload needs ~1.3K |

The harness sends a **unique prefix per request** so server-side prefix caching never hits — we measure real prefill, not cache replays.

## Known quirks

- **AEON CUDA-graph estimator bug:** The AEON vLLM build returns ~−20 GiB for CUDA-graph memory estimation, inflating the KV budget past physical RAM. Fix: explicit `--kv-cache-memory-bytes` (32–64 GiB) in every vLLM recipe.
- **llama.cpp KV:** Use `-np 32` (slots >= sweep ceiling) and `-kvu` (unified KV buffer). Explicit `-np` (no -kvu) measured ~30% slower per-stream at N=2–16.
- **Step-3.7-Flash** requires `force: true` in its profile because the RAM guard heuristic (weights × 1.3 = ~115 GB) exceeds the 107 GB free pool, but the real footprint (~92 GB with q8 KV) fits.
