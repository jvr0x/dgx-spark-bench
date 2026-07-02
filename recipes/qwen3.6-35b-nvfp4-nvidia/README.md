# Recipe: Qwen3.6-35B-A3B · NVFP4 · vLLM (AEON) — flagship

Reproducible concurrency-throughput sweep of NVIDIA's `nvidia/qwen3.6-35b-a3b-nvfp4`
(mixed-precision NVFP4 MoE, ~3B active) served on a DGX Spark, **as you actually run it** —
MTP speculative decoding and CUDA graphs on — with only the knobs that would otherwise
invalidate a sweep adjusted.

## Files
| File | Role |
|---|---|
| `qwen3.6-35b-nvfp4-bench.yaml` | lmswitch bench profile (serving layer). Goes in your lmswitch `ai-models/`. |
| `harness.yaml` | what the harness drives (measurement layer). |

## Why a separate "bench" profile (not the daily-driver config)
The interactive config is tuned for low-latency agent use; benchmarking it as-is measures the
wrong thing. This profile changes exactly three things and **labels them**:

| Knob | Daily | Bench | Why |
|---|---|---|---|
| `max_num_seqs` | 4 | 64 | otherwise everything past N=4 just queues — the chart's knee would be a config artifact |
| `gpu_memory_utilization` | 0.4 | 0.9 | solo & dedicated → max KV cache → real concurrency ceiling |
| `ctx` | 262144 | 32768 | 256K pre-reserves huge KV/seq and throttles batching; the 1024/256 workload needs ~1.3K |

Prefix caching stays on (as-served) — the harness sends a **unique prefix per request**, so it
never hits and we measure real prefill, not cache replays.

## Run it (on the Spark)
```bash
# 0. one-time: create the Linux venv + deps on the Spark
cd ~/dev/dgx-spark-bench/harness
uv venv .venv && uv pip install --python .venv/bin/python httpx pyyaml jsonschema

# 1. serving layer — copy the bench profile into lmswitch, start it
cp ~/dev/dgx-spark-bench/recipes/qwen3.6-35b-nvfp4-nvidia/qwen3.6-35b-nvfp4-bench.yaml \
   ~/utils/lmswitch/ai-models/
lmswitch on qwen3.6-35b-nvfp4-bench            # blocks until "Ready on port 8214"

# 2. measurement layer — run the sweep, pinned to a couple cores so the client
#    doesn't steal cycles from the engine on the 20-core Grace CPU
taskset -c 0,1 .venv/bin/python bench.py \
  ../recipes/qwen3.6-35b-nvfp4-nvidia/harness.yaml \
  -o ../results/qwen3.6-35b-nvfp4-nvidia.json

# 3. stop
lmswitch off qwen3.6-35b-nvfp4-bench
```
The emitted `results/qwen3.6-35b-nvfp4-nvidia.json` is schema-valid and drops straight into the
dashboard.

## Caveats / what the first run calibrates
- **MTP spec-decode + `max_num_seqs: 64` is unvalidated** on this mixed-precision checkpoint (the
  daily driver runs seqs:4). If it OOMs at load or the sweep degrades past ~16 concurrent, lower
  `max_num_seqs` or drop `--speculative-config` — and record that as a finding.
- **MoE + spec-decode** don't follow the dense memory-bound model: decode is driven by ~3B active
  params, and MTP gains shrink as the batch grows. Expect a different curve shape than a dense model.
- `weights_gb` in `harness.yaml` is approximate — refine from `docker logs vllm-qwen3.6-35b-nvfp4-bench`
  or `nvidia-smi` after the first run.

## Measured results
_Pending first run on the Spark._ Once `results/qwen3.6-35b-nvfp4-nvidia.json` exists, summarise the
knee (N, agg tok/s) and the single-stream tok/s here.
