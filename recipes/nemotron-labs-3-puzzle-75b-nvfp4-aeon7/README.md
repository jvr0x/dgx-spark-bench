# Recipe: Nemotron-Labs-3-Puzzle-75B-A9B · NVFP4 · vLLM (AEON) — Hybrid MoE

Reproducible concurrency-throughput sweep of NVIDIA's
`nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4` (NVFP4 hybrid Mamba2-Transformer
LatentMoE, ~9.3B active from 75.3B total) served on a DGX Spark, **as you actually run it** —
MTP speculative decoding and Marlin MoE backend on — with only the knobs that would
otherwise invalidate a sweep adjusted.

## Files
| File | Role |
|---|---|
| `nemotron-labs-3-puzzle-75b-nvfp4-aeon7.yaml` | lmswitch bench profile (serving layer). Goes in your lmswitch `ai-models/`. |
| `harness.yaml` | what the harness drives (measurement layer). |

## Architecture notes
- **Hybrid Mamba2-Transformer LatentMoE** — interleaved Mamba SSM, MoE (LatentMoE), and Attention layers.
- **75.3B total / 9.3B active** — much larger active set than the 30B-nano models (~3B active).
  Expect different throughput characteristics: decode is driven by ~9.3B active params, not ~3B.
- **MTP spec-decode** — 3 speculative tokens, recommended by NVIDIA for throughput gains.
- **Marlin MoE backend** — required for SM121 stability on GB10 (Blackwell).
- **Mamba2 SSM** — requires `--mamba-backend flashinfer` and specific cache tuning.

## Why a separate "bench" profile (not the daily-driver config)

The interactive config is tuned for low-latency agent use; benchmarking it as-is measures the
wrong thing. This profile changes exactly three things and **labels them**:

| Knob | Daily | Bench | Why |
|---|---|---|---|
| `max_num_seqs` | 12 | 64 | otherwise everything past N=12 just queues — the chart's knee would be a config artifact |
| `gpu_memory_utilization` | 0.85 | 0.50 | solo & dedicated → KV cache gets real budget; ceiling is measured, not configured |
| `ctx` / `max_model_len` | 131072 | 32768 | 256K/128K pre-reserves huge KV/seq and throttles batching; the 1024/256 workload needs ~1.3K |

Prefix caching stays on (as-served) — the harness sends a **unique prefix per request**, so it
never hits and we measure real prefill, not cache replays.

## Run it (on the Spark)
```bash
# 0. one-time: create the Linux venv + deps on the Spark
cd ~/dev/dgx-spark-bench/harness
uv venv .venv && uv pip install --python .venv/bin/python httpx pyyaml jsonschema

# 1. serving layer — copy the bench profile into lmswitch, start it
cp ~/dev/dgx-spark-bench/recipes/nemotron-labs-3-puzzle-75b-nvfp4-aeon7/nemotron-labs-3-puzzle-75b-nvfp4-aeon7.yaml \
   ~/utils/lmswitch/ai-models/
lmswitch on nemotron-labs-3-puzzle-75b-nvfp4-aeon7   # blocks until "Ready on port 8243"
  # Expected load time: ~10-20 min (49.9 GB weights on LPDDR5x @ 273 GB/s)

# 2. measurement layer — run the sweep, pinned to a couple cores so the client
#    doesn't steal cycles from the engine on the 20-core Grace CPU
taskset -c 0,1 .venv/bin/python bench.py \
  ../recipes/nemotron-labs-3-puzzle-75b-nvfp4-aeon7/harness.yaml \
  -o ../results/nemotron-labs-3-puzzle-75b-nvfp4-aeon7.json

# 3. stop
lmswitch off nemotron-labs-3-puzzle-75b-nvfp4-aeon7
```

The emitted `results/nemotron-labs-3-puzzle-75b-nvfp4-aeon7.json` is schema-valid and drops
straight into the dashboard.

## Caveats / what the first run calibrates
- **Memory pressure at high concurrency:** The 9.3B active-parameter MoE engine has a much
  larger per-request KV footprint than the 3B-active models. At N > 12, expect throughput to
  flatten sooner. If OOM occurs, lower `max_num_seqs` or reduce `gpu_memory_utilization`.
- **MTP spec-decode + Marlin MoE** — this combination is newly tested on the Spark. The first
  run validates that spec-decode actually improves throughput (vs. no-spec) on the 9.3B active
  set. If spec-decode degrades performance, record that as a finding.
- **Load time:** 49.9 GB weights on LPDDR5x @ 273 GB/s → ~15-20 minutes to load. The
  `ready_timeout` in lmswitch handles this.
- **`weights_gb`** is approximate — refine from `docker logs` after the first run.
- **KV cache budget:** With `gpu_memory_utilization: 0.50`, roughly 64 GB is available for KV
  cache (128 GB × 0.50 - 49.9 GB weights ≈ 14 GB overhead). At 32K context, each request
  consumes ~a few MB of KV cache. N=64 should fit but may be tight.

## Comparison targets
| Model | Active | Quant | Backend | N=1 tok/s | N=8 agg tok/s |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | ~3B | NVFP4 | vLLM AEON | — | — |
| Nemotron-3-Nano-30B | ~3B | NVFP4 | vLLM AEON | — | — |
| **Puzzle-75B-A9B** | **~9.3B** | **NVFP4** | **vLLM AEON** | **pending** | **pending** |

_Larger active set → lower per-session tok/s but potentially higher aggregate throughput if
the KV cache budget supports concurrent sessions._
