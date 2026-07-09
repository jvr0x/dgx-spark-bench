# Recipe: Nemotron Puzzle 75B-A9B · NVFP4 · vLLM 0.23 (AEON-7)

Concurrency-throughput sweep of NVIDIA
[`nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4)
on a single DGX Spark (GB10): hybrid Mamba2 + LatentMoE + attention, **75.3B total / ~9.3B active**,
served with **AEON-7** `ghcr.io/aeon-7/aeon-vllm-ultimate:latest`, MTP×3, fp8 KV cache.

## Files

| File | Role |
|---|---|
| `nemotron-labs-3-puzzle-75b-nvfp4-aeon7.yaml` | lmswitch bench profile → copy into `ai-models/` |
| `harness.yaml` | harness target + series metadata for `bench.py` |

## As-served stack (measured 2026-07-09)

| Setting | Value |
|---|---|
| Image | `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` (vLLM **0.23.0+aeon.sm121a.dflash**) |
| MoE backend | **`flashinfer_cutlass`** (mixed-precision checkpoint; marlin/triton rejected per layer) |
| NVFP4 GEMM | `VLLM_NVFP4_GEMM_BACKEND=marlin` |
| Mamba | `--mamba-backend flashinfer` |
| KV cache | `--kv-cache-dtype fp8` |
| Spec decode | MTP, `num_speculative_tokens=3` |
| `gpu_memory_utilization` | **0.85** |
| `max_num_seqs` | **64** (sweep ceiling) |
| `max_model_len` | **32768** (bench profile; daily driver may use longer ctx on another yaml) |
| Port / served id | **8243** / `nemotron-labs-3-puzzle-75b-nvfp4-aeon7` |

## Measured results (`results/nemotron-labs-3-puzzle-75b-nvfp4-aeon7.json`)

Workload: chat, 1024 prompt / 256 output tokens, closed-loop, unique prefixes (prefix cache defeated).

| N | Agg tok/s | Per-session tok/s |
|---:|---:|---:|
| 1 | 31.3 | 32.8 |
| 8 | 129.4 | 17.3 |
| 32 | 260.3 | 8.3 |
| 64 | **330.0** | 5.7 |

Peak aggregate **330 tok/s @ N=64**; scaling continues through the sweep ceiling (no artificial knee at N=12).

## Bench profile vs daily driver

Use **`nvidia-nemotron-labs-3-puzzle-75b`** (port 8125) for agent-style daily use. This recipe is the **bench** profile only:

| Knob | Daily (`nvidia-nemotron-labs-3-puzzle-75b`) | Bench (this recipe) |
|---|---|---|
| Port | 8125 | 8243 |
| Image | stock `vllm/vllm-openai:cu130-nightly` (default) | AEON-7 ultimate |
| `max_num_seqs` | 4 | 64 |
| `ctx` | 262144 | 32768 |
| `gpu_memory_utilization` | 0.88 | 0.85 |

Prefix caching stays on; the harness uses a unique prefix per request so TTFT reflects real prefill.

## Run it (on the Spark)

```bash
cd ~/dev/dgx-spark-bench/harness
uv venv .venv && uv pip install --python .venv/bin/python httpx pyyaml jsonschema

cp ~/dev/dgx-spark-bench/recipes/nemotron-labs-3-puzzle-75b-nvfp4-aeon7/nemotron-labs-3-puzzle-75b-nvfp4-aeon7.yaml \
   ~/utils/lmswitch/ai-models/
lmswitch on nemotron-labs-3-puzzle-75b-nvfp4-aeon7   # wait for Ready on port 8243

taskset -c 0,1 .venv/bin/python bench.py \
  ../recipes/nemotron-labs-3-puzzle-75b-nvfp4-aeon7/harness.yaml \
  -o ../results/nemotron-labs-3-puzzle-75b-nvfp4-aeon7.json

lmswitch off nemotron-labs-3-puzzle-75b-nvfp4-aeon7
```

First load is ~10–20 minutes (~50 GiB weights). Expect ~35+ minutes for the full N=1..64 sweep.

## Caveats

- **Unified memory:** avoid large co-tenants while serving; 0.85 util + ~50 GiB weights leaves KV for high N but is tight on 128 GiB.
- **MTP + reasoning:** clients may want `chat_template_kwargs: {"enable_thinking": false}` for cleaner `content` and better draft acceptance.
- **Filename id** must stay `nemotron-labs-3-puzzle-75b-nvfp4-aeon7` — it is the OpenAI `model` id the harness calls.