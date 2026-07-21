# Recipe: MiniMax-M3 428B · W4A16+EAGLE3 · vLLM (dual DGX Spark)

Concurrency-throughput sweep of [`Sebesky/MiniMax-M3-W4A16-GPTQ`](https://huggingface.co/Sebesky/MiniMax-M3-W4A16-GPTQ)
across **two DGX Sparks (GB10)**: tensor-parallel TP=2 over CX7 RoCE, W4A16 GPTQ weights + NVFP4
(b12x) KV cache, EAGLE3 speculative decoding (draft TP=2). As-served config — not a raised
bench-profile (see [Caveats](#caveats)).

## Files

| File | Role |
|---|---|
| `minimax-m3-dual.yaml` | lmswitch dual bench profile → copy into `ai-models/` |
| `harness.yaml` | harness target + series metadata for `bench.py` |

## As-served stack (measured 2026-07-21)

| Setting | Value |
|---|---|
| Runtime | `vllm-dual` — TP=2 across master (spark) + worker (gigabyte) over CX7 |
| Image | `vllm-node-minimax-m3-b12x-lmswitch` |
| Attention backend | `b12x` |
| KV cache | `nvfp4` (indexer KV also nvfp4) |
| Spec decode | EAGLE3, `draft_tensor_parallel_size=2`, 3 tokens |
| `gpu_memory_utilization` | **0.929** |
| `max_num_seqs` | **1** (genuine KV-capacity ceiling — weights alone are 105.8 GiB/node) |
| `max_model_len` | **196,608** |
| Port / served id | **8888** / `minimax-m3-dual` |

## Measured results (`results/minimax-m3-dual.json`)

Workload: chat, 1024 prompt / 256 output tokens, closed-loop, unique prefixes (prefix cache defeated).
Single N=1 point — this checkpoint has no concurrency headroom on this pair.

| N | Agg tok/s | Per-session tok/s | TTFT p50 | TTFT p95/p99 |
|---:|---:|---:|---:|---:|
| 1 | 10.0 | **30.2** | 1,317 ms | 97,294 / 129,918 ms (n=7) |

Steady-state per-session throughput is **30.2 tok/s** with a **1.3 s** p50 TTFT — the roofline
anchor for this 428B checkpoint on this pair. The p95/p99 TTFT are cold-start outliers (n=7
samples in the 180 s window; one or two requests hit first-call JIT/EAGLE3-warmup variance on
the exotic `b12x` attention path) rather than steady-state behavior — read p50 as the
representative number, not the tail.

## Run it (on the master Spark)

```bash
cd ~/dev/dgx-spark-bench/harness
uv venv .venv && uv pip install --python .venv/bin/python httpx pyyaml jsonschema

cp ~/dev/dgx-spark-bench/recipes/minimax-m3-dual/minimax-m3-dual.yaml \
   ~/utils/lmswitch/ai-models/
lmswitch on minimax-m3-dual   # wait for Ready on port 8888 (both nodes)

taskset -c 0,1 .venv/bin/python bench.py \
  ../recipes/minimax-m3-dual/harness.yaml \
  -o ../results/minimax-m3-dual.json

lmswitch off minimax-m3-dual
```

First load is long (~209 GiB weights + 1.6 GiB EAGLE3 drafter over NFS); budget 20-30+ minutes.

## Caveats

- **As-served, not a bench-profile:** `max_num_seqs=1` is a genuine KV-capacity ceiling at
  428B params on 2×128 GiB unified memory — not an artificially low daily-driver throttle.
  The single measured point is the roofline anchor, not a scaling curve.
- **`gpu_memory_utilization=0.929`** is hand-bracketed to this exact pair's boot-time free
  memory (0.93 fails, 0.925 undersizes KV) — do not reuse on other hardware without re-bracketing.
- **Dual-node telemetry:** GPU temperature/power in `results/*.json` come from `nvidia-smi` on
  the master node only — the worker node's GPU isn't captured.
- **Filename id** must stay `minimax-m3-dual` — it is the OpenAI `model` id the harness calls.
