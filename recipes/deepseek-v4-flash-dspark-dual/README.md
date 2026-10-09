# Recipe: DeepSeek-V4-Flash DSpark 1M · vLLM 0.26 SM121 (dual DGX Spark)

Concurrency-throughput sweep of [`deepseek-ai/DeepSeek-V4-Flash-DSpark`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark)
across **two DGX Sparks (GB10)**: tensor-parallel TP=2 over CX7 RoCE, NVFP4-DS-MLA KV cache,
b12x MoE backend, DSpark speculative decoding (K=6), 1M context. As-served config — not a raised
bench-profile (see [Caveats](#caveats)).

## Files

| File | Role |
|---|---|
| `deepseek-v4-flash-dspark-dual.yaml` | lmswitch dual bench profile → copy into `ai-models/` |
| `harness.yaml` | harness target + series metadata for `bench.py` |

## As-served stack (engine from r0b0tlab v0.26, adopted 2026-07-31)

Runtime values taken from [r0b0tlab/DeepSeek-V4-Flash-DSpark-v026-SM121](https://github.com/r0b0tlab/DeepSeek-V4-Flash-DSpark-v026-SM121)
(`profiles/dspark-r0b0tlab-production.env`, `scripts/run-dspark-dual-gb10.sh`); memory shape kept
from this pair's validated 1M profile.

| Setting | Value | Source |
|---|---|---|
| Runtime | `vllm-dual` — TP=2 across master (spark) + worker (gigabyte) over CX7 | ours |
| Image | `ghcr.io/r0b0tlab/deepseek-v4-flash-dspark-v026-sm121:v0.26.0-sm121-optimized` | r0b0tlab |
| Engine | vLLM `0.26.0+dspark.sm121.2` (known-fix overlay, pinned NCCL 2.30.4) | r0b0tlab |
| MoE backend | `flashinfer_b12x` | r0b0tlab |
| KV cache | `nvfp4_ds_mla`, **profiler-sized (unpinned)** | ours |
| Spec decode | dspark, **6** tokens (engine-default draft sampling) | r0b0tlab |
| Scheduling | `--optimization-level 2 --performance-mode balanced --enable-flashinfer-autotune --jit-monitor-mode error` | r0b0tlab |
| CUDA graphs | on — `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, capture size at engine default (24) | r0b0tlab |
| `gpu_memory_utilization` | **0.85** | ours |
| `max_num_seqs` | **12** | ours |
| `max_num_batched_tokens` | **8192** | both (their long-context lane caps here too) |
| `max_model_len` | **1,048,576** | ours |
| Port / served id | **8888** / `deepseek-v4-flash-dspark-dual` | ours |

Upstream's own dual-GB10 numbers, measured at **327,680** ctx / `max_num_seqs=16`: 80.58 tok/s c1
decode median, 402.0 tok/s aggregate at c16, ~300K-token request qualified, GSM8K smoke 50/50.
Those do not transfer to 1M — different memory shape, different concurrency.

### Why the memory values are not upstream's

r0b0tlab pin the KV pool explicitly (`--kv-cache-memory-bytes 15998753178`, ~14.9 GiB) and pair it
with `gpu_memory_utilization=0.835` and `16/16384`. That byte count is sized for 327,680 tokens.
On the **0.25 engine** this pair had ~7.5 GiB free for KV after weights and workspaces — a
2026-07-20 run died in `vllm/v1/core/kv_cache_utils.py` needing 7.54 GiB with 7.31 GiB available.
The 0.26 figure is unmeasured, but a ~14.9 GiB pin overshoots by roughly 2×, not by a few percent,
so 1M keeps: util 0.85, no pin (profiler sizes the pool), 8192 batched tokens, `max_num_seqs=12`,
and `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`.

`--max-cudagraph-capture-size` is left unset. It bounds the captured *batch size* and defaults to
`min(max_num_seqs * 2, 512)` = 24 at `max_num_seqs=12` — identical to the old explicit `24`, whose
`6 * 4` comment never matched the recipe's own 12 seqs. With the profiler told not to reserve for
cudagraphs, keeping capture at that validated footprint is what makes the pairing safe.

## Measured results — PENDING RE-MEASURE

⚠️ The numbers below (and `results/deepseek-v4-flash-dspark-dual.json`) were measured **2026-07-21
on the previous engine**: Anemll `dspark-vllm-gx10:0.1.1` (vLLM 0.25), spec-decode K=3
probabilistic, `--async-scheduling` on. Context, util and concurrency are unchanged, but the
engine is not. Kept as the prior-engine baseline until the v0.26 sweep is run.

Workload: chat, 1024 prompt / 256 output tokens, closed-loop, unique prefixes (prefix cache defeated).

| N | Agg tok/s | Per-session tok/s | TTFT p95 |
|---:|---:|---:|---:|
| 1 | 46.9 | 52.6 | 525 ms |
| 2 | 71.1 | 38.7 | 603 ms |
| 4 | 95.3 | 26.0 | 3,940 ms |
| 8 | 140.8 | 18.6 | 1,086 ms |
| 12 | **177.8** | 15.6 | 1,404 ms |

Peak aggregate **177.8 tok/s @ N=12** on vLLM 0.25.

## Run it (on the master Spark)

The image must exist on **both** nodes first:

```bash
docker pull ghcr.io/r0b0tlab/deepseek-v4-flash-dspark-v026-sm121:v0.26.0-sm121-optimized
ssh Gigabyte docker pull ghcr.io/r0b0tlab/deepseek-v4-flash-dspark-v026-sm121:v0.26.0-sm121-optimized
```

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

First start after the image switch pays a cold FlashInfer autotune pass on top of the TP=2 weight
load — `ready_timeout` is 3600 for that reason.

**If it fails at startup**, the two signatures to expect:
- `ValueError` from `vllm/v1/core/kv_cache_utils.py` — KV pool too small for 1M ctx. Free unified
  memory first (ComfyUI :8188, Forge :7860, other lmswitch models), then consider lowering
  `--max-cudagraph-capture-size`.
- OOM during cudagraph capture — capture is deliberately not counted by the memory profiler here;
  set `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`, or pin `--max-cudagraph-capture-size` below 24.

## Caveats

- **Engine values are upstream's, memory values are ours.** The 0.835 + 14.9 GiB KV pin +
  16/16384 shape only holds at 327,680 ctx; see the section above.
- **Upstream qualifies at ~300K, not 1M.** Nothing in the known-fix overlay hardcodes a ceiling
  (the sparse-indexer workspace scales as `max_model_len * 40`), but 1M on this engine is
  unvalidated until the sweep runs.
- **As-served, not a bench-profile:** `max_num_seqs=12` is this pair's validated concurrency
  ceiling at 1M ctx — not an artificially low daily-driver throttle.
- **Local deviations from upstream (deliberate):** `thinking:false` chat-template default (they
  serve `true`), plus our own tool-call/reasoning parser and `--generation-config=vllm` flags.
  Dropped to match them: `--async-scheduling`, `--enable-chunked-prefill` (v1 default),
  `--max-cudagraph-capture-size` (engine default reproduces the old value).
- **Symmetric NIC pinning kept:** upstream's launcher uses an asymmetric head/worker NIC pair
  (`enp1s0f0np0`/`rocep1s0f0` vs `f1np1`/`rocep1s0f1`) — that is their cabling. Ours stays
  symmetric, as validated on this pair.
- **Checkpoint revision unverified:** upstream pins model revision
  `913f0657a874f76844e2e91cbe706dbcaceeb6d7`; our local copy carries no HF revision metadata to
  check against.
- **1M-ctx checkpoint benched at 1024/256 tokens:** this is the short-context roofline number,
  not a long-context test.
- **Dual-node telemetry:** GPU temperature/power in `results/*.json` come from `nvidia-smi` on
  the master node only — the worker node's GPU isn't captured.
- **Filename id** must stay `deepseek-v4-flash-dspark-dual` — it is the OpenAI `model` id the
  harness calls.
