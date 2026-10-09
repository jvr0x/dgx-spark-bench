# Recipes — DGX Spark Benchmark Suite

Each subdirectory is one **reproducible benchmark recipe**. A recipe consists of:

| File | Role |
|---|---|
| `*.yaml` | lmswitch bench profile (serving layer). Goes in your lmswitch `ai-models/`. |
| `harness.yaml` | what the harness drives (measurement layer). |

## How to reproduce any recipe on a DGX Spark

### 0. Prerequisites

```bash
# Clone this repo + lmswitch (paths are your choice)
cd <path-to-dgx-spark-bench-repo>
uv venv harness/.venv && uv pip install --python harness/.venv/bin/python httpx pyyaml jsonschema
```

### 1. Start the model via lmswitch

Copy the bench profile into your lmswitch `ai-models/` directory, then start it.

```bash
cp <path-to-dgx-spark-bench-repo>/recipes/<recipe-id>/<recipe-id>.yaml <path-to-lmswitch>/ai-models/
lmswitch on <recipe-id>            # blocks until "Ready on port <port>"
```

### 2. Run the sweep

The harness drives a concurrency-throughput sweep against the served endpoint.

```bash
cd <path-to-dgx-spark-bench-repo>/harness
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

### Single-Spark trio (upstream recipes, not lmswitch)

Three of the recipes above (`qwen3.8-flash-next-single`, `deepseek-v4-flash-one-spark`,
`glm-5.3-flash-dflash2-single`) do **not** go through lmswitch: each is launched by its
upstream author's own `start.sh` / `serve_one_spark.sh` (MiaAI-Lab for the first two,
vcruz305 for the GLM one). Their harness configs' headers carry the exact launch command.
The ports (8890 / 8889 / 8891) avoid the daily driver on 8888. The GLM recipe is the only
llama.cpp lane here and accepts `min_tokens` but **not** `ignore_eos` (llama.cpp maps the
former; the latter is a vLLM/SGLang extra-body field — harmless to omit).

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
| [qwen3.8-27b-nvfp4-sglang](qwen3.8-27b-nvfp4-sglang/) (MTP) | Qwen3.8-27B | SGLang | NVFP4 mixed | 21 GB | First SGLang recipe — hybrid GDN VLM, in-checkpoint MTP head |
| [qwen3.8-27b-nvfp4-sglang](qwen3.8-27b-nvfp4-sglang/) (DSpark) | Qwen3.8-27B | SGLang | NVFP4 mixed | 23.6 GB | Same target, DSpark block-7 drafter — see caveat, not workload-neutral |
| [superqwen3.8-27b-ablit-nvfp4](superqwen3.8-27b-ablit-nvfp4/) | SuperQwen3.8-27B-abliterated | vLLM | NVFP4 W4A4 g16 | 19.2 GB | Eager mode (cudagraph capture has hard-reset this box) |
| [qwen3.8-flash-next-single](qwen3.8-flash-next-single/) | Qwen3.8-Flash-Next | vLLM (Mia's recipe) | NVFP4 | 99 GB | Single-Spark TP1, PLE offload, MTP k=3, 262k ctx |
| [deepseek-v4-flash-one-spark](deepseek-v4-flash-one-spark/) | DeepSeek-V4-Flash 0731 | sparkinfer (Mia's recipe) | EXL3 3.0 bpw | 107 GB | Single-Spark TP1, DSpark K5, 384k ctx |
| [glm-5.3-flash-dflash2-single](glm-5.3-flash-dflash2-single/) | GLM-5.3-Flash | llama.cpp (Cruz's recipe) | Q2_K GGUF + DFlash2 | 117 GB | Single-Spark, DFlash2 draft, 96k ctx |

## Bench profile design

All bench profiles change exactly three knobs from the daily-driver config and **label them**:

| Knob | Daily driver | Bench | Why |
|---|---|---|---|
| `max_num_seqs` | 4–16 | ≥ sweep ceiling (64) | Without it, everything past N=max_num_seqs just queues — the chart's knee would be a config artifact |
| `gpu_memory_utilization` | 0.4–0.85 | 0.3–0.5 | Bounds non-KV allocations; KV is sized by explicit byte cap |
| `ctx` | 262144 | 32768 (or 65536 for llama.cpp) | 256K pre-reserves huge KV/seq and throttles batching; the 1024/256 workload needs ~1.3K |

The harness sends a **unique prefix per request** so server-side prefix caching never hits — we measure real prefill, not cache replays.

### SGLang: what the three knobs translate to

The table above is written against vLLM. SGLang names these differently, and two of its
knobs have **no vLLM analogue at all** — miss them and the sweep measures the config, not
the hardware. Derived while building `qwen3.8-27b-nvfp4-sglang` (first sglang recipe here).

| vLLM (bench) | SGLang equivalent | Note |
|---|---|---|
| `max_num_seqs: 64` | `max_running_requests: 64` | Direct analogue. Below the sweep ceiling everything queues. |
| `gpu_memory_utilization: 0.5` | `mem_fraction_static` — **lower it only slightly** | **Inverted semantics.** In vLLM this bounds *non-KV* allocations, so bench drops it hard. In sglang the fraction *is* the weights+KV pool, so halving it starves KV instead. Go 0.95 → ~0.85, and read the `max_total_num_tokens=` line at boot rather than guessing. |
| `ctx: 32768` | `ctx: 32768` | Same. |
| — | `--cuda-graph-max-bs-decode` ≥ ceiling | **The silent one.** Daily configs pin this at 4. Leave it and decode falls off the graph path at N≥8, so you measure the eager fallback. |
| — | `--max-mamba-cache-size` = concurrency × 4 | Hybrid GDN models only. The GDN state pool is a *second* KV; 64 is sized for 16 running requests, not 64. |
| — | `--torch-compile-max-bs` ≥ ceiling | Only when `--enable-torch-compile` is on. Must track the decode-graph bound or compiled decode stops covering the swept batch sizes. |

Cost of the last one: on `qwen3.8-27b-nvfp4-sglang-dspark-bench`, raising it 4 → 64 made a cold
boot **~1600 s**, of which 1015 s was target-verify capture across 27 bs buckets and 409 s the
draft. That profile now mounts a bench-private inductor cache so re-boots don't re-pay it.
A ceiling-32 sweep only needs `--torch-compile-max-bs 32`, which roughly halves the compile —
the shipped profile keeps 64 because that is what produced the published JSON.

`min_tokens` and `ignore_eos` both work unchanged on SGLang's OpenAI endpoint (it maps
`min_tokens` → `min_new_tokens` internally), so `engine.extra_body` needs no per-engine casing.

## Known quirks

- **AEON CUDA-graph estimator bug:** The AEON vLLM build returns ~−20 GiB for CUDA-graph memory estimation, inflating the KV budget past physical RAM. Fix: explicit `--kv-cache-memory-bytes` (32–64 GiB) in every vLLM recipe.
- **llama.cpp KV:** Use `-np 32` (slots >= sweep ceiling) and `-kvu` (unified KV buffer). Explicit `-np` (no -kvu) measured ~30% slower per-stream at N=2–16.
- **Step-3.7-Flash** requires `force: true` in its profile because the RAM guard heuristic (weights × 1.3 = ~115 GB) exceeds the 107 GB free pool, but the real footprint (~92 GB with q8 KV) fits.
