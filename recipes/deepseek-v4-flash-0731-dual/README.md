# Recipe: DeepSeek-V4-Flash 0731 1M · Anemll vLLM 0.25 (dual DGX Spark)

Concurrency-throughput sweep of [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
across **two DGX Sparks (GB10)**: tensor-parallel TP=2 over CX7 RoCE, NVFP4-DS-MLA KV cache,
b12x MoE backend, DSpark speculative decoding (K=5), 1M context. As-served config, values taken
verbatim from the live lmswitch recipe — not a raised bench profile.

> **Frozen at the benchmarked config (2026-08-01).** The live lmswitch recipe moved on
> 2026-08-20: it now mounts MiaAI-Lab's runtime hotfix suite (upstream `main` @ `21da90f`)
> and adds `--long-prefill-token-threshold=1024`, `--enable-prompt-tokens-details`,
> `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`, `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096`
> and `DSPARK_MAX_INFLIGHT_PREFILLS=2`. The yaml here is deliberately *not* updated — it
> is what produced the numbers below. Re-copy it from `~/utils/lmswitch/ai-models/` and
> re-run the sweep before comparing against the current lane; upstream's own A/B puts
> 32K x c4 per-stream decode at 8.2 -> 24.6 tok/s from the #27 change alone.

Distinct from [`deepseek-v4-flash-dspark-dual`](../deepseek-v4-flash-dspark-dual): different
checkpoint (0731 vs the DSpark preview), different memory shape (util 0.80 / `max_num_seqs` 6 vs
0.85 / 12), different spec-decode depth (K=5 vs K=3 on the profile that produced that lane's
published numbers).

## Files

| File | Role |
|---|---|
| `deepseek-v4-flash-0731-dual.yaml` | lmswitch dual profile → copy into `ai-models/` |
| `harness.yaml` | harness target + series metadata for `bench.py` |
| `probe-decode-shapes.py` | N=1 decode-rate + draft-acceptance probe behind the workload-choice section |
| `upstream-shape-run.json` | MiaAI-Lab `benchmark-0731.py` run verbatim against this server (independent cross-check) |
| `forced-shape-run.json` | this server under the repo's **default** workload (46.9 tok/s N=1, 123.7 agg @N=6) |
| `run1-co-resident.json` | repeat of the default-workload run with the idle `qwen3-vl-8b` co-resident (repeatability record) |

None of the three side runs are in `results/manifest.json`, so the dashboard shows exactly one
series for this recipe.

## As-served stack

Ported from the [MiaAI-Lab two-Spark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
(`docker-compose.dspark.yml` + `docs/DEEPSEEK_V4_FLASH_0731.md`); memory shape and serving
preferences are ours.

| Setting | Value |
|---|---|
| Runtime | `vllm-dual` — TP=2 across master (spark) + worker (gigabyte) over CX7 RoCE |
| Image | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` |
| Engine | vLLM `0.25.2.dev0+g752a3a504` (reported by `/version` on the live server) |
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731`, rev `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| MoE backend | `flashinfer_b12x` |
| KV cache | `nvfp4_ds_mla`, profiler-sized (unpinned) — 2,031,837-token pool at boot |
| Spec decode | dspark, **5** tokens, probabilistic draft sampling (checkpoint `dspark_block_size=5`) |
| CUDA graphs | on, **regular** — `VLLM_USE_BREAKABLE_CUDAGRAPH=0`, `enforce_eager` off |
| Capture size | requested 36 (`max_num_seqs * (MTP+1)`), engine truncates to 32 |
| `gpu_memory_utilization` | **0.80** |
| `max_num_seqs` | **6** |
| `max_num_batched_tokens` | **8192**, `block_size` 256, chunked prefill + async scheduling on |
| `max_model_len` | **1,048,576** |
| Port / served id | **8888** / `deepseek-v4-flash-0731-dual` |

The Anemll image predates 0731 and ships the preview encoder, so
`ai-models/scripts/dspark-0731-bootstrap.sh` (mounted on **both** nodes as the entrypoint) installs
the checkpoint's `encoding/encoding_dsv4.py` into vLLM and patches the `deepseek_v4` tokenizer.
`DSPARK_ENCODING_FILE=/model/encoding/encoding_dsv4.py` is required because lmswitch bind-mounts
the checkpoint at `/model` rather than in the HF cache layout upstream's compose globs.

## Measured results

Workload: chat, 1024 prompt / 256 output tokens, closed-loop, unique prefixes (prefix cache
defeated), 30 s warmup + 180 s measure per point.

Measured 2026-08-02 (`results/deepseek-v4-flash-0731-dual.json`). Workload: 256-token prompt,
**natural stop**, temp 0.6 / top-p 0.95, output instruction "Return exactly 128 numbered lowercase
English words, then stop." — closed loop, unique prefixes, 30 s warmup + 180 s measure per point.

| N | Agg tok/s | Per-session tok/s | TTFT p50 | TTFT p95 | ITL p50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 89.8 | **95.5** | 327 ms | 340 ms | 64.5 ms |
| 2 | 135.1 | 71.8 | 422 ms | 453 ms | 80.0 ms |
| 4 | 192.9 | 51.8 | 502 ms | 667 ms | 105.1 ms |
| 6 | **233.2** | 42.1 | 560 ms | 752 ms | 128.2 ms |

Peak aggregate **233.2 tok/s @ N=6** — the profile's `max_num_seqs` ceiling, so the curve is still
climbing at the last point (2.60x scaling from N=1 to N=6).

> ⚠️ **This series does not use the repo's default workload**, and is therefore not directly
> comparable to the forced-output series in `results/`. The reason is below; the same server
> measured under the default workload is kept in this directory as `forced-shape-run.json`
> (46.9 tok/s N=1, 123.7 agg @N=6) along with a repeat run, `run1-co-resident.json`.

### Why the workload is not this repo's default

Because on a spec-decode profile there is no single decode rate, and the repo's default workload
lands at the bottom of the range. The engine's **step rate is
constant at 15.2 steps/s** (66 ms/step) no matter what you ask it; throughput is that number
multiplied by how many speculative tokens survive verification, and acceptance is a property of the
*text being generated*. Measured on this server at N=1, all rates from vLLM's own metrics
(`generation_tokens_total` / `request_decode_time_seconds`, so no client overhead in the number) —
reproduce with `probe-decode-shapes.py`:

| Output shape | Decode tok/s | Draft acceptance | tok/step | steps/s |
|---|---:|---:|---:|---:|
| numbered word list, temp 0.6, natural stop | 89.6 | 97.8% | 5.89 | 15.2 |
| numbered word list, temp 1.0, forced 512 | 71.2 | 73.8% | 4.69 | 15.2 |
| free-form prose, temp 0.0, natural stop | 55.3 | 51.1% | 3.56 | 15.5 |
| free-form prose, temp 1.0, natural stop | 51.2 | 47.2% | 3.36 | 15.2 |
| filler-prompt summarise, forced 256 (**repo default**) | 38.2 | 30.2% | 2.51 | 15.2 |
| filler-prompt summarise, forced 512 (**repo default**) | 41.8 | 34.4% | 2.72 | 15.4 |

Ceiling is `~15.2 x 6` ≈ 91 tok/s at perfect acceptance (K=5 plus the bonus token); floor is ~15
tok/s if nothing drafts. The repo default — filler prompt, generation forced past EOS — sits near the
bottom of that range at ~30% acceptance, which is why the published series above uses the draftable
shape instead. Interactive use lands between the two rows, drifting up on structured or repetitive
output.

(The published N=1 per-session figure, 95.5 tok/s, sits slightly above that 91 estimate: step rate
is a little higher than 15.2/s at these short sequence lengths, where the KV read per step is
smallest. The 15.2 figure was measured on 512-token generations.)

### Cross-check against upstream's own harness

[MiaAI-Lab's `scripts/benchmark-0731.py`](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
run verbatim against this same server, unchanged: it prompts for "exactly 128 numbered lowercase
English words", i.e. the highly-draftable end of the table above. Full output in
`upstream-shape-run.json`.

| Prompt tokens | c | Aggregate tok/s | Median decode tok/s | Median TTFT | Prefill tok/s |
|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 70.9 | **73.6** | 261 ms | 1,074 |
| 256 | 2 | 116.1 | 62.2 | 414 ms | 676 |
| 256 | 4 | 168.3 | 44.8 | 657 ms | 426 |
| 256 | 6 | **211.8** | 38.9 | 891 ms | 314 |
| 2048 | 1 | 63.5 | 73.5 | 1,104 ms | 1,877 |
| 2048 | 6 | 152.1 | 33.6 | 4,553 ms | 511 |

That reproduces upstream's published two-Spark figures (75.4 tok/s C1, 191.2 aggregate at c6) on
this pair, and confirms the serving profile is healthy: same box, same container, same hour, 73.6
vs 46.9 tok/s purely from workload shape.

The published series above runs the same shape through this repo's harness (which is why it carries
full TTFT/ITL percentiles that upstream's script does not emit) and reads a little higher again —
89.8 agg / 95.5 per-session at N=1, 233.2 agg at N=6 — because its generations are shorter and more
uniform than upstream's 512-token ones.

No cross-lane claim is made against `deepseek-v4-flash-dspark-dual`: that series is a different
checkpoint, K=3 vs K=5, util 0.85 vs 0.80 and `max_num_seqs` 12 vs 6, and its published numbers are
already flagged in-repo as a prior-engine record pending re-measure. Compare them in the dashboard,
where each series carries its own config.

## Caveats

- **Not comparable to the other series in `results/`.** They use the repo default (filler prompt,
  forced past EOS); this one uses draftable output with a natural stop, deliberately, because the
  default measures this profile at ~30% draft acceptance. Comparing this series' 89.8 against a
  forced-output series' number compares workloads, not hardware. `forced-shape-run.json` in this
  directory has the default-workload numbers if that is the comparison you want.
- **Every other spec-decode series in this repo is understated** by the same effect, by an amount
  nobody has measured. Only this recipe has been quantified so far.
- **As-served, not a bench profile:** `max_num_seqs=6` is the recipe's own value (paired with util
  0.80 and capture size 36 — the three move together), so the sweep stops at N=6.
- **1M-ctx checkpoint benched at a 256-token prompt:** short-context roofline, not a long-context
  test.
- **Both workloads were run on an otherwise empty box.** Under the default workload, a repeat run
  with the idle `qwen3-vl-8b` GGUF (4.8 G) co-resident matched within 2% at every point, so the
  daily driver's presence is not a confound at this shape.
- **Dual-node telemetry:** GPU temperature/power in `results/*.json` come from `nvidia-smi` on the
  master node only — the worker node's GPU isn't captured.
- **Local deviations from upstream (deliberate):** `thinking:false` chat-template default, our own
  tool-call/reasoning parser flags and `--generation-config=vllm`.
- **Filename id** must stay `deepseek-v4-flash-0731-dual` — it is the OpenAI `model` id the harness
  calls.

## Run it (on the master Spark)

The image must exist on **both** nodes first (`docker pull ghcr.io/anemll/dspark-vllm-gx10:0.1.1`,
same on `Gigabyte`).

```bash
cp ~/dev/dgx-spark-bench/recipes/deepseek-v4-flash-0731-dual/deepseek-v4-flash-0731-dual.yaml \
   ~/utils/lmswitch/ai-models/
lmswitch on deepseek-v4-flash-0731-dual   # wait for Ready on port 8888 (both nodes)

cd ~/dev/dgx-spark-bench/harness
uv venv .venv && uv pip install --python .venv/bin/python -e .

taskset -c 0,1 .venv/bin/python bench.py \
  ../recipes/deepseek-v4-flash-0731-dual/harness.yaml \
  -o ../results/deepseek-v4-flash-0731-dual.json
```

First start pays a cold FlashInfer autotune pass on top of the 48-shard TP=2 weight load over NFS —
`ready_timeout` is 3600 for that reason. The local daily driver should be off during **startup**
(a loaded 36 G GGUF shrinks profiled free memory and 1M KV sizing fails).
