# qwen3.6-27b-nvfp4-aeon7-dual — BLOCKED (not benchable on this image)

Intended as the TP=2 arm of a dual-vs-solo A/B against
[`qwen3.6-27b-nvfp4-aeon7-bench`](../qwen3.6-27b-nvfp4-aeon7-bench) (same
checkpoint, same image, same knobs, one Spark). **No result is published: the
engine dies before it can serve a single measured request.**

## The failure

Under TP=2 across spark + gigabyte, the engine hangs mid-decode and the worker
RPC times out:

```
INFO  shm_broadcast.py:705  No available shared memory broadcast block found in 60 seconds.
ERROR core.py:1233          TimeoutError: RPC call to sample_tokens timed out.
ERROR async_llm.py:704      vllm.v1.engine.exceptions.EngineDeadError
```

Every subsequent request 500s, so a full sweep returns all-zero — schema-valid
garbage. Both attempts are kept here as evidence, deliberately **outside**
`results/` and **not** in `results/manifest.json`:

- `FAILED-2026-07-30-all-zero.json` — with DFlash n=10
- `FAILED-2026-07-30-nospec-all-zero.json` — with `--speculative-config` removed

## What was ruled out

| Hypothesis | Verdict |
|---|---|
| Config error in this bench profile | **No.** The daily driver `qwen3.6-27b-nvfp4-aeon7-dual.yaml` hit the identical `sample_tokens` timeout on 2026-07-23 (13 matching log lines in its stopped container). Dual 27B has never served a token. |
| The DFlash drafter under TP=2 | **No.** `qwen3.6-27b-nvfp4-aeon7-dual-nospec.yaml` drops `--speculative-config` entirely and dies the same way at the N=1 warmup. The `scheduled_spec_decode_tokens: [-1 x 10]` in the DFlash run is a symptom, not the cause. |
| KV cache starvation | **No.** The dual boot reports `GPU KV cache size: 401,844 tokens` (2x the solo run's 200,867 — confirming `--kv-cache-memory-bytes` is per rank). The sweep's working set is ~41K tokens. |
| Boot / weight loading | **No.** Both profiles reach `/health` and answer a plain non-streaming request with 200 OK. |

## What actually triggers it

The engine survives simple non-streaming completions but dies on the harness
request shape: **streaming + `ignore_eos` + `min_tokens: 256` over 1024-token
prompts**. That is the same shape that fork-blocked
[`deepseek-v4-flash-spark-q3-llamacpp`](../deepseek-v4-flash-spark-q3-llamacpp),
and the `shm_broadcast` hang matches the reproducible TP=2 stall seen on the
Laguna S 2.1 NVFP4 dual profile. Suspected common cause: this vLLM build's
cross-node shm broadcast path, not anything model-specific.

Next thing to try, if someone picks this up: a different image (the
`vllm/vllm-openai:cu130-nightly` line rather than the AEON build), or
`--distributed-timeout-seconds` raised the way `step-3.7-flash-dual` does — that
recipe hit a related NCCL watchdog teardown and worked around it.

## Files

| File | Purpose |
|---|---|
| `qwen3.6-27b-nvfp4-aeon7-dual-bench.yaml` | TP=2 bench profile, knobs matched to the solo arm |
| `qwen3.6-27b-nvfp4-aeon7-dual-nospec.yaml` | Same, minus spec-decode — the isolation test |
| `harness.yaml` / `harness-nospec.yaml` | Sweep configs for each |

Both profiles hold ctx 32768, `max_num_seqs` 64, `--max-num-batched-tokens`
16384, chunked prefill + prefix caching, `--kv-cache-memory-bytes` 32 GiB per
rank, `gpu_memory_utilization` 0.5 — identical to the solo bench profile, so
only topology differs. Note this makes both arms *matched but untuned*: the
tuned daily profiles run util 0.85 solo / 0.75 dual.

## The solo arm (measured, for reference)

`results/qwen3.6-27b-nvfp4-aeon7-bench.json`, re-run 2026-07-30:

| N | agg tok/s | per-session | TTFT p95 |
|---|---|---|---|
| 1 | 22.8 | 27.9 | 13.1 s |
| 2 | 42.7 | 23.8 | 0.9 s |
| 4 | 71.1 | 19.9 | 1.4 s |
| 8 | 112.4 | 14.9 | 2.0 s |
| 16 | 140.8 | 10.1 | 13.8 s |
| 32 | 140.8 | 9.9 | 47.2 s |

Two independent runs 11 days apart agree within ~3% (140.8 vs 145.1 agg at
N=32), and the N=1 TTFT spike appears in **both** — it is a reproduced
characteristic of the DFlash cold path on the first request after boot, not a
one-off warmup artifact.
