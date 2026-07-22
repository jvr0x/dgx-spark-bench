# Recipe: Laguna-S-2.1-NVFP4 · vLLM (dual DGX Spark)

Concurrency-throughput sweep of [`poolside/Laguna-S-2.1-NVFP4`](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4)
across **two DGX Sparks (GB10)**: tensor-parallel TP=2 over CX7, NVFP4 weights, fp8 KV cache.
See the solo recipe at [`../laguna-s-2.1-nvfp4/`](../laguna-s-2.1-nvfp4/) for the working,
benchmarked, DFlash-accelerated config — this dual copy exists only to document why TP=2
doesn't currently work for this model.

## Files

| File | Role |
|---|---|
| `laguna-s-2.1-nvfp4-dual.yaml` | lmswitch dual recipe (final attempt: AEON image, no DFlash) → copy into `ai-models/` |
| `laguna-bootstrap.sh` | flashinfer-install entrypoint used by the earlier stock-image attempts |
| `harness.yaml` | harness target + series metadata (unused — no successful run exists) |

## Measured results

**Blocked — not benched.** All three attempts hung under load; no `results/laguna-s-2.1-nvfp4-dual.json`
is published. See [Known blocker](#known-blocker-2026-07-22-shm_broadcast-hang-under-tp2) below.

## Known blocker (2026-07-22): shm_broadcast hang under TP=2

Three configurations were tried, each a genuine attempt to get a working dual server, not a
repeat of the same mistake:

1. **Stock `vllm/vllm-openai:v0.25.1` + `--trust-remote-code` + DFlash speculative decoding**
   (mirroring the solo recipe's own working setup). Booted clean — 65.44 GiB KV cache,
   3,738,604 tokens, 14.26x concurrency @ 262144 ctx — and served several requests correctly
   (spec-decoding metrics logged, ~27% draft acceptance). Hung a few minutes into the benchmark
   sweep.
2. **Same stock image, DFlash removed** (isolating whether DFlash itself was the cause). Hung
   after exactly one successful request.
3. **AEON image (`ghcr.io/aeon-7/aeon-vllm-ultimate`), no DFlash, no `--trust-remote-code`**
   (native Laguna support, ruling out the remote-code path as the cause). Hung before any
   request even arrived — right after `torch.compile` finished, during CUDA graph
   capture/profiling.

All three failed with the **identical signature**: the head node's local shared-memory RPC to
its own `Worker_TP0` process goes silent —

```
[shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds.
```

— repeating every 60s for 4 minutes, then a hard `TimeoutError: RPC call to sample_tokens
timed out.` kills the EngineCore. The head process exits (clean shutdown, exit 0), but the
**worker on the second node is left orphaned**, spinning on broken-pipe `TCPStore` errors for
several more minutes until manually `docker rm -f`'d on both nodes.

Reproducing across two different images, with and without speculative decoding, and at three
different points in the request lifecycle (mid-sweep, after one request, before any request)
rules out DFlash and the stock-image `--trust-remote-code` workaround as the cause. The most
likely explanation is something in Laguna's architecture itself — mixed sliding-window/global
attention (36:12 ratio) plus its custom MoE routing — desyncing vLLM 0.25.1's native
`--nnodes 2` multi-node TP launcher on this hardware, though the exact mechanism hasn't been
root-caused.

Needs investigation into whether this reproduces on a non-Laguna model of similar size under
the same launcher (to isolate model-architecture vs. general TP=2-launcher causes), and/or a
newer vLLM point release, before this recipe can be benched.
