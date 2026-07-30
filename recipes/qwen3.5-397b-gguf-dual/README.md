# qwen3.5-397b-gguf-dual

Qwen3.5-397B-A17B (unsloth **UD-IQ4_NL** GGUF, 180.5 GiB, all 512 experts) served
across **two DGX Sparks** with llama.cpp **RPC over CX7 RDMA**.

| | |
|---|---|
| Checkpoint | `unsloth/Qwen3.5-397B-A17B-GGUF/UD-IQ4_NL` |
| Backend | llama.cpp b9811 (`9df06805e`), RPC, 2 nodes |
| Topology | head = spark (`CUDA0`), worker = gigabyte (`RPC0`), `--tensor-split 0.45,0.55` |
| KV | q8_0, flash-attn on, `-kvu` unified buffer |
| Bench profile | `-np 8`, ctx 32768, port 8228 |

## Why GGUF and not NVFP4

`nvidia/Qwen3.5-397B-A17B-NVFP4-V2` is 226.8 GiB (NVFP4 `group_size=16` lands at
4.5 bpw once the FP8 microscales are counted). TP=2+EP=2 splits it ~107 GiB/node
against ~119 GiB of unified memory, which hard-reset the head node four times on
2026-07-23. This quant is 180.5 GiB → ~81/99 GiB with the split above, so it
actually boots.

vLLM's `vllm-dual` runtime cannot serve GGUF and the single-node llama runtime
cannot span nodes, hence the llama.cpp RPC path.

## Bench profile vs. daily driver

The daily driver (`qwen3.5-397b-gguf-dual.yaml`, port 8105) runs at
llama.cpp's default `--parallel 1`, which would flatten any concurrency sweep.
The bench profile here is a **separate file and port** (`-bench.yaml`, 8228) with
`-np 8 -kvu` and ctx dropped 65536 → 32768: with `-kvu` the ctx is one shared
pool, and 8 streams × 1280 tok ≈ 10K cells, so 32768 is ample while leaving
headroom (measured ~17 GiB free on the head after boot). Extra slots cost KV on
the 15 full-attention layers only — the other 45 are linear/GDN with a
fixed-size, ctx-independent recurrent state.

## Reproduce

```bash
ln -s "$PWD/recipes/qwen3.5-397b-gguf-dual/qwen3.5-397b-gguf-dual-bench.yaml" \
      <your-lmswitch-dir>/ai-models/
lmswitch on qwen3.5-397b-gguf-dual-bench      # Ready on port 8228

cd harness
.venv/bin/python bench.py ../recipes/qwen3.5-397b-gguf-dual/harness.yaml \
  -o ../results/qwen3.5-397b-gguf-dual.json

lmswitch off qwen3.5-397b-gguf-dual-bench
```

`rpc_cache: true` keeps the worker's ~99 GiB share under
`~/.cache/llama.cpp/rpc`, so only the head re-reads weights on a restart.

## Result (2026-07-30)

| N | aggregate tok/s | per-session tok/s | TTFT p95 |
|---|---|---|---|
| 1 | 12.8 | 13.6 | 2.0 s |
| 2 | 17.1 | 9.7 | 3.6 s |
| 4 | 22.8 | 6.5 | 6.3 s |
| 8 | 22.8 | 3.6 | 12.1 s |

Aggregate saturates at ~22.8 tok/s by N=4; from N=4 to N=8 only latency grows
(ITL p50 159.8 → 276.7 ms, TTFT p95 6.3 → 12.1 s). Slots are not the limit —
`-np 8` was served, and the N=8 point sat at 8 concurrent streams. What the
sweep shows is the flat aggregate; it does not isolate *why* (memory bandwidth
vs. RPC-link round trips per token would need a separate single-node or
link-instrumented run).
