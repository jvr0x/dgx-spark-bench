# Recipe: Laguna-S-2.1-FP8 · bare-metal vLLM+Ray (dual DGX Spark, TP=2)

Copied verbatim (IPs adapted to this cluster) from
https://howtospark.com/recipes/laguna-s-2.1-fp8-dual-spark-tp2 on 2026-07-22.
**Untested on this cluster.** Saved because it's a plausible fix for the
`shm_broadcast` hang that killed the NVFP4 dual attempt
([[poolside-laguna-s21-lmswitch]], `../laguna-s-2.1-nvfp4-dual/README.md`):
that attempt used vLLM's native `--nnodes 2` launcher with `enforce_eager: false`,
while this recipe explicitly requires `--enforce-eager` ("prevents cross-node
CUDA-graph deadlock on GB10") and uses `--distributed-executor-backend ray`
instead of the native multi-node launcher — both differences target exactly
the failure mode we hit.

**Not lmswitch.** This is bare-metal (systemd-run + a Python venv on both
nodes), not a Docker recipe — doesn't fit `runtime: vllm-dual` in
`~/utils/lmswitch/ai-models/`. Needs `~/venvs/vllm-025` (vLLM 0.25.1 + torch
2.11.0+cu130) built on both nodes first — does not exist yet on spark as of
2026-07-22.

Also needs the FP8 checkpoint (`poolside/Laguna-S-2.1-FP8`, ~121G) and its
W4A16 DFlash draft (`sapidlabs/Sparkulator-Laguna-S-2.1`, ~0.82G) downloaded —
neither is present in `~/models/poolside/` yet (only the NVFP4 pair is).

## Cluster IP mapping

Source recipe used generic `192.168.100.1/.2` placeholders over a direct RoCE
link. This cluster's real CX7 addresses (confirmed via `ip addr` + ssh config,
matching `[[lmswitch-vllm-dual-runtime]]` convention):

| Placeholder | Real (this cluster) |
|---|---|
| `192.168.100.1` (head) | `10.100.224.2` (spark) |
| `192.168.100.2` (worker) | `10.100.224.1` (Gigabyte) |

NIC/HCA names (`enp1s0f1np1` / `rocep1s0f1`) already matched the existing
dual convention — no change needed there.

## Networking env vars

```
NCCL_IB_DISABLE=0
NCCL_IB_HCA=rocep1s0f1
NCCL_SOCKET_IFNAME=enp1s0f1np1
GLOO_SOCKET_IFNAME=enp1s0f1np1
RAY_memory_monitor_refresh_ms=0
```

## Prerequisites (both nodes)

1. **Enable user lingering** — without it, systemd kills the worker raylet on
   SSH disconnect and vLLM hangs indefinitely:
   ```bash
   ssh 10.100.224.2 'loginctl enable-linger $USER'   # spark
   ssh 10.100.224.1 'loginctl enable-linger $USER'   # Gigabyte
   ```
2. vLLM 0.25.1 + torch 2.11.0+cu130 in `~/venvs/vllm-025` on both nodes.
3. NVIDIA driver ≥580, passwordless SSH between nodes, ninja+nvcc on PATH.
4. Stage checkpoints on both nodes (each TP rank reads all shards):
   ```bash
   hf download poolside/Laguna-S-2.1-FP8 --local-dir ~/models/hf/Laguna-S-2.1-FP8
   hf download sapidlabs/Sparkulator-Laguna-S-2.1 --local-dir ~/models/hf/Laguna-S-2.1-DFlash-W4A16

   for d in Laguna-S-2.1-FP8 Laguna-S-2.1-DFlash-W4A16; do
     rsync -a --partial --inplace \
       -e 'ssh -c aes128-gcm@openssh.com -o Compression=no' \
       ~/models/hf/$d/ 10.100.224.1:~/models/hf/$d/
   done
   ```
5. **vLLM patch (both nodes)**, in `vllm/model_executor/models/laguna_dflash.py`,
   guards `get_cache_scale` so the compressed-tensors W4A16 draft can load:
   ```python
   if self.quant_config is not None and hasattr(
       self.quant_config, "get_cache_scale"
   ) and (
       scale_name := self.quant_config.get_cache_scale(name)
   ):
   ```

## Ray cluster + serve

See `serve.sh` for the full copy-pasted sequence (GID resolution, Ray head/worker
start via `systemd-run --user`, and the `vllm serve` invocation) with IPs
already substituted for this cluster.

## Key parameters

| Parameter | Value | Purpose |
|---|---|---|
| `--enforce-eager` | required | prevents cross-node CUDA-graph deadlock on GB10 |
| `--distributed-executor-backend` | `ray` | NOT the native `--nnodes 2` launcher used by the failed NVFP4 dual attempt |
| `--speculative-config num_speculative_tokens` | 6 | W4A16 draft length, peak perf point per source |
| `--kv-cache-memory-bytes` | 5368709120 (5 GiB/rank) | ~273K tokens, ~1.04x context |
| `--tensor-parallel-size` | 2 | required for fit (checkpoint ~121G) |
| `--max-model-len` | 262144 | full native context |

## Gotchas (from source)

1. Lingering disabled → worker raylet dies on SSH disconnect → vLLM hangs.
2. Missing vLLM patch → quantized draft fails with AttributeError.
3. Mismatched draft (e.g. NVFP4 draft against FP8 target) → 0% acceptance.
4. Without `--enforce-eager` → CUDA-graph deadlock, engine times out sampling.
5. `qkv_proj` must stay BF16 — quantizing it breaks DFlash precompute.
6. Ignore-list format needs `re:` regex patterns, not literal names.

## Reported performance (source, not reproduced here)

60.85 tok/s decode (W4A16 DFlash k=6) vs 26.5 tok/s baseline; 62.2% draft
acceptance at k=6; TTFT 0.7s@2K / 8.9s@32K / 47s@128K / 118s@248K; ~81 GiB
head / ~74 GiB worker / ~40 GiB free each.

## Next steps before this can actually run here

- Build `~/venvs/vllm-025` on both nodes (or confirm an existing venv already
  satisfies vllm==0.25.1 + torch 2.11.0+cu130).
- Download FP8 checkpoint + W4A16 draft (~122G total) and mirror to Gigabyte.
- Apply the `laguna_dflash.py` patch on both nodes.
- Stop the daily-driver llama-server on both nodes first (see
  [[spark-vllm-gpu-mem-ceiling]] — same rule as every other dual recipe).
