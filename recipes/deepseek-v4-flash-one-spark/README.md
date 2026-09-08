# Recipe: DeepSeek-V4-Flash 0731 · Mia's single-Spark (EXL3, TP=1)

Concurrency-throughput sweep of [`0xSero/deepseek-v4-flash-0731-spark`](https://huggingface.co/0xSero/deepseek-v4-flash-0731-spark)
on **one DGX Spark (GB10)**: sparkinfer (formerly b12x) kernel stack, EXL3 3.0 bpw, TP=1,
NVFP4-DS-MLA compressed KV cache, DSpark K5 speculative decoding, 384K context. As-served
config from the upstream recipe; only `MAX_NUM_SEQS` (default 1 -> 4) and `SERVING_PORT`
(8888 -> 8889) were changed for the bench.

> **Frozen at the benchmarked config (2026-09-04).** This is the upstream
> [MiaAI-Lab one-DGX-Spark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark),
> measured on this host.

## Files

| File | Role |
|---|---|
| `harness.yaml` | harness target + series metadata for `bench.py` |

## As-served stack

| Setting | Value |
|---|---|
| Runtime | `ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer` (NVIDIA vLLM 26.02 base) |
| Engine | vLLM (sparkinfer / b12x) |
| Checkpoint | `0xSero/deepseek-v4-flash-0731-spark`, rev `22f28d32b9b29b4352eaa380ff8c2c170b2847ab` (~107 GB) |
| Architecture | MoE 284B total / 13B active, hybrid attention (DSA), text-only |
| KV cache | `nvfp4_ds_mla`, 432-byte native NVFP4 records (`KV_RECORD=stock432`) |
| Context | 384,000 (bench; default) |
| Spec decode | DSpark K5 (K64 draft model) |
| GPU utilization | 0.94 |

## Reproduce

```bash
cd ~/dev/miaai-lab/DeepSeek-v4-Flash-One-DGX-Spark
./download.sh                   # ~107 GB checkpoint + TP1 coalesce
MAX_NUM_SEQS=4 SERVING_PORT=8889 ./start.sh
# then, from ~/dev/dgx-spark-bench/harness:
taskset -c 0,1 .venv/bin/python bench.py \
  ../recipes/deepseek-v4-flash-one-spark/harness.yaml \
  -o ../results/deepseek-v4-flash-one-spark.json
```

## Notes

- The recipe defaults to `MAX_NUM_SEQS=1` (single deep-context request); the concurrency
  sweep needs the override above. The KV pool shrinks with seq count (hybrid cache split:
  2 seqs observed ~337K total), so keep `MAX_MODEL_LEN` lower when raising max seqs.
- `GPU_MEMORY_UTILIZATION=0.94` is the boot-safe max on this host (~114.5 GiB free).
- Weights download into `./hf-hub` (local) or a LAN mount (`REMOTE_HOST`).
