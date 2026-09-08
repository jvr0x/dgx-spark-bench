# Recipe: Qwen3.8-Flash-Next · Mia's single-Spark (TP=1)

Concurrency-throughput sweep of [`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4)
on **one DGX Spark (GB10)**: vLLM TP=1, PLE n-gram table offloaded to host RAM and
memory-mapped (`MADV_RANDOM`), FP8 KV, MTP k=3 speculative decoding, 262K native context.
As-served config from the upstream recipe; only `PORT=8890` and `KV_TARGET_GIB` were touched
for the bench (port rendered moot once the run starts, asleep on the old daily driver).

> **Frozen at the benchmarked config (2026-09-04).** This is the upstream
> [MiaAI-Lab single-Spark recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark),
> measured on this host the same day the recipe was published.

## Files

| File | Role |
|---|---|
| `harness.yaml` | harness target + series metadata for `bench.py` |

## As-served stack

| Setting | Value |
|---|---|
| Runtime | vLLM (`vllm/vllm-openai:qwen38-flash-next`) |
| Engine | vLLM qwen38-flash-next image (single-container, TP=1) |
| Checkpoint | `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` (~99 GB) |
| Architecture | MoE 125B total / 6B active, hybrid GDN + QSA, vision-language |
| KV cache | FP8 (`KV_CACHE_DTYPE=fp8`), ~22 GiB target (KV_TARGET_GIB=20 in env sample) |
| Context | 262,144 (native; YaRN off) |
| Spec decode | MTP k=3 (in-checkpoint MTP module) |
| PLE table | offloaded to system RAM, memory-mapped (MADV_RANDOM) |

## Reproduce

```bash
cd ~/dev/miaai-lab/Qwen3.8-Flash-Next-Single-DGX-Spark
cp .env.sample .env   # set PORT=8890
./download.sh         # ~99 GB checkpoint
./start.sh            # ~10-12 min to /health; serves :8890
# then, from ~/dev/dgx-spark-bench/harness:
taskset -c 0,1 .venv/bin/python bench.py \
  ../recipes/qwen3.8-flash-next-single/harness.yaml \
  -o ../results/qwen3.8-flash-next-single.json
```

## Notes

- The PLE table (~27 GB) is built on first launch beside the checkpoint; budget ~130 GiB disk.
- `MAX_NUM_SEQS=4` (env sample) caps the concurrency sweep at 4.
- The daily-driver port 8888 is checked by the `comfy-h3.service` watch; 8890 avoids it.
