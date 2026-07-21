# Recipe: Step-3.7-Flash NVFP4 · vLLM (dual DGX Spark)

Concurrency-throughput sweep of [`stepfun-ai/Step-3.7-Flash-NVFP4`](https://huggingface.co/stepfun-ai/Step-3.7-Flash-NVFP4)
across **two DGX Sparks (GB10)**: tensor-parallel TP=2 over CX7 RoCE, NVFP4 weights (modelopt),
fp8 KV cache. Baseline no-MTP checkpoint. As-served config — not a raised bench-profile
(see [Caveats](#caveats)).

## Files

| File | Role |
|---|---|
| `step-3.7-flash-dual.yaml` | lmswitch dual bench profile → copy into `ai-models/` |
| `harness.yaml` | harness target + series metadata for `bench.py` |

## As-served stack (measured 2026-07-21)

| Setting | Value |
|---|---|
| Runtime | `vllm-dual` — TP=2 across master (spark) + worker (gigabyte) over CX7 |
| Image | `vllm/vllm-openai:stepfun37` |
| Quantization | `modelopt` (NVFP4) |
| KV cache | `fp8` |
| Spec decode | none (baseline, no-MTP checkpoint) |
| `gpu_memory_utilization` | **0.85** |
| `max_num_seqs` | **8** (validated ceiling on this pair) |
| `max_model_len` | **262,144** |
| Port / served id | **8888** / `step-3.7-flash-dual` |

## Measured results

**Blocked — not yet benched.** Two attempts (2026-07-21) crashed mid-decode with the same
cross-node NCCL failure; see [Caveats](#caveats). No `results/step-3.7-flash-dual.json` exists
yet.

## Run it (on the master Spark)

```bash
cd ~/dev/dgx-spark-bench/harness
uv venv .venv && uv pip install --python .venv/bin/python httpx pyyaml jsonschema

cp ~/dev/dgx-spark-bench/recipes/step-3.7-flash-dual/step-3.7-flash-dual.yaml \
   ~/utils/lmswitch/ai-models/
lmswitch on step-3.7-flash-dual   # wait for Ready on port 8888 (both nodes)

taskset -c 0,1 .venv/bin/python bench.py \
  ../recipes/step-3.7-flash-dual/harness.yaml \
  -o ../results/step-3.7-flash-dual.json

lmswitch off step-3.7-flash-dual
```

## Caveats

- **As-served, not a bench-profile:** `max_num_seqs=8` is this pair's validated concurrency
  ceiling — not an artificially low daily-driver throttle.
- **Baseline, no MTP:** the MTP-grafted variant needs a separate BF16-grafted checkpoint
  (upstream's `graft-mtp.sh`), not used for this recipe.
- **Dual-node telemetry:** GPU temperature/power in `results/*.json` come from `nvidia-smi` on
  the master node only — the worker node's GPU isn't captured.
- **Filename id** must stay `step-3.7-flash-dual` — it is the OpenAI `model` id the harness calls.

### Known blocker (2026-07-21): cross-node NCCL hang mid-decode

Two bench attempts both crashed with the same failure, well into generation (not at startup):
a cross-node `_ALLGATHER_BASE` collective between spark and Gigabyte (CX7 RoCE) stalled and was
killed by PyTorch's `ProcessGroupNCCL` watchdog after exactly 600s, at a similar collective
sequence number both times (`SeqNum=187` then `SeqNum=167`). The engine declares itself dead
(`EngineDeadError`) and every subsequent request 500s.

Tried: `env.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900` (rules out slow JIT compile as the cause —
it wasn't) and `extra_args: --distributed-timeout-seconds=1800` (confirmed applied via
`non-default args` in the log, but the watchdog that actually fired belongs to a *different*
process group than the one that flag controls, so it had no effect and still fired at 600000ms).

The consistent mid-generation timing and near-identical sequence number across two independent
runs point at a reproducible deadlock in this recipe's dual-node collective path — not
one-time network flakiness or JIT warmup — so further timeout bumps are unlikely to help.
Needs investigation into which process group is stalling (likely tied to `step3p5`'s MoE/EP
all-gather under TP=2) and/or RoCE fabric health, before this recipe can be benched.
