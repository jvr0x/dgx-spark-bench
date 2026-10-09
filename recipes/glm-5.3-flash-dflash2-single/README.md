# Recipe: GLM-5.3-Flash · Cruz's single-Spark (llama.cpp + DFlash2)

Concurrency-throughput sweep of [`vcruz305/GLM-5.3-Flash-GGUF`](https://huggingface.co/vcruz305/GLM-5.3-Flash-GGUF)
(Q2_K, 2.91 bpw) on **one DGX Spark (GB10)**: llama.cpp llama-server (Cruz's fork, CUDA
`121a`), Q2_K target + DFlash2 BF16 draft, FP8 KV, 98K context, single sequence slot.
As-served config from the upstream recipe; paths changed to this host's checkout.

> **Frozen at the benchmarked config (2026-09-04).** This is the upstream
> [vcruz305/GLM-5.3-Flash-DFlash2-DGX-Spark-recipe](https://github.com/vcruz305/GLM-5.3-Flash-DFlash2-DGX-Spark-recipe),
> measured on this host. Q2_K is a fit-vs-quality trade-off: 4bpw EXL3 (~164 GiB) does not
> fit one 128 GiB Spark, so the single-box GLM lane must be 3 bpw or lower.

## Files

| File | Role |
|---|---|
| `harness.yaml` | harness target + series metadata for `bench.py` |

## As-served stack

| Setting | Value |
|---|---|
| Runtime | llama.cpp `llama-server` (Cruz fork `4a06ec6`, CUDA `121a`) |
| Engine | llama.cpp (llama-server, `llama-speculative-simple` for probes) |
| Target | `GLM-5.3-Flash-Q2_K.gguf` (116.7 GB, 2.91 bpw) |
| Draft | `GLM-5.3-Flash-DFlash2-BF16.gguf` (2.35 GB, Inco AI DFlash2) |
| KV cache | q8_0 (`-ctk q8_0 -ctv q8_0`) |
| Context | 98,304 (`-c 98304`; 96K is the last measured OK at ~114.8 GiB) |
| Spec decode | `--spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-p-min 0.30` |
| Concurrency | `-np 1` (single sequence slot) |

## Reproduce

```bash
# build llama.cpp fork (once)
cd ~/dev/community-recipes
git clone --depth 1 https://github.com/vcruz305/llama.cpp.git llama.cpp-glm5next
cd llama.cpp-glm5next && cmake -S . -B build-cuda -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121a
cmake --build build-cuda --target llama-server llama-speculative-simple llama-quantize -j 20

# weights (~119 GB total)
hf download vcruz305/GLM-5.3-Flash-GGUF GLM-5.3-Flash-Q2_K.gguf --local-dir ~/models/GLM-5.3-Flash-GGUF
hf download vcruz305/GLM-5.3-Flash-DFlash2-GGUF GLM-5.3-Flash-DFlash2-BF16.gguf --local-dir ~/models/GLM-5.3-Flash-DFlash2-GGUF

# serve
cd ~/dev/community-recipes/GLM-5.3-Flash-DFlash2-DGX-Spark-recipe
LLAMA_CPP=$HOME/dev/community-recipes/llama.cpp-glm5next \
  TARGET_GGUF=$HOME/models/GLM-5.3-Flash-GGUF/GLM-5.3-Flash-Q2_K.gguf \
  DRAFT_GGUF=$HOME/models/GLM-5.3-Flash-DFlash2-GGUF/GLM-5.3-Flash-DFlash2-BF16.gguf \
  PORT=8891 ./scripts/serve_one_spark.sh

# then, from ~/dev/dgx-spark-bench/harness:
taskset -c 0,1 .venv/bin/python bench.py \
  ../recipes/glm-5.3-flash-dflash2-single/harness.yaml \
  -o ../results/glm-5.3-flash-dflash2-single.json
```

## Notes

- Do not mix `--spec-type draft-mtp` with the DFlash2 draft; do not serve on the MTP-only
  branch (`ea37b8b`); do not use `-fit on` at 128K.
- The draft license is CC BY-NC-ND 4.0 (Inco AI) - research/eval use.
- DFlash2 acceptance ~31% on unique prose vs ~94% on repetitive/structured output; quote
  the structured ladder only as tok/s, not as a model quality score.
