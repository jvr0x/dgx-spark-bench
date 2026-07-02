# dgx-spark-bench

Real, reproducible LLM inference benchmarks on the **NVIDIA DGX Spark** (GB10 Grace Blackwell,
128 GB unified LPDDR5x @ 273 GB/s) — with a focus on what the box actually does under **agentic
workloads**: many parallel long-context sessions, not just single-stream chat.

**📊 Live dashboard:** [jvr0x.github.io/dgx-spark-bench](https://jvr0x.github.io/dgx-spark-bench/)

Every published number is a real run of this harness on real hardware. No projections, no
vendor numbers — and every series links to the exact recipe that produced it.

## How it works

```
lmswitch on <recipe>  ──►  harness/bench.py  ──►  results/<run>.json  ──►  web/ dashboard
     (serves model            (concurrency          (schema-validated,       (GitHub Pages)
      on its own port)         sweep + metrics)      pinned provenance)
```

Models are served with [**lmswitch**](https://github.com/jvr0x/lmswitch) — a small tool that
launches local LLMs (vLLM via Docker, GGUF via llama.cpp) from per-model YAML configs and
handles readiness polling. A **recipe** = one lmswitch bench-profile yaml + one harness config,
so anyone with a Spark can reproduce a published series with two clones and three commands.

The harness drives a closed-loop concurrency sweep (N parallel sessions, each streaming
back-to-back requests) against the OpenAI-compatible endpoint and reports, per N:

- **aggregate tok/s** — total decode throughput of the box
- **per-session tok/s** — the stream speed one user actually experiences
- **TTFT / ITL percentiles** (p50/p95/p99)

Prompts carry a unique per-request prefix so prefix caching can't fake prefill numbers.
Models are benchmarked **as served** (spec-decode on, real parsers) and the full serving
config is embedded in the results and shown in the dashboard's Config view.

## Reproduce a result

```bash
# 1. install lmswitch (see its README), then:
git clone https://github.com/jvr0x/dgx-spark-bench && cd dgx-spark-bench

# 2. register the recipe with lmswitch and serve it
ln -s "$PWD/recipes/qwen3.6-35b-nvfp4-nvidia/qwen3.6-35b-nvfp4-bench.yaml" <your-lmswitch-dir>/ai-models/
lmswitch on qwen3.6-35b-nvfp4-bench        # waits until the endpoint is ready

# 3. run the sweep (on the Spark itself)
cd harness && uv venv && uv pip install -e .
.venv/bin/python bench.py ../recipes/qwen3.6-35b-nvfp4-nvidia/harness.yaml -o my-run.json

lmswitch off qwen3.6-35b-nvfp4-bench
```

Compare `my-run.json` against the published `results/qwen3.6-35b-nvfp4-nvidia.json`.

## Repo layout

```
harness/     backend-agnostic async load generator (Python, tests in harness/tests/)
recipes/     one dir per benchmarked model: lmswitch bench-profile + harness config + README
results/     published runs; results/manifest.json lists ONLY real measured runs
schema/      results.schema.json — the harness ↔ dashboard data contract
web/         the dashboard (vanilla HTML/CSS/JS, no build step)
```

## Adding a recipe

1. Copy an existing dir under `recipes/`, point it at your model, and tune the bench knobs
   (`max_num_seqs` ≥ your sweep ceiling; on Spark, pin KV size explicitly — see the comments
   in the flagship recipe for the hard-won details).
2. Fill in `series:` metadata in `harness.yaml`, including `config:` (the load-bearing engine
   flags) and `recipe_url`.
3. Run the sweep, drop the json into `results/`, add it to `results/manifest.json`.

## Reproducibility principles

On fixed hardware the only real variable is the software stack, so everything is pinned:
container image, model revision, engine flags, harness version, workload definition. Each
`results.json` records the environment it was produced in.

## License

[MIT](LICENSE)
