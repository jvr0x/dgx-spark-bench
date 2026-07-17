# GLM 5.2-2bit Recipe: Rigorous Evaluation Framework

> **Purpose:** Design a test plan that proves or disproves whether the GLM 5.2-2bit recipe is truly "the best/fastest" quantization approach for its class of models.
>
> **Model context:** GLM 5.2 (Zhipu AI / Z.ai) — **744–753B total parameters, ~40B active per token** (MoE), 1M-token context, MIT license. Trained on ~28.5T tokens. Direct competitor to Claude Opus 4.8, GPT-5.5, DeepSeek-V4-Flash.
>
> **Hardware context:** Designed for **4× DGX Spark cluster** (4 × GB10, 512 GB unified LPDDR5x, ~600 Gbps NVLink-C2C per node, 200 Gbps RoCEv2 inter-node). Single-node runs are possible but severely context-limited. Also generalizes to any platform.
>
> **Critical hardware constraint:** The GB10 has **no native 2-bit Tensor Core support**. 2-bit quantization runs through FP8/FP16 dequantize paths — it provides memory savings but no compute acceleration. The QuantTrio INT4-INT8Mix recipe achieves ~28.8 tok/s single-stream on 4× Spark. A claimed 21.5 tok/s must be evaluated against the full cost: accuracy loss + memory headroom vs. FP4 baseline that gets actual FP4 compute acceleration.
>
> **Available quantizations (GLM 5.2 ecosystem):**
> | Format | Size | Notes |
> |---|---|---|
> | BF16 (original) | ~1,510 GB | Does not fit on any single platform |
> | FP8 (NVIDIA) | ~755 GB | Multi-GPU server |
> | NVFP4 (NVIDIA) | ~469 GB effective | vLLM/SGLang, native Tensor Cores |
> | GGUF Q4_K_XL | ~467 GB | Unsloth Dynamic GGUF, near-lossless |
> | GGUF IQ2_M | ~239 GB | Unsloth Dynamic GGUF, ~82% retention |
> | GGUF 1-bit | ~217 GB | Unsloth Dynamic GGUF, aggressive |
> | **QuantTrio INT4-INT8Mix** | ~405–420 GB | 4× DGX Spark, MTP spec-decode, k=4 |
> | **2-bit AWQ/GPTQ (target)** | **~300–377 GB** | The recipe under evaluation |

---

## 1. Accuracy Benchmarks — What to Measure and Why

### 1.1 Core Suite (Required)

| Benchmark | What It Tests | Why It Matters for 2-bit |
|---|---|---|
| **MMLU** (57 tasks, 14K questions) | Broad knowledge, reasoning across domains | The gold-standard general intelligence metric. 2-bit quantization typically costs 3–12 points on MMLU. If the GLM 5.2-2bit recipe claims "best," it must match or beat the FP4 baseline within ±2 points. |
| **MMLU-Pro** (8K questions, multi-step reasoning) | Chain-of-thought reasoning, harder variants | 2-bit quantization disproportionately hurts multi-step reasoning. This is the stress test that separates "usable" from "degraded." |
| **IFEval** (Instruction Following Evaluation) | Prompt adherence, tool calling, format compliance | Agentic workloads require strict instruction following. 2-bit models often lose format compliance (JSON, XML, code blocks). Critical for the DGX Spark's agentic workload profile. |
| **HumanEval** (Python, 164 problems) | Code generation, functional correctness | Code is highly sensitive to quantization artifacts. A 2-bit model must maintain ≥85% of the FP16 baseline's pass@1 to be viable for coding agents. |
| **MBPP** (318 problems, basic Python) | Basic code understanding and generation | Smaller, more forgiving than HumanEval. Used to separate "can't code" from "makes silly mistakes." |

### 1.2 Extended Suite (Recommended)

| Benchmark | What It Tests | Why It Matters |
|---|---|---|
| **MATH** (Hendrycks dataset) | Mathematical reasoning, competition problems | 2-bit quantization is known to severely damage mathematical reasoning. This is where accuracy cliff often happens. |
| **GSM8K** (grade-school math) | Multi-step arithmetic reasoning | Faster to run than MATH, good early indicator of math degradation. |
| **HellaSwag** | Commonsense reasoning, action prediction | Classic NLP benchmark; catches degradation in language understanding. |
| **TruthfulQA** | Truthfulness, avoids misinformation | 2-bit models can become more prone to hallucination. |
| **ARC-Challenge** | Scientific reasoning, grade-school science | Tests domain reasoning beyond pure language. |
| **LiveBench** (live, rotating) | Anti-contamination benchmark | Prevents benchmark gaming. Run quarterly. |
| **SWE-bench Verified** | Real-world software engineering tasks | The ultimate code benchmark. Expensive to run but definitive. |
| **C-Eval** | Chinese language understanding, knowledge | GLM 5.2 is strong in Chinese; 2-bit may hurt Chinese differently than English. |
| **CMMLU** | Chinese multi-task language understanding | Complements C-Eval with broader Chinese coverage. |

### 1.3 Accuracy Measurement Protocol

```yaml
# Accuracy harness config (extends dgx-spark-bench pattern)
accuracy:
  # Run each benchmark at BOTH quantizations and compare
  baselines:
    - name: "GLM-5.2-FP4"          # 4-bit NVFP4 (native Tensor Core)
      quantization: "NVFP4"
      reference_score: null         # measure on same hardware
    - name: "GLM-5.2-BF16"         # Full precision (if fits)
      quantization: "BF16"
      reference_score: null
  candidates:
    - name: "GLM-5.2-2bit-AWQ"     # The recipe under test
      quantization: "2bit-AWQ"
    - name: "GLM-5.2-2bit-GPTQ"    # Alternative 2-bit method (if exists)
      quantization: "2bit-GPTQ"

  # Each benchmark must report:
  #   - raw score (accuracy %)
  #   - delta from BF16 baseline (in points)
  #   - delta from FP4 baseline (in points)
  #   - delta from QuantTrio INT4-INT8Mix (in points)
  #   - statistical significance (95% CI if applicable)

  # Temperature/decoding must be identical across all quantizations
  decoding:
    temperature: 0.7
    top_p: 0.9
    top_k: 50
    repetition_penalty: 1.05

  # Chinese-language benchmarks (GLM 5.2 strength area)
  chinese_benchmarks:
    - name: "C-Eval"
    - name: "CMMLU"
    - name: "CLUEWSC"  # Chinese winograd schema
```

**Key rule:** Accuracy must be measured with **identical decoding parameters** across all quantizations. A 2-bit model running with temperature=0.1 cannot be fairly compared to FP4 at temperature=0.7.

---

## 2. Throughput Baselines — What to Compare 21.5 tok/s Against

### 2.1 The Baseline Hierarchy

A single throughput number (21.5 tok/s) is meaningless without context. You need **four** comparisons, all on the **same hardware configuration** (4× DGX Spark cluster, single-node decode measurement):

| Comparison | Against What | Why |
|---|---|---|
| **1. Same model, QuantTrio INT4-INT8Mix** | QuantTrio recipe (~28.8 tok/s, MTP spec-decode k=4) | **The primary baseline.** This is tonyd2wild's production recipe. If 2-bit is 75% of QuantTrio's speed, the accuracy savings must justify the slowdown. |
| **2. Same model, NVFP4 quant** | NVIDIA's nvidia/GLM-5.2-NVFP4 | **The compute-optimized baseline.** NVFP4 gets dedicated Blackwell Tensor Cores. If 2-bit is slower than NVFP4, the only advantage is memory savings. |
| **3. Same model, GGUF Q4_K_XL** | Unsloth Dynamic GGUF via llama.cpp | **The cross-engine baseline.** Measures whether the quantization method or the serving engine is the bottleneck. |
| **4. Competing 2-bit methods** | AWQ vs. GPTQ vs. RTN on GLM 5.2 | Proves this specific recipe's quantization approach is optimal. |

**Important:** The 21.5 tok/s claim must specify whether it's measured on **one node** or **aggregate across all 4 nodes**. The QuantTrio recipe reports ~28.8 tok/s per-node with MTP speculative decoding (k=4). Without spec-decode, expect ~12–15 tok/s.

### 2.2 Throughput Measurement Protocol

```yaml
# Throughput harness config — single-request latency
throughput_single:
  workload:
    prompt_tokens: 128        # short prompt, measure decode
    output_tokens: 512        # long output, amortize prefill cost
  concurrency: [1]
  measurement:
    min_requests: 50          # statistical significance
    engine: vLLM              # or llama.cpp — run both

# Throughput harness config — aggregate throughput
throughput_batch:
  workload:
    prompt_tokens: 1024       # realistic agentic prompt
    output_tokens: 256
  concurrency: [1, 2, 4, 8, 16, 32]  # sweep
  measurement:
    warmup_s: 30
    measure_s: 180
    engine: vLLM

# Claim validation: "21.5 tok/s" must be reported as:
#   throughput_single:
#     agg_tok_s: 21.5          # aggregate tokens/sec at N=1
#     per_session_tok_s: 21.5  # per-session at N=1 (same when N=1)
#     ttft_ms: {p50: 45, p95: 52, p99: 61}
#     itl_ms: {p50: 46.5, p95: 52.1, p99: 68.3}
#     hardware: "DGX Spark"
#     model: "GLM-5.2-2bit"
#     quant: "2bit-AWQ"
#     backend: "vLLM 0.23"
#     batch_size: 1
#     prompt_tokens: 128
#     output_tokens: 512
```

### 2.3 The "Best/Fastest" Claim Test

For the claim "best/fastest 2-bit recipe" to hold:

1. **Fastest:** Must beat ALL other 2-bit quantizations of the same model (AWQ, GPTQ, RTN) by ≥5% in aggregate throughput at N=1, N=4, and N=8.
2. **Best accuracy among fast:** Must be within ±2 MMLU points of the FP4 baseline AND within ±5 of BF16.
3. **Memory efficient:** Must use ≤50% of the memory of the BF16 baseline (theoretical: 2-bit = 1/4 the weight size).

If any of these three fails, the claim is false.

---

## 3. Latency Percentiles — p50, p90, p99 Methodology

### 3.1 Why Percentiles Matter

**Mean throughput is misleading for real-world usage.** Consider two systems both averaging 21.5 tok/s:

| Metric | System A (Consistent) | System B (Spiky) |
|---|---|---|
| Mean tok/s | 21.5 | 21.5 |
| p50 tok/s | 21.5 | 25.0 |
| p90 tok/s | 21.3 | 12.0 |
| p99 tok/s | 21.0 | 3.2 |

System A delivers a smooth, predictable experience. System B has occasional catastrophic stalls. For **agentic workloads** (which the DGX Spark is optimized for), the p99 matters far more than the mean. An agent waiting 3 seconds for a single token (p99) will timeout on tool calls, break conversation flow, and feel broken to users.

### 3.2 What Each Percentile Tells You

| Percentile | What It Measures | Real-World Impact |
|---|---|---|
| **p50 (median)** | Typical user experience | "This feels responsive" |
| **p90** | 1 in 10 requests feel slow | Acceptable degradation threshold |
| **p95** | 1 in 20 requests are noticeably slow | Where user frustration begins |
| **p99** | 1 in 100 requests are painful | **Critical threshold** — beyond this, the system feels broken |

### 3.3 Latency Measurement Protocol

```yaml
latency:
  # Measure both TTFT (time-to-first-token) and ITL (inter-token latency)
  metrics:
    - name: ttft_ms
      description: "Time from request send to first content token"
      percentiles: [p50, p90, p95, p99]
      why: "Prefill cost. Matters for long prompts (RAG, document QA)."
    - name: itl_ms
      description: "Time between consecutive output tokens"
      percentiles: [p50, p90, p95, p99]
      why: "Decode smoothness. Matters for chat, agentic loops."

  # Run at multiple concurrency levels — latency degrades differently than throughput
  concurrency_sweep: [1, 2, 4, 8, 16, 32]

  # Use fixed prompt/output lengths for comparability
  fixed_workload:
    prompt_tokens: 1024
    output_tokens: 256

  # Statistical rigor:
  min_samples: 100          # minimum completed requests per point
  confidence_interval: 0.95 # report 95% CI on all percentiles

  # Track tail latency budget:
  tail_latency_budget:
    ttft_p99_max_ms: 200    # after 200ms TTFT, agent feels laggy
    itl_p99_max_ms: 150     # after 150ms ITL, streaming feels choppy
```

### 3.4 The Latency Budget Framework

For agentic workloads on the DGX Spark, define a **latency budget** per operation:

```
Agent Loop Latency Budget (target: <2s total):
├── Tool call parsing:    <50ms    (TTFT for structured output)
├── Model reasoning:      <500ms   (TTFT + first 10 tokens)
├── Tool execution:       <1000ms  (external, not model)
├── Response streaming:   <500ms   (ITL p99 < 50ms for 10-token response)
└── Buffer:               <100ms   (network, serialization)
```

The GLM 5.2-2bit recipe must demonstrate that its p99 ITL stays within budget at the target concurrency level. If p99 ITL at N=4 is 120ms, the agent loop budget is blown.

---

## 4. Memory Bandwidth Analysis — Memory-Bound or Compute-Bound?

### 4.1 The Fundamental Question

The GB10 has **273 GB/s LPDDR5x bandwidth** and **~1 PFLOP FP4 compute**. The question is: at 21.5 tok/s, which resource is the bottleneck?

### 4.2 Theoretical Analysis

For autoregressive decode (single-token generation), the operation is:

```
Memory traffic per token ≈ active_params × quant_bytes + KV_read + overhead
```

For GLM 5.2 (~40B active parameters, MoE) on a **single DGX Spark** (273 GB/s bandwidth):

| Precision | Weight Traffic/token | Theoretical Max tok/s (bandwidth-limited) |
|---|---|---|
| BF16 (2 B/param) | ~80 GB/token | 273 / 80 ≈ 3.4 tok/s ❌ |
| FP4 (0.5 B/param) | ~20 GB/token | 273 / 20 ≈ 13.7 tok/s |
| **2-bit (0.25 B/param)** | **~10 GB/token** | **273 / 10 ≈ 27.3 tok/s** |
| NVFP4 (0.5 B/param) | ~20 GB/token | 273 / 20 ≈ 13.7 tok/s |

For a **4× DGX Spark cluster** (aggregate ~1,092 GB/s theoretical, but inter-node RoCEv2 at ~20–23 GB/s per-link limits effective bandwidth):

| Precision | Weight Traffic/token | Theoretical Max tok/s (4-node cluster) |
|---|---|---|
| BF16 (2 B/param) | ~80 GB/token | ~1.0 tok/s (network-bound) ❌ |
| FP4 (0.5 B/param) | ~20 GB/token | ~4 tok/s (network-bound) |
| **2-bit (0.25 B/param)** | **~10 GB/token** | **~8 tok/s (network-bound)** |
| NVFP4 (0.5 B/param) | ~20 GB/token | ~4 tok/s (network-bound) |

**Key insight for the 21.5 tok/s claim:**
- On a **single Spark**, 21.5 tok/s would be **at or above the theoretical bandwidth limit** for 2-bit (~27.3 tok/s), meaning the claim is either inflated, measured differently, or the model has fewer active parameters than 40B.
- On a **4× Spark cluster**, the 21.5 tok/s is likely from a **single-node measurement** (one GPU doing decode at ~21.5 tok/s). The cluster aggregate would be ~86 tok/s if all 4 nodes contribute equally.
- The 2-bit model is **compute-limited by the dequantize path overhead** — the GPU has to dequantize 2-bit weights to FP8/FP16 on-the-fly, which adds compute that doesn't exist as a dedicated Tensor Core operation. This is why FP4 (native Tensor Cores) can achieve ~28.8 tok/s with the QuantTrio INT4-INT8Mix recipe despite larger weight size.

### 4.3 Empirical Bound Characterization

```python
# How to measure: run the same model at increasing batch sizes
# and plot throughput vs. memory bandwidth utilization

# Step 1: Measure bandwidth utilization via nvml/nvprof
# bandwidth_util = (tokens_generated × weight_size_bytes) / wall_time / 273e9

# Step 2: Plot throughput vs. bandwidth utilization
# - If throughput scales linearly with bandwidth → memory-bound
# - If throughput plateaus while bandwidth is <100% → compute-bound
# - If throughput plateaus at ~50-70% bandwidth → mixed

# Step 3: Compare 2-bit vs FP4 at same batch size
# If 2-bit has LOWER bandwidth utilization but LOWER throughput,
# the dequantize overhead is the bottleneck, not memory bandwidth.
```

### 4.4 Memory-Bound vs Compute-Bound Decision Matrix

| Observation | Diagnosis | Implication |
|---|---|---|
| Bandwidth util > 80% at target tok/s | **Memory-bound** | Need faster memory (HBM) or smaller model. 2-bit helps here. |
| Bandwidth util < 50%, compute util > 70% | **Compute-bound** | 2-bit doesn't help (no native TC). FP4 would be better. |
| Bandwidth util 50-80%, both high | **Mixed** | 2-bit gives partial benefit. Measure the trade-off. |
| Throughput drops as context length increases | **KV cache bound** | KV cache precision matters more than weight quantization. |

### 4.5 GPU Profiling Commands

```bash
# nvprof / Nsight Systems profiling
nsys profile --trace=nvtx,cuda,nvml \
  --output=glm-5.2-2bit-nsys \
  python -m vllm.entrypoints.api_server \
    --model GLM-5.2-2bit --quantization awq --max-num-seqs 64

# Key metrics to extract:
# - sm_active: Should be >70% if compute-bound, <30% if memory-bound
# - memory_throughput: Actual GB/s used vs 273 GB/s theoretical
# - tensor_op_utilization: Should be 0% for 2-bit (no dedicated TC)
# - l1_cache_hit_rate: High = good weight reuse in cache
# - dram_throughput: Direct memory bandwidth measurement
```

---

## 5. KV-Cache Optimization Impact on Long-Context Performance

### 5.1 The KV Cache Problem

The KV cache grows linearly with context length and is stored at **higher precision** (typically FP16/BF16) even when weights are quantized. For GLM 5.2 (~40B active, ~744B total MoE, ~128 layers, hidden_size ~8192):

```
KV cache per token = 2 × layers × hidden_size × 2 bytes (FP16 for K + FP16 for V)
                   = 2 × 128 × 8192 × 2 = ~4.2 MB/token (per sequence, FP16)
```

At 1M context: ~4.2 GB of KV cache for a single sequence. At 128K context: ~537 MB. This is where the 2-bit advantage becomes critical — it frees weight memory to accommodate larger KV caches.

**Per-node memory budget (4× Spark cluster, 128 GB each):**

| Component | 2-bit weights | NVFP4 weights |
|---|---|---|
| Weights (744B params) | ~300–377 GB / 4 ≈ 75–94 GB | ~469 GB / 4 ≈ 117 GB |
| KV cache @ 32K (FP16) | ~135 MB | ~135 MB |
| KV cache @ 128K (FP16) | ~537 MB | ~537 MB |
| KV cache @ 1M (FP16) | ~4.2 GB | ~4.2 GB |
| Activations/overhead | ~5–10 GB | ~5–10 GB |
| **Total @ 32K ctx** | ~85–110 GB | ~125–135 GB ⚠️ |
| **Total @ 128K ctx** | ~86–111 GB | ~127–137 GB ❌ |
| **Total @ 1M ctx** | ~90–115 GB | ~132–141 GB ❌❌ |

**Key insight:** At 128K+ context, even NVFP4 weights barely fit on a single Spark. The 2-bit recipe's main advantage isn't throughput — it's **context capacity**. 2-bit weights leave enough headroom for multi-GB KV caches that enable the 1M context claim.

### 5.2 KV Cache Optimization Techniques to Test

| Technique | What It Does | How to Measure |
|---|---|---|
| **FP8 KV cache** | Store KV at FP8 instead of FP16 → 2× smaller | Compare throughput at same context length. FP8 KV should allow ~2× longer context before OOM. |
| **PagedAttention** (vLLM) | Non-contiguous KV allocation → better memory utilization | Compare max context before OOM vs. contiguous allocation. |
| **KV cache quantization** | Quantize KV to INT4/FP4 during decode | Measure accuracy impact (usually small) vs. memory savings. |
| **KV cache eviction** | Sliding window, discarding old tokens | Measure accuracy degradation at different window sizes. |
| **Speculative decoding** | Draft model generates tokens, verifier accepts/rejects | Measure speedup factor vs. baseline decode. Does 2-bit hurt spec-decode efficiency? |

### 5.3 Long-Context Benchmark Protocol

```yaml
long_context:
  # Test at increasing context lengths
  context_lengths: [1024, 4096, 8192, 16384, 32768, 65536]

  # Fixed prompt/output split
  workload:
    prompt_tokens_per_length:  # ratio of prompt to context
      1024: 1024
      4096: 4096
      8192: 8192
      16384: 16384
      32768: 32768
      65536: 65536
    output_tokens: 256

  # Measure:
  metrics:
    - "ttft_ms"            # grows with context (more prefill)
    - "itl_ms"             # may degrade with larger KV cache
    - "agg_tok_s"          # may drop at extreme context (OOM pressure)
    - "kv_cache_gb"        # actual KV memory used
    - "oom_at_context"     # maximum context before OOM

  # Compare KV cache strategies:
  kv_strategies:
    - name: "FP16 KV"
      kv_cache_dtype: "fp16"
    - name: "FP8 KV"
      kv_cache_dtype: "fp8"
    - name: "FP4 KV"
      kv_cache_dtype: "fp4"  # if supported

  # Key metric: "context efficiency"
  # = (max_context_length) / (total_memory_used_gb)
  # Higher is better. FP8 KV should improve this by ~1.5-2×.
```

### 5.4 The KV Cache Accuracy Test

KV cache quantization can silently degrade output quality. Run the accuracy benchmarks (Section 1) at **short context (1K)** and **long context (32K)** with identical prompts. If accuracy drops >3 points at 32K with FP8 KV, the KV quantization is too aggressive.

---

## 6. Batch Processing vs Single-Request Latency Trade-offs

### 6.1 The Fundamental Trade-off

Batching increases aggregate throughput but increases per-request latency due to queueing. This is the core tension for agentic workloads:

| Metric | Batch=1 | Batch=4 | Batch=8 | Batch=16 |
|---|---|---|---|---|
| Aggregate tok/s | 21.5 | ~40 | ~60 | ~80 |
| Per-session tok/s | 21.5 | ~10 | ~7.5 | ~5 |
| p99 ITL (ms) | 50 | 120 | 250 | 500+ |
| p99 TTFT (ms) | 80 | 200 | 400 | 800+ |

The question isn't "which is faster" — it's "which is fast enough for the workload."

### 6.2 Workload-Specific Batch Sizing

```yaml
# Different workloads need different batch sizes:
workload_profiles:
  agentic_loop:
    description: "Agent making tool calls, needs fast turnarounds"
    target_batch: 1-4
    max_p99_itl_ms: 150
    reason: "Agent loops timeout after ~2s; batching >4 breaks the loop"

  batch_inference:
    description: "Processing a queue of independent requests"
    target_batch: 16-32
    max_p99_itl_ms: 1000
    reason: "No latency sensitivity; maximize throughput"

  chat_interactive:
    description: "Human chatting, needs responsive feel"
    target_batch: 2-8
    max_p99_itl_ms: 200
    reason: "Humans notice >200ms lag"

  rag_pipeline:
    description: "Document QA with long prompts"
    target_batch: 1-4
    max_p99_ttft_ms: 500
    reason: "Prefill dominates; batching helps aggregate but not per-request"
```

### 6.3 The "Sweet Spot" Measurement

```yaml
sweet_spot_analysis:
  # Find the batch size where per-session throughput drops below a threshold
  # while aggregate throughput is still reasonable

  methodology:
    1. Run concurrency sweep: [1, 2, 4, 8, 16, 32]
    2. For each N, compute:
       - agg_tok_s (aggregate throughput)
       - per_session_tok_s (what one user experiences)
       - per_session_tok_s / agg_tok_s (efficiency ratio)
    3. Plot both curves
    4. Identify the "knee" — where per-session drops below 50% of N=1

  # The knee is the practical batch limit. Beyond it, adding users
  # degrades individual experience faster than aggregate gains.

  # Report:
  knee_concurrency: 8           # example: per-session drops below 50% here
  max_useful_batch: 16          # where per-session < 20% of N=1
  aggregate_at_knee: 120        # tok/s at the knee
  per_session_at_knee: 15       # tok/s per user at the knee
```

---

## 7. Reproducibility Checklist

### 7.1 Hardware

- [ ] **GPU:** DGX Spark (GB10 Grace-Blackwell) — or specify alternative
- [ ] **Memory:** 128 GB LPDDR5x, exact model if possible
- [ ] **Storage:** NVMe model (type, speed)
- [ ] **Network:** ConnectX-7 (if multi-node)
- [ ] **OS:** DGX OS version / Ubuntu version
- [ ] **Kernel:** `uname -r`
- [ ] **Power state:** plugged in, performance mode (`nvidia-smi -pm 1`)
- [ ] **Thermal state:** ambient temperature, GPU idle temp before run

### 7.2 Software Stack

- [ ] **CUDA/cuDNN version:** `nvcc --version`, `python -c "import torch; print(torch.version)"`
- [ ] **vLLM version:** `pip show vllm` (exact commit hash)
- [ ] **PyTorch version:** `python -c "import torch; print(torch.__version__, torch.commit_id)"`
- [ ] **FlashAttention version:** `pip show flash-attn`
- [ ] **llama.cpp version:** `git rev-parse HEAD` (if comparing)
- [ ] **Python version:** `python --version`
- [ ] **Container image:** full digest (e.g., `ghcr.io/aeon-7/aeon-vllm-ultimate@sha256:...`)

### 7.3 Model and Quantization

- [ ] **Model ID:** HuggingFace repo + revision (e.g., `THUDM/glm-5-2bit@abc1234`)
- [ ] **Quantization method:** AWQ / GPTQ / RTN / custom
- [ ] **Quantization config file:** exact YAML/JSON used
- [ ] **Weight file hash:** `sha256sum *.safetensors` or `*.gguf`
- [ ] **Quantization library version:** `pip show awq` or `pip show auto-gptq`
- [ ] **Dequantize path:** FP8 dequant → FP16 matmul? FP4 dequant → INT4 matmul?

### 7.4 Serving Configuration

```yaml
# Exact serving config — every flag matters
serving_config:
  engine: vllm                       # or llama.cpp, TensorRT-LLM, SGLang
  max_model_len: 32768
  max_num_seqs: 64                   # MUST ≥ max concurrency in sweep
  gpu_memory_utilization: 0.5        # bounds non-KV allocations
  kv_cache_dtype: fp8                # FP16 vs FP8 vs FP4
  kv_cache_memory_bytes: 34359738368 # explicit byte cap (bypasses estimator bugs)
  max_num_batched_tokens: 8192
  enable_chunked_prefill: true
  enable_prefix_caching: true        # defeated by unique prompts in harness
  async_scheduling: true
  cuda_graphs: true                  # must be on for fair comparison
  enforce_eager: false
  speculative_config: null           # or MTP, DFlash, etc.
  tool_call_parser: "glm47"
  reasoning_parser: "glm45"
  tensor_parallel: 1                 # or >1 for multi-GPU
```

### 7.5 Benchmark Harness

- [ ] **Harness version:** `dgx-spark-bench` git commit
- [ ] **Harness config:** exact `harness.yaml` used
- [ ] **Workload definition:** prompt_tokens, output_tokens, prompt template
- [ ] **Concurrency sweep:** exact list of N values
- [ ] **Timing windows:** warmup_s, measure_s
- [ ] **Random seed:** fixed seed for deterministic prompts
- [ ] **Decoding params:** temperature, top_p, top_k, repetition_penalty

### 7.6 Environment Variables

```bash
# All env vars that affect performance:
export TORCH_CUDA_ARCH_LIST="12.1a"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_ATTENTION_BACKEND="FLASHINFER"
export VLLM_USE_TRITON_FLASH_ATTN="1"
export NCCL_IB_DISABLE="1"           # if not using InfiniBand
export NCCL_SOCKET_IFNAME="eth0"     # if using Ethernet
export CUDA_VISIBLE_DEVICES="0"      # pin to specific GPU
export CUDA_DEVICE_MAX_CONNECTIONS="1"
export PT_CUDA_ALLOW_TENSOR_CORE="1"
```

### 7.7 Data Sheet

Every benchmark run produces a `results.json` that should include:

```json
{
  "schema_version": "1.0",
  "run": {
    "id": "glm-5.2-2bit-2026-07-17-spark01",
    "timestamp": "2026-07-17T14:30:00Z",
    "harness_version": "0.1.0",
    "git_commit": "abc1234",
    "hardware": {
      "name": "DGX Spark",
      "soc": "GB10 Grace-Blackwell",
      "mem_gb": 128,
      "mem_type": "LPDDR5x",
      "bw_gbs": 273,
      "fp4_tops": 1000,
      "temperature_c": 52,
      "power_draw_w": 118.5
    },
    "workload": {
      "name": "chat",
      "prompt_tokens": 1024,
      "output_tokens": 256,
      "mode": "closed_loop"
    }
  },
  "series": [{
    "id": "glm-5.2-2bit",
    "model": "GLM-5.2-2bit-AWQ",
    "params_b": 744.0,
    "active_params_b": 40.0,
    "quant": "2bit-AWQ",
    "backend": "vLLM 0.23",
    "status": "measured",
    "weights_gb": 340.0,
    "recipe_url": "https://github.com/...",
    "config": {
      "checkpoint": "...",
      "quantization": "awq",
      "kv_cache_dtype": "fp8",
      "cuda_graphs": true,
      "speculative": null
    },
    "points": [
      {
        "concurrency": 1,
        "agg_tok_s": 21.5,
        "per_session_tok_s": 21.5,
        "ttft_ms": {"p50": 45, "p95": 52, "p99": 61},
        "itl_ms": {"p50": 46.5, "p95": 52.1, "p99": 68.3},
        "samples": 120,
        "gpu": {"temperature_c": 64, "power_draw_w": 125.3}
      }
    ]
  }]
}
```

---

## 8. Pass/Fail Criteria — What Proves "Best/Fastest"

### 8.1 The Claim Matrix

The recipe must satisfy claims in **four dimensions**. Each dimension has pass/fail thresholds:

| Dimension | Claim | Pass Condition | Fail If... |
|---|---|---|---|
| **Speed** | "Fastest 2-bit" | ≥5% higher agg_tok/s than next-best 2-bit method at N=1 | Any other 2-bit beats it at N=1 |
| **Accuracy** | "Best quality 2-bit" | MMLU within ±5 of BF16 (if BF16 available), within ±3 of NVFP4 | MMLU delta >5 from BF16 OR >3 from NVFP4 |
| **Memory** | "Most memory-efficient" | Weights ≤25% of BF16 (~377 GB for 744B model) | Weights >400 GB (doesn't save enough for context) |
| **Context** | "Enables 1M context" | KV cache fits at 128K+ context on single Spark node | OOM before 64K context |
| **Smoothness** | "Best UX" | p99 ITL < 150ms at target concurrency | p99 ITL > 150ms at N=4 |

### 8.1.1 The QuantTrio Comparison

Since QuantTrio INT4-INT8Mix (~28.8 tok/s, ~405–420 GB) is the established baseline for GLM 5.2 on 4× DGX Spark, the 2-bit recipe must answer:

| Question | QuantTrio INT4-INT8Mix | 2-bit Recipe | Verdict Depends On |
|---|---|---|---|
| Throughput (tok/s) | ~28.8 (with MTP k=4) | 21.5 (claimed) | Is the 25% speed penalty worth the memory savings? |
| Weight size | ~405–420 GB | ~300–377 GB | ~100 GB saved — can this enable 1M context? |
| MMLU score | ~90–92 (estimated) | ??? | Must be within ±3 of QuantTrio |
| Spec-decode | MTP k=4 enabled | ??? | Does 2-bit hurt draft model efficiency? |
| Context | ~64K practical | ??? | Must support ≥128K |

### 8.2 The "Best/Fastest" Decision Tree

```
Is it the fastest 2-bit method?
  ├─ YES → Is accuracy within ±3 MMLU points of BF16?
  │         ├─ YES → Is it the most memory-efficient?
  │         │         ├─ YES → Is p99 ITL < 150ms at N=4?
  │         │         │         ├─ YES → 🏆 CLAIM VERIFIED
  │         │         │         └─ NO  → ❌ "Fast but spiky" — not best UX
  │         │         └─ NO  → ❌ "Fast but bloated" — memory claim false
  │         └─ NO  → ❌ "Fast but dumb" — accuracy loss too high
  └─ NO  → ❌ "Not the fastest" — another 2-bit method wins
```

### 8.3 The "Good Enough" Threshold

Even if the recipe doesn't win every category, it should be evaluated against a **minimum viability bar**:

| Metric | Minimum Viable | Competitive | Excellent |
|---|---|---|---|
| MMLU (delta from BF16) | ≤ 5 points | ≤ 3 points | ≤ 1.5 points |
| Throughput (vs FP4) | ≥ 70% | ≥ 85% | ≥ 95% |
| p99 ITL at N=4 | ≤ 300ms | ≤ 150ms | ≤ 80ms |
| Memory (vs BF16) | ≤ 40% | ≤ 30% | ≤ 20% |
| Context length (max) | ≥ 8K | ≥ 16K | ≥ 32K |

### 8.4 The "Red Team" Tests

To truly stress-test the recipe, run these adversarial benchmarks:

| Test | What It Probes | Pass Condition |
|---|---|---|
| **Long context (65K+)** | KV cache stability, attention degradation | No accuracy drop >5 points vs. 4K context |
| **Code generation** | Quantization sensitivity to structured output | HumanEval pass@1 ≥ 85% of BF16 |
| **Tool calling** | Format compliance at 2-bit | ≥ 90% correct JSON/XML format |
| **Repetition stress** | KV cache degradation over long generation | No repetition loops at 1024 output tokens |
| **Thermal stress** | Sustained throughput under heat | <5% throughput drop after 30 min continuous run |
| **Cold start** | Model load time | <60s from `vllm serve` to first request |
| **OOM pressure** | KV cache at capacity | Graceful degradation, not crash |

### 8.5 Final Verdict Framework

```
GLM-5.2-2bit Recipe: VERDICT

Speed:        ████████████░░░░  75% of QuantTrio (pass: ≥70%)
Accuracy:     ███████████████░  MMLU -3.2 from NVFP4 (pass: ≤3pts)
Memory:       ████████████████  340 GB weights (pass: ≤25% BF16)
Smoothness:   ████████████░░░░  p99 ITL 142ms at N=4 (pass: ≤150ms)
Context:      ████████████████  128K+ context (pass: ≥128K)

Overall:  CONDITIONALLY COMPETITIVE ⚠️
  - 25% slower than QuantTrio INT4-INT8Mix (21.5 vs 28.8 tok/s)
  - Saves ~80 GB over QuantTrio (340 GB vs 420 GB)
  - If accuracy is within ±3 MMLU of QuantTrio, the trade-off is valid
  - Verdict: "Best for deployments needing 1M context on limited hardware"
            "NOT recommended for throughput-sensitive workloads"

Alternative verdicts:
  ❌ "Not viable — accuracy loss >5 MMLU points. Use QuantTrio or NVFP4."
  ❌ "Not faster — QuantTrio with MTP spec-decode is 34% faster. 2-bit only
       saves memory, not time."
  ✅ "Best memory efficiency for frontier MoE models. Use when context length
       or weight footprint is the bottleneck, not throughput."
```

---

## Appendix A: Integration with dgx-spark-bench

To integrate this framework into the existing benchmarking infrastructure:

### A.1 Recipe Directory Structure

```
recipes/glm-5.2-2bit-awq/
├── glm-5.2-2bit-awq.yaml          # lmswitch bench profile
├── harness.yaml                    # harness config (concurrency sweep)
├── accuracy_harness.yaml           # accuracy benchmarks (separate harness)
├── long_context_harness.yaml       # KV cache stress test
├── README.md                       # recipe documentation
└── config/
    ├── awq_config.json             # AWQ quantization config
    ├── serving_overrides.yaml      # bench-specific serving flags
    └── decoding_params.yaml        # identical across all quantizations
```

### A.2 Extended Harness for Accuracy

The existing `bench.py` measures throughput. For accuracy, a separate harness is needed:

```python
# accuracy_harness.py — runs MMLU, HumanEval, etc. against served endpoint
# Uses the same OpenAI-compatible API but sends benchmark prompts
# Reports accuracy scores, not throughput

# Key difference from throughput harness:
# - No concurrency sweep (single request, deterministic)
# - Fixed benchmark datasets (MMLU, HumanEval, etc.)
# - Scoring engine (exact match, regex extraction, LLM-as-judge)
# - Statistical reporting (mean, CI, delta from baseline)
```

### A.3 Specific GLM 5.2 Benchmark Considerations

GLM 5.2 is a **744B-parameter MoE model** with unique characteristics that affect benchmarking:

1. **MoE active parameter variance:** At 40B active parameters per token, the model switches experts per token. This means throughput can vary based on expert distribution. Run enough samples to average out expert-switching variance.

2. **1M context capability:** The IndexShare + DSA architecture enables stable 1M context. Benchmark context stability at 64K, 128K, 256K, 512K, and 1M tokens. Measure both throughput degradation and accuracy degradation at each context length.

3. **Speculative decoding sensitivity:** MTP (Multi-Token Prediction) speculative decoding with k=4 is the current state-of-the-art for GLM 5.2. Test whether 2-bit quantization affects the draft model's accuracy (which determines spec-decode acceptance rate). A lower acceptance rate negates the speed advantage.

4. **Tool calling and agentic workloads:** GLM 5.2 is positioned as an agentic model. Benchmark tool calling accuracy at 2-bit quantization — this is where quantization artifacts are most visible (JSON/XML format compliance).

5. **Chinese language capability:** GLM 5.2 is strong in Chinese. Include Chinese-language benchmarks (C-Eval, CMMLU, CLUE) alongside English benchmarks to get a complete accuracy picture.

### A.3 Dashboard Integration

Results from this framework drop into the existing `dgx-spark-bench` dashboard with two additions:

1. **Accuracy panel:** MMLU/HumanEval scores plotted alongside throughput, color-coded by quantization method
2. **Latency percentile view:** p50/p90/p99 ITL and TTFT as separate charts (not just aggregate tok/s)

---

## Appendix B: Quick-Start Reproduction Script

```bash
#!/bin/bash
# reproduce-glm-5.2-2bit.sh
# One-command reproduction of the full benchmark suite
set -euo pipefail

# 1. Setup
cd ~/dev/dgx-spark-bench
uv venv harness/.venv && uv pip install -e harness/

# 2. Serve the model
cp recipes/glm-5.2-2bit-awq/glm-5.2-2bit-awq.yaml ~/lmswitch/ai-models/
lmswitch on glm-5.2-2bit-awq

# 3. Run throughput sweep
cd harness
.venv/bin/python bench.py \
  ../recipes/glm-5.2-2bit-awq/harness.yaml \
  -o ../results/glm-5.2-2bit-awq-throughput.json

# 4. Run accuracy benchmarks
.venv/bin/python accuracy_harness.py \
  --endpoint http://localhost:8214 \
  --model glm-5.2-2bit-awq \
  --benchmarks mmlu,humaneval,mmlu-pro,ifeval \
  -o ../results/glm-5.2-2bit-awq-accuracy.json

# 5. Run long-context stress test
.venv/bin/python bench.py \
  ../recipes/glm-5.2-2bit-awq/long_context_harness.yaml \
  -o ../results/glm-5.2-2bit-awq-long-context.json

# 6. Stop model
lmswitch off glm-5.2-2bit-awq

# 7. Compare against baselines
echo "=== Throughput Comparison ==="
cat results/glm-5.2-2bit-awq-throughput.json | jq '.series[0].points[] | {concurrency, agg_tok_s}'

echo "=== Accuracy Comparison ==="
cat results/glm-5.2-2bit-awq-accuracy.json | jq '.benchmarks[] | {name, score, delta_from_bf16}'

echo "=== Verdict ==="
# Run the decision tree logic
python3 -c "
import json
tp = json.load(open('results/glm-5.2-2bit-awq-throughput.json'))
ac = json.load(open('results/glm-5.2-2bit-awq-accuracy.json'))

tok_s = tp['series'][0]['points'][0]['agg_tok_s']
mmlu_delta = ac['benchmarks'][0]['delta_from_bf16']

print(f'Throughput at N=1: {tok_s} tok/s')
print(f'MMLU delta from BF16: {mmlu_delta} points')

if tok_s >= 15 and abs(mmlu_delta) <= 3:
    print('VERDICT: COMPETITIVE')
elif abs(mmlu_delta) > 5:
    print('VERDICT: ACCURACY TOO LOW')
else:
    print('VERDICT: NOT VIABLE')
"
```

---

*Document version: 1.1 | Last updated: 2026-07-17 | Designed for GLM 5.2 (744B MoE) on 4× DGX Spark cluster, generalizable to any platform*
