#!/bin/bash
# Bootstrap entrypoint for stock vllm/vllm-openai:v0.25.1 serving
# poolside/Laguna-S-2.1-NVFP4. That image doesn't ship FlashInfer's NVFP4
# kernels, so install them before handing off to `vllm serve`. Ported from
# github.com/MiaAI-Lab/Laguna-S-2.1-DGX-Spark-RTX-6000-PRO/start.sh, which
# validated this exact model+draft combo (unlike the AEON image, which has a
# real bug in its own DFlash integration for Laguna — see
# laguna-s-2.1-nvfp4.yaml).
set -e
if python3 -c "import flashinfer" 2>/dev/null; then
  echo "[bootstrap] FlashInfer already installed in image"
else
  echo "[bootstrap] Installing FlashInfer for FP4 support ..."
  if pip install \
    flashinfer-python==0.6.15.dev20260712 \
    --extra-index-url https://flashinfer.ai/whl/nightly/ \
  ; then
    echo "[bootstrap] Installed flashinfer nightly"
  elif pip install "flashinfer-python>=0.6.15"; then
    echo "[bootstrap] Installed flashinfer stable"
  else
    echo "[bootstrap] WARNING: flashinfer install skipped -- vLLM may fall back gracefully"
  fi
fi
echo "[bootstrap] Starting vLLM ..."
exec vllm serve "$@"
