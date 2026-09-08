#!/usr/bin/env bash
# trio-watch.sh — keep an eye on the three single-Spark trio downloads.
# Runs in tmux (`trio-watch`), polls every 10 min, logs to ~/logs/trio-watch.log.
# Exits 0 with a banner when ALL THREE have landed (bench-day can start).
set -u
QWEN_SNAP=~/models/qwen38-flashnext-hf/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/snapshots
DS_MANIFEST=~/dev/miaai-lab/DeepSeek-v4-Flash-One-DGX-Spark/data/tp1/rank-sliced-tp1-manifest.json
GLM_DIR=~/models/GLM-5.3-Flash-EXL3-K2
qwen_done=0; ds_done=0; glm_done=0
ts() { date '+%F %T'; }
while true; do
  if [[ $qwen_done -eq 0 ]]; then
    cfg=$(ls "$QWEN_SNAP"/*/config.json 2>/dev/null | head -1 || true)
    if [[ -n "$cfg" ]]; then
      bytes=$(du -sb ~/models/qwen38-flashnext-hf 2>/dev/null | cut -f1)
      if [[ "$bytes" -ge 95000000000 ]]; then
        qwen_done=1; echo "$(ts) QWEN LANDED ($(numfmt --to=iec "$bytes"))"
      fi
    fi
  fi
  if [[ $ds_done -eq 0 && -f "$DS_MANIFEST" ]]; then
    ds_done=1; echo "$(ts) DEEPSEEK LANDED ($DS_MANIFEST present)"
  fi
  if [[ $glm_done -eq 0 && -d "$GLM_DIR" ]]; then
    n=$(ls "$GLM_DIR"/model-*.safetensors 2>/dev/null | wc -l)
    if [[ "$n" -ge 120 ]]; then
      glm_done=1; echo "$(ts) GLM LANDED ($n/120 shards)"
    fi
  fi
  if [[ $qwen_done -eq 1 && $ds_done -eq 1 && $glm_done -eq 1 ]]; then
    echo "$(ts) TRIO COMPLETE — Qwen + DeepSeek + GLM weights all on disk. Bench-day can start (see TASK.md runbook)."
    exit 0
  fi
  echo "$(ts) status qwen=$qwen_done ds=$ds_done glm=$glm_done"
  sleep 600
done
