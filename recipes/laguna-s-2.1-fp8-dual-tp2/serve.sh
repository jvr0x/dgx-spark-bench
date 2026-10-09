#!/bin/bash
# Laguna-S-2.1-FP8 dual-Spark TP=2 serve sequence, bare-metal vLLM+Ray.
# Copied from https://howtospark.com/recipes/laguna-s-2.1-fp8-dual-spark-tp2
# (2026-07-22), IPs adapted to this cluster (spark=10.100.224.2, Gigabyte=
# 10.100.224.1). UNTESTED here — see README.md. Run each section by hand,
# don't blindly `bash serve.sh` first time.
set -euo pipefail

SPARK=10.100.224.2
GIGABYTE=10.100.224.1
IFNAME=enp1s0f1np1
HCA=rocep1s0f1
VENV="$HOME/venvs/vllm-025"

# --- 0. prereqs (one-time, both nodes) ---
# ssh $SPARK    'loginctl enable-linger $USER'
# ssh $GIGABYTE 'loginctl enable-linger $USER'

# --- 1. resolve RoCEv2 GID indices (both nodes) ---
gid() {
  ssh "$1" "for g in /sys/class/infiniband/$HCA/ports/1/gids/*; do i=\${g##*/}; \
    [ \"\$(cat /sys/class/infiniband/$HCA/ports/1/gid_attrs/types/\$i 2>/dev/null)\" = 'RoCE v2' ] || continue; \
    case \$(cat \$g) in *ffff:$(printf '%02x%02x:%02x%02x' $(echo $2|tr . ' ')) ) echo \$i; break;; esac; done"
}
GID_SPARK=$(gid "$SPARK" "$SPARK")
GID_GIGABYTE=$(gid "$GIGABYTE" "$GIGABYTE")

ENV="--setenv=PATH=$VENV/bin:$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --setenv=NCCL_IB_DISABLE=0 --setenv=NCCL_IB_HCA=$HCA \
  --setenv=NCCL_SOCKET_IFNAME=$IFNAME --setenv=GLOO_SOCKET_IFNAME=$IFNAME \
  --setenv=RAY_memory_monitor_refresh_ms=0"

# --- 2. Ray head on spark ---
ssh "$SPARK" "systemd-run --user --collect --unit=laguna-ray-head $ENV \
  --setenv=VLLM_HOST_IP=$SPARK --setenv=NCCL_IB_GID_INDEX=$GID_SPARK \
  -p MemoryMax=98G -p MemorySwapMax=0 -p LimitMEMLOCK=infinity \
  $VENV/bin/ray start --block --head --node-ip-address=$SPARK --port=6379 --object-store-memory=2000000000"
sleep 8

# --- 3. Ray worker on Gigabyte ---
ssh "$GIGABYTE" "systemd-run --user --collect --unit=laguna-ray-worker $ENV \
  --setenv=VLLM_HOST_IP=$GIGABYTE --setenv=NCCL_IB_GID_INDEX=$GID_GIGABYTE \
  -p MemoryMax=110G -p MemorySwapMax=0 -p LimitMEMLOCK=infinity \
  $VENV/bin/ray start --block --address=$SPARK:6379 --object-store-memory=2000000000 --node-ip-address=$GIGABYTE"
sleep 10

ssh "$SPARK" "$VENV/bin/ray status"

# --- 4. vllm serve (on spark) ---
ssh "$SPARK" "export VLLM_HOST_IP=$SPARK NCCL_SOCKET_IFNAME=$IFNAME GLOO_SOCKET_IFNAME=$IFNAME \
  NCCL_IB_DISABLE=0 NCCL_IB_HCA=$HCA NCCL_IB_GID_INDEX=$GID_SPARK RAY_memory_monitor_refresh_ms=0; \
  systemd-run --user --scope --collect -p MemoryMax=12G -p MemorySwapMax=0 \
  $VENV/bin/vllm serve ~/models/hf/Laguna-S-2.1-FP8 \
  --served-model-name laguna-s-fp8 \
  --distributed-executor-backend ray --tensor-parallel-size 2 \
  --enforce-eager \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-memory-bytes 5368709120 \
  --max-num-seqs 2 --max-num-batched-tokens 8192 \
  --speculative-config '{\"method\":\"dflash\",\"model\":\"'\$HOME'/models/hf/Laguna-S-2.1-DFlash-W4A16\",\"num_speculative_tokens\":6}' \
  --host 0.0.0.0 --port 8000"

# --- 5. smoke test ---
# curl -s $SPARK:8000/v1/chat/completions -H 'Content-Type: application/json' \
#   -d '{"model":"laguna-s-fp8","messages":[{"role":"user","content":"capital of France? one word"}],"max_tokens":8,"temperature":0}'
