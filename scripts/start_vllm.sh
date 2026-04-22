#!/bin/bash
# Script to start vLLM server with Ray backend
# Usage: ./start_vllm.sh

# Exit on any error
set -e

# Check if required environment variables are set
if [ -z "$VLLM_HOST_IP" ]; then
  echo "Error: VLLM_HOST_IP environment variable is not set"
  exit 1
fi

if [ -z "$NCCL_SOCKET_IFNAME" ]; then
  echo "Error: NCCL_SOCKET_IFNAME environment variable is not set"
  exit 1
fi

echo "Starting vLLm server with Ray backend..."
vllm serve /ai/models/NVIDIA-Nemotron-3-120B/ \
  --api-key LLM \
  --tensor-parallel-size 4 \
  --distributed-executor-backend ray \
  --port 8000 \
  --trust-remote-code \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --enable-chunked-prefill \
  --reasoning-parser nemotron_v3 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.85

echo "vLLM server stopped."