#!/bin/bash
# Script to start Ray head node
# Usage: ./start_ray_head.sh

# Check if we are in the correct conda environment
if [ "$CONDA_DEFAULT_ENV" != "vllm-2" ]; then
  echo "Error: This script must be run in the 'vllm-2' conda environment."
  echo "Please activate it with: conda activate vllm-2"
  exit 1
fi

# Exit on any error
set -e

echo "Starting Ray head node..."
ray start \
  --head \
  --port=6379 \
  --dashboard-port=8265 \
  --num-gpus=2 \
  --node-ip-address=$VLLM_HOST_IP \
  --metrics-export-port=8001

echo "Ray head node started. To stop, run: ray stop"