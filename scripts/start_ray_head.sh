#!/bin/bash
# Script to start Ray head node
# Usage: ./start_ray_head.sh

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