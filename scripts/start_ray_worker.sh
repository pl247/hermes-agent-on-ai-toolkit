#!/bin/bash
# Script to start Ray worker node
# Usage: ./start_ray_worker.sh <head_node_ip>

# Check if we are in the correct conda environment
if [ "$CONDA_DEFAULT_ENV" != "vllm-2" ]; then
  echo "Error: This script must be run in the 'vllm-2' conda environment."
  echo "Please activate it with: conda activate vllm-2"
  exit 1
fi

# Exit on any error
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <head_node_ip>"
  exit 1
fi

HEAD_NODE_IP=$1

echo "Starting Ray worker node connecting to head at $HEAD_NODE_IP..."
ray start \
  --address=$HEAD_NODE_IP:6379 \
  --node-ip-address=$VLLM_HOST_IP

echo "Ray worker node started. To stop, run: ray stop"