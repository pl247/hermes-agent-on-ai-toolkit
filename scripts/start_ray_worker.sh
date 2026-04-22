#!/bin/bash
# Script to start Ray worker node
# Usage: ./start_ray_worker.sh <head_node_ip>

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