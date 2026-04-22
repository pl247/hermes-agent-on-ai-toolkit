# Hermes Agent on AI Toolkit

See the AI Toolkit: https://github.com/pl247/ai-toolkit-2.0

## Overview
This repository provides instructions for running Hermes Agent with a custom LLM served via vLLM on the AI Toolkit.

## Prerequisites
- AI Toolkit already installed and running on Ubuntu with CUDA, Docker, and vLLM operational.
- Two hosts (each with 2 GPUs, total 4 GPUs) connected via high-speed network (e.g., InfiniBand or RoCE).
- A Ray cluster initialized across the two hosts for distributed tensor parallelism.
- The Nemotron-3-120B model (or compatible) available for deployment.

## Step 1: Download the Nemotron Model
Use Hugging Face CLI to download the model weights to your local storage:
```bash
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 --local-dir /ai/models/NVIDIA/Nemotron-3-120B
```
Ensure you have sufficient disk space and that the directory is accessible from both hosts.

## Step 2: Deploy the LLM with vLLM using Ray for Tensor Parallelism
vLLM runs on a single host (designated as the VLLM host) but uses Ray to distribute tensor parallelism across both hosts (TP=4 over 4 GPUs total: 2 GPUs per host).

### Environment Variables
Set these on the host where you will run the vLLm server (adjust values as needed for your network):
```bash
export VLLM_HOST_IP=1.1.1.11          # IP address of the vLLM host (for Hermes Agent to connect)
export NCCL_SOCKET_IFNAME=ens7f0np0   # Backend interface name used for NCCL/TP traffic
export NCCL_DEBUG=TRACE               # Optional: for troubleshooting first runs
export PYTORCH_ALLOC_CONF=expandable_segments:True  # Helps with memory fragmentation
```

### vLLM Serve Command
Run the following command on the vLLM host (ensure Ray cluster is up and the model path is correct):
```bash
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
```

**Key points about this setup:**
- **Single vLLM host**: The vLLM server process runs on only one host (the host with IP `VLLM_HOST_IP`).
- **Ray cluster for distribution**: Ray manages distributing the tensor parallelism across both hosts. You must have a Ray cluster running with one host as head node and the other as worker.
- **Tensor Parallelism 4**: The model is split across 4 GPUs total (2 GPUs on each host).
- **Backend network**: The `NCCL_SOCKET_IFNAME` (ens7f0np0) should be configured for high-speed communication between hosts for NCCL-based tensor parallelism traffic.
- **First-time troubleshooting**: Keep `NCCL_DEBUG=TRACE` to see detailed logs; remove or set to WARN/ERROR once stable.

**Ray cluster setup (prerequisite):**
On the head node (typically one of your two hosts):
```bash
ray start --head --port=6379
```
On the worker node (the other host):
```bash
ray start --address <head_node_ip>:6379
```
Verify with `ray status` or `ray list nodes`.

3. Verify the API is accessible from the Hermes Agent host: `curl http://<VLLM_HOST_IP>:8000/v1/models` should return the model list.

## Step 3: Install Hermes Agent
1. Clone the Hermes Agent repository (if not already present):
   ```bash
   git clone https://github.com/hermesagent/hermes-agent.git
   cd hermes-agent
   ```
2. Follow the Hermes Agent installation instructions (typically via pip or conda). Ensure you install any required dependencies.
   ```bash
   pip install -e .
   ```
   or follow the official guide.

## Step 4: Configure Hermes Agent to Use the Custom vLLM Endpoint
1. Create or edit the Hermes Agent configuration file (e.g., `config.yaml`) to point to your vLLM server.
   Example configuration:
   ```yaml
   model:
     type: custom
     base_url: "http://<VLLM_HOST_IP>:8000/v1"   # Use the VLLM_HOST_IP set above
     api_key: "LLM"                               # Must match the --api-key used in vllm serve
     model_name: "nemotron-3-120B"
     # Additional parameters for chat/completions
     max_tokens: 2048
     temperature: 0.7
   ```
2. Ensure Hermes Agent can reach the vLLM host (network connectivity, no firewall blocks).

## Step 5: Test the Integration
1. Start Hermes Agent (or your custom agent script) and send a prompt that requires chain‑of‑thought reasoning.
2. Verify that the response is generated via the vLLM endpoint (check logs on the vLLM server for incoming requests).
3. If successful, you have Hermes Agent running with a powerful LLM backend.

## Troubleshooting
- **Connection refused**: Confirm the vLLM server is running and accessible from the Hermes host (`telnet <VLLM_HOST_IP> 8000`).
- **Model not found**: Ensure the model name in the request matches what vLLM serves.
- **Performance issues**: Verify GPU utilization and that tensor parallelism is correctly set (TP=4). Check Ray and vLLM logs for errors. Ensure the backend network (NCCL_SOCKET_IFNAME) is healthy and not saturated.
- **Ray issues**: Verify Ray cluster status with `ray status` or `ray list nodes`. Ensure both hosts can communicate on the Ray port (default 6379).
- **NCCL issues**: If seeing NCCL errors, verify `NCCL_SOCKET_IFNAME` is correct and that the interface is operational on both hosts.

## Notes
- The instructions assume a working AI Toolkit with Docker and CUDA; deploying vLLM may require the `vllm` Docker image or a native pip install adjusted for your environment.
- For production, consider using a reverse proxy, load balancer, or Kubernetes service to front the vLLM instance.
- The `--api-key LLM` value must match exactly between the vLLM serve command and Hermes Agent configuration.
