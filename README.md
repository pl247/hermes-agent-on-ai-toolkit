# Hermes Agent on AI Toolkit

## Overview
This repository provides instructions for running Hermes Agent with a custom LLM served via vLLM on the AI Toolkit.

## Prerequisites
- AI Toolkit already installed and running on Ubuntu with CUDA 12.2, Docker, and vLLM operational. See: https://github.com/pl247/ai-toolkit-2.0
- Two hosts (each with 2 GPUs, total 4 GPUs) connected via high-speed network (e.g., RDMA over Converged Ethernet (RoCE)).
- A Ray cluster initialized across the two hosts for distributed tensor parallelism.
- The Nemotron-3-120B model (or compatible) available for deployment.

## Step 1: Setup Environment and Dependencies
On **both hosts**, create a conda environment and install required packages:

```bash
# Create conda environment with Python 3.12
conda create -n vllm-2 python=3.12 -y

# Activate the environment
conda activate vllm-2

# Install vLLM and Ray on both hosts
pip install vllm
pip install "ray[default]"
```

## Step 2: Download the Nemotron Model
The AI Toolkit has created a directory on each host called `/ai/models` for storing models. Use Hugging Face CLI to download the model weights to your local storage:
```bash
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 --local-dir /ai/models/NVIDIA/Nemotron-3-120B
```
Ensure you have sufficient disk space. If you don't have shared storage between the hosts, you need to make sure the model is copied to the local disk of each host (you can run this command on both hosts or copy the directory between them).

## Step 3: Deploy the LLM with vLLM using Ray for Tensor Parallelism
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
You can run the vLLM serve command directly, or use the provided script:
```bash
# Direct command:
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

# Or using the script (make sure to set environment variables first):
chmod +x scripts/start_vllm.sh
./scripts/start_vllm.sh
```

### Network Communication Diagram
The following diagram shows how the two hosts communicate for Ray clustering and how Hermes Agent talks to the LLM locally:

```
                      FRONTEND NETWORK (192.168.1.x)
               Host 1 (Head)         Host 2
                 .------.              .------.
                 | vLLM |              |      |
                 |Server|              |      |
                 '------'              '------'
                   1.1.1.11              1.1.1.12
                     ^                      ^
                     |     Hermes Agent     |
                     |    (on Host 1)       |
                     +--------> http://192.168.1.11:8000/v1  <----------+
                               (private, internal network only)

                      BACKEND NETWORK (1.1.1.x) 
               Host 1 (Head)         Host 2
                 .------.              .------.
                 | Ray  |              | Ray  |
                 |Head  |<------------>|Worker|  <-- Ray communication (control plane + tensor parallelism via NCCL over RoCE)
                 '------'              '------'
                   1.1.1.11              1.1.1.12
```

**Key points about this setup:**
- **Single vLLM host**: The vLLM server process runs on only one host (Host 1 with IP `VLLM_HOST_IP`).
- **Ray cluster for distribution**: Ray manages distributing the tensor parallelism across both hosts. Host 1 is the head node, Host 2 is the worker.
- **Tensor Parallelism 4**: The model is split across 4 GPUs total (2 GPUs on each host).
- **Frontend Network (192.168.1.x)**: Used for the vLLM API server that Hermes Agent connects to (http://192.168.1.11:8000/v1).
- **Backend Network (1.1.1.x)**: Used for Ray cluster communication, including:
  - Control plane messages (task scheduling, metadata)
  - High-performance NCCL-based tensor parallelism traffic between hosts for serving the Nemotron AI model over vLLM/ray (GPU-GPU communication for model parallelism using RoCE)
- **Local Connection**: Hermes Agent talks directly to the vLLM server on Host 1 via the frontend network (192.168.1.11) - this is still a private network connection.
- **First-time troubleshooting**: Keep `NCCL_DEBUG=TRACE` to see detailed logs; remove or set to WARN/ERROR once stable.

### Benefit of running Hermes to a local LLM
- **Reduced internet exposure**: Most communication happens on private networks (both frontend and backend networks are internal)
- **Data privacy**: Data between the Hermes agent and the LLM is not exposed to a public AI model - your prompts and responses stay within your infrastructure
- **Cost savings**: You don't have to pay for tokens that the agent uses. An agent like OpenClaw can consume thousands of tokens per day in normal operation, which would incur significant costs with public APIs
- **Performance**: Low-latency, high-throughput communication within your private network
- **Control**: Full control over the model, versions, and infrastructure

**Ray cluster setup (prerequisite):**
You can run the Ray start commands directly, or use the provided scripts:

**Manually from command line:**
```bash
# On the head node (Host 1):
export VLLM_HOST_IP=1.1.1.11
conda activate vllm-2
ray start \
  --head \
  --port=6379 \
  --dashboard-port=8265 \
  --num-gpus=2 \
  --node-ip-address 1.1.1.11 \
  --metrics-export-port=8001

# On the worker node (Host 2):
export VLLM_HOST_IP=1.1.1.12
conda activate vllm-2
ray start \
  --address=1.1.1.11:6379 \
  --node-ip-address 1.1.1.12
```

**Using the provided scripts:**
```bash
# On the head node (Host 1):
export VLLM_HOST_IP=1.1.1.11
conda activate vllm-2
chmod +x scripts/start_ray_head.sh
./scripts/start_ray_head.sh

# On the worker node (Host 2):
export VLLM_HOST_IP=1.1.1.12
conda activate vllm-2
chmod +x scripts/start_ray_worker.sh
./scripts/start_ray_worker.sh 1.1.1.11  # Pass the head node IP as argument
```

Verify with `ray status` or `ray list nodes`.

3. Verify the API is accessible from the Hermes Agent host: `curl http://<VLLM_HOST_IP>:8000/v1/models` should return the model list.

## Step 4: Install Hermes Agent
To install Hermes Agent on the AI Toolkit (Ubuntu 22.04), follow these steps:

```bash
# Download and run the installer
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash

# After installation, source your shell profile to update the PATH
source ~/.bashrc  # or source ~/.zshrc if using zsh

# Then when running Hermes:
# 1. Choose a provider
# 2. Select "Hermes model"
# 3. Choose "Custom model"
# 4. Provide the URL for VLLM (http://<VLLM_HOST_IP>:8000/v1) and key (LLM)
```

For more info: https://hermes-agent.nousresearch.com/docs/getting-started/quickstart

## Step 5: Configure Hermes Agent to Use the Custom vLLM Endpoint
Once Hermes Agent is installed and running, configure it to use your vLLM server:
- Base URL: http://<VLLM_HOST_IP>:8000/v1
- API Key: LLM
- Model Name: nemotron-3-120B
- Additional parameters as needed (max_tokens, temperature, etc.)

## Step 6: Test the Integration
1. Start Hermes Agent (or your custom agent script) and send a prompt that requires chain‑of‑thought reasoning.
2. Verify that the response is generated via the vLLM endpoint (check logs on the vLLM server for incoming requests).
3. If successful, you have Hermes Agent running with a powerful LLM backend.

## Troubleshooting
- **Connection refused**: Confirm the vLLM server is running and accessible from the Hermes host (`telnet <VLLM_HOST_IP> 8000`).
- **Model not found**: Ensure the model name in the request matches what vLLM serves.
- **Performance issues**: Verify GPU utilization and that tensor parallelism is correctly set (TP=4). Check Ray and vLLM logs for errors. Ensure the backend network (NCCL_SOCKET_IFNAME) is healthy and not saturated.
- **Ray issues**: Verify Ray cluster status with `ray status` or `ray list nodes`. Ensure both hosts can communicate on the Ray port (default 6379).
- **NCCL issues**: If seeing NCCL errors, verify `NCCL_SOCKET_IFNAME` is correct and that the interface is operational on both hosts.
- **Frontend/Backend confusion**: Ensure frontend network is used for the vLLM API (Hermes Agent connection), backend network (ens7f0np0) is used for Ray cluster communication and NCCL/GPU communication.

## Notes
- The instructions assume a working AI Toolkit with Docker and CUDA; deploying vLLM may require the `vllm` Docker image or a native pip install adjusted for your environment.
- For production, consider using a reverse proxy, load balancer, or Kubernetes service to front the vLLM instance.
- The `--api-key LLM` value must match exactly between the vLLM serve command and Hermes Agent configuration.
- All traffic between hosts and to Hermes Agent remains on private networks - reduced internet exposure for LLM interactions.
- Helper scripts are available in the `scripts/` directory to simplify Ray cluster and vLLM server startup.