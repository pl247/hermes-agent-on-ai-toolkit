# Hermes Agent on AI Toolkit

Last updated: 2026-04-22 08:26:41

## Overview
This repository provides instructions for running Hermes Agent with a custom LLM served via vLLM on the AI Toolkit.

## Prerequisites
- AI Toolkit already installed and running on Ubuntu with CUDA, Docker, and vLLM operational.
- Access to two hosts (or a multi-node setup) each with 2 GPUs (total 4 GPUs) for tensor parallelism.
- The Nemotron-3-120B model (or compatible) available for deployment.

## Step 1: Deploy the LLM with vLLM (Tensor Parallelism)

1. Pull the Nemotron-3-120B model weights (ensure you have access and sufficient storage).
2. On each host, start the vLLM server with tensor parallelism size 2 (since 2 GPUs per host) and configure for distributed serving across hosts.

   Example command on **Host A** (adjust IPs, ports, and model path):
   ```bash
   python -m vllm.entrypoints.api_server \
       --model /path/to/nemotron-3-120B \
       --tensor-parallel-size 2 \
       --pipeline-parallel-size 1 \
       --distributed-executor-backend mp \
       --host 0.0.0.0 \
       --port 8000
   ```

   Example command on **Host B** (same settings, ensure they can communicate):
   ```bash
   python -m vllm.entrypoints.api_server \
       --model /path/to/nemotron-3-120B \
       --tensor-parallel-size 2 \
       --pipeline-parallel-size 1 \
       --distributed-executor-backend mp \
       --host 0.0.0.0 \
       --port 8000
   ```

   > Note: For multi-host tensor parallelism, you may need to set environment variables like `VLLM_HOSTS` or use a launcher script; refer to vLLM documentation for multi-node TP setup. Ensure the hosts can reach each other on the specified ports (open firewall if needed).

3. Verify the API is accessible from each host: `curl http://<host_ip>:8000/v1/models` should return the model list.

## Step 2: Install Hermes Agent

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

## Step 3: Configure Hermes Agent to Use the Custom vLLM Endpoint

1. Create or edit the Hermes Agent configuration file (e.g., `config.yaml`) to point to your vLLM server.
   Example configuration:
   ```yaml
   model:
     type: custom
     base_url: "http://<vllm_host_ip>:8000/v1"   # Use one of the host IPs; if load‑balanced, use the LB address.
     api_key: "your-api-key-if-required"          # Optional, set if vLLM requires auth.
     model_name: "nemotron-3-120B"
     # Additional parameters for chat/completions
     max_tokens: 2048
     temperature: 0.7
   ```
2. Ensure Hermes Agent can reach the vLLM host (network connectivity, no firewall blocks).

## Step 4: Test the Integration

1. Start Hermes Agent (or your custom agent script) and send a prompt that requires chain‑of‑thought reasoning.
2. Verify that the response is generated via the vLLM endpoint (check logs on the vLLM servers for incoming requests).
3. If successful, you have Hermes Agent running with a powerful LLM backend.

## Troubleshooting

- **Connection refused**: Confirm the vLLM server is running and accessible from the Hermes host (`telnet <vllm_ip> 8000`).
- **Model not found**: Ensure the model name in the request matches what vLLM serves.
- **Performance issues**: Verify GPU utilization and that tensor parallelism is correctly set (check vLLM logs for TP size).

## Notes

- The instructions assume a working AI Toolkit with Docker and CUDA; deploying vLLM may require the `vllm` Docker image or a native pip install adjusted for your environment.
- For production, consider using a reverse proxy, load balancer, or Kubernetes service to front the vLLM instances.

---

*Happy coding!*