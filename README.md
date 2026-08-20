# High-Throughput LLM Interface

A GPU-backed LLM inference system built around vLLM, FastAPI, and Llama 3.2 1B.

## Architecture

Client
→ FastAPI Gateway
→ Native vLLM Server
→ Llama 3.2 1B
→ NVIDIA RTX 3050 4 GB

## Hardware

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- VRAM: 4 GB
- Environment: WSL2 Ubuntu
- Python: 3.12.13
- vLLM: 0.27.1
- PyTorch: 2.13.0+cu130

## Milestones

### M1 — Environment and GPU Verification

Verified:

- Python 3.12.13
- CUDA availability
- RTX 3050 visibility
- vLLM installation

### M2 — Base Model Inference

Model:

`meta-llama/Llama-3.2-1B-Instruct`

Verified:

- Model loading
- CUDA execution
- Generation
- KV-cache allocation

Working configuration:

- `gpu_memory_utilization=0.78`
- `max_model_len=1024`
- `max_num_seqs=4`
- `max_num_batched_tokens=512`

WSL2 compatibility settings:

- `VLLM_USE_V2_MODEL_RUNNER=0`
- `VLLM_USE_FLASHINFER_SAMPLER=0`

### M3 — FastAPI Gateway

Endpoints:

- `GET /health`
- `POST /generate`

The gateway forwards requests to the native vLLM server.

### M4 — Benchmarking

Application-level benchmark:

| Concurrency | Throughput |
| ---: | ---: |
| 1 | 67.90 tok/s |
| 2 | 127.57 tok/s |
| 4 | 248.92 tok/s |

Benchmark configuration:

- 8 requests per concurrency level
- 64 maximum output tokens
- warmup request
- FastAPI → vLLM serving path

## Running the system

### Start vLLM

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --dtype auto \
  --gpu-memory-utilization 0.78 \
  --max-model-len 1024 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 512 \
  --host 0.0.0.0 \
  --port 8000
