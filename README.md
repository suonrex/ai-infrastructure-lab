# GPU Performance & AI Infrastructure Lab

Hands-on GPU performance and AI infrastructure experiments using
NVIDIA GeForce RTX 4070 Ti, CUDA, PyTorch, and Docker.

## Objectives

- Understand NVIDIA GPU architecture
- Explore CUDA execution and memory hierarchy
- Benchmark GPU compute performance
- Analyze memory bandwidth and access patterns
- Understand FP32, FP16, and BF16
- Build practical foundations for LLM inference infrastructure

## Hardware

- GPU: NVIDIA GeForce RTX 4070 Ti
- VRAM: 12 GB
- CPU: Intel Core i5-13500K
- RAM: 32 GB

## Software

- Windows 11
- WSL2
- Ubuntu 24.04 LTS
- Docker Desktop
- NVIDIA Container Toolkit
- PyTorch
- CUDA

## Project Structure

```text
01-gpu-computing/
    GPU compute and precision benchmarks

02-gpu-memory/
    Memory bandwidth, access pattern and cache experiments

03-cuda-execution/
    CUDA execution model and warp experiments

docs/
    Benchmark results and technical notes
```

## Benchmark Results

| Benchmark | Result |
|---|---:|
| FP32 | 37.53 TFLOPS |
| FP16 | 76.91 TFLOPS |
| BF16 | 76.62 TFLOPS |
| Memory Bandwidth | 426.66 GB/s |
| Random / Sequential Access | 12.40x |

## Key Concepts

- CUDA
- GPU Memory Hierarchy
- Memory Bandwidth
- Memory Latency
- Cache Locality
- Coalesced Memory Access
- Thread
- Warp
- Thread Block
- Streaming Multiprocessor (SM)
- SIMT
- Warp Divergence
- Tensor Core

## Next Steps

- CUDA profiling with NVIDIA Nsight
- Local LLM serving with vLLM
- GPU inference benchmarking
- Kubernetes GPU workloads
- NVIDIA GPU Operator
- Prometheus / Grafana / DCGM
- GKE GPU infrastructure
