# GPU Benchmark Results

## Test Environment

### Hardware

- GPU: NVIDIA GeForce RTX 4070 Ti
- VRAM: 12 GB
- CPU: Intel Core i5-13500K
- RAM: 32 GB

### Software

- Windows 11
- WSL2
- Ubuntu 24.04 LTS
- Docker Desktop
- NVIDIA PyTorch Container
- PyTorch 2.8.0
- CUDA Runtime 12.9

## 1. Matrix Multiplication

A 5000 × 5000 matrix multiplication was executed on the GPU using PyTorch.

The benchmark measures GPU compute performance and demonstrates CUDA-accelerated tensor operations.

Result:

- Matrix size: 5000 × 5000
- GPU: RTX 4070 Ti
- Execution time: approximately 0.0504 seconds

## 2. Precision Benchmark

The same matrix workload was tested using different numerical precisions.

| Precision | Average Time | Performance |
|---|---:|---:|
| FP32 | 0.006660 s | 37.53 TFLOPS |
| FP16 | 0.003251 s | 76.91 TFLOPS |
| BF16 | 0.003263 s | 76.62 TFLOPS |

### Observation

FP16 and BF16 achieved substantially higher throughput than FP32 for this workload.

Lower-precision formats also reduce tensor memory requirements.

## 3. Memory Bandwidth

A GPU memory copy benchmark was used to estimate effective memory bandwidth.

Result:

```text
Effective Memory Bandwidth: 426.66 GB/s
The RTX 4070 Ti has a theoretical memory bandwidth of approximately 504 GB/s.

The measured result represents effective bandwidth for this particular benchmark rather than the theoretical hardware maximum.
```

## 4. Memory Access Pattern

Sequential and random memory access patterns were compared.

| Access Pattern | Time |
|---|---:|
| Sequential | 0.016921 s |
| Random | 0.209777 s |

Random access was approximately 12.40x slower.

Sequential access provides a more efficient pattern for coalesced GPU memory access.

Random access can reduce memory efficiency because threads may access widely separated memory locations.

## 5. Cache / Locality Experiment

A small tensor and a large tensor were processed repeatedly.

| Workload | Time |
|---|---:|
| Small tensor | 0.022243 s |
| Large tensor | 0.132783 s |

The large-tensor workload took approximately 5.97x longer.

This experiment demonstrates the importance of working-set size and memory locality.

It should not be interpreted as a direct measurement of L2 cache hit rate. A rigorous cache analysis would use NVIDIA Nsight Compute.

## 6. Key Findings

1. Lower numerical precision can increase tensor throughput.
2. GPU performance depends on both computation and memory behavior.
3. Sequential/coalesced access is generally more efficient than poorly localized random access.
4. Cache locality and working-set size can influence performance.
5. Theoretical hardware specifications do not necessarily equal application-level performance.

## Next

Future work will include NVIDIA Nsight profiling and local LLM serving with vLLM.
