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

## 7. LLM Serving Benchmark

### 7.1 vLLM Single-Request Benchmark

- Model: Qwen3-8B-AWQ
- GPU: NVIDIA GeForce RTX 4070 Ti 12 GB
- vLLM: 0.30.0
- Quantization: AWQ
- Compute dtype: FP16
- Max model length: 4096
- GPU memory utilization: 0.85

| Metric | Result |
|---|---:|
| Requests | 8 |
| Prompt tokens | 180 total |
| Generation tokens | 1084 total |
| Avg. prompt tokens/request | 22.5 |
| Avg. generation tokens/request | 135.5 |
| Avg. TTFT | 1.543 s |
| Avg. prefill time | 1.474 s |
| Avg. decode time | 1.843 s |
| Time/output token | 13.74 ms |
| Decode throughput | ~72.8 tok/s |
| Avg. queue time | ~0.31 ms |

### Calculation

Average time per output token:

```text
0.1099438 / 8
= 0.013743 seconds/token
```

Decode throughput:

```text
1 / 0.013743
≈ 72.8 tokens/s
```

### Interpretation
The vLLM metrics show that single-request inference achieved approximately
72.8 output tokens/s at the decode stage.
This should be distinguished from end-to-end request latency, which includes
prefill, decode, and request/streaming overhead.

Detailed experiment:

`04-llm-serving/benchmarks/vllm-single-request.md`

### 7.2 Concurrent Inference

Four requests were submitted concurrently using `xargs -P4`.

Key results:

| Metric | 4 Concurrent Requests |
|---|---:|
| Generated tokens | 800 |
| Aggregate E2E throughput | ~72.3 tok/s |
| Per-request decode throughput | ~76.2 tok/s |
| Average TTFT | ~8.38 s |
| Average decode time | ~2.61 s |

The experiment demonstrated that request concurrency can substantially change the throughput characteristics of the serving system. The controlled sequential-vs-concurrent experiment in Section 7.5 further quantifies this effect.

Detailed experiment:

`04-llm-serving/benchmarks/vllm-concurrency.md`

## 7.3 Eight Concurrent Requests

Eight requests were submitted concurrently using `xargs -P8`.

Key results:

| Metric | 8 Concurrent Requests |
|---|---:|
| Generated tokens | 1,600 |
| Wall-clock time | ~2.99 s |
| Aggregate E2E throughput | ~535.7 tok/s |
| Per-request decode throughput | ~74.7 tok/s |
| Average TTFT | ~0.30 s |
| Average decode time | ~2.66 s |

The aggregate throughput increased substantially compared with the 4-request
experiment, while per-request decode throughput remained in a similar range.

This indicates that concurrency primarily improves **system-level aggregate
throughput**, rather than making an individual request generate tokens
significantly faster.

---

## 7.4 Sixteen Concurrent Requests

Sixteen requests were submitted concurrently using `xargs -P16`.

Key results:

| Metric | 16 Concurrent Requests |
|---|---:|
| Generated tokens | 3,200 |
| Wall-clock time | ~2.98 s |
| Aggregate E2E throughput | ~1,075 tok/s |
| Per-request decode throughput | ~74 tok/s |
| Average TTFT | ~0.42 s |
| Average decode time | ~2.68 s |

GPU monitoring during the workload showed:

| GPU Metric | Observed |
|---|---:|
| SM utilization | ~90–93% |
| Memory-controller utilization | ~96–99% |
| Memory clock | ~10,251 MHz |
| GPU/core clock | ~2,805–2,820 MHz |
| GPU temperature | ~46–50 °C |
| GPU power | ~160–180 W |

The GPU remained heavily utilized throughout the active inference period.

The important distinction is between **aggregate throughput** and
**per-request decode throughput**:

- **Aggregate throughput** measures the total output tokens generated by
  all concurrent requests per second.
- **Per-request decode throughput** describes the token generation rate
  experienced by an individual request.

Increasing concurrency therefore increases aggregate throughput without
producing a proportional increase in the decode speed of each individual
request.

---

## 7.5 Sequential vs. Concurrent Inference

To isolate the effect of concurrency, the same workload was executed in two
different ways:

1. 16 requests executed sequentially.
2. 16 requests submitted concurrently.

Both experiments used:

- 16 requests
- `max_tokens = 200`
- 3,200 total output tokens
- The same model and inference configuration

### Results

| Metric | 16 Sequential | 16 Concurrent |
|---|---:|---:|
| Requests | 16 | 16 |
| Total generated tokens | 3,200 | 3,200 |
| Wall-clock time | 41.536 s | 2.976 s |
| Aggregate throughput | ~77.1 tok/s | ~1,075 tok/s |
| Relative throughput | 1× | ~14× |

The concurrent workload completed in approximately 3 seconds, while the
sequential workload required approximately 42 seconds.

The aggregate throughput improvement was:

\[
\frac{1075}{77.1} \approx 13.9\times
\]

Therefore, this experiment measured approximately a **14× improvement in
aggregate throughput** when multiple requests were allowed to execute
concurrently.

### Interpretation

The result demonstrates the importance of concurrency for LLM serving.

With sequential execution, the system processes one request at a time. Even
though GPU utilization can remain high during token generation, the inference
engine has only one active sequence to process.

With concurrent execution, multiple requests are active at the same time.
The inference scheduler can organize computation across these active
sequences, allowing the GPU to process substantially more output tokens
during the same wall-clock interval.

This should be understood as a **concurrency and scheduling/batching effect**,
rather than as an improvement in the raw generation speed of a single
request.

An important observation is that GPU utilization alone does not describe
serving performance. The sequential workload showed high GPU utilization as
well, but its aggregate throughput was only about 77 tok/s. The concurrent
workload achieved approximately 1,075 tok/s aggregate throughput with a
similar high GPU utilization level.

---

## 7.6 Concurrency Scaling Summary

The experiments produced the following scaling results:

| Concurrency | Total Tokens | Wall Time | Aggregate Throughput |
|---:|---:|---:|---:|
| 1 | ~200 | — | ~single-request decode rate |
| 4 | 800 | ~11.06 s | ~72.3 tok/s |
| 8 | 1,600 | ~2.99 s | ~535.7 tok/s |
| 16 | 3,200 | ~2.98 s | ~1,075 tok/s |

The results show that increasing concurrency can significantly increase
aggregate serving throughput.

However, aggregate throughput should not be interpreted as the latency or
generation speed of an individual request. Higher concurrency can change
queueing and TTFT characteristics while improving total system throughput.

This distinction is important when evaluating an LLM serving system:

- **Latency:** How quickly does an individual request receive its response?
- **TTFT (Time to First Token):** How long until the first generated token?
- **Decode throughput:** How quickly are tokens generated for a request?
- **Aggregate throughput:** How many output tokens can the entire serving
  system generate per second?

A production system must balance these metrics according to its workload.

---

# 8. Key Findings

The vLLM experiments demonstrated several important characteristics of GPU
LLM serving.

### 8.1 GPU utilization is not equivalent to serving throughput

High GPU utilization does not automatically mean that the serving system is
achieving maximum useful throughput.

The sequential experiment showed approximately 90%+ SM utilization while
producing only about 77 aggregate output tokens/s.

---

### 8.2 Concurrency can dramatically increase aggregate throughput

The controlled 16-request experiment produced:

- Sequential: ~77 tok/s
- Concurrent: ~1,075 tok/s

This represents approximately a **14× increase in aggregate throughput** for
the same total amount of generated output.

---

### 8.3 Per-request performance and system throughput are different metrics

The 16-concurrent experiment maintained approximately 74 tok/s of
per-request decode throughput, while aggregate throughput reached
approximately 1,075 tok/s.

Therefore:

```text
Per-request throughput
        ≠
Aggregate serving throughput
```

### 8.4 GPU monitoring provides additional context

During high-concurrency inference:

| GPU Metric | Observed |
|---|---:|
| SM utilization | ~90–93% |
| Memory-controller utilization | ~96–99% |
| Memory clock | ~10,251 MHz |
| GPU clock | ~2.8 GHz |

These measurements indicate that the GPU was actively processing the
workload and that both compute resources and memory traffic were heavily
engaged.

---

### 8.5 Practical lesson

For LLM serving, performance analysis should not rely on a single metric.

A useful benchmark should measure at least:

1. Request concurrency
2. TTFT (Time to First Token)
3. Decode throughput
4. Aggregate throughput
5. GPU utilization
6. GPU memory utilization
7. Power and temperature

This provides a more complete picture of the serving system than simply
checking GPU utilization or VRAM usage.

---

# 9. Conclusion

This lab established a complete local GPU inference benchmark using:

- NVIDIA GeForce RTX 4070 Ti
- Qwen3-8B-AWQ
- vLLM
- Docker
- NVIDIA CUDA
- `nvidia-smi dmon`
- vLLM Prometheus metrics

The experiments progressed from single-request inference to concurrent
serving and finally to a controlled sequential-vs-concurrent comparison.

The most significant result was the 16-request experiment:

- 16 sequential requests: ~77 tok/s aggregate
- 16 concurrent requests: ~1,075 tok/s aggregate
- Aggregate throughput improvement: ~14×

The experiment demonstrates why concurrency, request scheduling, and
batching behavior are fundamental concepts in production LLM serving.

An important distinction is that the ~1,075 tok/s figure represents
aggregate output-token throughput across the concurrent workload. It does
not mean that a single request generates tokens at 1,075 tok/s.

The measured per-request decode throughput remained around 74 tok/s.

Therefore:

Per-request throughput ≠ Aggregate serving throughput

The sequential-vs-concurrent experiment also showed that high GPU utilization
alone is not sufficient to evaluate an LLM serving system. The sequential
workload could maintain high GPU utilization while achieving only about
77 tok/s aggregate throughput. Increasing concurrency allowed the serving
system to process substantially more useful work during the same period.

The next phase of the project will move the vLLM workload from the local
Docker environment into Kubernetes and investigate:

Client
  ↓
Kubernetes Service
  ↓
vLLM Pod
  ↓
NVIDIA GPU
  ↓
CUDA
  ↓
GPU Hardware

The Kubernetes phase will focus on:

- Container orchestration
- GPU scheduling
- Kubernetes resource requests and limits
- Service exposure
- GPU-aware workloads
- vLLM deployment
- Multi-replica inference serving