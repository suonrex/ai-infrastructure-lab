# vLLM Single-Request Benchmark

## Environment

- GPU: NVIDIA GeForce RTX 4070 Ti 12 GB
- Model: Qwen3-8B-AWQ
- Serving engine: vLLM 0.30.0
- Quantization: AWQ
- dtype: FP16
- max_model_len: 4096
- gpu_memory_utilization: 0.85

## Metrics

| Metric | Result |
|---|---:|
| Requests | 8 |
| Prompt tokens | 180 total |
| Generation tokens | 1084 total |
| Average prompt tokens/request | 22.5 |
| Average generation tokens/request | 135.5 |
| Average TTFT | 1.543 s |
| Average prefill time | 1.474 s |
| Average decode time | 1.843 s |
| Time per output token | 13.74 ms |
| Decode throughput | ~72.8 tok/s |
| Average queue time | ~0.31 ms |

## Calculation

Average time per output token:

0.1099438 / 8
= 0.013743 s/token

Decode throughput:

1 / 0.013743
≈ 72.8 tokens/s
