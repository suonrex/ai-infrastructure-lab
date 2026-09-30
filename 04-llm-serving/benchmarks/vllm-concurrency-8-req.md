# vLLM Concurrency Benchmark

## 1. Test Environment

- GPU: NVIDIA GeForce RTX 4070 Ti
- VRAM: 12 GB
- Model: Qwen3-8B-AWQ
- Quantization: AWQ
- Compute dtype: FP16
- vLLM: 0.30.0
- Max model length: 4096
- GPU memory utilization: 0.85
- API: OpenAI-compatible `/v1/chat/completions`

The model is served locally through Docker and vLLM.

---

## 2. Objective

The purpose of this benchmark is to evaluate how vLLM behaves under concurrent inference requests.

The main metrics are:

- End-to-end throughput
- Per-request decode throughput
- Time to First Token (TTFT)
- Decode time
- Aggregate serving throughput

The experiment also demonstrates the difference between:

- **Single-request performance**
- **Per-request generation speed**
- **Aggregate throughput under concurrency**

---

## 3. Test Workload

Eight requests were sent concurrently.

Each request used:

- `max_tokens = 200`
- `enable_thinking = false`
- Same prompt

The concurrency was generated using `xargs -P8`.

The benchmark command was:

    time seq 1 8 | xargs -P8 -I{} curl -s \
      'http://localhost:8000/v1/chat/completions' \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "/model",
        "messages": [{"role":"user","content":"Explain GPU memory bandwidth in detail."}],
        "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": false}
      }'

---

## 4. Wall-Clock Result

The measured wall-clock time was:

    real    0m2.987s

The vLLM Prometheus metrics showed:

    Total requests: 20
    Total generation tokens: 3484

The previous benchmark state was:

    Requests: 12
    Generation tokens: 1884

Therefore, this experiment generated:

    20 - 12 = 8 requests

and:

    3484 - 1884 = 1600 tokens

Since each request was configured for 200 tokens:

    8 × 200 = 1600 tokens

All eight requests therefore reached the 200-token limit.

---

## 5. Aggregate End-to-End Throughput

Aggregate throughput is calculated as:

    generated tokens / wall-clock time

Therefore:

    1600 / 2.987
    = 535.7 tokens/s

Result:

    Aggregate E2E throughput ≈ 535.7 tok/s

This represents the total output-token throughput of the serving system across all eight concurrent requests.

It does NOT mean that a single request generates 535.7 tok/s.

---

## 6. Per-Request Decode Throughput

vLLM reports:

    request_time_per_output_token_seconds

The cumulative metric before this experiment was:

    0.1624388742 seconds

The cumulative metric after this experiment was:

    0.2695476290 seconds

Difference:

    0.2695476290 - 0.1624388742
    = 0.1071087548 seconds

There were eight new requests:

    0.1071087548 / 8
    = 0.0133886 seconds/token

Therefore:

    1 / 0.0133886
    ≈ 74.7 tok/s

Result:

    Per-request decode throughput ≈ 74.7 tok/s

This is close to the previous single-request and four-request decode performance.

---

## 7. Time to First Token (TTFT)

The cumulative TTFT metric before this experiment was:

    45.8632796 seconds

After the experiment:

    48.2342162 seconds

Difference:

    48.2342162 - 45.8632796
    = 2.3709366 seconds

Average across eight new requests:

    2.3709366 / 8
    ≈ 0.296 seconds

Result:

    Average TTFT ≈ 296 ms

This value should not be directly compared with the previous four-concurrency TTFT because the experiments were not controlled as a rigorous latency benchmark.

---

## 8. Decode Time

The cumulative decode time before this experiment was:

    25.19365565 seconds

After the experiment:

    46.50829785 seconds

Difference:

    46.50829785 - 25.19365565
    = 21.31464220 seconds

Average decode time per request:

    21.31464220 / 8
    ≈ 2.664 seconds

Result:

    Average decode time ≈ 2.66 s/request

---

## 9. Results Summary

| Metric | 8 Concurrent Requests |
|---|---:|
| Concurrent requests | 8 |
| Tokens per request | 200 |
| Total generated tokens | 1600 |
| Wall-clock time | 2.987 s |
| Aggregate E2E throughput | **535.7 tok/s** |
| Per-request decode throughput | **~74.7 tok/s** |
| Average TTFT | ~296 ms |
| Average decode time | ~2.66 s |

---

## 10. Key Observation

The most important observation is the difference between **per-request performance** and **aggregate throughput**.

Per-request decode throughput remained around:

    ~75 tok/s

while the aggregate serving throughput reached:

    ~536 tok/s

This demonstrates that concurrent inference can substantially increase the amount of work processed by the GPU at the system level.

Conceptually:

    Individual request
            |
            v
        ~75 tok/s
            |
            |
    +-------+-------+
    |       |       |
    v       v       v
   R1      R2      ... R8
    \       |       /
     \      |      /
      +-----+-----+
            |
            v
    vLLM scheduler
            |
            v
        GPU workload
            |
            v
    ~536 tok/s aggregate

The aggregate throughput is therefore not equivalent to the generation speed of a single request.

---

## 11. Continuous Batching Concept

This experiment illustrates why LLM serving systems use techniques such as continuous batching.

Instead of processing requests completely independently:

    Request 1 → GPU
    Request 2 → GPU
    Request 3 → GPU
    ...

vLLM can schedule multiple active sequences together:

    Request 1 ─┐
    Request 2 ─┤
    Request 3 ─┤
    Request 4 ─┤
    Request 5 ─┤
    Request 6 ─┤ → vLLM scheduler → GPU
    Request 7 ─┤
    Request 8 ─┘

This allows the GPU to process a larger amount of work during each scheduling step.

The practical objective of an inference server is therefore not necessarily to maximize the throughput of one request, but to maximize useful system throughput while maintaining acceptable latency.

---

## 12. Important Interpretation

The result:

    ~535.7 tok/s

should be described as:

    Aggregate output-token throughput under 8 concurrent requests

It should NOT be described as:

    RTX 4070 Ti generates 535.7 tok/s for one request

The latter would be incorrect.

A more accurate performance model is:

    Single-request latency
            +
    Per-request decode throughput
            +
    Concurrency
            +
    Batching efficiency
            =
    Aggregate serving throughput

---

## 13. Comparison with Previous Tests

| Concurrency | Aggregate Throughput | Per-request Decode |
|---:|---:|---:|
| 1 | ~57 tok/s* | ~72.8 tok/s |
| 4 | ~72.3 tok/s | ~76.2 tok/s |
| 8 | **~535.7 tok/s** | **~74.7 tok/s** |

*The single-request measurement was an end-to-end `curl` measurement and is not directly comparable to the vLLM internal decode-throughput metric.

The 8-request result should therefore be interpreted primarily as evidence of high aggregate throughput under concurrency rather than as a direct apples-to-apples latency comparison.

---

## 14. Next Experiment

Next, observe GPU-level behavior during the 8-concurrent workload.

The goal is to compare:

    Request concurrency
            ↓
    vLLM batching
            ↓
    GPU utilization
            ↓
    GPU power consumption
            ↓
    Aggregate throughput

This will connect the vLLM serving metrics with the GPU metrics observed through `nvidia-smi`.

Potential metrics:

- GPU Utilization
- VRAM Usage
- Power Draw
- GPU Temperature
- Aggregate Token Throughput
- TTFT
- Decode Throughput

This is the next step toward understanding production-style LLM inference performance.
