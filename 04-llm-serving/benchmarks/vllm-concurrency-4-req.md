# vLLM Concurrent Request Benchmark

## 1. Objective

Measure the behavior of vLLM under concurrent inference requests and compare it with the single-request baseline.

The experiment focuses on:

- Concurrent request handling
- End-to-end throughput
- Decode throughput
- Time To First Token (TTFT)
- Decode latency
- The relationship between throughput and latency

---

## 2. Environment

| Component | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti |
| VRAM | 12 GB |
| Model | Qwen3-8B-AWQ |
| Quantization | AWQ |
| Compute dtype | FP16 |
| Serving engine | vLLM 0.30.0 |
| Max model length | 4096 |
| GPU memory utilization | 0.85 |
| Concurrent requests | 4 |
| Max output tokens/request | 200 |

---

## 3. Test Command

Four requests were submitted concurrently using `xargs -P4`:

```bash
time seq 1 4 | xargs -P4 -I{} curl -s \
'http://localhost:8000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-d '{
  "model": "/model",
  "messages": [
    {
      "role": "user",
      "content": "Explain GPU memory bandwidth in detail."
    }
  ],
  "max_tokens": 200,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}' > /tmp/vllm_concurrent_{}.json
```

# vLLM Concurrent Request Benchmark — Results

## 4. Concurrency Model

Four requests were executed concurrently.

**Concurrency flow:**

`R1` ─┐  
`R2` ─┤  
`R3` ─┤ → **vLLM → GPU**  
`R4` ─┘

`xargs -P4` allows up to four `curl` processes to run concurrently.

The objective is to observe how vLLM handles multiple simultaneous inference requests.

## 5. Wall-Clock Result

The four concurrent requests completed in:

`real    0m11.059s`  
`user    0m0.031s`  
`sys     0m0.015s`

Each request used:

`max_tokens = 200`

The vLLM metrics confirmed that the four new requests generated:

`800 tokens total`

Therefore:

`4 requests × 200 tokens = 800 tokens`

## 6. Metrics Interpretation

vLLM metrics are cumulative since the server started.

Therefore, to isolate this experiment:

`experiment value = new cumulative value - previous cumulative value`

### Generation Tokens

Previous cumulative value:

`1084 tokens`

New cumulative value:

`1884 tokens`

Calculation:

`1884 - 1084 = 800 tokens`

Therefore, the four concurrent requests generated exactly 800 tokens.

## 7. End-to-End Aggregate Throughput

Wall-clock time:

`11.059 seconds`

Total newly generated tokens:

`800 tokens`

Calculation:

`800 / 11.059 ≈ 72.3 tokens/s`

### Result

**End-to-end aggregate throughput ≈ 72.3 tok/s**

This includes the overall request duration and should not be interpreted as pure GPU decode throughput.

## 8. Decode Throughput

The vLLM `request_time_per_output_token_seconds` metric changed from:

Previous cumulative sum:

`0.1099437967 seconds`

New cumulative sum:

`0.1624388742 seconds`

Difference:

`0.1624388742 - 0.1099437967 = 0.0524950775 seconds`

There were four new requests:

`0.0524950775 / 4 = 0.01312377 seconds/token`

Convert to milliseconds:

`0.01312377 × 1000 ≈ 13.12 ms/token`

Convert to tokens/second:

`1 / 0.01312377 ≈ 76.2 tok/s`

### Result

**Per-request decode throughput ≈ 76.2 tok/s**

## 9. Time To First Token (TTFT)

Previous cumulative TTFT:

`12.3411403 seconds`

New cumulative TTFT:

`45.8632796 seconds`

Difference:

`45.8632796 - 12.3411403 = 33.5221393 seconds`

Average across the four new requests:

`33.5221393 / 4 ≈ 8.38 seconds`

### Result

**Average TTFT ≈ 8.38 seconds**

TTFT increased significantly compared with the previous single-request workload.

## 10. Decode Time

Previous cumulative decode time:

`14.7471352 seconds`

New cumulative decode time:

`25.1936557 seconds`

Difference:

`25.1936557 - 14.7471352 = 10.4465205 seconds`

Average across four new requests:

`10.4465205 / 4 ≈ 2.612 seconds`

### Result

**Average decode time ≈ 2.61 seconds/request**

## 11. Results Summary

| Metric | Single-request baseline | 4 concurrent requests |
|---|---:|---:|
| Requests | 1 workload | 4 |
| Generated tokens | ~200/request | 800 total |
| Decode throughput | ~72.8 tok/s | ~76.2 tok/s |
| Aggregate E2E throughput | — | ~72.3 tok/s |
| Average TTFT | ~1.54 s* | ~8.38 s |
| Average decode time | ~1.84 s* | ~2.61 s |

\* The single-request values are derived from the previous 8-request cumulative metrics and therefore represent an earlier workload average rather than an isolated single request.

## 12. Observations

### Throughput

The four-concurrent workload achieved approximately:

**72.3 tok/s aggregate E2E throughput**

The measured per-request decode throughput was approximately:

**76.2 tok/s**

This is close to the previous single-request decode result of approximately:

**72.8 tok/s**

Therefore, this experiment does **not yet demonstrate a large throughput improvement from concurrency**.

### Latency

The major change was TTFT.

Average TTFT increased to approximately:

**8.38 seconds**

This demonstrates the basic serving trade-off between concurrency and latency:

**More concurrent requests → potentially better batching/throughput, but potentially higher latency.**

### Important Conclusion

The experiment demonstrates that vLLM can handle concurrent requests, but this workload alone does not prove that continuous batching significantly increases aggregate throughput.

A higher-concurrency experiment is required.

## 13. Next Experiment

Test higher concurrency:

`1 concurrent request → 4 concurrent requests → 8 concurrent requests`

Compare:

- TTFT
- Decode tok/s
- Aggregate throughput
- GPU utilization

The next experiment will determine whether increasing concurrency produces meaningful improvements in aggregate serving throughput on the RTX 4070 Ti.
