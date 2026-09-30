# vLLM 16-Concurrent Request Benchmark

## Results

- Concurrent requests: 16
- Output tokens per request: 200
- Total generated tokens: 3200
- Wall-clock time: 3.175 s
- Aggregate E2E throughput: ~1007.9 tok/s
- Average TTFT: ~423 ms
- Per-request decode throughput: ~74.2 tok/s
- Average decode time: ~2.68 s

## GPU Monitoring

During the active workload:

- SM utilization: ~93%
- Memory-controller utilization: ~98%
- Power: ~180 W
- GPU temperature: ~46°C
- GPU clock: ~2820 MHz
- Memory clock: ~10251 MHz

## Key Observation

At 16 concurrent requests, vLLM generated approximately 1008 output
tokens per second in aggregate.

Per-request decode throughput remained approximately 74 tok/s,
while aggregate throughput increased substantially with concurrency.

The GPU reached approximately 93% SM utilization and 98% memory-controller
utilization during the workload, indicating that the GPU was approaching
saturation for this workload.

## Interpretation

Increasing concurrency provides vLLM with more active sequences to
schedule and batch. This increases the amount of work available to the GPU
and therefore increases aggregate serving throughput.

The result demonstrates the distinction between:

- Per-request decode throughput
- Aggregate serving throughput
- GPU utilization
- GPU memory-controller utilization

## Next Experiment

Continuous batching experiment.
