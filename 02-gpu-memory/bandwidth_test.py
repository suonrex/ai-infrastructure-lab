import torch
import time

device = torch.device("cuda")

# 1 GB FP32 Tensor
N = 256 * 1024 * 1024 // 4

x = torch.randn(N, device=device)
y = torch.empty_like(x)

# Warm-up
for _ in range(5):
    y.copy_(x)

torch.cuda.synchronize()

# Benchmark
start = time.time()

for _ in range(10):
    y.copy_(x)

torch.cuda.synchronize()

elapsed = time.time() - start

# One copy: read X + write Y
bytes_transferred = x.numel() * x.element_size() * 2

bandwidth = bytes_transferred * 10 / elapsed / 1e9

print(f"Tensor size: {x.numel() * x.element_size() / 1024**3:.2f} GB")
print(f"Total time: {elapsed:.4f} seconds")
print(f"Bandwidth: {bandwidth:.2f} GB/s")
