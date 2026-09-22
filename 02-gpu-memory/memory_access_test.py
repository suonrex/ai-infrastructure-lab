import torch
import time

device = "cuda"

N = 256 * 1024 * 1024 // 4   # 256 MiB of FP32

x = torch.randn(N, device=device)
y = torch.empty_like(x)

# --------------------------------
# 1. Sequential Access
# --------------------------------

torch.cuda.synchronize()

start = time.perf_counter()

for _ in range(10):
    y.copy_(x)

torch.cuda.synchronize()

sequential_time = time.perf_counter() - start


# --------------------------------
# 2. Random Access
# --------------------------------

indices = torch.randperm(N, device=device)

torch.cuda.synchronize()

start = time.perf_counter()

for _ in range(10):
    y = x[indices]

torch.cuda.synchronize()

random_time = time.perf_counter() - start


print(f"Tensor size: {x.numel() * x.element_size() / 1024**2:.0f} MiB")
print(f"Sequential access: {sequential_time:.6f} sec")
print(f"Random access:     {random_time:.6f} sec")
print(f"Random / Sequential: {random_time / sequential_time:.2f}x")
