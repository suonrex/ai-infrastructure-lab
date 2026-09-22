import torch
import time

device = "cuda"

# Small tensor
small = torch.randn(1024 * 1024, device=device)

# Large tensor
large = torch.randn(256 * 1024 * 1024 // 4, device=device)


def benchmark(tensor, name, iterations=100):
    torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(iterations):
        y = tensor * 2.0

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print(f"{name}: {elapsed:.6f} sec")


benchmark(small, "Small tensor")
benchmark(large, "Large tensor")
