import torch
import time

device = torch.device("cuda")

sizes = 5000

for dtype in [torch.float32, torch.float16, torch.bfloat16]:

    print(f"\n=== {dtype} ===")

    x = torch.randn(sizes, sizes, device=device, dtype=dtype)
    y = torch.randn(sizes, sizes, device=device, dtype=dtype)

    # Warm-up
    z = x @ y
    torch.cuda.synchronize()

    start = time.time()

    z = x @ y

    torch.cuda.synchronize()

    elapsed = time.time() - start

    print(f"Time: {elapsed:.4f} seconds")
    print(f"Memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MiB")
