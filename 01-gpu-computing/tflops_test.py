import torch
import time

device = torch.device("cuda")
N = 5000

for dtype in [torch.float32, torch.float16, torch.bfloat16]:

    print(f"\n=== {dtype} ===")

    x = torch.randn(N, N, device=device, dtype=dtype)
    y = torch.randn(N, N, device=device, dtype=dtype)

    # Warm-up
    for _ in range(3):
        z = x @ y

    torch.cuda.synchronize()

    start = time.time()

    for _ in range(10):
        z = x @ y

    torch.cuda.synchronize()

    elapsed = time.time() - start

    # Average execution time
    avg_time = elapsed / 10

    # Matrix multiplication FLOPs
    flops = 2 * (N ** 3)

    # TFLOPS
    tflops = flops / avg_time / 1e12

    print(f"Average time: {avg_time:.6f} seconds")
    print(f"Performance: {tflops:.2f} TFLOPS")
