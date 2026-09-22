import torch
import time

print("=== PyTorch CUDA Test ===")

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

device = torch.device("cuda")

print("\nCreating tensors on GPU...")

x = torch.randn(5000, 5000, device=device)
y = torch.randn(5000, 5000, device=device)

print("x device:", x.device)
print("y device:", y.device)

torch.cuda.synchronize()

start = time.time()

z = x @ y

torch.cuda.synchronize()

elapsed = time.time() - start

print("\n=== Result ===")
print("z device:", z.device)
print("z shape:", z.shape)
print(f"Matrix multiplication time: {elapsed:.4f} seconds")

print("\nGPU memory allocated:",
      round(torch.cuda.memory_allocated() / 1024**2, 2), "MB")

print("GPU memory reserved:",
      round(torch.cuda.memory_reserved() / 1024**2, 2), "MB")
