import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <torch/extension.h>

__global__ void branch_kernel(
    const float* x,
    float* y,
    int N,
    bool divergent
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N)
        return;

    if (divergent) {
        if (threadIdx.x % 2 == 0) {
            y[i] = x[i] * 2.0f;
        } else {
            y[i] = x[i] * 3.0f;
        }
    } else {
        y[i] = x[i] * 2.0f;
    }
}

void launch_branch(
    torch::Tensor x,
    torch::Tensor y,
    bool divergent
) {
    int N = x.numel();

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    branch_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N,
        divergent
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_branch", &launch_branch);
}
'''

module = load_inline(
    name="warp_divergence",
    cpp_sources="",
    cuda_sources=cuda_source,
    functions=None,
    extra_cuda_cflags=["-O3"],
)

device = "cuda"
N = 10_000_000

x = torch.randn(N, device=device)
y = torch.empty_like(x)

# Warm-up
for _ in range(10):
    module.launch_branch(x, y, False)
    module.launch_branch(x, y, True)

torch.cuda.synchronize()


def benchmark(divergent):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(100):
        module.launch_branch(x, y, divergent)

    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / 100


normal = benchmark(False)
divergent = benchmark(True)

print(f"Normal:     {normal:.6f} ms")
print(f"Divergent:  {divergent:.6f} ms")
print(f"Ratio:      {divergent / normal:.2f}x")
