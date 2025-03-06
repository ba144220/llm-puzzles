import torch

import triton
import triton.language as tl

# Get the current device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@triton.jit
def vec_add_kernel(
    x_ptr, # 1D input tensor
    y_ptr, # 1D input tensor
    output_ptr, # 1D output tensor
    n_elements, # number of elements in the input tensors
    BLOCK_SIZE: tl.constexpr, # block size
):
    pid = tl.program_id(axis=0) # get the current program id
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
    
def vec_add(x, y):
    output = torch.empty_like(x).cuda()
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    vec_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE)
    return output
    
    
torch.manual_seed(0)
size = 98432
x = torch.rand(size).cuda()
y = torch.rand(size).cuda()
output_torch = x + y
output_triton = vec_add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
    

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, dtype=torch.float32).cuda()
    y = torch.rand(size, dtype=torch.float32).cuda()
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: vec_add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True, save_path='./triton_kernel/kernels/performance')
