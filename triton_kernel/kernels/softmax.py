import torch
import lovely_tensors
import triton
import triton.language as tl

lovely_tensors.monkey_patch()
def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    x: [M, N]
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0] # [M]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None] # [M, N]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z) # [M, N]
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1) # [M]
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None] # [M, N]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret

@triton.jit
def softmax_kernel(
    x_ptr,
    x_stride,
    y_ptr,
    y_stride,
    n_cols,
    block_size: tl.constexpr
):
    row_idx = tl.program_id(0)
    
    row_start_ptr = x_ptr + row_idx * x_stride
    # tl.device_print("row_start_ptr", row_start_ptr)
    
    col_offsets = tl.arange(0, block_size)
    input_ptrs = row_start_ptr + col_offsets
    
    mask = col_offsets < n_cols
    
    row_vals = tl.load(input_ptrs, mask=mask, other=-float("inf"))
    
    row_max = tl.max(row_vals, axis=0)
    
    row_vals = row_vals - row_max
    
    row_exp = tl.exp(row_vals)
    
    row_sum = tl.sum(row_exp, axis=0)
    
    row_softmax = row_exp / row_sum
    
    y_row_start_ptr = y_ptr + row_idx * y_stride
    
    y_row_ptrs = y_row_start_ptr + col_offsets
    
    tl.store(y_row_ptrs, row_softmax, mask=mask)
    
    
    

def triton_softmax(x):
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    
    # Get block size
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Allocate output
    y = torch.empty_like(x)
    
    grid = (n_rows,)
    softmax_kernel[grid](
        x,
        x.stride(0),
        y, 
        y.stride(0),
        n_cols, 
        block_size = BLOCK_SIZE
    )
    
    return y

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch'],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, dtype=torch.float32).cuda()
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: triton_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)



def main():
    x = torch.randn(603, 1432).cuda()
    naive_ret = naive_softmax(x)
    print(naive_ret)
    triton_ret = triton_softmax(x)
    print(triton_ret)
    if torch.allclose(naive_ret, triton_ret):
        print("All close")
    else:
        print("Not all close")
        
    benchmark.run(show_plots=True, print_data=True, save_path="./triton_kernel/kernels/performance")

if __name__ == '__main__':
    main()
