import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr,          # pointer to first input vector
    y_ptr,          # pointer to second input vector
    output_ptr,     # pointer to output vector
    n_elements,     # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """Compute output = x + y element-wise."""
    # Each program instance processes one block of elements.
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask out out-of-bounds accesses when n_elements is not a multiple of BLOCK_SIZE.
    mask = offsets < n_elements

    # Load inputs.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute.
    output = x + y

    # Store result.
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Vector addition: output = x + y."""
    assert x.shape == y.shape and x.is_cuda and y.is_cuda
    output = torch.empty_like(x)
    n_elements = x.numel()

    # Launch grid: one kernel instance per block.
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


if __name__ == "__main__":
    # Quick correctness test.
    N = 1024 * 1024  # 1M elements
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    y = torch.randn(N, device="cuda", dtype=torch.float32)

    out_triton = vector_add(x, y)
    out_torch = x + y

    assert torch.allclose(out_triton, out_torch), "Mismatch!"
    print(f"✓ vector-add passed: {N} elements, max error = {(out_triton - out_torch).abs().max().item():.2e}")

    # Quick throughput check.
    ms_triton = triton.testing.do_bench(lambda: vector_add(x, y))
    ms_torch = triton.testing.do_bench(lambda: x + y)
    print(f"Triton: {ms_triton:.4f} ms,  Torch: {ms_torch:.4f} ms")
