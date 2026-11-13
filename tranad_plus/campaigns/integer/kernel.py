"""INT8 x INT8 -> INT32 GEMM/BMM on CUDA; no floating-point dot fallback.

W4A4 also uses these INT8 instructions, with operands restricted to [-7,7].
It is NOT native packed INT4 Tensor Core execution. See README.md.
"""
import re
import torch

_KERNEL = None
_AUDIT = {}


def _kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    from ._cuda import integer_dot
    _KERNEL = integer_dot
    return _KERNEL


def int_bmm(a, b, backend='cuda_triton'):
    if a.dtype != torch.int8 or b.dtype != torch.int8 or a.ndim != 3 or b.ndim != 3:
        raise TypeError('Integer GEMM requires rank-3 INT8 operands')
    groups, m, k = a.shape
    if b.shape[-2] != k or b.shape[0] not in (1, groups) or a.device != b.device:
        raise ValueError('Incompatible integer GEMM operands')
    # Worst-case sum cannot overflow INT32 under the supported symmetric range.
    if k * 127 * 127 > torch.iinfo(torch.int32).max:
        raise OverflowError('INT32 accumulator bound exceeded')
    if backend == 'reference_cpu':
        if a.device.type != 'cpu':
            raise ValueError('Reference test backend is CPU only')
        return torch.bmm(a.to(torch.int32), b.expand(groups, -1, -1).to(torch.int32))
    if backend != 'cuda_triton' or a.device.type != 'cuda':
        raise RuntimeError('Campaign requires CUDA/Triton; no CPU or floating-point fallback')
    import triton
    n = b.shape[-1]
    c = torch.empty((groups, m, n), device=a.device, dtype=torch.int32)
    compiled = _kernel()[(triton.cdiv(m, 16), triton.cdiv(n, 32), groups)](
        a, b, c, m, n, k, *a.stride(), 0 if b.shape[0] == 1 else b.stride(0),
        b.stride(1), b.stride(2), 16, 32, 32, num_warps=4)
    # Inspect generated instructions, rather than infer integer execution from dtype names.
    key = (str(a.device), m, n, k, a.stride(), b.stride())
    if key not in _AUDIT:
        ptx = compiled.asm['ptx']
        instructions = sorted(set(re.findall(r'mma\.sync[^;\n]*\.s32\.s8\.s8\.s32', ptx)))
        if not instructions:
            raise RuntimeError('Compiled kernel does not contain the required INT8/INT32 MMA instructions')
        _AUDIT[key] = {'shape': [groups, m, n, k], 'mma_instructions': instructions,
                       'input_dtype': 'int8', 'accumulator_dtype': 'int32'}
    return c


def kernel_audit():
    return list(_AUDIT.values())


def preflight(device):
    """Compile and compare awkward/padded shapes against exact CPU arithmetic."""
    import triton
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for the integer campaign')
    if torch.cuda.get_device_capability(device)[0] < 8:
        raise RuntimeError('This campaign targets Ampere or newer NVIDIA GPUs')
    cases = []
    generator = torch.Generator().manual_seed(2026)
    for bits in (8, 4):
        limit = 2**(bits-1)-1
        for g, m, n, k, broadcast in [(2, 3, 2, 2, False), (3, 13, 25, 55, False),
                                     (1, 33, 55, 110, True), (2, 10, 50, 25, True)]:
            a = torch.randint(-limit, limit+1, (g, m, k), dtype=torch.int8, generator=generator)
            b = torch.randint(-limit, limit+1, (1 if broadcast else g, k, n), dtype=torch.int8, generator=generator)
            expected = int_bmm(a, b, 'reference_cpu')
            actual = int_bmm(a.to(device), b.to(device)).cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            cases.append({'bits': bits, 'shape': [g, m, n, k], 'exact_integer_match': True})
    return {'triton_version': triton.__version__, 'device': torch.cuda.get_device_name(device),
            'cases': cases, 'instructions': kernel_audit(), 'no_float_dot_fallback': True,
            'int4_native_instructions': False, 'int4_execution': 'W4A4_values_on_INT8_MMA'}
