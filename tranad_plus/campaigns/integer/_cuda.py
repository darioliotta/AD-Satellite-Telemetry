"""CUDA-only Triton kernel. Imported lazily; INT8 inputs, INT32 accumulator."""
import triton
import triton.language as tl

@triton.jit
def integer_dot(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                a0: tl.constexpr, a1: tl.constexpr, a2: tl.constexpr,
                b0: tl.constexpr, b1: tl.constexpr, b2: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    group = tl.program_id(2)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.int32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + kk
        x = tl.load(A + group*a0 + row[:, None]*a1 + k[None, :]*a2,
                    (row[:, None] < M) & (k[None, :] < K), other=0)
        y = tl.load(B + group*b0 + k[:, None]*b1 + col[None, :]*b2,
                    (k[:, None] < K) & (col[None, :] < N), other=0)
        acc += tl.dot(x, y, out_dtype=tl.int32)
    tl.store(C + group*M*N + row[:, None]*N + col[None, :], acc,
             (row[:, None] < M) & (col[None, :] < N))
