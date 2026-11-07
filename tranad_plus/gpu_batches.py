"""Optional CUDA-resident window batches for small/medium telemetry datasets.

Only enabled for non-shuffled, float32 datasets. Keeps window and batch order,
loss, model, batch size and optimizer unchanged. The GPU may still be underutilized
for small TranAD architectures; memory occupancy is not the optimization goal.
"""
from __future__ import annotations
import math
import os
import torch


class CUDABatches:
    def __init__(self, windows:torch.Tensor, batch_size:int):
        self.windows=windows
        self.batch_size=batch_size
    def __iter__(self):
        for start in range(0,len(self.windows),self.batch_size):
            yield self.windows[start:start+self.batch_size]
    def __len__(self):
        return math.ceil(len(self.windows)/self.batch_size)


def maybe_cuda_batches(dataset, *, batch_size, shuffle, device):
    """Return GPU batches or None (fall back to existing DataLoader).

    Native datasets store contiguous `big_track`, OPS-SAT stores already
    fragment-safe `windows`. `as_strided` creates a view, not materialized
    overlapping copies. Fail closed for unsupported layouts/dtypes/shuffle.
    """
    setting=os.environ.get('TRANADPLUS_GPU_CACHE','auto').lower()
    if setting not in ('auto','off'):
        raise ValueError('TRANADPLUS_GPU_CACHE must be auto or off')
    if setting=='off' or shuffle or torch.device(device).type!='cuda':
        return None
    try:
        cap_gib=float(os.environ.get('TRANADPLUS_GPU_CACHE_MAX_GIB','16'))
    except ValueError as exc:
        raise ValueError('TRANADPLUS_GPU_CACHE_MAX_GIB must be a number') from exc
    if not 0 < cap_gib <= 32:
        raise ValueError('TRANADPLUS_GPU_CACHE_MAX_GIB must be in (0,32]')
    is_opssat=hasattr(dataset,'windows')
    storage=dataset.windows if is_opssat else getattr(dataset,'big_track',None)
    if not isinstance(storage,torch.Tensor) or storage.dtype!=torch.float32 or storage.ndim not in (2,3):
        print('GPU CACHE: unsupported storage/dtype; using DataLoader',flush=True)
        return None
    if not storage.is_contiguous():
        return None
    required=storage.numel()*storage.element_size()
    free,total=torch.cuda.mem_get_info(device)
    budget=min(int(cap_gib*1024**3),int(free*0.25))
    if required>budget:
        print(f'GPU CACHE: data {required/1024**2:.1f} MiB exceeds {budget/1024**2:.1f} MiB budget; using DataLoader',flush=True)
        return None
    try:
        gpu_storage=storage.to(device)
    except torch.cuda.OutOfMemoryError:
        print('GPU CACHE: unable to allocate; using DataLoader',flush=True)
        return None
    if is_opssat:
        gpu_windows=gpu_storage
    else:
        count=len(dataset)
        window=int(dataset.window_size)
        stride_t,stride_f=gpu_storage.stride()
        if count <= 0 or count+window-1>len(gpu_storage):
            raise RuntimeError('Unexpected native window bounds')
        gpu_windows=gpu_storage.as_strided((count,window,gpu_storage.size(1)),(stride_t,stride_t,stride_f))
    print(f'GPU CACHE: {required/1024**2:.1f} MiB dataset resident; '
          f'{len(gpu_windows)} windows, batch {batch_size}, original order retained',flush=True)
    return CUDABatches(gpu_windows,batch_size)


def training_batches(dataset, *, batch_size, shuffle, device, num_workers=0):
    """Single ordered loader policy for OPS-SAT, SMAP and MSL.

    Keeps exact mini-batch boundaries. For CUDA, use resident data when
    eligible; otherwise pin host memory for non-blocking copies.
    """
    from torch.utils.data import DataLoader
    cached = maybe_cuda_batches(dataset, batch_size=batch_size, shuffle=shuffle, device=device)
    if cached is not None:
        print(f'[DATA] path=cuda_resident windows={len(dataset)} '
              f'batch={batch_size} batches={len(cached)}', flush=True)
        return cached, 'cuda_resident'
    cuda = torch.device(device).type=='cuda'
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,
                        num_workers=num_workers,pin_memory=cuda)
    print(f'[DATA] path={"cpu_pinned" if cuda else "cpu"} windows={len(dataset)} '
          f'batch={batch_size} batches={len(loader)}',flush=True)
    return loader, ('cpu_pinned' if cuda else 'cpu')
