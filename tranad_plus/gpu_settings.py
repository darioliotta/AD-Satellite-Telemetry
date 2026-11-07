"""Numerical execution policy. Never silently enable mixed precision or TF32.

A100 PCIe 40 GB benefits chiefly from caching batches on the GPU, avoiding
per-batch CPU synchronizations, and caching decoder predictions for all alphas.
The default FP32 mode preserves the numerical protocol. TF32 is opt-in and
must be treated as an additional experiment, not mixed into the main grid.
"""
import os


def configure():
    import torch
    setting=os.environ.get('TRANADPLUS_CUDA_PRECISION','faithful')
    if setting not in ('faithful','tf32'):
        raise ValueError('TRANADPLUS_CUDA_PRECISION must be faithful or tf32')
    allow=setting=='tf32'
    torch.backends.cuda.matmul.allow_tf32=allow
    torch.backends.cudnn.allow_tf32=allow
    if torch.cuda.is_available():
        name=torch.cuda.get_device_name(0)
        print(f'CUDA execution policy: {setting} | TF32={allow} | device={name}',flush=True)
    return setting


def adam_implementation():
    """Standard preserves v10 optimizer; fused changes floating-point execution.

    Fused must be explicitly selected and the output namespace v11 keeps it
    separate from v10. Do NOT mix different Adam implementations under one run ID.
    """
    setting = os.environ.get('TRANADPLUS_ADAM_IMPL', 'standard').lower()
    if setting not in ('standard', 'fused'):
        raise ValueError('TRANADPLUS_ADAM_IMPL must be standard or fused')
    return setting


def make_adam(parameters, *, lr, device):
    import torch
    impl = adam_implementation()
    if impl == 'fused' and torch.device(device).type != 'cuda':
        raise ValueError('Fused Adam requires CUDA')
    return torch.optim.Adam(parameters, lr=lr, **({'fused': True} if impl == 'fused' else {}))


def select_optimizer(model, optimizer, scheduler):
    """Replace upstream Adam only in explicitly enabled fused mode.

    Scheduler and restored optimizer state are attached to the replacement.
    The normal mode returns the exact upstream optimizer unchanged.
    """
    impl = adam_implementation()
    if impl == 'standard':
        return optimizer, scheduler
    from torch import optim
    if next(model.parameters()).device.type != 'cuda':
        raise ValueError('Fused Adam requires CUDA')
    if len(optimizer.param_groups) != 1:
        raise ValueError('Unexpected optimizer parameter groups')
    new = optim.Adam(model.parameters(), lr=optimizer.param_groups[0]['lr'], fused=True)
    scheduler.optimizer = new
    return new, scheduler


def sync_upstream_optimizer(upstream, active):
    """Mirror replacement Adam state into upstream's saved optimizer handle."""
    if upstream is not active:
        upstream.load_state_dict(active.state_dict())
