"""Small archive estimates from real exports, and a campaign output budget."""
from pathlib import Path
import shutil

GiB=1024**3


class StorageBudgetError(RuntimeError):
    """Output must stop without discarding existing artifacts."""


def output_bytes(root):
    root=Path(root)
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0


def check_output_budget(root,max_gib):
    """Refuse further work after the budget; never delete scientific results."""
    used=output_bytes(root)
    if used>max_gib*GiB:
        raise StorageBudgetError(f'Output budget exceeded: {used/GiB:.3f} > {max_gib} GiB; results preserved')
    if shutil.disk_usage(Path(root).parent).free<128*1024**2:
        raise StorageBudgetError('Less than 128 MiB free: stopping before more output; results preserved')
    return used


def model_archive_estimate(profile,formats,checkpoint=None,smd=False):
    """Measure actual packed files, excluding inference and permanent writes."""
    import torch
    from tranad_plus.core.models import build_model
    from .quantization import export_model
    with torch.random.fork_rng(devices=[]):model=build_model(profile)
    if checkpoint is not None:model.load_state_dict(checkpoint['model_state'],strict=True)
    formats=list(dict.fromkeys(['fp32']+list(formats)))
    packed=sum(export_model(model,None,profile,name,measure_only=True)['artifact_file_bytes'] for name in formats)
    # Native SMD keeps one latest recovery plus one final checkpoint. Adam has
    # two moment tensors; state_dict aliases use shared torch storage on disk.
    parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
    buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers())
    checkpoints=2*(3*parameter_bytes+buffer_bytes+256*1024) if smd else 0
    return {'packed_model_bytes':packed,'checkpoint_estimate_bytes':checkpoints,'formats':len(formats),'smd':smd}


def storage_estimate(parts,max_gib):
    packed=sum(p['packed_model_bytes'] for p in parts)
    checkpoints=sum(p['checkpoint_estimate_bytes'] for p in parts)
    formats=sum(p['formats'] for p in parts)
    # Allow substantial slack for JSON operator/alpha diagnostics, manifests,
    # raw latency samples, epoch losses/logs, and regenerated summary tables.
    overhead=formats*512*1024+len(parts)*512*1024+32*1024**2
    total=packed+checkpoints+overhead
    if total>max_gib*GiB:
        raise RuntimeError(f'Conservative archive estimate {total/GiB:.3f} GiB exceeds {max_gib} GiB budget')
    return {'jobs':len(parts),'format_evaluations':formats,'measured_packed_model_GiB':round(packed/GiB,4),
            'estimated_SMD_checkpoints_GiB':round(checkpoints/GiB,4),
            'diagnostics_and_logs_allowance_GiB':round(overhead/GiB,4),
            'conservative_total_GiB':round(total/GiB,4),'output_budget_GiB':max_gib,
            'pointwise_arrays_saved':False,'scope':'model payload measured locally; diagnostics/checkpoints estimated, not guaranteed final size'}
