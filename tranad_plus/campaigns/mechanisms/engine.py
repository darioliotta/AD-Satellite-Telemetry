"""Fixed-horizon matched training and score-only streaming inference."""
from pathlib import Path
import json
import time
import numpy as np
import torch

from tranad_plus.core.common import save_torch, append_json, atomic_destination
from tranad_plus.core.training.engine import seed_all, rng_state, restore_rng, TrainingInterrupted
from tranad_plus.core.training.objectives import reconstruction_loss
from tranad_plus.campaigns.exploration.progress import bar, record
from .control import build_control
from .data import batches


def stream_scores(model, dataset, device, batch_size, alpha, title, labels=None, baseline=None):
    source, transfer = batches(dataset, batch_size, device)
    if model is not None:
        model.eval()
    pieces, elapsed, offset = [], time.perf_counter(), 0
    by_class = {str(i): {'values': 0, 'squared_error_sum_by_feature': None} for i in (0, 1)}
    cuda = torch.device(device).type == 'cuda'
    if cuda:
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter()
    with torch.inference_mode(), bar(source, total=len(source), desc=title, position=2, unit='batch', leave=False) as progress:
        for batch in progress:
            windows = batch[0] if isinstance(batch, (tuple, list)) else batch
            windows = windows.to(device, non_blocking=transfer == 'cpu_pinned')
            target = windows[:, -1]
            if baseline:
                past = windows[:, :-1]
                if baseline == 'persistence':
                    reconstruction = past[:, -1]
                elif baseline == 'past_median':
                    reconstruction = torch.quantile(past, .5, dim=1)
                else:
                    raise ValueError('Unknown deterministic baseline')
            else:
                outputs = model(windows.permute(1, 0, 2), target[None])
                if isinstance(outputs, tuple):
                    if alpha is None:
                        raise ValueError('Two outputs require fixed alpha')
                    reconstruction = (alpha*outputs[0]+(1-alpha)*outputs[1])[0]
                else:
                    if alpha is not None:
                        raise ValueError('Single-output model has no alpha')
                    reconstruction = outputs[0]
            errors = (reconstruction-target).square()
            if not bool(torch.isfinite(errors).all()):
                raise FloatingPointError('Nonfinite errors; no silent masking/dropping')
            pieces.append(errors.mean(dim=1).cpu().numpy())
            if labels is not None:
                flags = np.asarray(labels)[offset:offset+len(target)]
                for cls in (0, 1):
                    mask = torch.as_tensor(flags == cls, device=device)
                    if bool(mask.any()):
                        sums = errors[mask].double().sum(dim=0).cpu().numpy()
                        current = by_class[str(cls)]
                        current['values'] += int(mask.sum().item())
                        previous = current['squared_error_sum_by_feature']
                        current['squared_error_sum_by_feature'] = sums if previous is None else previous+sums
            offset += len(target)
    if cuda:
        torch.cuda.synchronize(device)
    for value in by_class.values():
        total = value.pop('squared_error_sum_by_feature')
        value['mean_squared_error_by_feature_index'] = (total/value['values']).tolist() if total is not None else None
    scores = np.concatenate(pieces)
    if len(scores) != len(dataset):
        raise AssertionError('Inference/target count mismatch')
    return scores, {'seconds_including_transfer': time.perf_counter()-elapsed, 'points': len(scores),
                    'batch_size': batch_size, 'class_errors': by_class,
                    'feature_note': 'Indices only; no physical sensor identity is inferred'}


def reconcile(folder, committed):
    path = Path(folder)/'epochs.jsonl'
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row['epoch'] <= committed:
                rows[row['epoch']] = row
    if set(rows) != set(range(1, committed+1)):
        raise RuntimeError('Missing committed training telemetry')
    with atomic_destination(path) as f:
        for epoch in sorted(rows):
            f.write(json.dumps(rows[epoch])+'\n')


def nominal_mse(model, dataset, labels, device, batch_size, alpha):
    state, mode = rng_state(), model.training
    try:
        scores, _ = stream_scores(model, dataset, device, batch_size, alpha, 'V20 nominal monitoring')
        nominal = scores[np.asarray(labels) == 0]
        if not len(nominal):
            raise ValueError('Missing nominal monitoring samples')
        return float(nominal.astype(np.float64).mean())
    finally:
        restore_rng(state)
        model.train(mode)


def train(profile, bundle, folder, signature, device, settings, stop_after_epoch=None):
    folder = Path(folder)
    final, recovery = folder/'checkpoints/final.ckpt', folder/'checkpoints/resume.ckpt'
    if final.exists():
        value = torch.load(final, map_location='cpu', weights_only=True)
        if value['signature'] != signature or value['profile'] != profile or value['epoch'] != profile['epochs']:
            raise RuntimeError('Final control checkpoint mismatch')
        return final
    seed_all(profile['seed'])
    model = build_control(profile).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=profile['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=.9)
    source, transfer = batches(bundle.datasets['train'], profile['batch_sz'], device)
    start, accumulated = 0, 0.
    if recovery.exists():
        saved = torch.load(recovery, map_location='cpu', weights_only=True)
        if saved['signature'] != signature or saved['profile'] != profile:
            raise RuntimeError('Recovery/source/data/profile mismatch')
        model.load_state_dict(saved['model_state'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state'])
        scheduler.load_state_dict(saved['scheduler_state'])
        restore_rng(saved['rng_state'])
        start, accumulated = saved['epoch'], saved['training_seconds']
        if scheduler.last_epoch != start:
            raise RuntimeError('Recovered scheduler mismatch')
    reconcile(folder, start)
    alpha = settings['fixed_alpha'] if model.output_count == 2 else None
    cuda = torch.device(device).type == 'cuda'
    title = f"V20 TRAIN {profile['dataset']}/{profile['variant']} {profile['condition']} s{profile['seed']} W{profile['window_sz']} FF{profile['dim_feedforward']}"
    with bar(total=profile['epochs'], initial=start, desc=title, position=1, unit='epoch', leave=False) as progress:
        for epoch in range(start+1, profile['epochs']+1):
            if cuda:
                torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
            begun = time.perf_counter()
            model.train()
            total, count = torch.zeros((), device=device, dtype=torch.float64), 0
            with bar(source, total=len(source), desc=f'V20 training batches | epoch {epoch}', position=2, unit='batch', leave=False) as phase:
                for batch in phase:
                    x = batch[0] if isinstance(batch, (tuple, list)) else batch
                    x = x.to(device, non_blocking=transfer == 'cpu_pinned').permute(1, 0, 2)
                    target = x[-1:]
                    outputs = model(x, target)
                    loss = reconstruction_loss(outputs, target, epoch,
                                               implementation='square' if profile['dataset'] == 'OPS-SAT' else 'mse_loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward(); optimizer.step()
                    total.add_(loss.detach().double()); count += 1
            loss_value = float((total/count).item()) if count else float('nan')
            if not np.isfinite(loss_value):
                raise FloatingPointError('Nonfinite training objective; previous recovery retained')
            scheduler.step()
            monitor = {split: nominal_mse(model, bundle.datasets[split], bundle.labels[split], device,
                                         settings['batch_size_by_dataset'][profile['dataset']], alpha)
                       for split in ('train', 'validation')}
            if cuda:
                torch.cuda.synchronize(device)
            seconds = time.perf_counter()-begun; accumulated += seconds
            row = {'epoch': epoch, 'training_objective': loss_value, 'fixed_alpha': alpha,
                   'train_nominal_mse': monitor['train'], 'validation_nominal_mse': monitor['validation'],
                   'lr_after_scheduler': optimizer.param_groups[0]['lr'], 'seconds_including_monitoring': seconds,
                   'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if cuda else None}
            append_json(folder/'epochs.jsonl', row)
            save_torch(recovery, {'schema_version': 1, 'signature': signature, 'profile': profile, 'epoch': epoch,
                       'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                       'scheduler_state': scheduler.state_dict(), 'rng_state': rng_state(), 'training_seconds': accumulated})
            print(f"[TRAIN] {profile['identifier']} {epoch}/{profile['epochs']} loss={loss_value:.6g} "
                  f"train={monitor['train']:.6g} validation={monitor['validation']:.6g} {seconds:.2f}s", flush=True)
            progress.update(1)
            progress.set_postfix(train=f"{monitor['train']:.4g}", val=f"{monitor['validation']:.4g}", epoch_s=f'{seconds:.1f}')
            record(folder, 'training', identifier=profile['identifier'], **row,
                   remaining_seconds_estimate=seconds*(profile['epochs']-epoch))
            if stop_after_epoch and epoch >= stop_after_epoch:
                raise TrainingInterrupted('V20 recovery committed')
    saved = torch.load(recovery, map_location='cpu', weights_only=True)
    save_torch(final, {k: saved[k] for k in ('schema_version', 'signature', 'profile', 'epoch', 'model_state')})
    return final
