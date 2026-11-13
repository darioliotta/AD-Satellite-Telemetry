"""Original ordered Adam trajectory plus RNG-neutral validation diagnostics.

Audit mode follows the full trajectory after the first stopping decision, but
freezes the early-stopped checkpoint at that decision. This gives paired final,
50-epoch, and early-stopped endpoints without separate baseline retraining.
"""
from pathlib import Path
import time
import numpy as np
import torch

from tranad_plus.core.common import append_json, atomic_destination, save_torch, write_json
from tranad_plus.core.models import build_model
from tranad_plus.core.training.engine import seed_all, rng_state, restore_rng, TrainingInterrupted
from tranad_plus.core.training.objectives import reconstruction_loss
from tranad_plus.campaigns.exploration.progress import bar, record
from .policy import EarlyStopping
from gpu_batches import training_batches


def configured_model(profile, arm):
    model = build_model(profile)
    # Existing dropout stays at 0.1 in both declared arms. Expose the choice in
    # provenance and fail closed if the architecture's defaults ever change.
    modules = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    attention = [m for m in model.modules() if isinstance(m, torch.nn.MultiheadAttention)]
    if not modules or any(m.p != arm['dropout'] for m in modules):
        raise RuntimeError('Unexpected baseline dropout configuration')
    if any(m.dropout != arm['dropout'] for m in attention):
        raise RuntimeError('Unexpected attention dropout configuration')
    return model


def nominal_loader(bundle, split, batch_size, device):
    dataset = bundle.datasets[split]
    labels = np.asarray(bundle.labels[split])
    indices = np.flatnonzero(labels == 0)
    if not len(indices):
        raise ValueError(f'No nominal windows in {split}')
    if len(indices) == len(dataset):
        return training_batches(dataset, batch_size=batch_size, shuffle=False, device=device, num_workers=0)
    # Materialization retains CUDA-resident ordered batching for this small
    # dataset; all original windows were built within individual fragments.
    class Windows(torch.utils.data.Dataset):
        def __init__(self):
            self.windows = dataset.windows[torch.as_tensor(indices)].contiguous()
        def __len__(self):
            return len(self.windows)
        def __getitem__(self, i):
            return self.windows[i]
    return training_batches(Windows(), batch_size=batch_size, shuffle=False, device=device, num_workers=0)


def validation_detection(model, bundle, profile, spec, folder, device):
    """Secondary OPS-SAT checkpoint criterion, never a test-driven choice."""
    from tranad_plus.core.evaluation.scoring import collect_outputs, reconstruction_scores
    from tranad_plus.core.evaluation.metrics import continuous_metrics
    from tranad_plus.campaigns.temporal.evaluation import bf200
    from tranad_plus.campaigns.thresholds.metrics import point_metrics
    from tranad_plus.campaigns.exploration.progress import scoring_bars
    state, was_training = rng_state(), model.training
    try:
        with scoring_bars():
            values = collect_outputs(model, bundle.datasets['validation'], device,
                                     spec['evaluation'].get('batch_size_by_dataset', {}).get(profile['dataset'],
                                     spec['evaluation']['batch_size']), 'V19 validation detection', folder/'nonfinite.json')
        alpha = spec['opssat_detection']['fixed_alpha'] if model.output_count == 2 else None
        scores = reconstruction_scores(values, alpha)
        labels = bundle.labels['validation']
        result = bf200(scores, labels, bundle.groups['validation'])
        result.update(alpha=alpha, selection_uses_test_labels=False,
                      raw_without_PA=point_metrics(labels, scores > result['bf_threshold']),
                      **continuous_metrics(scores, labels))
        return result
    finally:
        model.train(was_training)
        restore_rng(state)


def evaluate_mse(model, batches, transfer, device):
    """Fixed definition at every epoch, without dropout or epoch loss weights."""
    state, was_training = rng_state(), model.training
    total, count = None, 0
    model.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(device, non_blocking=transfer == 'cpu_pinned').permute(1, 0, 2)
                target = x[-1:]
                outputs = model(x, target)
                outputs = outputs if isinstance(outputs, tuple) else (outputs,)
                sums = torch.stack([(o-target).double().square().sum() for o in outputs])
                total = sums if total is None else total+sums
                count += target.numel()
        values = (total/count).cpu().tolist() if count else []
        if not values or not np.isfinite(values).all():
            raise FloatingPointError('Invalid held-out reconstruction error')
        return {'last_output_mse': float(values[-1]), 'output_mse': values, 'target_values': count}
    finally:
        model.train(was_training)
        restore_rng(state)  # Diagnostic DataLoader iteration must not change dropout RNG.


def _history(folder, committed):
    path = folder/'epochs.jsonl'
    if not path.exists():
        if committed:
            raise RuntimeError('Missing committed epoch telemetry')
        return
    rows = {}
    for line in path.read_text().splitlines():
        try:
            row = __import__('json').loads(line)
        except ValueError:
            continue
        if row['epoch'] <= committed:
            rows[row['epoch']] = row
    if set(rows) != set(range(1, committed+1)):
        raise RuntimeError('Missing committed epochs')
    with atomic_destination(path) as f:
        for epoch in sorted(rows):
            f.write(__import__('json').dumps(rows[epoch], allow_nan=False)+'\n')


def train(profile, arm, bundle, folder, signature, spec, device, *, stop_on_patience=False,
          stop_after_epoch=None):
    folder = Path(folder)
    checkpoints = folder/'checkpoints'
    recovery = checkpoints/'resume.ckpt'
    done = folder/'training.json'
    if done.exists():
        import json
        value = json.loads(done.read_text())
        if value['signature'] != signature:
            raise RuntimeError('Training signature mismatch')
        return value
    seed_all(profile['seed'])
    model = configured_model(profile, arm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=profile['lr'], weight_decay=arm['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=.9)
    batches, transfer = training_batches(bundle.datasets['train'], batch_size=profile['batch_sz'],
                                        shuffle=False, device=device, num_workers=0)
    diagnostic_batch = spec['evaluation'].get('batch_size_by_dataset', {}).get(profile['dataset'], spec['evaluation']['batch_size'])
    diagnostic_loaders = {split: nominal_loader(bundle, split, diagnostic_batch, device)
                          for split in ('train', 'validation')}
    policy = EarlyStopping(**{k: spec['early_stopping'][k]
                              for k in ('patience', 'min_epochs', 'relative_min_delta')})
    start, cumulative, snapshots = 0, 0., {}
    detection_best, detection_state = None, None
    fixed_epoch = profile.get('fixed_epoch_checkpoint', spec['fixed_epoch_checkpoint'])
    fixed_key = f'fixed_{fixed_epoch}' if 'fixed_epoch_checkpoint' in profile else 'fixed_50'
    if recovery.exists():
        saved = torch.load(recovery, map_location='cpu', weights_only=True)
        if saved['signature'] != signature or saved['profile'] != profile or saved['arm'] != arm:
            raise RuntimeError('Recovery source/data/profile mismatch')
        model.load_state_dict(saved['model_state'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state'])
        scheduler.load_state_dict(saved['scheduler_state'])
        policy = EarlyStopping(**saved['policy'])
        restore_rng(saved['rng_state'])
        start, cumulative, snapshots = saved['epoch'], saved['seconds'], saved['snapshots']
        detection_best, detection_state = saved.get('detection_best'), saved.get('detection_model_state')
        if scheduler.last_epoch != start:
            raise RuntimeError('Scheduler recovery mismatch')
    _history(folder, start)
    # If a crash occurred between writing a best snapshot and committing resume,
    # restore the authoritative best state embedded in that same resume file.
    if recovery.exists() and saved.get('best_model_state') is not None:
        save_torch(checkpoints/'early_stopped.ckpt', {
            'signature': signature, 'profile': profile, 'arm': arm,
            'epoch': policy.best_epoch, 'model_state': saved['best_model_state']})
    best_state = saved.get('best_model_state') if recovery.exists() else None
    if detection_state is not None:
        save_torch(checkpoints/'validation_detection.ckpt', {
            'signature': signature, 'profile': profile, 'arm': arm,
            'epoch': snapshots['validation_detection'], 'model_state': detection_state})
    cuda = torch.device(device).type == 'cuda'
    title = (f"V19 TRAIN {profile['dataset']}/{profile['variant']} W{profile['window_sz']} "
             f"FF{profile['dim_feedforward']} s{profile['seed']} wd={arm['weight_decay']:g}")
    with bar(total=profile['epochs'], initial=start, desc=title, position=1, unit='epoch', leave=False) as progress:
        for epoch in range(start+1, profile['epochs']+1):
            if stop_on_patience and policy.stop_epoch is not None:
                break
            if cuda:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            model.train()
            loss_total, count = torch.zeros((), device=device, dtype=torch.float64), 0
            with bar(batches, total=len(batches), desc=f'TRAIN batches | epoch {epoch}',
                     position=2, unit='batch', leave=False) as batch_progress:
                for batch in batch_progress:
                    x = batch[0] if isinstance(batch, (tuple, list)) else batch
                    x = x.to(device, non_blocking=transfer == 'cpu_pinned').permute(1, 0, 2)
                    target = x[-1:]
                    loss = reconstruction_loss(model(x, target), target, epoch,
                                               implementation='square' if profile['dataset'] == 'OPS-SAT' else 'mse_loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    loss_total.add_(loss.detach().double())
                    count += 1
            loss_value = float((loss_total/count).item()) if count else float('nan')
            if not np.isfinite(loss_value):
                raise FloatingPointError('Nonfinite training loss; last durable recovery preserved')
            scheduler.step()
            optimization_done = time.perf_counter()
            diagnostics = {split: evaluate_mse(model, loader, path, device)
                           for split, (loader, path) in diagnostic_loaders.items()}
            improved = policy.step(epoch, diagnostics['validation']['last_output_mse'])
            payload = {'signature': signature, 'profile': profile, 'arm': arm,
                       'epoch': epoch, 'model_state': model.state_dict()}
            if improved:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                save_torch(checkpoints/'early_stopped.ckpt', payload)
                snapshots['early_stopped'] = epoch
            if epoch == fixed_epoch:
                save_torch(checkpoints/f'{fixed_key}.ckpt', payload)
                snapshots[fixed_key] = epoch
            detection = None
            if profile['dataset'] == 'OPS-SAT' and spec.get('opssat_detection') and (
                    epoch == 1 or epoch % spec['opssat_detection']['every_epochs'] == 0 or epoch == profile['epochs']):
                detection = validation_detection(model, bundle, profile, spec, folder, device)
                if detection_best is None or detection['bf_f1'] > detection_best['bf_f1']:
                    detection_best = detection
                    detection_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    save_torch(checkpoints/'validation_detection.ckpt', payload)
                    snapshots['validation_detection'] = epoch
            if cuda:
                torch.cuda.synchronize(device)
            seconds = time.perf_counter()-started
            cumulative += seconds
            row = {'epoch': epoch, 'training_objective': loss_value,
                   'train_nominal': diagnostics['train'], 'validation_nominal': diagnostics['validation'],
                   'policy': policy.state(), 'learning_rate_after_step': optimizer.param_groups[0]['lr'],
                   'seconds_including_diagnostics': seconds,
                   'optimization_seconds': optimization_done-started,
                   'validation_detection': detection,
                   'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if cuda else None}
            append_json(folder/'epochs.jsonl', row)
            save_torch(recovery, {**payload, 'optimizer_state': optimizer.state_dict(),
                       'scheduler_state': scheduler.state_dict(), 'rng_state': rng_state(),
                       'policy': policy.state(), 'seconds': cumulative, 'snapshots': snapshots,
                       'best_model_state': best_state, 'detection_best': detection_best,
                       'detection_model_state': detection_state})
            print(f"[TRAIN] {profile['identifier']} | epoch {epoch}/{profile['epochs']} | "
                  f"train_MSE={diagnostics['train']['last_output_mse']:.7g} "
                  f"validation_MSE={diagnostics['validation']['last_output_mse']:.7g} "
                  f"selected_epoch={policy.best_epoch} stop_epoch={policy.stop_epoch} | {seconds:.2f}s", flush=True)
            progress.update(1)
            progress.set_postfix(train=f"{diagnostics['train']['last_output_mse']:.4g}",
                                 val=f"{policy.best:.4g}", best=policy.best_epoch,
                                 bad=policy.bad_epochs, stop=policy.stop_epoch or '-', epoch_s=f'{seconds:.2f}')
            record(folder, 'training', identifier=profile['identifier'], **row,
                   remaining_seconds_estimate=seconds*(profile['epochs']-epoch))
            if stop_after_epoch and epoch >= stop_after_epoch:
                raise TrainingInterrupted(f'V19 recovery committed at epoch {epoch}')
    saved = torch.load(recovery, map_location='cpu', weights_only=True)
    save_torch(checkpoints/'final.ckpt', {k: saved[k] for k in ('signature', 'profile', 'arm', 'epoch', 'model_state')})
    snapshots['final'] = saved['epoch']
    result = {'signature': signature, 'snapshots': snapshots, 'policy': policy.state(),
              'training_seconds_including_diagnostics': cumulative,
              'mode': 'stop_on_patience' if stop_on_patience else 'audit_full_horizon',
              'monitor_uses_test': False, 'monitor_uses_validation_anomalies': False,
              'secondary_detection_monitor': detection_best,
              'nominal_diagnostic_values': {s: int((np.asarray(bundle.labels[s]) == 0).sum())
                                            for s in ('train', 'validation')}}
    write_json(done, result)
    return result
