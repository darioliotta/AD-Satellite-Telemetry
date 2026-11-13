"""Three terminal rows: campaign, current phase, current batch loop."""
from contextlib import contextmanager, redirect_stdout
import os
from pathlib import Path
import sys
import time

from tranad_plus.core.common import now, write_json, append_json, save_npz

NAMES = {'reference': 'TranAD+', 'shared_decoder': 'One-decoder',
         'single_pass': 'One-pass', 'encoder_only': 'Encoder-only'}


def bar(iterable=None, **kw):
    from tqdm import tqdm
    kw.setdefault('bar_format','{desc} {n_fmt}/{total_fmt} |{bar:7}| [{elapsed}<{remaining}]{postfix}')
    return tqdm(iterable, dynamic_ncols=True, mininterval=.7, ascii=True,
                disable=os.environ.get('V16_NO_PROGRESS') == '1', **kw)


def announce(text):
    from tqdm import tqdm
    stream = sys.stdout
    if isinstance(stream, Log):
        stream.file.write(text+'\n')
        stream.file.flush()
        stream = stream.stream
    tqdm.write(text, file=stream)


def record(folder, phase, **fields):
    value={'updated': now(), 'phase': phase, **fields}
    write_json(Path(folder)/'progress.json',value)
    append_json(Path(folder)/'events.jsonl',value)


class Log:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file
    def write(self, value):
        self.file.write(value)
        self.file.flush()
        # Routine loader/epoch chatter stays in run.log. The runner explicitly
        # displays configuration headers and bars through announce().
        return len(value)
    def flush(self):
        self.file.flush()
    def isatty(self):
        return False


@contextmanager
def logged(folder):
    with (Path(folder)/'run.log').open('a', buffering=1) as stream:
        with redirect_stdout(Log(sys.stdout, stream)):
            yield


@contextmanager
def scoring(title):
    from tranad_plus.core.evaluation import scoring as module
    previous = module.progress_batches
    module.progress_batches = lambda values, title: bar(values, total=len(values), desc=title,
                                                         position=2, leave=False, unit='batch')
    try:
        yield
    finally:
        module.progress_batches = previous


@contextmanager
def threshold_search(folder, total, title):
    begun=time.perf_counter();previous_step=begun
    with bar(total=total, desc=title, position=1, leave=False, unit='alpha') as progress:
        def update(index, alpha, bf, pot):
            nonlocal previous_step
            current=time.perf_counter()
            progress.update(1)
            progress.set_postfix(BF=f'{bf:.4f}', POT='NA' if pot is None else f'{pot:.4f}')
            record(folder, 'threshold_search', description=title, alpha=alpha,
                   alphas_complete=index+1, alphas_total=total,
                   last_alpha_seconds=current-previous_step,total_seconds=current-begun)
            previous_step=current
        yield update


@contextmanager
def metric_bars():
    from tranad_plus.campaigns.temporal import evaluation
    old = evaluation.alpha_search
    evaluation.alpha_search = threshold_search
    try:
        yield
    finally:
        evaluation.alpha_search = old


@contextmanager
def native_training(folder, profile, device=None):
    import torch
    from tranad_plus.core.training import engine
    resume = Path(folder)/'checkpoints/resume.ckpt'
    initial = torch.load(resume, map_location='cpu', weights_only=True)['epoch']+1 if resume.exists() else 0
    old_emit, old_batches, old_loss = engine.emit_epoch, engine.training_batches, engine.reconstruction_loss
    losses=[]
    current, times = initial, []
    with bar(total=profile['epochs'], initial=initial, desc='SMD TRAIN | epochs', position=1,
             unit='epoch', leave=False) as epochs:
        class Batches:
            def __init__(self, source):self.source=source
            def __len__(self):return len(self.source)
            def __iter__(self):
                yield from bar(self.source, total=len(self.source), position=2, leave=False, unit='batch',
                               desc=f'SMD epoch {current+1}/{profile["epochs"]}')
        def batches(*args, **kw):
            source, transfer = old_batches(*args, **kw)
            return Batches(source), transfer
        def emit(**values):
            nonlocal current
            old_emit(**values)
            current=values['epoch']; times.append(values['seconds'])
            epochs.update(current-epochs.n)
            epochs.set_postfix(L=f'{values["loss"]:.4g}', s=f'{values["seconds"]:.1f}')
            record(folder, 'smd_training', epoch=current, epochs=profile['epochs'],
                   loss=values['loss'], remaining_seconds_estimate=sum(times[-5:])/len(times[-5:])*(profile['epochs']-current))
            from .resources import process_resources,analytic_operations
            append_json(Path(folder)/'epoch_resources.jsonl',{'epoch':current,**process_resources(device or 'cpu'),
                         'architecture_forward_MACs_this_epoch':analytic_operations(profile)['matmul_MACs_per_window']*values['windows']})
            if losses:
                save_npz(Path(folder)/f'batch_losses/epoch_{current:03d}.npz',loss=torch.stack(losses).cpu().numpy())
                losses.clear()
        def objective(*args,**kwargs):
            value=old_loss(*args,**kwargs);losses.append(value.detach());return value
        engine.emit_epoch, engine.training_batches, engine.reconstruction_loss = emit, batches, objective
        try:yield
        finally:engine.emit_epoch, engine.training_batches, engine.reconstruction_loss = old_emit, old_batches, old_loss
