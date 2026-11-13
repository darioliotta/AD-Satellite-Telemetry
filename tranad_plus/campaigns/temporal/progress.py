"""Readable configuration, epoch, batch and alpha progress with durable timing."""
from contextlib import contextmanager
from pathlib import Path
import time

from tranad_plus.campaigns.exploration.progress import bar, logged, record, scoring_bars


@contextmanager
def training(folder, profile):
    import torch
    from tranad_plus.core.training import engine
    resume = Path(folder) / "checkpoints/resume.ckpt"
    start = torch.load(resume, map_location="cpu", weights_only=True)["epoch"] + 1 if resume.exists() else 0
    old_emit, old_batches = engine.emit_epoch, engine.training_batches
    epoch = start
    times = []
    title = f"V15 {profile['dataset']} {profile['variant']} W{profile['window_sz']} {profile['input_condition']} s{profile['seed']}"
    with bar(total=profile["epochs"], initial=start, desc=title, position=1, unit="epoch", leave=False) as epochs:
        class Batches:
            def __init__(self, source):
                self.source = source
            def __len__(self):
                return len(self.source)
            def __iter__(self):
                yield from bar(self.source, total=len(self.source), position=2,
                               desc=f"Batches epoch {epoch+1}/{profile['epochs']}", unit="batch", leave=False)
        def batches(*args, **kwargs):
            source, transfer = old_batches(*args, **kwargs)
            return Batches(source), transfer
        def emit(**values):
            nonlocal epoch
            old_emit(**values)
            epoch = values["epoch"]
            times.append(values["seconds"])
            epochs.update(epoch - epochs.n)
            epochs.set_postfix(loss=f"{values['loss']:.5g}", epoch_s=f"{values['seconds']:.2f}")
            record(folder, "training", epoch=epoch, epochs=profile["epochs"],
                   identifier=profile["identifier"], loss=values["loss"],
                   latest_epoch_seconds=values["seconds"],
                   remaining_seconds_estimate=sum(times[-5:])/len(times[-5:])*(profile["epochs"]-epoch))
        engine.emit_epoch, engine.training_batches = emit, batches
        try:
            yield
        finally:
            engine.emit_epoch, engine.training_batches = old_emit, old_batches


@contextmanager
def alpha_search(folder, total, title):
    started = time.monotonic()
    with bar(total=total, desc=title, position=1, unit="alpha", leave=False) as progress:
        def advance(index, alpha, bf, pot):
            progress.update(1)
            progress.set_postfix(alpha=alpha, BF=f"{bf:.4f}", POT="NA" if pot is None else f"{pot:.4f}")
            record(folder, "threshold_search", alpha_index=index+1, alphas=total,
                   elapsed_seconds=time.monotonic()-started)
        yield advance
