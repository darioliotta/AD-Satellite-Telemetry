"""V14 progress bars and durable timing, outside the frozen v13 implementation."""
from contextlib import contextmanager, redirect_stdout
import json
from pathlib import Path
import os
import sys
import time
import traceback

from tranad_variants.common import now, write_json

_observer = None


def enabled():
    return os.environ.get("TOAST_NO_TQDM") != "1"


def bar(iterable=None, *, total=None, desc, position=0, unit="job", initial=0, leave=True):
    from tqdm import tqdm
    return tqdm(iterable, total=total, desc=desc, position=position, unit=unit,
                initial=initial, leave=leave, disable=not enabled(), dynamic_ncols=True,
                mininterval=.5, ascii=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")


def record(folder, phase, **values):
    write_json(Path(folder) / "progress.json", {"phase": phase, "updated": now(), **values})


class LogStream:
    """Readable full text log; suppress repetitive epoch/alpha lines on screen."""
    def __init__(self, console, logfile):
        self.console, self.logfile, self.pending = console, logfile, ""

    def write(self, text):
        self.logfile.write(text)
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if _observer is not None:
                _observer(line)
            if not enabled() or not line.startswith(("[TRAIN]", "[EVAL] alpha=")):
                if enabled():
                    from tqdm import tqdm
                    tqdm.write(line, file=self.console)
                else:
                    self.console.write(line + "\n")
        return len(text)

    def flush(self):
        self.logfile.flush()
        self.console.flush()

    def isatty(self):
        return False


@contextmanager
def logged(path):
    with Path(path).open("a", buffering=1) as stream:
        with redirect_stdout(LogStream(sys.stdout, stream)):
            try:
                yield
            except Exception:
                traceback.print_exc(file=stream)
                raise


@contextmanager
def training(folder, parent_epochs, total_epochs, dataset):
    """Observe existing batches/epoch callbacks; preserve order and RNG calls."""
    from tranad_variants.training import engine
    resume = Path(folder) / "checkpoints/resume.ckpt"
    start = parent_epochs
    if resume.exists():
        import torch
        start = torch.load(resume, map_location="cpu", weights_only=True)["epoch"] + 1
    b = bar(total=total_epochs-parent_epochs, initial=start-parent_epochs,
            desc=f"V14 TRAIN {dataset} | epochs {parent_epochs+1}-{total_epochs}", position=1, unit="epoch", leave=False)
    old_emit, old_batches = engine.emit_epoch, engine.training_batches
    completed, durations, started = start, [], time.monotonic()

    def emit(**kw):
        nonlocal completed
        old_emit(**kw)  # Retains the exact existing text in run.log.
        completed = kw["epoch"]
        durations.append(kw["seconds"])
        b.update(completed - parent_epochs - b.n)
        b.set_postfix(loss=f"{kw['loss']:.5g}", epoch_s=f"{kw['seconds']:.2f}")
        record(folder, "training", epoch=completed, target_epoch=total_epochs,
               added_epochs_completed=completed-parent_epochs, loss=kw["loss"],
               latest_epoch_seconds=kw["seconds"], elapsed_session_seconds=time.monotonic()-started,
               remaining_seconds_estimate=(sum(durations[-5:])/len(durations[-5:]))*(total_epochs-completed))

    class Batches:
        def __init__(self, original):
            self.original = original
        def __len__(self):
            return len(self.original)
        def __iter__(self):
            yield from bar(self.original, total=len(self.original),
                           desc=f"Batches | epoch {completed+1}/{total_epochs}",
                           position=2, unit="batch", leave=False)

    def batches(*args, **kwargs):
        original, transfer = old_batches(*args, **kwargs)
        return Batches(original), transfer

    engine.emit_epoch, engine.training_batches = emit, batches
    try:
        yield
    finally:
        engine.emit_epoch, engine.training_batches = old_emit, old_batches
        b.close()


@contextmanager
def evaluation(folder, alphas):
    global _observer
    previous = _observer
    b = bar(total=alphas, desc="V14 evaluation | alpha search", position=1, unit="alpha", leave=False)
    started = time.monotonic()
    completed = 0
    def observe(line):
        nonlocal completed
        if line.startswith("[EVAL] alpha="):
            completed += 1
            b.update(1)
            record(folder, "evaluating", alphas_complete=completed, alphas_total=alphas,
                   elapsed_seconds=time.monotonic()-started)
    _observer = observe
    try:
        record(folder, "evaluating", alphas_complete=0, alphas_total=alphas)
        yield
    finally:
        _observer = previous
        b.close()


@contextmanager
def scoring_bars():
    """Place reused scoring bars below V14 job/condition progress."""
    from tranad_variants.evaluation import scoring
    previous = scoring.progress_batches
    scoring.progress_batches = lambda iterable, title: bar(iterable, total=len(iterable), desc=title,
                                                          position=2, unit="batch", leave=False)
    try:
        yield
    finally:
        scoring.progress_batches = previous
