"""One dataset-independent console format for TranAD+ epoch telemetry.

Logging is not a source of truth: epoch_telemetry.py writes JSONL first.
"""
from __future__ import annotations


def epoch_line(*, dataset: str, identifier: str, epoch: int, epochs: int,
               loss: float, lr: float, seconds: float, batches: int, windows: int,
               peak_allocated_bytes: int | None = None) -> str:
    if not (epochs > 0 and 1 <= epoch <= epochs and batches > 0 and seconds >= 0):
        raise ValueError('Invalid epoch telemetry')
    rate = windows / seconds if seconds > 0 else float('nan')
    peak = (f'{peak_allocated_bytes / (1024 ** 3):.2f} GiB'
            if peak_allocated_bytes is not None else 'n/a')
    return (f'[TRAIN] {dataset:<7} | {identifier} | {epoch:03d}/{epochs:03d} '
            f'| loss={loss:.7g} | lr={lr:.3g} | {seconds:.3f}s '
            f'| batches={batches} | windows={windows} | {rate:,.0f} win/s '
            f'| GPU peak={peak}')


def emit_epoch(**kwargs):
    print(epoch_line(**kwargs), flush=True)


def evaluation_line(*, dataset: str, identifier: str, index: int, total: int, alpha: float) -> str:
    if total < 1 or not 1 <= index <= total or not 0 <= alpha <= 1:
        raise ValueError('Invalid evaluation progress')
    return (f'[EVAL]  {dataset:<7} | {identifier} | alpha {index:02d}/{total:02d}'
            f' | alpha={alpha:.2f} | beta={1-alpha:.2f}')


def emit_evaluation(**kwargs):
    print(evaluation_line(**kwargs), flush=True)


def progress_batches(iterable, *, title: str):
    """Optional read-only tqdm: no change to loader order, batches or numerics.

    Force display from v12 launcher, even when stdout is captured by tee/logging.
    Unit tests and library calls default to quiet unless attached to a TTY.
    """
    import os
    import sys
    from tqdm import tqdm
    disable = os.environ.get('TOAST_NO_TQDM') == '1' or not (
        os.environ.get('TOAST_FORCE_TQDM') == '1' or sys.stderr.isatty()
    )
    return tqdm(iterable, desc=title, total=len(iterable), unit='batch',
                ascii=True, ncols=96, mininterval=0.5, leave=False, disable=disable)
