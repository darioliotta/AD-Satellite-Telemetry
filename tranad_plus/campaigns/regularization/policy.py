"""Checkpoint decisions depend exclusively on finite nominal-validation MSE."""
from dataclasses import dataclass, asdict
import math


@dataclass
class EarlyStopping:
    patience: int = 10
    min_epochs: int = 20
    relative_min_delta: float = .001
    best: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0
    stop_epoch: int | None = None

    def __post_init__(self):
        if self.patience < 1 or self.min_epochs < 1 or not 0 <= self.relative_min_delta < 1:
            raise ValueError('Invalid early-stopping settings')

    def step(self, epoch, value):
        if epoch < 1 or not math.isfinite(value) or value < 0:
            raise ValueError('Finite nonnegative MSE and positive epoch required')
        if self.stop_epoch is not None:
            return False  # Never select a checkpoint using observations after stopping.
        improved = self.best is None or value < self.best * (1-self.relative_min_delta)
        if improved:
            self.best, self.best_epoch, self.bad_epochs = float(value), epoch, 0
        else:
            self.bad_epochs += 1
        if epoch >= self.min_epochs and self.bad_epochs >= self.patience:
            self.stop_epoch = epoch
        return improved

    def state(self):
        return asdict(self)
