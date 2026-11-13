"""Experimental input intervention; the four shared architectures stay unchanged."""
import torch
from torch import nn
from tranad_plus.core.models import build_model


class CurrentAccess(nn.Module):
    def __init__(self, base, hidden):
        super().__init__()
        self.base, self.hidden = base, hidden
        self.output_count, self.variant = base.output_count, base.variant
    def forward(self, src, target):
        if self.hidden:
            src = src.clone()
            src[-1] = 0
            # True current values must not enter decoder queries either.
            query = torch.zeros_like(target)
        else:
            query = target
        return self.base(src, query)


def build_control(profile):
    if profile['condition'] not in ('observed_current', 'hidden_current'):
        raise ValueError('Unknown current-sample intervention')
    return CurrentAccess(build_model(profile), profile['condition'] == 'hidden_current')
