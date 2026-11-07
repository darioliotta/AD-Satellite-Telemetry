import torch
from ..reference.model import ReferenceTranAD


class SinglePassTranAD(ReferenceTranAD):
    output_count = 1
    variant = "single_pass"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.transformer_decoder2

    def forward(self, src, tgt):
        # Keep 2F embedding and zero context, isolating removal of stage two.
        return self.fcn(self.transformer_decoder1(*self.encode(src, torch.zeros_like(src), tgt)))
