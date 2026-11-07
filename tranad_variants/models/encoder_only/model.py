import torch
from ..reference.model import ReferenceTranAD


class EncoderOnlyTranAD(ReferenceTranAD):
    output_count = 1
    variant = "encoder_only"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.transformer_decoder1
        del self.transformer_decoder2

    def forward(self, src, tgt):
        _, memory = self.encode(src, torch.zeros_like(src), tgt)
        return self.fcn(memory[-1:, :, :])
