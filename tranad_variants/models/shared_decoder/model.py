from ..reference.model import ReferenceTranAD


class SharedDecoderTranAD(ReferenceTranAD):
    variant = "shared_decoder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Preserve both stages, self-conditioning, and the original loss.
        # Constructing the reference first preserves the retained weights' RNG
        # initialization relative to a paired reference run with the same seed.
        self.transformer_decoder2 = self.transformer_decoder1
