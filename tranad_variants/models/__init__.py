from .reference.model import ReferenceTranAD
from .shared_decoder.model import SharedDecoderTranAD
from .single_pass.model import SinglePassTranAD
from .encoder_only.model import EncoderOnlyTranAD

MODELS = {"reference": ReferenceTranAD, "shared_decoder": SharedDecoderTranAD,
          "single_pass": SinglePassTranAD, "encoder_only": EncoderOnlyTranAD}


def build_model(profile):
    return MODELS[profile["variant"]](
        n_feats=profile["features"], window_sz=profile["window_sz"],
        batch_sz=profile["batch_sz"], dim_feedforward=profile["dim_feedforward"],
        num_encoder_layers=profile["num_layers"],
        num_decoder_layers=profile["num_layers"]).float()
