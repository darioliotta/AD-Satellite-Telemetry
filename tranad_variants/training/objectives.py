import torch.nn.functional as functional


def reconstruction_loss(outputs, target, epoch_one_based, *, implementation="square"):
    if epoch_one_based < 1:
        raise ValueError("Epoch numbers are one-based")
    if implementation not in ("square", "mse_loss"):
        raise ValueError("Unknown MSE kernel")
    def squared_error(value):
        return (functional.mse_loss(value, target, reduction="none")
                if implementation == "mse_loss" else (value - target).square())
    if isinstance(outputs, tuple):
        if len(outputs) != 2:
            raise ValueError("Expected exactly two reconstruction outputs")
        first, second = outputs
        weight = 1.0 / epoch_one_based
        return (weight * squared_error(first) + (1.0 - weight) * squared_error(second)).mean()
    return squared_error(outputs).mean()
