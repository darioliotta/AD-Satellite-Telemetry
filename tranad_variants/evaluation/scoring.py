import time
import numpy as np
import torch

from ..bootstrap import activate_baseline
from ..common import write_json
activate_baseline()
from gpu_batches import training_batches
from console_telemetry import progress_batches


def collect_outputs(model, dataset, device, batch_size, title, diagnostic_path):
    batches, transfer = training_batches(dataset, batch_size=batch_size, shuffle=False,
                                        device=device, num_workers=0)
    parts = [[] for _ in range(model.output_count)]
    targets = []
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for batch in progress_batches(batches, title=title):
            windows = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = windows.to(device, non_blocking=(transfer == "cpu_pinned")).permute(1, 0, 2)
            y = x[-1:, :, :]
            outputs = model(x, y)
            outputs = outputs if isinstance(outputs, tuple) else (outputs,)
            if len(outputs) != model.output_count:
                raise ValueError("Output count differs from architecture contract")
            for part, output in zip(parts, outputs):
                if output.shape != y.shape:
                    raise ValueError("Reconstruction/target shapes differ")
                part.append(output[0].cpu().numpy())
            targets.append(y[0].cpu().numpy())
    if cuda:
        torch.cuda.synchronize(device)
    result = {"outputs": [np.concatenate(p, axis=0) for p in parts],
              "target": np.concatenate(targets, axis=0)}
    arrays = result["outputs"] + [result["target"]]
    diagnostics = [{"array": f"output_{i + 1}" if i < model.output_count else "target",
                    "nonfinite": int((~np.isfinite(a)).sum()), "values": int(a.size)}
                   for i, a in enumerate(arrays)]
    if any(r["nonfinite"] for r in diagnostics):
        write_json(diagnostic_path, {"status": "invalid_nonfinite_outputs", "split": title,
                                      "arrays": diagnostics})
        raise FloatingPointError(f"Nonfinite outputs: see {diagnostic_path}")
    if len(result["target"]) != len(dataset):
        raise AssertionError("Scored points differ from dataset length")
    result["performance"] = {"seconds_including_transfer": time.perf_counter() - started,
                              "points": len(dataset), "batch_size": batch_size}
    return result


def reconstruction_scores(outputs, alpha=None, *, float64=False):
    values = outputs["outputs"]
    target = outputs["target"]
    if float64:
        values = [v.astype(np.float64) for v in values]
        target = target.astype(np.float64)
    if len(values) == 1:
        if alpha is not None:
            raise ValueError("Alpha is not applicable to single-output models")
        reconstruction = values[0]
    else:
        if alpha is None or not 0 <= float(alpha) <= 1:
            raise ValueError("Two-output models require alpha in [0,1]")
        reconstruction = float(alpha) * values[0] + (1.0 - float(alpha)) * values[1]
    scores = np.mean((reconstruction - target) ** 2, axis=1)
    if not len(scores) or not np.isfinite(scores).all():
        raise FloatingPointError("Nonfinite/empty reconstruction scores")
    return scores
