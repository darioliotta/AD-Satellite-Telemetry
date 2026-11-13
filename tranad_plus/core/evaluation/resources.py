"""Measured server forward latency and explicitly analytical matmul cost."""
import time
import numpy as np
import torch


def measure(model, profile, sample, device, settings):
    model.eval()
    x = sample.to(device).unsqueeze(1)
    target = x[-1:, :, :]
    cuda = torch.device(device).type == "cuda"
    with torch.inference_mode():
        for _ in range(settings["warmup"]):
            model(x, target)
        if cuda:
            torch.cuda.synchronize(device)
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        durations = []
        for _ in range(settings["iterations"]):
            if cuda:
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            model(x, target)
            if cuda:
                torch.cuda.synchronize(device)
            durations.append((time.perf_counter() - started) * 1000)
    f, w, h, layers = (profile[k] for k in ("features", "window_sz", "dim_feedforward", "num_layers"))
    d = 2 * f
    encoder = layers * (4*w*d*d + 2*w*w*d + 2*w*d*h)
    decoder = layers * (4*d*d + 2*d + (2+2*w)*d*d + 2*w*d + 2*d*h)
    head = d * f
    macs = encoder + head if model.variant == "encoder_only" else encoder + decoder + head
    if model.output_count == 2:
        macs *= 2
    return {"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "parameter_bytes_fp32": sum(p.numel() * p.element_size() for p in model.parameters()),
            "buffer_bytes": sum(b.numel() * b.element_size() for b in model.buffers()),
            "output_count": model.output_count,
            "batch_one_forward": {"mean_ms": float(np.mean(durations)), "median_ms": float(np.median(durations)),
                                  "p95_ms": float(np.quantile(durations, .95)),
                                  "warmup": settings["warmup"], "iterations": settings["iterations"],
                                  "scope": "model forward only; wall time includes launch/synchronization, excludes input transfer and scoring",
                                  "gpu_peak_incremental_allocated_bytes":
                                      max(0, torch.cuda.max_memory_allocated(device) - baseline) if cuda else None},
            "analytical_matmul_macs_per_window": int(macs),
            "macs_scope": "QKV/out projections, QK/AV, FFN and output linear maps; excludes biases, nonlinearities, softmax, elementwise operations and I/O",
            "hardware_scope": "Measured on the declared CPU/GPU; not onboard satellite performance"}
