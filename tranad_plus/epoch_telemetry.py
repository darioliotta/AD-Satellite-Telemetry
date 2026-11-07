"""Append-only durable epoch instrumentation shared by both native training wrappers."""
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path


def write_epoch(dataset, identifier, epoch, loss, lr, elapsed, num_batches,
                num_windows, device, gpu_peak_allocated=None, gpu_peak_reserved=None):
    from config import RESULTS_DIR
    path = RESULTS_DIR / dataset / f'{identifier}_epochs.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import resource
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        max_rss_bytes = int(max_rss * 1024) if sys.platform.startswith('linux') else int(max_rss)
    except (ImportError, AttributeError):
        max_rss_bytes = None
    record = {
        'process_peak_rss_bytes': max_rss_bytes,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'epoch_one_based': int(epoch), 'loss_mean_of_batch_means': float(loss),
        'learning_rate_after_scheduler_step': float(lr),
        'elapsed_seconds': float(elapsed), 'batches': int(num_batches),
        'dataset_windows': int(num_windows), 'device': str(device),
        'gpu_peak_allocated_bytes': int(gpu_peak_allocated) if gpu_peak_allocated is not None else None,
        'gpu_peak_reserved_bytes': int(gpu_peak_reserved) if gpu_peak_reserved is not None else None,
    }
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    return path
