"""Run the two SWaT20 seed replicas with progress bars and durable outputs.

From tranad_plus: python -u run_swat20.py
Preview without loading data or writing results: python run_swat20.py --dry-run
Uses the existing training/evaluation code; numerical settings are unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


CODE = Path(__file__).resolve().parent
HP = dict(dataset_str="SWaT", features=51, window_sz=20,
          dim_feedforward=816, num_layers=4, batch_sz=2048,
          lr=0.0001, epochs=100, pot_q=0.001)


def identifier(seed):
    return f"papercheck_SWaT_bf_w20_ff816_l4_bs2048_s{seed}"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", choices=("both", "train", "evaluate"), default="both")
    parser.add_argument("--output-root", type=Path, default=CODE.parent / "results")
    parser.add_argument("--data-root", type=Path, default=CODE.parent / "data")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", choices=("train", "evaluate"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct nonnegative integers")
    if args.worker and len(args.seeds) != 1:
        parser.error("A worker requires exactly one seed")
    args.output_root = args.output_root.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    if args.output_root == CODE or CODE in args.output_root.parents:
        parser.error("Results must be outside tranad_plus")
    return args


def environment(args):
    env = os.environ.copy()
    env.pop("TRANADPLUS_CAMPAIGN_PLAN", None)
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        "TRANADPLUS_OUTPUT_ROOT": str(args.output_root),
        "TRANADPLUS_RESULTS_VERSION": "v12",
        "TRANADPLUS_CAMPAIGN_SUBDIR": "",
        "TRANADPLUS_DATA_ROOT": str(args.data_root),
        "TRANADPLUS_PREPROCESSED_DIR": str(args.data_root / "processed/tranad_plus"),
        "TRANADPLUS_NATIVE_CHECKPOINT_DIR": str(args.output_root / "v12/SWaT"),
        "TRANADPLUS_NATIVE_PICKLE_DIR": str(args.output_root / "v12/SWaT/_native_pickles"),
        "TRANADPLUS_CUDA_PRECISION": "faithful",
        "TRANADPLUS_ADAM_IMPL": "standard",
        "TRANADPLUS_GPU_CACHE": "auto",
        "TOAST_FORCE_TQDM": "0" if args.no_progress else "1",
        "TOAST_NO_TQDM": "1" if args.no_progress else "0",
    })
    return env


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=float) + "\n")
    temporary.replace(path)


def duration(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def check_checkpoint(path, seed):
    import torch
    saved = torch.load(path, map_location="cpu", weights_only=False)
    hp = saved["hp_dict"]
    for key, expected in HP.items():
        if key != "dim_feedforward" and hp.get(key) != expected:
            raise ValueError(f"Checkpoint {path}: {key}={hp.get(key)!r}, expected {expected!r}")
    if hp["identifier"] != identifier(seed):
        raise ValueError(f"Checkpoint identifier differs: {hp['identifier']}")
    state = saved["model_states"]
    if state["transformer_encoder.layers.0.linear1.weight"].shape[0] != 816:
        raise ValueError("Checkpoint effective feedforward width is not 816")
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("Checkpoint contains nonfinite weights")
    return saved


class EpochBatches:
    """Wrap iteration only; preserve tensors, batches, ordering and CUDA work."""
    def __init__(self, batches, *, seed, start_epoch, disabled):
        self.batches = batches
        self.seed = seed
        self.epoch = start_epoch
        self.disabled = disabled

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        from tqdm import tqdm
        self.epoch += 1
        with tqdm(self.batches, total=len(self.batches), unit="batch",
                  desc=f"Seed {self.seed} epoca {self.epoch}/100", position=1,
                  leave=False, dynamic_ncols=True, mininterval=1,
                  disable=self.disabled) as progress:
            yield from progress


def train(args, seed, folder):
    import torch
    from tqdm import tqdm
    import gpu_batches
    import resumable_training as training

    name = identifier(seed)
    final = folder / f"{name}.ckpt"
    if final.exists():
        check_checkpoint(final, seed)
        print(f"Training gia completato e checkpoint verificato: {name}", flush=True)
        return

    resume = folder / f"{name}.resume.ckpt"
    start = 0
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        start = int(saved["epoch"]) + 1
        del saved
    original_batches = gpu_batches.training_batches
    original_emit = training.emit_epoch
    previous_argv = sys.argv
    with tqdm(total=100, initial=start, unit="epoca", position=0,
              desc=f"SWaT20 seed {seed}", dynamic_ncols=True,
              disable=args.no_progress) as epochs:
        def batches(*positional, **keywords):
            loader, mode = original_batches(*positional, **keywords)
            return EpochBatches(loader, seed=seed, start_epoch=start,
                                disabled=args.no_progress), mode

        def emit(**values):
            with tqdm.external_write_mode(file=sys.stderr):
                original_emit(**values)
            epochs.set_postfix(loss=f"{values['loss']:.6g}",
                               sec_epoca=f"{values['seconds']:.1f}", refresh=False)
            epochs.update(values["epoch"] - epochs.n)

        gpu_batches.training_batches = batches
        training.emit_epoch = emit
        sys.argv = [str(CODE / "resumable_training.py"),
                    "--model_str", "TranAD", "--identifier", name,
                    "--device", args.device, "--scaling", "True",
                    "--shuffle", "0", "--padding", "1", "--num_workers", "0",
                    "--seed", str(seed), "--checkpoint-every", "1"]
        for key, value in HP.items():
            if key != "features":
                sys.argv.extend([f"--{key}", str(value)])
        try:
            training.main()
        finally:
            gpu_batches.training_batches = original_batches
            training.emit_epoch = original_emit
            sys.argv = previous_argv


def evaluate(args, seed, folder):
    import pickle
    import numpy as np
    import torch
    from tqdm import tqdm
    import config
    import console_telemetry
    import evaluation_progress
    import paper_metrics as metrics

    name = identifier(seed)
    check_checkpoint(folder / f"{name}.ckpt", seed)
    config.PROFILES.setdefault("SWaT", {})[name] = {}
    original_metrics = metrics.native_metrics_from_arrays
    original_emit = console_telemetry.emit_evaluation
    original_load = torch.load
    previous_argv = sys.argv
    alpha_progress = None
    best_bf = best_pot = 0.0

    def load_on_cpu(*positional, **keywords):
        keywords["map_location"] = "cpu"
        keywords.setdefault("weights_only", False)
        return original_load(*positional, **keywords)

    def checked(train_arrays, test_arrays, alpha, q):
        nonlocal alpha_progress, best_bf, best_pot
        # Start this ETA after inference: all 21 steps now do comparable work.
        if alpha_progress is None:
            alpha_progress = tqdm(total=21, desc=f"Seed {seed} soglie/alpha",
                                  unit="alpha", dynamic_ncols=True,
                                  disable=args.no_progress)
        for split, arrays in (("train", train_arrays), ("test", test_arrays)):
            if not np.isfinite(metrics.anomaly_scores(arrays, alpha)).all():
                raise FloatingPointError(f"Nonfinite {split} scores at alpha={alpha}")
        result, dense = original_metrics(train_arrays, test_arrays, alpha, q)
        best_bf = max(best_bf, result["bf_f1"])
        best_pot = max(best_pot, result["pot_f1"])
        alpha_progress.set_postfix(BF=f"{best_bf:.4f}", POT=f"{best_pot:.4f}",
                                   refresh=False)
        alpha_progress.update(1)
        return result, dense

    def emit(**values):
        with tqdm.external_write_mode(file=sys.stderr):
            original_emit(**values)

    torch.load = load_on_cpu
    metrics.native_metrics_from_arrays = checked
    console_telemetry.emit_evaluation = emit
    sys.argv = [str(CODE / "evaluation_progress.py"), "--identifier", name,
                "--device", args.device, "--alpha_search", "True"]
    try:
        evaluation_progress.main()
    finally:
        if alpha_progress is not None:
            alpha_progress.close()
        torch.load = original_load
        metrics.native_metrics_from_arrays = original_metrics
        console_telemetry.emit_evaluation = original_emit
        sys.argv = previous_argv

    summary = {"identifier": name, "seed": seed, "hyperparameters": HP}
    for method in ("pot", "bf"):
        with (folder / "_native_pickles" / f"{name}_{method}.pkl").open("rb") as stream:
            result = pickle.load(stream)
        summary[method] = {key: value for key, value in result.items()
                           if key.startswith(method + "_") or key in ("alpha", "beta")}
    write_json(folder / f"{name}_summary.json", summary)
    print("REPLICA COMPLETATA\n" + json.dumps(summary, indent=2), flush=True)


def worker(args):
    import torch
    device = torch.device(args.device)
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device unavailable: {args.device}")
    folder = args.output_root / "v12/SWaT"
    if args.worker == "train":
        train(args, args.seeds[0], folder)
    else:
        evaluate(args, args.seeds[0], folder)


def main():
    args = arguments()
    os.chdir(CODE)
    if args.worker:
        os.environ.update(environment(args))
        worker(args)
        return 0

    stages = ("train", "evaluate") if args.stage == "both" else (args.stage,)
    plan = [(seed, stage) for seed in args.seeds for stage in stages]
    folder = args.output_root / "v12/SWaT"
    print(f"SWaT20: seeds {args.seeds}; W20 FF816 L4 BS2048, 100 epoche", flush=True)
    print(f"Risultati: {folder}", flush=True)
    if args.dry_run:
        for index, (seed, stage) in enumerate(plan, 1):
            print(f"[{index}/{len(plan)}] {stage.upper()} {identifier(seed)}")
        return 0

    data = args.data_root / "processed/tranad_plus/SWaT"
    for part in ("Both/Tracks", "Train/Tracks", "Train/Labels", "Test/Tracks", "Test/Labels"):
        if not any((data / part).glob("*.npy")):
            raise FileNotFoundError(f"Missing SWaT data: {data / part}")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "_native_pickles").mkdir(exist_ok=True)
    import fcntl
    with (folder / "_swat20.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another run_swat20.py campaign is already running") from None
        return campaign(args, plan, folder)


def campaign(args, plan, folder):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = folder / f"_swat20_campaign_{stamp}.json"
    report = {"status": "running", "seeds": args.seeds, "stages": []}
    started = time.perf_counter()
    for index, (seed, stage) in enumerate(plan, 1):
        name = identifier(seed)
        log = folder / f"{name}_{stage}.log"
        record = dict(identifier=name, stage=stage, status="running", log=str(log))
        report["stages"].append(record)
        write_json(report_path, report)
        print(f"\n[{index}/{len(plan)}] {stage.upper()} seed {seed} | "
              f"tempo totale {duration(time.perf_counter() - started)}", flush=True)
        command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                   "--worker", stage, "--seeds", str(seed), "--device", args.device,
                   "--output-root", str(args.output_root), "--data-root", str(args.data_root)]
        if args.no_progress:
            command.append("--no-progress")
        phase_start = time.perf_counter()
        try:
            # Stream raw chunks so tqdm's carriage returns remain live in tmux.
            with log.open("ab", buffering=0) as stream, subprocess.Popen(
                command, cwd=CODE, env=environment(args), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, bufsize=0
            ) as process:
                try:
                    while chunk := process.stdout.read(65536):
                        stream.write(chunk)
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            record.update(status="complete" if code == 0 else "failed", exit_code=code)
        except BaseException:
            record["status"] = "interrupted"
            report["status"] = "interrupted"
            raise
        finally:
            record["seconds"] = time.perf_counter() - phase_start
            report["elapsed_seconds"] = time.perf_counter() - started
            if record["status"] == "failed":
                report["status"] = "failed"
            write_json(report_path, report)
        print(f"Fase {record['status']}: {duration(record['seconds'])}", flush=True)
        if code:
            print(f"Interrotto. Consultare: {log}", flush=True)
            return 1
    report["status"] = "complete"
    write_json(report_path, report)
    print(f"\nCOMPLETATO in {duration(time.perf_counter() - started)}\n"
          f"Report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrotto: rilanciare lo stesso comando per riprendere dal checkpoint.", flush=True)
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
