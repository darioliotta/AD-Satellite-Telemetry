"""Evaluate original author weights on local data; no training or benchmark overwrites.

python -u verify_author_checkpoints.py
python verify_author_checkpoints.py --dry-run
python -u verify_author_checkpoints.py --cases wadi

Defaults: two MSL W50 configurations, SMD W20, WADI W50. WADI is evaluated
in both FP32 (original formula, including any genuine NaNs) and FP64.
The dense BF result is supplementary; reference comparisons use BF200.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import http.cookiejar
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request
import zipfile

CODE = Path(__file__).resolve().parent
REPO = "https://github.com/yyexela/TranADPlus"
CASES = {
    "msl50_old": {
        "file": "TranAD_MSL_lr0-0001_dff2x_wz50_model.ckpt",
        "drive_id": "1o3ap85Kke4FgIm3VvQQbW-Yy-jY6SKPk",
        "dataset": "MSL", "features": 55, "window": 50, "ff": 110, "layers": 1,
        "batch": 2048, "epochs": 50, "lr": 0.0001, "q": 0.01,
        "log": "TranAD_MSL_lr0-0001_dff2x_ws50.txt",
        "reference": {
            "pot": {"f1":0.8711340206185567,"precision":0.8321219226260258,"recall":0.9139840329642029,"ROC/AUC":0.9463431178102393},
            "bf": {"f1":0.6205815632089091,"precision":0.4565204199283937,"recall":0.968709760494463,"ROC/AUC":0.917754645257568},
        },
    },
    "msl50": {
        "file": "Aug-20-24_03-34-10-129705.ckpt",
        "drive_id": "1JU1EJpf-pUgF8bn5nLe5j1-smk4x5d6i",
        "dataset": "MSL", "features": 55, "window": 50, "ff": 880, "layers": 2,
        "batch": 2048, "epochs": 100, "lr": 0.0001, "q": 0.01,
        "log": "Aug-20-24_03-34-10-129705_eval.txt",
        "reference": {
            "pot": {"f1":0.828732956434985,"precision":0.8570642454257807,"recall":0.8022147823847541,"ROC/AUC":0.8933809905498736},
            "bf": {"f1":0.8195600136783313,"precision":0.7351738241308794,"recall":0.9258305433942827,"ROC/AUC":0.9436550429540388},
        },
    },
    "smd20": {
        "file": "Aug-20-24_00-59-24-881901.ckpt",
        "drive_id": "1CJjzBB1-5NLe4i8kAYI6MO4L23RkBFll",
        "dataset": "SMD", "features": 38, "window": 20, "ff": 76, "layers": 5,
        "batch": 8192, "epochs": 100, "lr": 0.0001, "q": 0.001,
        "log": "Aug-20-24_00-59-24-881901_eval.txt",
        "reference": {
            "pot": {"f1":0.7129061265981155,"precision":0.5784042125742804,"recall":0.9289159081646515,"ROC/AUC":0.9497881065939925},
            "bf": {"f1":0.8217416536852071,"precision":0.7466004329244602,"recall":0.9137005841597609,"ROC/AUC":0.9501312723459334},
        },
    },
    "wadi": {
        "file": "Aug-26-24_18-03-12-989875.ckpt",
        "drive_id": "1iXxV_28N4bZcPMvZBn4_OrdFArrcwnwE",
        "dataset": "WADI", "features": 128, "window": 50, "ff": 2048, "layers": 2,
        "batch": 1024, "epochs": 50, "lr": 0.001, "q": 0.001,
        "log": "Aug-26-24_18-03-12-989875_eval.txt",
        "reference": {
            "pot": {"f1":0.5084310441094361,"precision":0.5739316778015883,"recall":0.4563496040894056,"ROC/AUC":0.7177954967825793},
            "bf": {"f1":0,"precision":0,"recall":0,"ROC/AUC":0.5},
        },
    },
}


def clean(value):
    """Write strict JSON: retain nonfinite counts and represent NaN thresholds as null."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}
        self.inside = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and "download" in attrs.get("action", ""):
            self.action = attrs["action"]
            self.inside = True
        elif tag == "input" and self.inside and attrs.get("name"):
            self.fields[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.inside = False


def checkpoint_file(case, cache, disabled=False):
    from tqdm import tqdm
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / case["file"]
    if destination.exists():
        if not zipfile.is_zipfile(destination):
            raise ValueError(f"Invalid cached checkpoint: {destination}")
        print(f"Checkpoint gia scaricato: {destination}", flush=True)
        return destination
    url = ("https://drive.usercontent.google.com/download?" +
           urllib.parse.urlencode(dict(id=case["drive_id"], export="download", confirm="t")))
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    for _ in range(3):
        with opener.open(url, timeout=120) as response:
            first = response.read(8192)
            if "text/html" in response.headers.get("Content-Type", "") or first.lstrip().startswith(b"<"):
                form = DownloadForm()
                form.feed((first + response.read()).decode("utf-8", errors="replace"))
                if not form.action:
                    raise RuntimeError("Drive returned HTML instead of weights; download confirmation unavailable")
                parsed = urllib.parse.urlparse(form.action)
                if parsed.scheme != "https" or parsed.hostname not in ("drive.google.com", "drive.usercontent.google.com"):
                    raise RuntimeError("Unexpected Drive download confirmation URL")
                url = form.action + ("&" if "?" in form.action else "?") + urllib.parse.urlencode(form.fields)
                continue
            length = response.headers.get("Content-Length")
            total = int(length) if length else None
            partial = destination.with_suffix(".ckpt.part")
            with partial.open("wb") as stream, tqdm(
                total=total, unit="B", unit_scale=True, desc="Download " + case["dataset"],
                dynamic_ncols=True, disable=disabled
            ) as progress:
                stream.write(first)
                progress.update(len(first))
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block)
                    progress.update(len(block))
            if (total is not None and partial.stat().st_size != total) or not zipfile.is_zipfile(partial):
                raise RuntimeError(f"Incomplete checkpoint download: {partial}")
            partial.replace(destination)
            return destination
    raise RuntimeError("Drive download confirmation did not yield a checkpoint")


def original_bf_nonfinite(scores, labels):
    """Preserve original BF200 NaN behavior without repeating 200 identical empty predictions."""
    import numpy as np
    from TranADPlus.src import omnianomaly
    with np.errstate(invalid="ignore", over="ignore"):
        start = float(np.min(scores))
        end = float(np.mean(scores) + 3 * np.std(scores))
        thresholds = np.linspace(start, end, 200)
    if np.isnan(thresholds).all():
        stats, latency = omnianomaly.calc_seq(scores, labels.copy(), float("nan"))
        keys = ("f1", "precision", "recall", "TP", "TN", "FP", "FN", "ROC/AUC")
        return {**{"bf_" + k: v for k, v in zip(keys, stats)},
                "bf_threshold": float("nan"), "bf_latency": latency}
    return omnianomaly.bf_search(scores, labels.copy(), start, end, 200, verbose=False)


def score_metrics(train, test, labels, q):
    import numpy as np
    import paper_metrics as metrics
    from TranADPlus.src import omnianomaly
    train_ok, test_ok = bool(np.isfinite(train).all()), bool(np.isfinite(test).all())
    result = {"pot": None, "bf": None, "dense": None,
              "valid_finite_scores": train_ok and test_ok}
    if train_ok:
        # Do not filter test NaNs: comparison with threshold must follow the original code.
        result["pot"] = omnianomaly.pot_eval(train, test, labels.copy(), q)[0]
    else:
        result["pot_error"] = "Nonfinite training scores: POT not fitted"
    result["bf"] = (metrics.paper_best_f1(test, labels) if test_ok
                    else original_bf_nonfinite(test, labels))
    if test_ok:
        result["dense"] = metrics.dense_best_f1(test, labels)
    return result


def nonfinite_info(scores, labels=None):
    import numpy as np
    bad = ~np.isfinite(scores)
    indices = np.flatnonzero(bad)
    result = dict(points=int(len(scores)), nonfinite=int(bad.sum()),
                  first_nonfinite=int(indices[0]) if len(indices) else None)
    if labels is not None:
        result["nonfinite_nominal"] = int((bad & (labels == 0)).sum())
        result["nonfinite_anomalous"] = int((bad & (labels > 0)).sum())
    return result


def compare(observed, reference):
    result = {}
    for method in ("pot", "bf"):
        values = observed.get(method)
        result[method] = {
            key: dict(reference=expected,
                      observed=values.get(method + "_" + key) if values else None,
                      delta_pp=100 * (values[method + "_" + key] - expected) if values else None)
            for key, expected in reference[method].items()
        }
    return result


def evaluate_case(case, path, output, mode, args):
    import numpy as np
    import pickle
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from gpu_settings import configure
    from TranADPlus.src import models
    from TranADPlus.src.datasets import TSDataset_tracks
    import paper_metrics as metrics

    configure()
    device = torch.device(args.device)
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device unavailable: {device}")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    hp, state = saved["hp_dict"], saved["model_states"]
    ff = int(state["transformer_encoder.layers.0.linear1.weight"].shape[0])
    actual = {}
    for module in ("transformer_encoder", "transformer_decoder1", "transformer_decoder2"):
        indices = {int(match.group(1)) for name in state
                   if (match := re.match(re.escape(module) + r"\.layers\.(\d+)\.", name))}
        actual[module] = len(indices)
    for key, expected in (("dataset_str", case["dataset"]), ("features", case["features"]),
                          ("window_sz", case["window"]), ("batch_sz", case["batch"]),
                          ("epochs", case["epochs"]), ("lr", case["lr"]), ("pot_q", case["q"])):
        if hp.get(key) != expected:
            raise ValueError(f"Unexpected author checkpoint {key}: {hp.get(key)!r} != {expected!r}")
    if ff != case["ff"] or set(actual.values()) != {case["layers"]}:
        raise ValueError(f"Unexpected tensor architecture: FF={ff}, layers={actual}")

    dtype = torch.float64 if mode == "fp64" else torch.float32

    class InferenceModel(models.TranAD):
        def forward(self, source, target):
            return super().forward(source.to(dtype=dtype), target.to(dtype=dtype))

    model = InferenceModel(
        n_feats=hp["features"], window_sz=hp["window_sz"], batch_sz=hp["batch_sz"],
        dim_feedforward=ff, num_encoder_layers=case["layers"],
        num_decoder_layers=case["layers"]
    ).to(device=device, dtype=dtype)
    model.load_state_dict(state, strict=True)
    metadata = {
        "purpose": "author_weights_control_not_our_training",
        "checkpoint_file": case["file"], "checkpoint_sha256": sha256(path),
        "checkpoint_hp_dict": hp, "effective_ff": ff, "actual_layers": actual,
        "inference_precision": mode, "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "source_log": REPO + "/blob/main/Results/TranADPlus/" + case["log"],
        "reference": case["reference"], "score_formula": "mean((alpha*D1+(1-alpha)*D2-target)**2)",
        "point_adjustment": "upstream_global", "bf_original_thresholds": 200,
        "dense_bf_uses_test_labels": True, "input_sha256": {},
        "runtime_source_sha256": {
            name: sha256(CODE / name) for name in (
                "TranADPlus/src/models.py", "TranADPlus/src/datasets.py",
                "TranADPlus/src/helpers.py", "TranADPlus/src/omnianomaly.py",
                "TranADPlus/src/spot.py", "paper_metrics.py", "gpu_settings.py",
                "verify_author_checkpoints.py")
        },
    }
    del saved, state

    base = args.data_root / "processed/tranad_plus" / case["dataset"]
    loaders = []
    scaler = None
    for split in ("Train", "Test"):
        tracks = sorted((base / split / "Tracks").glob("*.npy"))
        labels = sorted((base / split / "Labels").glob("*.npy"))
        if not tracks or len(tracks) != len(labels):
            raise ValueError(f"Missing or unaligned {split} tracks/labels: {base}")
        for item in tracks + labels:
            metadata["input_sha256"][str(item.relative_to(base))] = sha256(item)
        data = TSDataset_tracks(
            track_paths=[str(p) for p in tracks], anom_paths=[str(p) for p in labels],
            window_size=hp["window_sz"], padding=hp["padding"], scaling=True,
            scaler=scaler, downsample=hp["downsample"])
        if data.big_track.shape[1] != hp["features"] or not torch.isfinite(data.big_track).all():
            raise ValueError("Wrong feature count or nonfinite normalized input")
        scaler = data.scaler
        metadata[split.lower()] = dict(points=len(data), normalization_dtype=str(data.big_track.dtype),
                                       normalized_min=float(data.big_track.min()),
                                       normalized_max=float(data.big_track.max()))
        loaders.append(DataLoader(data, batch_size=hp["batch_sz"], shuffle=False,
                                  num_workers=hp["num_workers"]))
    write_json(output / "metadata.json", metadata)
    arrays = [metrics.decoder_arrays(model, loader, device,
              f"{case['dataset']} {mode} {split}") for loader, split in zip(loaders, ("TRAIN", "TEST"))]
    del model, loaders
    if device.type == "cuda":
        torch.cuda.empty_cache()
    labels = arrays[1][3]
    np.save(output / "labels_test.npy", labels)
    masks = {}
    for split, data in zip(("train", "test"), arrays):
        for decoder, values in zip(("first", "second"), data[:2]):
            masks[split + "_" + decoder] = ~np.isfinite(values).all(axis=1)
    np.savez_compressed(output / "nonfinite_masks.npz", **masks)

    best = {"pot": None, "bf": None, "dense": None}
    all_finite = True
    with (output / "checks.jsonl").open("w") as checks:
        for index, alpha in enumerate(tqdm(np.linspace(0, 1, 21), desc="POT / BF200 / BF denso",
                                          unit="alpha", dynamic_ncols=True,
                                          disable=args.no_progress)):
            with np.errstate(invalid="ignore", over="ignore"):
                train = metrics.anomaly_scores(arrays[0], alpha)
                test = metrics.anomaly_scores(arrays[1], alpha)
            record = {"alpha": float(alpha), "train": nonfinite_info(train),
                      "test": nonfinite_info(test, labels)}
            all_finite &= record["train"]["nonfinite"] == record["test"]["nonfinite"] == 0
            np.savez_compressed(output / f"scores_alpha_{index:02d}.npz",
                                train=train, test=test, alpha=float(alpha))
            observed = score_metrics(train, test, labels, hp["pot_q"])
            record["metrics"] = observed
            checks.write(json.dumps(clean(record), allow_nan=False) + "\n")
            checks.flush()
            for method, field in (("pot", "pot_f1"), ("bf", "bf_f1"), ("dense", "bf_dense_f1")):
                value = observed[method]
                if value is not None and (best[method] is None or value[field] > best[method][field]):
                    best[method] = {**value, "alpha": float(alpha), "beta": float(1-alpha)}

    # An explicitly separate endpoint diagnostic: omit the unused decoder instead
    # of computing 0*NaN. Original scores above remain untouched.
    endpoint = None
    if case["dataset"] == "WADI":
        direct = [np.mean((data[0] - data[2]) ** 2, axis=1) for data in arrays]
        np.savez_compressed(output / "endpoint_first_decoder_only.npz",
                            train=direct[0], test=direct[1])
        endpoint = {"formula": "mean((D1-target)**2); diagnostic, not original alpha formula",
                    "train": nonfinite_info(direct[0]), "test": nonfinite_info(direct[1], labels),
                    "metrics": score_metrics(*direct, labels, hp["pot_q"])}
    summary = {
        "case": args.worker, "dataset": case["dataset"], "precision": mode,
        "purpose": "fp64_numerical_control" if mode == "fp64" else "fp32_replication",
        "status": "finite_control" if all_finite else "nonfinite_diagnostic",
        "all_original_scores_finite": bool(all_finite),
        "original_protocol_best": best, "comparison_same_checkpoint": compare(best, case["reference"]),
        "decoder_nonfinite_counts": {key: int(mask.sum()) for key, mask in masks.items()},
        "endpoint_control": endpoint, "source_log": metadata["source_log"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
    }
    for method, result in best.items():
        if result is not None:
            with (output / f"author_{method}.pkl").open("wb") as stream:
                pickle.dump(result, stream)
    write_json(output / "summary.json", summary)
    print(f"\nCONTROLLO COMPLETATO: {args.worker} {mode} — {summary['status']}", flush=True)
    for method in ("pot", "bf"):
        result = best[method]
        if result is None:
            print(method.upper() + ": non calcolabile; vedere checks.jsonl", flush=True)
            continue
        values = " | ".join(f"{key}={100 * result[method + '_' + key]:.4f}%"
                            for key in ("f1", "precision", "recall", "ROC/AUC"))
        delta = 100 * (result[method + "_f1"] - case["reference"][method]["f1"])
        print(f"{method.upper()}: {values} | delta F1 dal log={delta:+.4f} pp", flush=True)
    if best["dense"] is not None:
        print(f"BF denso (separato): {100 * best['dense']['bf_dense_f1']:.4f}%", flush=True)
    print("Punti non finiti nei decoder:", summary["decoder_nonfinite_counts"], flush=True)
    print("Riepilogo completo:", output / "summary.json", flush=True)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--precision", choices=("auto", "fp32", "fp64", "both"), default="auto",
                        help="auto: FP32 for all; also FP64 for WADI")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path, default=CODE.parent / "data")
    parser.add_argument("--results-root", type=Path, default=CODE.parent / "results")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", choices=tuple(CASES), help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("fp32", "fp64"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for key in ("data_root", "results_root"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.results_root == CODE or CODE in args.results_root.parents:
        parser.error("Results must be outside tranad_plus")
    if len(set(args.cases)) != len(args.cases):
        parser.error("Duplicate cases")
    args.root = args.results_root / "diagnostics/author_weights"
    args.checkpoint_dir = (args.checkpoint_dir or args.root / "_checkpoints").expanduser().resolve()
    if args.worker and (not args.mode or args.worker_output is None):
        parser.error("Internal worker arguments incomplete")
    return args


def environment(args):
    env = os.environ.copy()
    env.pop("TRANADPLUS_CAMPAIGN_PLAN", None)
    env.update(PYTHONDONTWRITEBYTECODE="1", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
               TRANADPLUS_CUDA_PRECISION="faithful",
               TRANADPLUS_PREPROCESSED_DIR=str(args.data_root / "processed/tranad_plus"),
               TOAST_NO_TQDM="1" if args.no_progress else "0",
               TOAST_FORCE_TQDM="0" if args.no_progress else "1")
    return env


def main():
    args = arguments()
    os.chdir(CODE)
    if args.worker:
        os.environ.pop("TRANADPLUS_CAMPAIGN_PLAN", None)
        os.environ.update(environment(args))
        def timeout(*_):
            raise TimeoutError("Author checkpoint control exceeded two hours")
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(7200)
        path = checkpoint_file(CASES[args.worker], args.checkpoint_dir, args.no_progress)
        evaluate_case(CASES[args.worker], path, args.worker_output, args.mode, args)
        return 0
    plan = []
    for key in args.cases:
        modes = (("fp32", "fp64") if args.precision == "both" or
                 (args.precision == "auto" and CASES[key]["dataset"] == "WADI")
                 else ("fp32" if args.precision == "auto" else args.precision,))
        plan.extend((key, mode) for mode in modes)
    print("CONTROLLI SU PESI AUTORI — nessun training", flush=True)
    for index, (key, mode) in enumerate(plan, 1):
        case = CASES[key]
        print(f"[{index}/{len(plan)}] {key} {mode}: "
              f"POT log={case['reference']['pot']['f1']:.6f}, BF200 log={case['reference']['bf']['f1']:.6f}",
              flush=True)
    print("Risultati:", args.root, flush=True)
    if args.dry_run:
        return 0
    return campaign(args, plan)


def campaign(args, plan):
    import fcntl
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "_campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another author weights campaign is running") from None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        report_file = args.root / f"campaign_{stamp}.json"
        report = {"status": "running", "reference_scope": "same checkpoint official log", "tasks": []}
        started = time.perf_counter()
        try:
            for index, (key, mode) in enumerate(plan, 1):
                output = args.root / key / (mode + "_" + stamp)
                output.mkdir(parents=True)
                record = {"case": key, "precision": mode, "status": "running", "output": str(output)}
                report["tasks"].append(record)
                write_json(report_file, report)
                print(f"\n[{index}/{len(plan)}] START {key} {mode}", flush=True)
                command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                           "--worker", key, "--mode", mode, "--worker-output", str(output),
                           "--device", args.device, "--data-root", str(args.data_root),
                           "--results-root", str(args.results_root),
                           "--checkpoint-dir", str(args.checkpoint_dir)]
                if args.no_progress:
                    command.append("--no-progress")
                phase_start = time.perf_counter()
                with (output / "evaluation.log").open("ab", buffering=0) as log, subprocess.Popen(
                    command, env=environment(args), cwd=CODE, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, bufsize=0
                ) as process:
                    try:
                        while chunk := process.stdout.read(65536):
                            log.write(chunk)
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
                record.update(status="complete" if code == 0 else "failed", exit_code=code,
                              seconds=time.perf_counter() - phase_start)
                if code == 0:
                    summary = json.loads((output / "summary.json").read_text())
                    record["result_status"] = summary["status"]
                    record["comparison"] = summary["comparison_same_checkpoint"]
                write_json(report_file, report)
        except BaseException:
            report["status"] = "interrupted"
            if report["tasks"] and report["tasks"][-1]["status"] == "running":
                report["tasks"][-1]["status"] = "interrupted"
            raise
        else:
            report["status"] = "complete" if all(t["status"] == "complete" for t in report["tasks"]) else "complete_with_errors"
        finally:
            report["seconds"] = time.perf_counter() - started
            write_json(report_file, report)
        print("\nReport campagna:", report_file, flush=True)
        for task in report["tasks"]:
            print(task["case"], task["precision"], task.get("result_status", task["status"]), flush=True)
        return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrotto. Checkpoint scaricati e risultati gia salvati conservati.", flush=True)
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

