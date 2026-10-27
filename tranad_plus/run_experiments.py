import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from config import (
    BASE_DIR,
    DATASETS,
    LOGS_DIR,
    NATIVE_PACKAGE,
    PROFILES,
    RESULTS_DIR,
    checkpoint_path,
    get_profile,
    list_profiles,
    result_directory,
)

SCHEMA_VERSION = 2
EPOCH_PATTERN = re.compile(r"\bEpoch\s+(\d+)\b", re.IGNORECASE)


def now():
    return datetime.now(timezone.utc).isoformat()


def run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(part)
    return hasher.hexdigest()


def profile_digest(profile):
    payload = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return digest_bytes(payload)


def normalize(value):
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if hasattr(value, "tolist"):
        return normalize(value.tolist())
    if hasattr(value, "item"):
        return normalize(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=".atomic_", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(normalize(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_copy(source, destination):
    source = Path(source)
    destination = Path(destination)
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty artifact: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise RuntimeError(f"Artifact collision with different content: {destination}")
        return destination
    temporary = destination.parent / ("." + destination.name + "." + uuid4().hex + ".tmp")
    try:
        shutil.copy2(source, temporary)
        if sha256_file(source) != sha256_file(temporary):
            raise RuntimeError(f"Copy verification failed: {destination}")
        if destination.exists():
            if sha256_file(destination) != sha256_file(temporary):
                raise RuntimeError(f"Artifact collision: {destination}")
        else:
            os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_json(path):
    if not Path(path).is_file():
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def status_path(dataset, identifier):
    return result_directory(dataset) / f"{identifier}_status.json"


def history_path(dataset, identifier):
    return result_directory(dataset) / f"{identifier}_history.jsonl"


def event(dataset, identifier, action, **details):
    path = history_path(dataset, identifier)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": now(), "action": action, "details": details}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(normalize(record), ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def set_status(profile, stage, **details):
    dataset = profile["dataset"]
    identifier = profile["identifier"]
    path = status_path(dataset, identifier)
    previous = read_json(path) or {}
    value = {
        **previous,
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "identifier": identifier,
        "profile": profile,
        "profile_sha256": profile_digest(profile),
        "stage": stage,
        "updated_at": now(),
        **details,
    }
    atomic_json(path, value)
    return value


def ensure_profile_consistency(profile):
    old = read_json(status_path(profile["dataset"], profile["identifier"]))
    if old is not None and old.get("profile_sha256") != profile_digest(profile):
        raise RuntimeError(
            f"Recorded configuration changed for {profile['identifier']}. "
            "Use a new profile identifier instead of reusing one."
        )


def dataset_inventory(dataset):
    base = NATIVE_PACKAGE / "Datasets" / "Preprocessed" / dataset
    result = {}
    for split in ("Train", "Test"):
        directory = base / split
        split_files = []
        for kind in ("Tracks", "Labels"):
            folder = directory / kind
            paths = sorted(folder.glob("*.npy")) if folder.is_dir() else []
            if not paths:
                raise FileNotFoundError(f"Missing preprocessed data: {folder}")
            for path in paths:
                stats = path.stat()
                split_files.append({
                    "name": str(path.relative_to(base)),
                    "size_bytes": stats.st_size,
                    "mtime_ns": stats.st_mtime_ns,
                })
        result[split] = split_files
    return result


def environment_details():
    details = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "timestamp": now(),
    }
    try:
        import torch
        details["torch"] = torch.__version__
        details["cuda_available"] = torch.cuda.is_available()
        details["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            details["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        details["torch"] = None
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=90, check=True)
        details["pip_freeze"] = result.stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        details["pip_freeze"] = None
    try:
        result = subprocess.run(["git", "-C", str(NATIVE_PACKAGE), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True)
        details["native_git_commit"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        details["native_git_commit"] = None
    return details


def check_device(device):
    import torch
    if device == "cpu":
        return
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available")
        return
    if device.startswith("cuda:"):
        try:
            index = int(device.partition(":")[2])
        except ValueError as exc:
            raise RuntimeError(f"Invalid device: {device}") from exc
        if not torch.cuda.is_available() or index not in range(torch.cuda.device_count()):
            raise RuntimeError(f"Unavailable CUDA device: {device}")
        return
    raise RuntimeError(f"Unsupported device: {device}")


def checkpoint_metadata(path, profile):
    import torch
    from TranADPlus.src import helpers
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or not isinstance(saved.get("hp_dict"), dict):
        raise RuntimeError(f"Checkpoint lacks hyperparameters: {path}")
    meta = saved["hp_dict"]
    expected = {
        "dataset_str": profile["dataset"],
        "identifier": profile["identifier"],
        "model_str": "TranAD",
        "lr": profile["lr"],
        "epochs": profile["epochs"],
        "batch_sz": profile["batch_sz"],
        "window_sz": profile["window_sz"],
        "dim_feedforward": helpers.parse_dim_feedforward(meta.get("features"), profile["dim_feedforward"]),
        "num_layers": profile["num_layers"],
        "pot_q": profile["pot_q"],
        "padding": profile["padding"],
        "shuffle": profile["shuffle"],
        "num_workers": profile["num_workers"],
        "downsample": profile["downsample"],
    }
    for key, wanted in expected.items():
        actual = meta.get(key)
        if isinstance(wanted, (int, float)) and not isinstance(wanted, bool):
            valid = isinstance(actual, (int, float)) and math.isclose(actual, wanted, rel_tol=1e-8, abs_tol=1e-12)
        else:
            valid = actual == wanted
        if not valid:
            raise RuntimeError(f"Checkpoint mismatch {path.name}: {key} expected {wanted!r}, found {actual!r}")
    return normalize(meta)


def intermediate_checkpoint(profile):
    import torch

    native = checkpoint_path(profile["identifier"])
    archive = result_directory(profile["dataset"]) / f"{profile['identifier']}.ckpt"
    if not native.is_file():
        return False
    saved = torch.load(native, map_location="cpu", weights_only=False)
    if isinstance(saved, dict) and "hp_dict" in saved:
        return False
    required = {"epoch", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict"}
    if not isinstance(saved, dict) or not required.issubset(saved):
        raise RuntimeError(f"Unrecognized native checkpoint: {native}")
    if archive.is_file():
        raise RuntimeError(f"Partial checkpoint conflicts with archived final checkpoint: {archive}")
    return True


def resume_command(profile, device):
    return [
        sys.executable, "-u", str(BASE_DIR / "resume_native.py"),
        "--profile", profile["identifier"], "--device", device,
    ]


def checkpoint_ready(profile):
    dataset, identifier = profile["dataset"], profile["identifier"]
    native = checkpoint_path(identifier)
    archive = result_directory(dataset) / f"{identifier}.ckpt"
    if native.is_file() and archive.is_file() and sha256_file(native) != sha256_file(archive):
        raise RuntimeError(f"Native and archived checkpoints differ: {identifier}")
    source = archive if archive.is_file() else native
    if not source.is_file() or source.stat().st_size == 0:
        return False
    checkpoint_metadata(source, profile)
    if not archive.is_file():
        atomic_copy(source, archive)
        event(dataset, identifier, "existing_checkpoint_archived", source=str(source))
    if not native.is_file():
        atomic_copy(archive, native)
        event(dataset, identifier, "checkpoint_restored", source=str(archive))
    return True


def training_command(profile, device):
    args = [
        "--model_str", "TranAD", "--dataset_str", profile["dataset"],
        "--identifier", profile["identifier"], "--device", device,
        "--scaling", "True", "--lr", str(profile["lr"]),
        "--epochs", str(profile["epochs"]), "--batch_sz", str(profile["batch_sz"]),
        "--window_sz", str(profile["window_sz"]),
        "--dim_feedforward", str(profile["dim_feedforward"]),
        "--num_layers", str(profile["num_layers"]), "--pot_q", str(profile["pot_q"]),
        "--shuffle", "1" if profile["shuffle"] else "0",
        "--padding", "1" if profile["padding"] else "0",
        "--num_workers", str(profile["num_workers"]),
    ]
    if profile["downsample"] is not None:
        args.extend(["--downsample", str(profile["downsample"])])
    return [sys.executable, "-u", str(BASE_DIR / "resumable_training.py"), *args]


def evaluation_command(profile, device):
    return [
        sys.executable, "-u", str(BASE_DIR / "evaluation_progress.py"),
        "--identifier", profile["identifier"], "--device", device,
        "--alpha_search", "True",
    ]


def run_logged(profile, command, stage, device, position, total):
    dataset, identifier = profile["dataset"], profile["identifier"]
    stamp = run_id()
    directory = LOGS_DIR / dataset
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{identifier}_{stage}_{stamp}.log"
    log_path = directory / filename
    archive_path = result_directory(dataset) / filename
    environment = os.environ.copy()
    if stage == "evaluate":
        environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    event(dataset, identifier, stage + "_started", command=command, log=str(log_path))
    set_status(profile, stage, current_command=command, current_log=str(log_path), error=None)
    started = time.monotonic()
    returncode = None
    try:
        with log_path.open("x", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command, cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, env=environment,
            )
            try:
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    if stage == "train":
                        match = EPOCH_PATTERN.search(line)
                        if match is not None:
                            epoch = int(match.group(1)) + 1
                            print(f"[CAMPAIGN {position}/{total}] {identifier}: epoch {epoch}/{profile['epochs']}", flush=True)
                            set_status(profile, stage, epoch=epoch, total_epochs=profile["epochs"])
                returncode = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
            finally:
                process.stdout.close()
    finally:
        if log_path.is_file():
            atomic_copy(log_path, archive_path)
    elapsed = round(time.monotonic() - started, 3)
    if returncode != 0:
        raise RuntimeError(f"{stage} exited with code {returncode}; log saved: {log_path}")
    event(dataset, identifier, stage + "_finished", duration_seconds=elapsed, log=str(archive_path))
    set_status(profile, stage + "_finished", last_duration_seconds=elapsed)
    return {"run_id": stamp, "log": archive_path.name, "duration_seconds": elapsed, "command": command}


def move_old_pickles(profile):
    dataset, identifier = profile["dataset"], profile["identifier"]
    directory = result_directory(dataset)
    for method in ("pot", "bf"):
        path = NATIVE_PACKAGE / "Pickles" / f"{identifier}_{method}.pkl"
        if path.exists():
            destination = directory / f"{identifier}_native_preexisting_{run_id()}_{method}.pkl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(destination))
            event(dataset, identifier, "preexisting_pickle_preserved", source=str(path), archived=str(destination))


def read_results(profile):
    directory = NATIVE_PACKAGE / "Pickles"
    identifier = profile["identifier"]
    results = {}
    sources = {}
    for method in ("pot", "bf"):
        source = directory / f"{identifier}_{method}.pkl"
        if not source.is_file():
            raise FileNotFoundError(f"Evaluation did not produce {source}")
        with source.open("rb") as handle:
            data = pickle.load(handle)
        if not isinstance(data, dict):
            raise RuntimeError(f"Invalid metric dictionary in {source}")
        data = normalize(data)
        score = data.get(f"{method}_f1")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise RuntimeError(f"Invalid {method} F1 for {identifier}: {score!r}")
        results[method] = data
        sources[method] = source
    return results, sources


def archive_evaluation(profile, invocation, checkpoint_hash, environment_snapshot):
    dataset, identifier = profile["dataset"], profile["identifier"]
    directory = result_directory(dataset)
    results, sources = read_results(profile)
    artifacts = {}
    for method, source in sources.items():
        destination = directory / f"{identifier}_evaluation_{invocation['run_id']}_{method}.pkl"
        atomic_copy(source, destination)
        artifacts[method] = {"file": destination.name, "sha256": sha256_file(destination), "size_bytes": destination.stat().st_size}
    report = {
        "schema_version": SCHEMA_VERSION,
        "model": "TranAD+",
        "dataset": dataset,
        "identifier": identifier,
        "profile": profile,
        "profile_sha256": profile_digest(profile),
        "run_id": invocation["run_id"],
        "created_at": now(),
        "results": results,
        "checkpoint": {"file": f"{identifier}.ckpt", "sha256": checkpoint_hash},
        "artifacts": artifacts,
        "evaluation": invocation,
        "environment_file": environment_snapshot,
    }
    destination = directory / f"{identifier}_evaluation_{invocation['run_id']}.json"
    if destination.exists():
        raise FileExistsError(destination)
    atomic_json(destination, report)
    event(dataset, identifier, "evaluation_archived", report=destination.name)
    set_status(profile, "completed", evaluation_report=destination.name, checkpoint_sha256=checkpoint_hash)
    return destination, results


def latest_reports(dataset):
    directory = result_directory(dataset)
    selected = {}
    for path in sorted(directory.glob("*_evaluation_*.json")):
        try:
            data = read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
            continue
        identifier = data.get("identifier")
        if data.get("dataset") != dataset or identifier not in PROFILES[dataset]:
            continue
        profile = get_profile(identifier)
        if data.get("profile_sha256") != profile_digest(profile):
            continue
        checkpoint = data.get("checkpoint") or {}
        archived = directory / f"{identifier}.ckpt"
        if not archived.is_file() or sha256_file(archived) != checkpoint.get("sha256"):
            continue
        artifacts = data.get("artifacts") or {}
        if not all(
            isinstance(artifacts.get(method), dict)
            and (directory / artifacts[method].get("file", "")).is_file()
            and sha256_file(directory / artifacts[method]["file"]) == artifacts[method].get("sha256")
            for method in ("pot", "bf")
        ):
            continue
        if not all(
            isinstance((data.get("results") or {}).get(method), dict)
            and isinstance(data["results"][method].get(f"{method}_f1"), (int, float))
            for method in ("pot", "bf")
        ):
            continue
        if identifier not in selected or data.get("created_at", "") > selected[identifier]["report"].get("created_at", ""):
            selected[identifier] = {"file": path.name, "report": data}
    return selected


def rebuild_summary(dataset):
    valid = latest_reports(dataset)
    best = {}
    for method in ("pot", "bf"):
        candidates = [
            (item["report"]["results"][method][f"{method}_f1"], identifier, item)
            for identifier, item in valid.items()
        ]
        if candidates:
            score, identifier, item = max(candidates, key=lambda entry: (entry[0], entry[1]))
            best[method] = {
                "identifier": identifier,
                "f1": score,
                "metrics": item["report"]["results"][method],
                "report": item["file"],
            }
        else:
            best[method] = None
    missing = sorted(set(PROFILES[dataset]) - set(valid))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "updated_at": now(),
        "status": "complete" if PROFILES[dataset] and not missing else "partial",
        "expected_profiles": len(PROFILES[dataset]),
        "completed_profiles": len(valid),
        "missing_profiles": missing,
        "profiles": {
            identifier: {
                "report": item["file"],
                "pot_f1": item["report"]["results"]["pot"]["pot_f1"],
                "best_f1": item["report"]["results"]["bf"]["bf_f1"],
            }
            for identifier, item in sorted(valid.items())
        },
        "best_results": best,
    }
    atomic_json(result_directory(dataset) / "summary.json", summary)
    return summary


def plan_from_args(args):
    if args.profile:
        profile = get_profile(args.profile)
        return [(profile["dataset"], profile["identifier"])]
    if args.dataset:
        return list_profiles(args.dataset)
    return list_profiles()


def execute_profile(profile, args, position, total, env_file):
    dataset, identifier = profile["dataset"], profile["identifier"]
    ensure_profile_consistency(profile)
    inventory = dataset_inventory(dataset)
    current = latest_reports(dataset).get(identifier)
    if current and args.stage in ("both", "evaluate") and not args.force_evaluation:
        print(f"[{position}/{total}] SKIPPED: {identifier}, complete validated evaluation found", flush=True)
        event(dataset, identifier, "skipped_completed", report=current["file"])
        return "skipped"
    partial = intermediate_checkpoint(profile)
    existing = False if partial else checkpoint_ready(profile)
    if args.stage in ("train", "both") and not existing:
        command = resume_command(profile, args.device) if partial else training_command(profile, args.device)
        if partial:
            print(f"[{position}/{total}] RESUMING: {identifier} from native checkpoint", flush=True)
        invocation = run_logged(profile, command, "train", args.device, position, total)
        checkpoint = checkpoint_path(identifier)
        if not checkpoint.is_file():
            raise RuntimeError(f"Training ended without checkpoint: {checkpoint}")
        checkpoint_metadata(checkpoint, profile)
        archived = result_directory(dataset) / f"{identifier}.ckpt"
        atomic_copy(checkpoint, archived)
        checkpoint_hash = sha256_file(archived)
        training_record = {
            "schema_version": SCHEMA_VERSION,
            "profile": profile,
            "profile_sha256": profile_digest(profile),
            "checkpoint": {"file": archived.name, "sha256": checkpoint_hash, "size_bytes": archived.stat().st_size},
            "training": invocation,
            "dataset_inventory": inventory,
            "environment_file": env_file,
            "completed_at": now(),
        }
        atomic_json(result_directory(dataset) / f"{identifier}_training_{invocation['run_id']}.json", training_record)
        set_status(profile, "trained", checkpoint_sha256=checkpoint_hash)
        event(dataset, identifier, "training_archived", checkpoint=archived.name)
        existing = True
    elif existing:
        print(f"[{position}/{total}] CHECKPOINT: {identifier} already available and validated", flush=True)
        set_status(profile, "trained", checkpoint_sha256=sha256_file(result_directory(dataset) / f"{identifier}.ckpt"))
    if args.stage in ("evaluate", "both"):
        if not existing:
            raise RuntimeError(f"No validated checkpoint for {identifier}")
        move_old_pickles(profile)
        invocation = run_logged(profile, evaluation_command(profile, args.device), "evaluate", args.device, position, total)
        checkpoint_hash = sha256_file(result_directory(dataset) / f"{identifier}.ckpt")
        report_path, metrics = archive_evaluation(profile, invocation, checkpoint_hash, env_file)
        print(f"[{position}/{total}] ARCHIVED: {report_path.name}", flush=True)
        print(f"[{position}/{total}] POT F1={metrics['pot']['pot_f1']:.6f} | Best F1={metrics['bf']['bf_f1']:.6f}", flush=True)
    rebuild_summary(dataset)
    return "processed"


def main():
    parser = argparse.ArgumentParser(description="Sequential TranAD+ benchmark campaign with artifact preservation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--dataset", choices=DATASETS)
    group.add_argument("--profile")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stage", choices=("train", "evaluate", "both"), default="both")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-evaluation", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    try:
        plan = plan_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not plan:
        parser.error("No profiles registered for selection")
    print(f"TranAD+ campaign: {len(plan)} profiles across {len(set(dataset for dataset, _ in plan))} datasets", flush=True)
    for position, (dataset, identifier) in enumerate(plan, 1):
        profile = get_profile(identifier)
        print(f"[{position}/{len(plan)}] {dataset} | {identifier} | {profile['epochs']} epochs | window {profile['window_sz']}", flush=True)
        command = training_command(profile, args.device) if args.stage in ("train", "both") else evaluation_command(profile, args.device)
        if not Path(command[2]).is_file():
            parser.error(f"Native source missing: {command[2]}")
    if args.dry_run:
        print("Dry run successful. No training or evaluation started.", flush=True)
        return
    check_device(args.device)
    campaign_id = run_id()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    environment_file = LOGS_DIR / f"campaign_{campaign_id}_environment.json"
    atomic_json(environment_file, environment_details())
    successes = 0
    skipped = 0
    failures = []
    started = time.monotonic()
    for position, (dataset, identifier) in enumerate(plan, 1):
        profile = get_profile(identifier)
        print(f"\n{'=' * 64}\n[{position}/{len(plan)}] START {dataset} / {identifier}\n{'=' * 64}", flush=True)
        try:
            result = execute_profile(profile, args, position, len(plan), str(environment_file))
            if result == "skipped":
                skipped += 1
            else:
                successes += 1
        except (Exception, KeyboardInterrupt) as exc:
            message = f"{type(exc).__name__}: {exc}"
            set_status(profile, "failed", error=message)
            event(dataset, identifier, "failed", error=message)
            failures.append({"dataset": dataset, "identifier": identifier, "error": message})
            print(f"[{position}/{len(plan)}] FAILED: {identifier}: {message}", flush=True)
            rebuild_summary(dataset)
            if isinstance(exc, KeyboardInterrupt) or not args.keep_going:
                break
        print(f"CAMPAIGN PROGRESS: {position}/{len(plan)} | processed={successes} | skipped={skipped} | failed={len(failures)}", flush=True)
    for dataset in sorted(set(dataset for dataset, _ in plan)):
        rebuild_summary(dataset)
    campaign_report = {
        "campaign_id": campaign_id,
        "created_at": now(),
        "device": args.device,
        "stage": args.stage,
        "profiles": plan,
        "completed_now": successes,
        "skipped": skipped,
        "failures": failures,
        "duration_seconds": round(time.monotonic() - started, 3),
        "environment": str(environment_file),
    }
    report_path = LOGS_DIR / f"campaign_{campaign_id}_summary.json"
    atomic_json(report_path, campaign_report)
    print(f"Campaign report: {report_path}", flush=True)
    if failures:
        raise SystemExit(1)
    print("Campaign completed successfully.", flush=True)


if __name__ == "__main__":
    main()
