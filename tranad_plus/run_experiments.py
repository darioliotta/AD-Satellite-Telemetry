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

# Bootstrap output selection before importing config: paths are resolved at import time.
_preparse = argparse.ArgumentParser(add_help=False)
_preparse.add_argument("--output-root", type=Path)
_output_opts, _ = _preparse.parse_known_args()
if _output_opts.output_root is not None:
    os.environ["TRANADPLUS_OUTPUT_ROOT"] = str(_output_opts.output_root.expanduser().resolve())

from config import (
    BASE_DIR,
    DATASETS,
    LOGS_DIR,
    NATIVE_PACKAGE,
    PROCESSED_DATA_DIR,
    PROFILES,
    RESULTS_DIR,
    checkpoint_path,
    pickle_dir, native_environment,
    get_profile,
    list_profiles,
    result_directory,
)

SCHEMA_VERSION = 2
EPOCH_PATTERN = re.compile(r"^\[TRAIN\].*?\|\s*(\d+)/(\d+)\b")


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
    # Dataset location can differ across CPU, CloudVeneto and INFN machines.
    # The full source SHA256 is validated separately by opssat_pipeline.
    portable = {k:v for k,v in profile.items() if k != 'segments_csv'}
    payload = json.dumps(portable, sort_keys=True, separators=(",", ":")).encode("utf-8")
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
    if dataset == 'OPS-SAT':
        # OPS-SAT inventory is checked in the dedicated segment-aware engine;
        # its source path belongs to each profile, not a native Preprocessed tree.
        return {'format': 'raw_fragments_csv', 'source': 'profile.segments_csv'}
    base = PROCESSED_DATA_DIR / dataset
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
        result = subprocess.run(["git", "-C", str(BASE_DIR), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True)
        details["code_git_commit"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        details["code_git_commit"] = None
    details["code_hashes_sha256"] = {
        name: sha256_file(BASE_DIR / name) for name in (
            "run_experiments.py", "config.py", "project_paths.py", "opssat_pipeline.py",
            "pipeline_metrics.py", "ablation.py", "TranADPlus/src/models.py",
            "TranADPlus/scripts/training_loop.py", "TranADPlus/scripts/evaluation.py",
            "launch_v12.py", "campaign_v12.py", "gpu_batches.py",
            "gpu_settings.py", "console_telemetry.py", "resumable_training.py",
            "paper_metrics.py", "evaluation_progress.py", "opssat_scoring.py"
        )
    }
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
    if archive.is_file() and archive.resolve() != native.resolve():
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
    if profile.get('seed') is not None:
        args.extend(['--seed', str(profile['seed'])])
    return [sys.executable, "-u", str(BASE_DIR / "resumable_training.py"), *args]


def evaluation_command(profile, device, alpha_search=True):
    command = [sys.executable, "-u", str(BASE_DIR / "evaluation_progress.py"),
               "--identifier", profile["identifier"], "--device", device]
    # Upstream uses argparse(type=bool): DO NOT pass the literal string "False".
    # Omit the flag to disable the paper alpha grid.
    if alpha_search:
        command.extend(["--alpha_search", "True"])
    return command


def run_logged(profile, command, stage, device, position, total):
    dataset, identifier = profile["dataset"], profile["identifier"]
    stamp = run_id()
    directory = result_directory(dataset)
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{identifier}_{stage}_{stamp}.log"
    log_path = directory / filename
    archive_path = log_path
    environment = os.environ.copy()
    environment.update(native_environment(dataset))
    environment['TOAST_FORCE_TQDM'] = '1'
    if profile.get('seed') is not None:
        environment['PYTHONHASHSEED'] = str(profile['seed'])
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
                            epoch = int(match.group(1))
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
        pass  # Log is written directly to its final immutable archive path.
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
        path = pickle_dir(dataset) / f"{identifier}_{method}.pkl"
        if path.exists():
            destination = directory / f"{identifier}_native_preexisting_{run_id()}_{method}.pkl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(destination))
            event(dataset, identifier, "preexisting_pickle_preserved", source=str(path), archived=str(destination))


def read_results(profile):
    directory = pickle_dir(profile["dataset"])
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
    if dataset == 'OPS-SAT':
        selected = {}
        for path in sorted(directory.glob('*_evaluation_*.json')):
            try:
                data = read_json(path)
            except (OSError, ValueError):
                continue
            identifier = data.get('identifier') if isinstance(data, dict) else None
            if identifier not in PROFILES.get(dataset, {}) or data.get('dataset') != dataset:
                continue
            profile = get_profile(identifier)
            if data.get('profile_sha256') != profile_digest(profile):
                continue
            ckpt = directory / (identifier + '.ckpt')
            if not ckpt.is_file() or sha256_file(ckpt) != (data.get('checkpoint') or {}).get('sha256'):
                continue
            metrics = data.get('results') or {}
            if not all(isinstance(metrics.get(k), dict) and isinstance(metrics[k].get(k+'_f1'), (int,float)) for k in ('bf',)):
                continue
            if identifier not in selected or data.get('created_at','') >= selected[identifier]['report'].get('created_at',''):
                selected[identifier] = {'file': path.name, 'report': data}
        return selected
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


def rebuild_summary(dataset, expected_profiles=None):
    """Campaign-scoped completeness, not the entire global profile catalog.

    A selected one-profile plan is complete when that profile is complete,
    regardless of unrelated catalog entries that have never been run.
    """
    all_valid = latest_reports(dataset)
    expected = set(expected_profiles) if expected_profiles is not None else set(PROFILES[dataset])
    valid = {identifier: item for identifier,item in all_valid.items() if identifier in expected}
    best = {}
    for method in ("pot", "bf"):
        candidates = [
            (item["report"]["results"][method][f"{method}_f1"], identifier, item)
            for identifier, item in valid.items()
        ]
        if candidates:
            candidates = [candidate for candidate in candidates if isinstance(candidate[0], (float,int))]
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
    missing = sorted(expected - set(valid))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "updated_at": now(),
        "status": "complete" if expected and not missing else "partial",
        "scope": "campaign_plan" if expected_profiles is not None else "catalog",
        "expected_profile_ids": sorted(expected),
        "other_completed_profiles": sorted(set(all_valid) - expected),
        "expected_profiles": len(expected),
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
    if args.plan:
        from campaign_plan import read_plan, register_plan_profiles
        file, entries, settings = read_plan(args.plan)
        from config import PROFILES as registry
        os.environ['TRANADPLUS_CAMPAIGN_PLAN'] = str(file)
        names = register_plan_profiles(registry, file)
        args.measurement_settings = settings
        args.plan_fingerprint = sha256_file(file)
        return [(get_profile(name)['dataset'], name) for name in names]
    if args.profile:
        profile = get_profile(args.profile)
        return [(profile["dataset"], profile["identifier"])]
    if args.dataset:
        return list_profiles(args.dataset)
    return list_profiles()


def _complete_measurement_report(profile, settings, device):
    from pipeline_metrics import measure_profile, measurement_key
    checkpoint = result_directory(profile['dataset']) / f"{profile['identifier']}.ckpt"
    if not checkpoint.is_file():
        return None
    digest = sha256_file(checkpoint)
    token, identity = measurement_key(settings, device)
    target = result_directory(profile['dataset']) / (
        f"{profile['identifier']}_integrated_{digest[:12]}_{device.replace(':', '-')}_{token}.json"
    )
    if target.is_file():
        try:
            value = read_json(target)
            eval_report = latest_reports(profile['dataset']).get(profile['identifier'])
            expected_source = eval_report['file'] if eval_report else None
            observed_source = (value.get('original_paper_evaluation') or {}).get('source')
            if (value.get('checkpoint', {}).get('sha256') == digest
                    and value.get('settings_digest') == token
                    and value.get('schema_version') == 'thesis-integrated-v3'
                    and observed_source == expected_source):
                return target
        except (OSError, ValueError, AttributeError):
            pass
    print(f"MEASURING: {profile['identifier']} ({device})", flush=True)
    output = measure_profile(profile, settings, device, code_digest=sha256_file(BASE_DIR / 'run_experiments.py'))
    event(profile['dataset'], profile['identifier'], 'integrated_metrics_saved', report=str(output))
    set_status(profile, 'observed', integrated_report=output.name)
    return output



def execute_opssat(profile, args, position, total):
    """OPS-SAT uses a fragment-aware data/evaluation engine, the SAME campaign command.

    Do not call the native track loaders, which would concatenate distinct fragments.
    """
    from opssat_pipeline import train, evaluate_and_measure, source_csv
    dataset, identifier = profile['dataset'], profile['identifier']
    source_csv(profile)  # fail early with a useful message
    completed = latest_reports(dataset).get(identifier)
    final_archive = result_directory(dataset) / f'{identifier}.ckpt'
    final_native = checkpoint_path(identifier)
    if completed and args.stage in ('evaluate', 'both') and not args.force_evaluation:
        from opssat_pipeline import prepare_data
        _, _, _, current_inventory = prepare_data(profile)
        archived = result_directory(dataset) / f'{identifier}.ckpt'
        import torch
        stored = torch.load(archived, map_location='cpu', weights_only=True)
        if stored.get('opssat_inventory',{}).get('source_sha256') != current_inventory['source_sha256']:
            raise RuntimeError('OPS-SAT dataset changed after training; do not reuse a completed report')
        # A completed evaluation is reusable, but device- or setting-specific
        # measurements must NOT be silently skipped (e.g. CPU -> T4 -> A100).
        from pipeline_metrics import measurement_key
        from observability.measure_experiment import hash_file
        token, _ = measurement_key(args.measurement_settings, args.device)
        expected = result_directory(dataset) / (
            f'{identifier}_integrated_{hash_file(archived)[:12]}_'
            f'{args.device.replace(":", "-")}_{token}.json')
        if expected.is_file():
            print(f'[{position}/{total}] OPS-SAT SKIPPED: {identifier} already evaluated and profiled on {args.device}', flush=True)
            event(dataset, identifier, 'skipped_completed', report=completed['file'])
            return 'skipped'
        print(f'[{position}/{total}] OPS-SAT EXISTING CHECKPOINT: profiling/evaluating on {args.device}', flush=True)
        evaluation, integrated = evaluate_and_measure(profile, args.measurement_settings, args.device)
        set_status(profile, 'completed', evaluation_report=evaluation.name, integrated_report=integrated.name)
        event(dataset, identifier, 'measurement_archived', report=integrated.name)
        return 'processed'
    if args.stage in ('train','both') and not final_archive.is_file() and not final_native.is_file():
        print(f'[{position}/{total}] OPS-SAT TRAIN: {identifier}', flush=True)
        set_status(profile, 'training')
        train(profile, args.device)
        atomic_copy(final_native, final_archive)
        checkpoint_metadata(final_archive, profile)
        set_status(profile, 'trained', checkpoint_sha256=sha256_file(final_archive))
        event(dataset, identifier, 'training_archived', checkpoint=final_archive.name)
    elif final_native.is_file() and not final_archive.is_file():
        checkpoint_metadata(final_native, profile)
        atomic_copy(final_native, final_archive)
    if args.stage in ('evaluate','both'):
        if not final_archive.is_file():
            raise FileNotFoundError('OPS-SAT evaluation requires a completed checkpoint')
        checkpoint_metadata(final_archive, profile)
        # Even for OPS-SAT all quality and profiling passes are generated together.
        evaluation, integrated = evaluate_and_measure(profile, args.measurement_settings, args.device)
        set_status(profile, 'completed', evaluation_report=evaluation.name, integrated_report=integrated.name,
                   checkpoint_sha256=sha256_file(final_archive))
        event(dataset, identifier, 'evaluation_archived', report=evaluation.name, integrated_report=integrated.name)
        print(f'[{position}/{total}] OPS-SAT REPORT: {integrated.name}', flush=True)
    rebuild_summary(dataset)
    return 'processed'


def execute_profile(profile, args, position, total, env_file):
    dataset, identifier = profile["dataset"], profile["identifier"]
    ensure_profile_consistency(profile)
    if dataset == 'OPS-SAT':
        return execute_opssat(profile, args, position, total)
    inventory = dataset_inventory(dataset)
    current = latest_reports(dataset).get(identifier)
    if current and args.stage in ("both", "evaluate") and not args.force_evaluation:
        print(f"[{position}/{total}] SKIPPED TRAIN/EVALUATION: {identifier}, validated evaluation found", flush=True)
        event(dataset, identifier, "skipped_completed", report=current["file"])
        if args.measurement_settings.get('enabled', True):
            measured = _complete_measurement_report(profile, args.measurement_settings, args.device)
            if measured:
                print(f"[{position}/{total}] MEASUREMENT: {measured.name}", flush=True)
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
        invocation = run_logged(profile, evaluation_command(profile, args.device, args.measurement_settings.get('paper_alpha_search',True)), "evaluate", args.device, position, total)
        checkpoint_hash = sha256_file(result_directory(dataset) / f"{identifier}.ckpt")
        report_path, metrics = archive_evaluation(profile, invocation, checkpoint_hash, env_file)
        print(f"[{position}/{total}] ARCHIVED: {report_path.name}", flush=True)
        print(f"[{position}/{total}] POT F1={metrics['pot']['pot_f1']:.6f} | Best F1={metrics['bf']['bf_f1']:.6f}", flush=True)
    if args.measurement_settings.get('enabled', True) and existing:
        measured = _complete_measurement_report(profile, args.measurement_settings, args.device)
        if measured:
            print(f"[{position}/{total}] MEASUREMENT: {measured.name}", flush=True)
    rebuild_summary(dataset)
    return "processed"


def main():
    parser = argparse.ArgumentParser(description="Sequential TranAD+ benchmark campaign with artifact preservation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--dataset", choices=DATASETS)
    group.add_argument("--profile")
    group.add_argument("--plan", type=Path, help="Single JSON file listing exact experiment configurations and metrics settings")
    parser.add_argument("--output-root", type=Path, help="Project-level results directory; versioned subfolder is created inside")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stage", choices=("train", "evaluate", "both"), default="both")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-evaluation", action="store_true")
    parser.add_argument("--only", action="append", help="Run only listed identifiers from --plan (may be repeated)")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    args.measurement_settings = {
        'enabled': True, 'warmup': 10, 'iterations': 50, 'score_batch_size': 512,
        'alpha': 0.5, 'threshold_train_quantile': 0.995,
        'save_raw_scores': True, 'hardware_json_dir': None, 'paper_alpha_search': True,
        'inference_ablation': False, 'export_decoder_arrays': False,
    }
    args.plan_fingerprint = None
    try:
        plan = plan_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.only:
        if not args.plan:
            parser.error("--only requires --plan")
        wanted=set(args.only)
        unknown=wanted-{identifier for _,identifier in plan}
        if unknown:
            parser.error(f"--only: identifiers not in plan: {sorted(unknown)}")
        plan=[(dataset,identifier) for dataset,identifier in plan if identifier in wanted]
    if not plan:
        parser.error("No profiles registered for selection")
    print(f"TranAD+ campaign: {len(plan)} profiles across {len(set(dataset for dataset, _ in plan))} datasets", flush=True)
    for position, (dataset, identifier) in enumerate(plan, 1):
        profile = get_profile(identifier)
        print(f"[{position}/{len(plan)}] {dataset} | {identifier} | {profile['epochs']} epochs | window {profile['window_sz']}", flush=True)
        command = ([sys.executable, '-u', str(BASE_DIR / 'opssat_pipeline.py')]
                   if dataset == 'OPS-SAT' else
                   (training_command(profile, args.device) if args.stage in ('train','both') else evaluation_command(profile, args.device, args.measurement_settings.get('paper_alpha_search',True))))
        if not Path(command[2]).is_file():
            parser.error(f"Native source missing: {command[2]}")
    if args.dry_run:
        print("Dry run successful. No training or evaluation started.", flush=True)
        return
    check_device(args.device)
    os.environ['TOAST_FORCE_TQDM'] = '1'
    campaign_id = run_id()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if args.plan:
        # Immutable snapshot: later edits to experiment_plan.json cannot change this run's specification.
        snapshot = LOGS_DIR / f'campaign_{campaign_id}_plan.json'
        atomic_copy(Path(args.plan), snapshot)
        print(f'Frozen plan: {snapshot}', flush=True)
    environment_file = LOGS_DIR / f"campaign_{campaign_id}_environment.json"
    atomic_json(environment_file, environment_details())
    successes = 0
    skipped = 0
    failures = []
    started = time.monotonic()
    from tqdm.auto import tqdm
    bar = tqdm(total=len(plan), desc=f'V12 {args.stage.upper()} (all datasets)',
               unit='run', dynamic_ncols=True, ascii=True, ncols=100,
               disable=os.environ.get('TOAST_NO_TQDM') == '1')
    for position, (dataset, identifier) in enumerate(plan, 1):
        profile = get_profile(identifier)
        global_position = position + (len(plan) if args.stage == 'evaluate' else 0)
        print(f'[V12 {args.stage.upper()} {position:03d}/{len(plan):03d} | GLOBAL {global_position:03d}/{len(plan)*2:03d}]', flush=True)
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
        bar.update(1)
        bar.set_postfix(dataset=dataset, ok=successes, skipped=skipped, failed=len(failures))
        print(f"CAMPAIGN PROGRESS: {position}/{len(plan)} | processed={successes} | skipped={skipped} | failed={len(failures)}", flush=True)
    bar.close()
    for dataset in sorted(set(dataset for dataset, _ in plan)):
        rebuild_summary(dataset, expected_profiles={ident for d, ident in plan if d==dataset})
    campaign_report = {
        "campaign_id": campaign_id,
        "created_at": now(),
        "device": args.device,
        "stage": args.stage,
        "plan_sha256": args.plan_fingerprint,
        "measurement_settings": args.measurement_settings,
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
