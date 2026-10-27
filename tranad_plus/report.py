
import json
import math
import os
import pickle
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from config import PROFILES, result_directory


PAPER_F1_REFERENCES = {
    "SMAP": {
        "pot": 0.8424,
        "bf": 0.8491,
    },
    "MSL": {
        "pot": 0.8901,
        "bf": 0.8196,
    },
    "SMD": {
        "pot": 0.7129,
        "bf": 0.8509,
    },
    "SWaT": {
        "pot": 0.3290,
        "bf": 0.8239,
    },
    "WADI": {
        "pot": 0.5084,
        "bf": 0.2788,
    },
}

REFERENCE_VERIFIED = {
    "SMAP": True,
    "MSL": False,
    "SMD": False,
    "SWaT": False,
    "WADI": False,
}


def normalize(value):
    if isinstance(value, dict):
        return {
            str(key): normalize(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]

    if hasattr(value, "tolist"):
        return normalize(value.tolist())

    if hasattr(value, "item"):
        return normalize(value.item())

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=".report_",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)

        json.dump(
            normalize(data),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")

    os.replace(temporary, path)


def read_native_metrics(identifier):
    from config import NATIVE_PACKAGE

    results = {}

    for method in ("pot", "bf"):
        source = (
            NATIVE_PACKAGE
            / "Pickles"
            / f"{identifier}_{method}.pkl"
        )

        if not source.is_file():
            raise FileNotFoundError(source)

        with source.open("rb") as handle:
            metrics = pickle.load(handle)

        if not isinstance(metrics, dict):
            raise ValueError(
                f"Invalid evaluation output: {source}"
            )

        results[method] = normalize(metrics)

        score = results[method].get(
            f"{method}_f1"
        )

        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(score)
        ):
            raise RuntimeError(
                f"Invalid {method} F1 for {identifier}"
            )

    return results


def create_run_id():
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")

    return f"{timestamp}_{uuid4().hex[:8]}"


def write_report(dataset, identifier, log_path):
    from config import NATIVE_PACKAGE

    metrics = read_native_metrics(identifier)
    directory = result_directory(dataset)
    directory.mkdir(parents=True, exist_ok=True)

    run_id = create_run_id()
    prefix = f"{identifier}_evaluation_{run_id}"

    artifacts = {}

    for method in ("pot", "bf"):
        source = (
            NATIVE_PACKAGE
            / "Pickles"
            / f"{identifier}_{method}.pkl"
        )

        filename = f"{prefix}_{method}.pkl"
        destination = directory / filename

        shutil.copy2(source, destination)
        artifacts[method] = filename

    log_source = Path(log_path)
    log_destination = directory / f"{prefix}.log"

    shutil.copy2(log_source, log_destination)

    reference = PAPER_F1_REFERENCES.get(dataset)

    comparison = None

    if reference is not None:
        comparison = {}

        for method in ("pot", "bf"):
            actual = metrics[method][f"{method}_f1"]

            comparison[method] = {
                "reference_f1": reference[method],
                "actual_f1": actual,
                "difference": actual - reference[method],
            }

    report = {
        "schema_version": 1,
        "model": "TranAD+",
        "dataset": dataset,
        "identifier": identifier,
        "run_id": run_id,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "results": metrics,
        "artifacts": {
            **artifacts,
            "evaluation_log": log_destination.name,
            "checkpoint": f"{identifier}.ckpt",
        },
        "paper_reference": {
            "model": "TranAD+",
            "scope": "dataset",
            "verified": REFERENCE_VERIFIED.get(
                dataset,
                False,
            ),
            "comparison": comparison,
        },
    }

    destination = directory / f"{prefix}.json"

    if destination.exists():
        raise FileExistsError(destination)

    atomic_json(destination, report)
    rebuild_summary(dataset)

    print(
        f"POT F1: {metrics['pot']['pot_f1']:.9f}",
        flush=True,
    )

    print(
        f"Best F1: {metrics['bf']['bf_f1']:.9f}",
        flush=True,
    )

    print(
        f"Saved evaluation: {destination}",
        flush=True,
    )

    return destination


def load_evaluations(dataset):
    directory = result_directory(dataset)
    reports = {}

    if not directory.is_dir():
        return reports

    for path in sorted(
        directory.glob("*_evaluation_*.json")
    ):
        try:
            data = json.loads(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            continue

        if not isinstance(data, dict):
            continue

        if data.get("schema_version") != 1:
            continue

        if data.get("dataset") != dataset:
            continue

        identifier = data.get("identifier")

        if identifier not in PROFILES[dataset]:
            continue

        results = data.get("results", {})
        artifacts = data.get("artifacts", {})

        if not isinstance(results, dict):
            continue

        if not isinstance(artifacts, dict):
            continue

        valid = True

        for method in ("pot", "bf"):
            item = results.get(method)

            if not isinstance(item, dict):
                valid = False
                break

            score = item.get(f"{method}_f1")

            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(score)
            ):
                valid = False
                break

            filename = artifacts.get(method)

            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not (directory / filename).is_file()
            ):
                valid = False
                break

        checkpoint_name = artifacts.get("checkpoint")

        if (
            not isinstance(checkpoint_name, str)
            or checkpoint_name != f"{identifier}.ckpt"
            or not (directory / checkpoint_name).is_file()
        ):
            valid = False

        if not valid:
            continue

        existing = reports.get(identifier)

        if (
            existing is None
            or data.get("created_at", "")
            > existing["data"].get("created_at", "")
        ):
            reports[identifier] = {
                "file": path.name,
                "data": data,
            }

    return reports


def has_evaluation(dataset, identifier):
    return identifier in load_evaluations(dataset)


def rebuild_summary(dataset):
    directory = result_directory(dataset)
    directory.mkdir(parents=True, exist_ok=True)

    reports = load_evaluations(dataset)
    expected = set(PROFILES[dataset])
    completed = set(reports)
    missing = sorted(expected - completed)

    configurations = {}

    for identifier, item in sorted(reports.items()):
        results = item["data"]["results"]

        configurations[identifier] = {
            "evaluation_file": item["file"],
            "pot_f1": results["pot"]["pot_f1"],
            "best_f1": results["bf"]["bf_f1"],
        }

    best = {}

    for method in ("pot", "bf"):
        candidates = []

        for identifier, item in reports.items():
            metrics = item["data"]["results"][method]
            score = metrics[f"{method}_f1"]

            candidates.append(
                (score, identifier, item, metrics)
            )

        if not candidates:
            best[method] = None
            continue

        score, identifier, item, metrics = max(
            candidates,
            key=lambda candidate: (
                candidate[0],
                candidate[1],
            ),
        )

        best[method] = {
            "identifier": identifier,
            "f1": score,
            "metrics": metrics,
            "evaluation_file": item["file"],
        }

    summary = {
        "schema_version": 1,
        "model": "TranAD+",
        "dataset": dataset,
        "status": (
            "complete"
            if expected and not missing
            else "partial"
        ),
        "expected_configurations": len(expected),
        "completed_configurations": len(completed),
        "missing_configurations": missing,
        "configurations": configurations,
        "best_results": best,
        "updated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    destination = directory / "summary.json"
    atomic_json(destination, summary)

    print(
        f"{dataset}: {len(completed)}/{len(expected)} "
        "configurations evaluated.",
        flush=True,
    )

    for method, label in (
        ("pot", "POT"),
        ("bf", "Best F1"),
    ):
        selected = best[method]

        if selected is not None:
            print(
                f"Best {label}: {selected['f1']:.9f} "
                f"from {selected['identifier']}",
                flush=True,
            )

    return summary
