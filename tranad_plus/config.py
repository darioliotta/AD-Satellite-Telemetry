
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
NATIVE_PACKAGE = BASE_DIR / "TranADPlus"
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "tranad_plus"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

DATASETS = ("SMAP", "MSL", "SMD", "SWaT", "WADI", "DSN_1k")

COMMON = {
    "num_layers": 1,
    "shuffle": False,
    "padding": True,
    "num_workers": 0,
    "downsample": None,
}

PROFILES = {
    "SMAP": {
        "smap_plus_ws10": {
            "epochs": 50,
            "batch_sz": 8192,
            "window_sz": 10,
            "lr": 0.001,
            "dim_feedforward": "2x",
            "pot_q": 0.07,
        },
        "smap_plus_ws50": {
            "epochs": 50,
            "batch_sz": 2048,
            "window_sz": 50,
            "lr": 0.001,
            "dim_feedforward": "2x",
            "pot_q": 0.07,
        },
    },
    "MSL": {
        "msl_plus_ws50_dff2x": {
            "epochs": 50,
            "batch_sz": 2048,
            "window_sz": 50,
            "lr": 0.0001,
            "dim_feedforward": "2x",
            "pot_q": 0.01,
        },
        "msl_plus_ws50_dff16": {
            "epochs": 50,
            "batch_sz": 2048,
            "window_sz": 50,
            "lr": 0.0001,
            "dim_feedforward": 16,
            "pot_q": 0.01,
        },
    },
    "SMD": {
        "smd_plus_ws20_dff16": {
            "epochs": 500,
            "batch_sz": 8192,
            "window_sz": 20,
            "lr": 0.0001,
            "dim_feedforward": 16,
            "pot_q": 0.001,
        },
    },
    "SWaT": {
        "swat_plus_ws50_dff2x_lr0-0001": {
            "epochs": 50,
            "batch_sz": 2048,
            "window_sz": 50,
            "lr": 0.0001,
            "dim_feedforward": "2x",
            "pot_q": 0.001,
        },
        "swat_plus_ws50_dff16_lr0-001": {
            "epochs": 50,
            "batch_sz": 2048,
            "window_sz": 50,
            "lr": 0.001,
            "dim_feedforward": 16,
            "pot_q": 0.001,
        },
    },
    "WADI": {
        "wadi_plus_ws20_dff2x_lr0-001": {
            "epochs": 50,
            "batch_sz": 8192,
            "window_sz": 20,
            "lr": 0.001,
            "dim_feedforward": "2x",
            "pot_q": 0.001,
        },
        "wadi_plus_ws50_dff2x_lr0-0001": {
            "epochs": 50,
            "batch_sz": 2048,
            "window_sz": 50,
            "lr": 0.0001,
            "dim_feedforward": "2x",
            "pot_q": 0.001,
        },
    },
    "DSN_1k": {
        "dsn_1k_plus_ws10_dff16": {
            "epochs": 500,
            "batch_sz": 8192,
            "window_sz": 10,
            "lr": 0.0001,
            "dim_feedforward": 16,
            "pot_q": 0.01,
        },
        "dsn_1k_plus_ws10_dff2x": {
            "epochs": 500,
            "batch_sz": 8192,
            "window_sz": 10,
            "lr": 0.0001,
            "dim_feedforward": "2x",
            "pot_q": 0.01,
        },
    },
}

EXPECTED_TRACK_COUNTS = {"SMAP": 54}


def list_profiles(dataset=None):
    datasets = (dataset,) if dataset else DATASETS
    profiles = []

    for name in datasets:
        if name not in PROFILES:
            raise ValueError(f"Unsupported dataset: {name}")

        for identifier in PROFILES[name]:
            profiles.append((name, identifier))

    return profiles


def get_profile(identifier):
    for dataset, profiles in PROFILES.items():
        if identifier in profiles:
            return {
                **COMMON,
                **profiles[identifier],
                "dataset": dataset,
                "identifier": identifier,
            }

    raise ValueError(f"Unknown training profile: {identifier}")


def checkpoint_path(identifier):
    return NATIVE_PACKAGE / "Checkpoints" / f"{identifier}.ckpt"


def result_directory(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return RESULTS_DIR / dataset


def log_directory(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return LOGS_DIR / dataset
