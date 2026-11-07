"""Shared project paths; v12 outputs isolated under results/v12/study."""
from __future__ import annotations
import os
import re
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
DATA_ROOT = Path(os.environ.get("TRANADPLUS_DATA_ROOT", str(PROJECT_ROOT / "data"))).expanduser().resolve()
RAW_ROOT = DATA_ROOT / "raw"
PREPROCESSED_ROOT = DATA_ROOT / "processed" / "tranad_plus"
NATIVE_DATA_ROOT = DATA_ROOT / "native"
OUTPUT_ROOT = Path(os.environ.get("TRANADPLUS_OUTPUT_ROOT", str(PROJECT_ROOT / "results"))).expanduser().resolve()

RESULTS_VERSION = os.environ.get("TRANADPLUS_RESULTS_VERSION", "v12")
if not re.fullmatch(r"v[1-9][0-9]*", RESULTS_VERSION):
    raise ValueError("TRANADPLUS_RESULTS_VERSION must be a plain version like v6 or v7")
CAMPAIGN_SUBDIR = os.environ.get("TRANADPLUS_CAMPAIGN_SUBDIR", "")
if CAMPAIGN_SUBDIR not in ("", "trial", "study", "robustness"):
    raise ValueError("TRANADPLUS_CAMPAIGN_SUBDIR must be trial, study, robustness, or empty")
RESULTS_ROOT = OUTPUT_ROOT / RESULTS_VERSION / CAMPAIGN_SUBDIR
CAMPAIGN_LOG_ROOT = RESULTS_ROOT / "_campaign_logs"

if OUTPUT_ROOT == CODE_ROOT or CODE_ROOT in OUTPUT_ROOT.parents:
    raise ValueError("Output root cannot be inside tranad_plus/; choose project-level results/")


def dataset_result(dataset: str) -> Path:
    if not dataset or dataset in {".", ".."} or "/" in dataset or "\\" in dataset:
        raise ValueError(f"Unsafe dataset name: {dataset!r}")
    return RESULTS_ROOT / dataset


def checkpoint_dir(dataset: str) -> Path:
    return dataset_result(dataset)


def pickle_dir(dataset: str) -> Path:
    return dataset_result(dataset) / "_native_pickles"


def native_environment(dataset: str) -> dict[str, str]:
    return {
        "TRANADPLUS_OUTPUT_ROOT": str(OUTPUT_ROOT),
        "TRANADPLUS_RESULTS_VERSION": RESULTS_VERSION,
        "TRANADPLUS_CAMPAIGN_SUBDIR": CAMPAIGN_SUBDIR,
        "TRANADPLUS_DATA_ROOT": str(DATA_ROOT),
        "TRANADPLUS_NATIVE_CHECKPOINT_DIR": str(checkpoint_dir(dataset)),
        "TRANADPLUS_NATIVE_PICKLE_DIR": str(pickle_dir(dataset)),
        "TRANADPLUS_PREPROCESSED_DIR": str(PREPROCESSED_ROOT),
        "TRANADPLUS_NATIVE_DATASET_DIR": str(NATIVE_DATA_ROOT),
    }
