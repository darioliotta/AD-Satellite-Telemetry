"""Resolve the shared baseline inside the single current source tree."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
BASELINE_ROOT = ROOT.parent


def activate_baseline():
    if not (BASELINE_ROOT / "TranADPlus/src/models.py").is_file():
        raise FileNotFoundError("The current tranad_plus/ tree is incomplete: TranADPlus/src/models.py is missing")
    path = str(BASELINE_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return BASELINE_ROOT
