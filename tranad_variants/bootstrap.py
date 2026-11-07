"""Use the sibling baseline explicitly; never copy or modify its implementation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
BASELINE_ROOT = PROJECT_ROOT / "tranad_plus"


def activate_baseline():
    if not (BASELINE_ROOT / "TranADPlus/src/models.py").is_file():
        raise FileNotFoundError("Keep tranad_variants/ and tranad_plus/ together in the project root")
    path = str(BASELINE_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return BASELINE_ROOT
