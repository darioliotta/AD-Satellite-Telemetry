import functools
import runpy
import sys
from pathlib import Path

from TranADPlus.src import helpers


TOTAL_ALPHA_TRIALS = 21


def main():
    native = Path(__file__).resolve().parent / "TranADPlus" / "scripts" / "evaluation.py"
    if not native.is_file():
        raise FileNotFoundError(native)

    search_enabled = False
    if "--alpha_search" in sys.argv:
        index = sys.argv.index("--alpha_search")
        if index + 1 < len(sys.argv):
            search_enabled = bool(sys.argv[index + 1])
    total = TOTAL_ALPHA_TRIALS if search_enabled else 1
    count = 0
    original = helpers.evaluate_TranAD

    @functools.wraps(original)
    def counted_evaluation(*args, **kwargs):
        nonlocal count
        count += 1
        alpha = kwargs.get("alpha")
        beta = kwargs.get("beta")
        details = ""
        if alpha is not None and beta is not None:
            details = f" | alpha={float(alpha):.2f}, beta={float(beta):.2f}"
        print(f"[EVALUATION {count}/{total}] START{details}", flush=True)
        result = original(*args, **kwargs)
        print(f"[EVALUATION {count}/{total}] COMPLETE{details}", flush=True)
        return result

    helpers.evaluate_TranAD = counted_evaluation
    try:
        sys.argv[0] = str(native)
        runpy.run_path(str(native), run_name="__main__")
    finally:
        helpers.evaluate_TranAD = original


if __name__ == "__main__":
    main()
