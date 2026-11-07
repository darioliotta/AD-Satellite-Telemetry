"""Read-only verification of a v12 campaign's dependencies, device and input paths."""
from __future__ import annotations
import argparse
import importlib
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))


def inventory(dataset, data_root, opssat_csv=None):
    errors = []
    if dataset == 'OPS-SAT':
        if opssat_csv is None or not opssat_csv.is_file():
            return [f'OPS-SAT segments.csv missing: {opssat_csv}']
        # Full CSV/content verification is performed by the OPS-SAT data adapter at training.
        return []
    base = data_root / 'processed' / 'tranad_plus' / dataset
    for split in ('Train', 'Test'):
        for kind in ('Tracks', 'Labels'):
            path = base / split / kind
            if not path.is_dir() or not any(path.glob('*.npy')):
                errors.append(f'Missing .npy in {path}')
        tracks = {p.stem for p in (base / split / 'Tracks').glob('*.npy')}
        labels = {p.stem for p in (base / split / 'Labels').glob('*.npy')}
        if tracks != labels:
            errors.append(f'{dataset}/{split}: Tracks/Labels mismatch')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--only', action='append')
    args = parser.parse_args()
    failures = []
    if sys.version_info < (3, 10):
        failures.append('Requires Python 3.10+')
    for name in ('torch', 'numpy', 'pandas', 'scipy', 'sklearn', 'tqdm'):
        try:
            package = importlib.import_module(name)
            print(f'{name}: {getattr(package, "__version__", "installed")}', flush=True)
        except ImportError as exc:
            failures.append(f'{name}: {exc}')
    if not failures:
        from run_experiments import check_device
        try: check_device(args.device)
        except Exception as exc: failures.append(f'Device unavailable: {exc}')
    from campaign_plan import register_plan_profiles
    from config import PROFILES
    from project_paths import DATA_ROOT, RESULTS_ROOT
    names = register_plan_profiles(PROFILES, args.plan)
    selected = set(args.only or names)
    if not selected.issubset(names):
        failures.append('Unknown --only experiment identifiers')
    by_dataset = {}
    for dataset, values in PROFILES.items():
        for name, profile in values.items():
            if name in selected:
                by_dataset.setdefault(dataset, []).append(profile)
    for dataset, profiles in by_dataset.items():
        opssat_csv = Path(profiles[0]['segments_csv']) if dataset == 'OPS-SAT' else None
        issues = inventory(dataset, DATA_ROOT, opssat_csv)
        failures.extend(issues)
        print(f'{dataset}: {len(profiles)} profiles -> {"READY" if not issues else "MISSING"}', flush=True)
    print(f'Data: {DATA_ROOT}\nOutputs: {RESULTS_ROOT}', flush=True)
    for failure in failures: print('ERROR:', failure, file=sys.stderr)
    if failures: return 1
    print('PREFLIGHT PASS. No training/evaluation launched.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
