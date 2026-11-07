"""Sequential train -> evaluate campaigns; completed jobs are verified and reused."""
import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import traceback
from datetime import datetime, timezone

from .bootstrap import ROOT, PROJECT_ROOT
from .common import digest, now, run_lock, source_manifest, write_json
from .planning import CAMPAIGNS, DATASETS, VARIANTS, load_campaign, validate_output_root


class Tee:
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile

    def write(self, value):
        self.console.write(value)
        self.logfile.write(value)
        return len(value)

    def flush(self):
        self.console.flush()
        self.logfile.flush()

    def isatty(self):
        return self.console.isatty()


@contextmanager
def logged(path):
    with Path(path).open("a", buffering=1, encoding="utf-8") as stream:
        with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
            try:
                yield
            except Exception:
                traceback.print_exc()
                raise


def bind_run(folder, profile, signature, manifest):
    path = folder / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("signature") != signature or json.loads((folder / "config.json").read_text()) != profile:
            raise RuntimeError(f"Existing experiment differs in code/data/configuration/environment: {folder}")
        return
    if (folder / "checkpoints").exists() or (folder / "metrics.json").exists():
        raise RuntimeError("Existing artifacts without provenance; refusing to adopt/overwrite them")
    write_json(folder / "config.json", profile)
    write_json(path, {"created_at": now(), "signature": signature, **manifest})


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    plan = parser.add_mutually_exclusive_group()
    plan.add_argument("--campaign", choices=CAMPAIGNS, default="pilot")
    plan.add_argument("--plan", type=Path)
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument("--variant", choices=VARIANTS, action="append")
    parser.add_argument("--only", help="Exact experiment identifier")
    parser.add_argument("--stage", choices=("train", "evaluate", "both"), default="both")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results",
                        help="Parent results directory; always appends v13/architectures")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-cache", choices=("auto", "off"), default="auto")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing results")
    parser.add_argument("--check", action="store_true", help="Validate source/data/splits without training or writing results")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failed job; default continues")
    parser.add_argument("--stop-after-epoch", type=int, help="Recovery test: keep configured total epochs; exit 75 after saving")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    plan, experiments, settings = load_campaign(args.campaign, args.plan)
    experiments = [e for e in experiments if
                   (not args.dataset or e.profile["dataset"] in args.dataset) and
                   (not args.variant or e.profile["variant"] in args.variant) and
                   (not args.only or e.identifier == args.only)]
    if not experiments:
        raise ValueError("The selected plan/filter contains no experiments")
    if args.cpu_threads is not None and args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        raise ValueError("stop-after-epoch must be positive")
    output = validate_output_root(args.output_root, args.data_root)
    sources = source_manifest()
    print(f"Campaign {plan['name']}: {len(experiments)} experiments; stage={args.stage}", flush=True)
    print("Order: TRAIN -> EVALUATE for each experiment. Output: " + str(output / "v13/architectures"), flush=True)
    for index, experiment in enumerate(experiments, 1):
        p = experiment.profile
        print(f"{index:03d}/{len(experiments):03d} {experiment.identifier} "
              f"W={p['window_sz']} FF={p['dim_feedforward']} batch={p['batch_sz']} epochs={p['epochs']}", flush=True)
    if args.dry_run:
        return 0
    # Set numerical policy explicitly; inherited benchmark campaign variables
    # must not redirect this runner's outputs or enable fused/mixed execution.
    os.environ["TRANADPLUS_CUDA_PRECISION"] = "faithful"
    os.environ["TRANADPLUS_ADAM_IMPL"] = "standard"
    os.environ["TRANADPLUS_GPU_CACHE"] = args.gpu_cache
    os.environ.pop("TRANADPLUS_CAMPAIGN_PLAN", None)
    os.environ["TRANADPLUS_RESULTS_VERSION"] = "v13"
    os.environ["TRANADPLUS_CAMPAIGN_SUBDIR"] = ""
    os.environ["TRANADPLUS_DATA_ROOT"] = str(args.data_root.resolve())
    os.environ["TRANADPLUS_OUTPUT_ROOT"] = str(output)
    from .data import inventory, load_data
    import torch
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    if args.check:
        checked = set()
        for e in experiments:
            key = (e.profile["dataset"], e.profile["configuration"])
            if key not in checked:
                bundle = load_data(e.profile, args.data_root)
                print(f"CHECK OK {key}: " + str({k: len(v) for k, v in bundle.datasets.items()}), flush=True)
                checked.add(key)
        return 0
    from gpu_settings import configure
    from .training.engine import train, TrainingInterrupted
    from .evaluation.engine import evaluate, verified_metrics
    from .reports.summarize import summarize
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This campaign validates CPU and CUDA only")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu explicitly for CPU work")
    configure()
    execution = {"hostname": platform.node(), "processor": platform.processor(),
                 "platform": platform.platform(), "python": platform.python_version(),
                 "torch": str(torch.__version__), "numpy": version("numpy"),
                 "pandas": version("pandas"), "sklearn": version("scikit-learn"), "scipy": version("scipy"),
                 "cuda": torch.version.cuda, "device": str(device), "cpu_threads": torch.get_num_threads(),
                 "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                 "precision": "fp32_tf32_disabled", "optimizer": "Adam_standard", "gpu_cache": args.gpu_cache}
    root = output / "v13/architectures"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = root / "_campaign_logs" / f"{plan['name']}_{stamp}.json"
    report = {"name": plan["name"], "started_at": now(), "stage": args.stage,
              "plan": plan, "planned_identifiers": [e.identifier for e in experiments], "runs": []}
    write_json(report_path, report)
    failures = 0
    for index, e in enumerate(experiments, 1):
        folder = e.folder(output)
        print(f"\n[{index}/{len(experiments)}] START {e.identifier}", flush=True)
        record = {"identifier": e.identifier, "started_at": now(), "folder": str(folder)}
        try:
            with run_lock(folder):
                files = inventory(e.profile, args.data_root)
                binding = {"profile": e.profile, "input_files": files, "sources_sha256": sources["sha256"],
                           "execution": execution, "evaluation_settings": settings}
                signature = digest(binding)
                bind_run(folder, e.profile, signature,
                         {"sources": sources, "input_files": files, "execution": execution,
                          "evaluation_settings": settings, "host": platform.node(), "platform": platform.platform()})
                try:
                    already = verified_metrics(folder, signature)
                    if already is not None:
                        record["status"] = "skipped_verified_complete"
                    else:
                        write_json(folder / "status.json", {**record, "status": "loading_data", "updated_at": now()})
                        bundle = load_data(e.profile, args.data_root, files)
                        write_json(folder / "data_inventory.json", bundle.inventory)
                        if args.stage in ("train", "both"):
                            write_json(folder / "status.json", {**record, "status": "training", "updated_at": now()})
                            with logged(folder / "train.log"):
                                train(e.profile, bundle, folder, signature, device, stop_after_epoch=args.stop_after_epoch)
                        if args.stage in ("evaluate", "both"):
                            write_json(folder / "status.json", {**record, "status": "evaluating", "updated_at": now()})
                            with logged(folder / "evaluation.log"):
                                evaluate(e.profile, bundle, folder, signature, device, settings)
                        record["status"] = "trained" if args.stage == "train" else "complete"
                        write_json(folder / "status.json", {**record, "updated_at": now()})
                except TrainingInterrupted:
                    write_json(folder / "status.json", {**record, "status": "interrupted_after_checkpoint", "updated_at": now()})
                    raise
                except Exception as exc:
                    # Write failure state while the run lock is still held.
                    # Do not replace a successful report when integrity checks fail.
                    if not (folder / "metrics.json").exists():
                        write_json(folder / "status.json", {**record, "status": "failed", "updated_at": now(),
                                   "reason": f"{type(exc).__name__}: {exc}"})
                    raise
        except TrainingInterrupted as exc:
            record.update({"status": "interrupted_after_checkpoint", "reason": str(exc)})
            report["runs"].append(record)
            report["status"] = "interrupted_after_checkpoint"
            write_json(report_path, report)
            return 75
        except Exception as exc:
            failures += 1
            record.update({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED {e.identifier}: {record['reason']}", flush=True)
        record["finished_at"] = now()
        report["runs"].append(record)
        write_json(report_path, report)
        summarize(root)
        if failures and args.fail_fast:
            break
    report.update({"finished_at": now(), "status": "complete" if not failures else "completed_with_failures",
                   "failures": failures})
    write_json(report_path, report)
    print(f"Campaign report: {report_path}", flush=True)
    return 1 if failures else 0
