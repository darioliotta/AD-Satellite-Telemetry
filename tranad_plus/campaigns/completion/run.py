"""V28 completion campaign: grid rescoring, grid seeds 2029-2030, univariate TranAD+ (SMAP/MSL).

Protocol: tranad_plus/campaigns/completion/README.md (fixed before running).
Training, evaluation, models and data adapters are the unchanged v13 implementations in
tranad_plus/core. Outputs: results/v28/{grid_rescore,grid_seeds,univariate}/...

Run from the repository root, e.g.
    python -B -m tranad_plus.campaigns.completion.run --dry-run
    python -B -m tranad_plus.campaigns.completion.run --check
    python -B -m tranad_plus.campaigns.completion.run --shard 0/3
"""
import argparse
import csv
import json
import os
import platform
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path

from tranad_plus.core.bootstrap import ROOT as CORE, PROJECT_ROOT, activate_baseline
from tranad_plus.core.common import digest, hash_file, now, run_lock, save_npz, source_manifest, write_json

activate_baseline()                     # puts tranad_plus/ on sys.path (gpu_settings, TranADPlus)
STUDIES = ("univariate", "grid_seeds", "grid_rescore")
GRID_TABLE = PROJECT_ROOT / "results/analysis_v10_v12/runs_270.csv"
NEW_SEEDS = (2029, 2030)
UNIVARIATE = {"datasets": ("SMAP", "MSL"), "variants": ("reference", "encoder_only"),
              "seeds": (2026, 2027, 2028, 2029, 2030), "dim_feedforward": 8, "coordinates": [0]}


# ------------------------------------------------------------------ plan
def baselines():
    return json.loads((CORE / "configs/baselines.json").read_text())


def profile_for(study, dataset, variant, window, ff, seed, features=None, extra=None):
    base = baselines(); info = base["datasets"][dataset]
    p = {**base["training"], **info["training"], "dataset": dataset, "features": features or info["features"],
         "variant": variant, "window_sz": int(window), "dim_feedforward": int(ff), "seed": int(seed),
         "configuration": f"w{int(window)}_ff{int(ff)}", "study": study, **(extra or {})}
    p["identifier"] = f"toast28_{study}_{dataset.lower().replace('-', '')}_{variant}_w{int(window):03d}_ff{int(ff):03d}_s{seed}"
    return p


def grid_rows():
    with open(GRID_TABLE) as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 270:
        raise ValueError("Expected the 270 audited v10/v12 grid runs")
    return rows


def plan(studies):
    jobs = []
    rows = grid_rows()
    if "univariate" in studies:
        for ds in UNIVARIATE["datasets"]:
            window = baselines()["datasets"][ds]["configurations"]["reference"]["window_sz"]
            for v in UNIVARIATE["variants"]:
                for s in UNIVARIATE["seeds"]:
                    jobs.append(("train", profile_for("univariate", ds, v, window, UNIVARIATE["dim_feedforward"], s,
                                                      features=1, extra={"input_coordinates": UNIVARIATE["coordinates"]})))
    if "grid_seeds" in studies:
        cells = sorted({(r["dataset"], int(r["window"]), int(r["feedforward"])) for r in rows})
        if len(cells) != 90:
            raise ValueError("Expected 30 grid cells per dataset")
        for ds, w, ff in cells:
            for s in NEW_SEEDS:
                jobs.append(("train", profile_for("grid_seeds", ds, "reference", w, ff, s)))
    if "grid_rescore" in studies:
        for r in rows:
            ckpt = PROJECT_ROOT / "results" / r["campaign"] / "study" / r["dataset"] / f"{r['identifier']}.ckpt"
            p = profile_for("grid_rescore", r["dataset"], "reference", r["window"], r["feedforward"], r["seed"],
                            extra={"source_checkpoint": str(ckpt.relative_to(PROJECT_ROOT)),
                                   "source_identifier": r["identifier"],
                                   "source_bf_f1": float(r["bf_f1"]), "source_pot_f1": float(r["pot_f1"]) if r["pot_f1"] else None})
            jobs.append(("rescore", p))
    if len({p["identifier"] for _, p in jobs}) != len(jobs):
        raise ValueError("Duplicate identifiers")
    return jobs


def folder_of(output, p):
    return Path(output) / "v28" / p["study"] / p["dataset"] / p["variant"] / f"w{p['window_sz']:03d}_ff{p['dim_feedforward']:03d}" / f"seed_{p['seed']}"


# ------------------------------------------------------------------ data
def load_profile(p):
    """Profile used to read files: the full input width of the dataset."""
    return {**p, "features": baselines()["datasets"][p["dataset"]]["features"]}


def load_bundle(p, data_root, files=None):
    from tranad_plus.core.data import load_data
    bundle = load_data(load_profile(p), data_root, files)
    coords = p.get("input_coordinates")
    if coords is not None:
        import torch
        for ds in bundle.datasets.values():           # min-max scaling is per coordinate, so slicing after it is exact
            ds.big_track = ds.big_track[:, coords].contiguous()
            ds.big_anom = ds.big_anom[:, coords].contiguous()
        bundle.inventory["input_coordinates"] = coords
    return bundle


# ------------------------------------------------------------------ jobs
def run_train(p, folder, data_root, device, settings, sources, execution):
    from tranad_plus.core.data import inventory
    from tranad_plus.core.runner import bind_run, logged
    from tranad_plus.core.training.engine import train
    from tranad_plus.core.evaluation.engine import evaluate, verified_metrics
    files = inventory(load_profile(p), data_root)
    binding = {"profile": p, "input_files": files, "sources_sha256": sources["sha256"],
               "execution": execution, "evaluation_settings": settings}
    signature = digest(binding)
    bind_run(folder, p, signature, {"sources": sources, "input_files": files, "execution": execution,
                                    "evaluation_settings": settings, "host": platform.node()})
    if verified_metrics(folder, signature) is not None:
        return "skipped_verified_complete"
    bundle = load_bundle(p, data_root, files)
    write_json(folder / "data_inventory.json", bundle.inventory)
    with logged(folder / "train.log"):
        train(p, bundle, folder, signature, device)
    with logged(folder / "evaluation.log"):
        evaluate(p, bundle, folder, signature, device, settings)
    return "complete"


def run_rescore(p, folder, data_root, device, settings, sources, execution):
    import numpy as np
    import torch
    from tranad_plus.core.runner import logged
    from tranad_plus.core.models import build_model
    from tranad_plus.core.evaluation.scoring import collect_outputs, reconstruction_scores
    from tranad_plus.core.evaluation.metrics import evaluate_oracle
    done = folder / "metrics.json"
    if done.exists():
        old = json.loads(done.read_text())
        if old.get("status") == "complete" and hash_file(folder / "scores.npz") == old["artifacts"]["scores.npz"]:
            return "skipped_verified_complete"
    ckpt = PROJECT_ROOT / p["source_checkpoint"]
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)      # project-owned v10/v12 checkpoint
    hp = saved["hp_dict"]
    for key in ("window_sz", "dim_feedforward", "batch_sz", "lr", "epochs", "features"):
        if int(hp[key]) != int(p[key]) if key != "lr" else float(hp[key]) != float(p[key]):
            raise RuntimeError(f"Checkpoint {key} differs from the plan: {hp[key]} vs {p[key]}")
    bundle = load_bundle(p, data_root)
    model = build_model(p).to(device)
    model.load_state_dict(saved["model_states"], strict=True)
    with logged(folder / "evaluation.log"):
        outputs = {k: collect_outputs(model, d, device, settings["score_batch_size_by_dataset"][p["dataset"]],
                                      f"RESCORE {p['dataset']} {k}", folder / "numerical_diagnostics.json")
                   for k, d in bundle.datasets.items()}
        alphas = [float(a) for a in np.linspace(0, 1, settings["alpha_steps"])]
        grids = {k: np.stack([reconstruction_scores(v, a) for a in alphas]) for k, v in outputs.items()}
        arrays = {f"{k}_scores": g for k, g in grids.items()}
        arrays.update({f"{k}_labels": v for k, v in bundle.labels.items()})
        arrays["alpha_values"] = np.array(alphas); arrays["alpha_applicable"] = np.array([True])
        for k, groups in bundle.groups.items():
            if groups:
                arrays[f"{k}_segment_ids"] = np.concatenate([np.repeat(g["id"], len(g["values"])) for g in groups])
                arrays[f"{k}_channels"] = np.concatenate([np.repeat(g["channel"], len(g["values"])) for g in groups])
        save_npz(folder / "scores.npz", **arrays)
        oracle = evaluate_oracle(grids["train"], grids["test"], bundle.labels["test"], alphas, p["pot_q"], settings,
                                 groups=bundle.groups.get("test"))
    bf = oracle["bf"]["bf_f1"]
    write_json(folder / "metrics.json", {
        "schema_version": 1, "status": "complete", "created_at": now(), "profile": p,
        "source_checkpoint_sha256": hash_file(ckpt), "paper_oracle": oracle,
        "agreement_with_source_report": {"source_bf_f1": p["source_bf_f1"], "recomputed_bf_f1": bf,
                                         "absolute_difference": abs(bf - p["source_bf_f1"])},
        "sources_sha256": sources["sha256"], "execution": execution, "evaluation_settings": settings,
        "artifacts": {"scores.npz": hash_file(folder / "scores.npz")}})
    return "complete"


# ------------------------------------------------------------------ compact copy for transfer
COMPACT_ALPHAS = (0.0, 0.5, 1.0)


def write_compact(folder):
    """scores_compact.npz: alpha rows 0, 0.5, 1 only (full 21-row scores.npz stays on the server).

    The continuous-score AUC for all 21 alpha values is already stored in metrics.json
    (paper_oracle.alpha_grid), so the compact file loses no reported quantity.
    """
    import numpy as np
    full = folder / "scores.npz"
    out = folder / "scores_compact.npz"
    if not full.is_file() or out.is_file():
        return
    z = np.load(full)
    alphas = z["alpha_values"]
    rows = [int(np.argmin(np.abs(alphas - a))) for a in COMPACT_ALPHAS] if len(alphas) else [0]
    arrays = {k: (z[k][rows] if k.endswith("_scores") else z[k]) for k in z.files}
    arrays["alpha_values"] = alphas[rows] if len(alphas) else alphas
    save_npz(out, **arrays)


# ------------------------------------------------------------------ main
def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--study", choices=STUDIES, action="append")
    a.add_argument("--shard", default="0/1", help="i/n: this process runs jobs with index %% n == i")
    a.add_argument("--device", default="cuda:0")
    a.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    a.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--check", action="store_true")
    a.add_argument("--smoke-epochs", type=int, help="testing only: override epochs (use a scratch --output-root)")
    a.add_argument("--limit", type=int, help="testing only: run the first N jobs of this shard")
    a.add_argument("--only", help="testing only: keep jobs whose identifier contains this text")
    args = a.parse_args(argv)
    i, n = (int(x) for x in args.shard.split("/"))
    jobs = [j for k, j in enumerate(plan(args.study or STUDIES)) if k % n == i]
    if args.only:
        jobs = [j for j in jobs if args.only in j[1]["identifier"]]
    if args.smoke_epochs:
        jobs = [(kind, {**p, "epochs": args.smoke_epochs}) for kind, p in jobs]
    if args.limit:
        jobs = jobs[:args.limit]
    counts = {}
    for kind, p in jobs:
        counts[p["study"]] = counts.get(p["study"], 0) + 1
    print(f"V28 shard {i}/{n}: {len(jobs)} jobs {counts}", flush=True)
    if args.dry_run:
        for kind, p in jobs:
            print(f"  {kind:7s} {p['identifier']}", flush=True)
        return 0
    os.environ["TRANADPLUS_CUDA_PRECISION"] = "faithful"
    os.environ["TRANADPLUS_ADAM_IMPL"] = "standard"
    os.environ["TRANADPLUS_GPU_CACHE"] = "auto"
    os.environ.pop("TRANADPLUS_CAMPAIGN_PLAN", None)
    os.environ["TRANADPLUS_DATA_ROOT"] = str(args.data_root.resolve())
    os.environ["TRANADPLUS_OUTPUT_ROOT"] = str(args.output_root.resolve())
    import torch
    from tranad_plus.core.planning import validate_output_root
    output = validate_output_root(args.output_root, args.data_root)
    settings = baselines()["evaluation"]
    sources = source_manifest()
    if args.check:
        missing = [p["source_checkpoint"] for k, p in jobs if k == "rescore" and not (PROJECT_ROOT / p["source_checkpoint"]).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} grid checkpoints missing, e.g. {missing[:3]}")
        seen = set()
        for kind, p in jobs:
            key = (p["dataset"], p["window_sz"], tuple(p.get("input_coordinates") or []))
            if key not in seen:
                b = load_bundle(p, args.data_root)
                print(f"CHECK OK {key}: " + str({k: (len(v), tuple(v[0][0].shape) if not isinstance(v[0], tuple) else tuple(v[0][0].shape))
                                                  for k, v in b.datasets.items()}), flush=True)
                seen.add(key)
        print(f"CHECK OK: sources pinned, {len(jobs)} jobs, checkpoints present", flush=True)
        return 0
    from gpu_settings import configure
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    configure()
    execution = {"hostname": platform.node(), "python": platform.python_version(), "torch": str(torch.__version__),
                 "numpy": version("numpy"), "sklearn": version("scikit-learn"), "cuda": torch.version.cuda,
                 "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                 "precision": "fp32_tf32_disabled", "optimizer": "Adam_standard"}
    log = output / "v28" / "_campaign_logs" / f"shard_{i}_of_{n}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    report = {"started_at": now(), "shard": args.shard, "studies": args.study or STUDIES, "runs": []}
    failures = 0
    for k, (kind, p) in enumerate(jobs, 1):
        folder = folder_of(output, p)
        print(f"\n[{k}/{len(jobs)}] {kind} {p['identifier']}", flush=True)
        started = time.time()
        record = {"identifier": p["identifier"], "kind": kind, "folder": str(folder)}
        try:
            with run_lock(folder):
                fn = run_train if kind == "train" else run_rescore
                record["status"] = fn(p, folder, args.data_root, device, settings, sources, execution)
                write_compact(folder)
        except Exception as exc:
            failures += 1
            record.update({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
            traceback.print_exc()
            print(f"FAILED {p['identifier']}: {record['reason']}", flush=True)
        record["seconds"] = round(time.time() - started, 1)
        report["runs"].append(record)
        write_json(log, report)
        print(f"[{k}/{len(jobs)}] {record['status']} in {record['seconds']}s", flush=True)
    report.update({"finished_at": now(), "failures": failures})
    write_json(log, report)
    print(f"V28 shard {i}/{n} finished: {len(jobs) - failures} ok, {failures} failed. Log: {log}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
