"""Recovery/target-preservation tests for the independent diagnostic studies."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import os
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import TensorDataset

from tranad_variants.data import DataBundle
from tranad_variants.training.engine import train, TrainingInterrupted
from tranad_variants.common import hash_file
from tranad_v14.continuation import initialize
from tranad_v14.context import perturb, raw_best_f1, real_point_mask
from tranad_v14.run import arguments, plan, main
from tranad_v14.progress import training as training_progress
from tranad_v14.metrics import pa_stats, dense_pa_f1


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_interventions_preserve_target_and_original_input(self):
        x = torch.arange(24.).reshape(4, 3, 2)
        original = x.clone()
        for condition in ("original", "reverse_past", "repeat_current"):
            changed = perturb(x, condition)
            torch.testing.assert_close(changed[-1], original[-1])
            torch.testing.assert_close(x, original)
        torch.testing.assert_close(perturb(x, "reverse_past")[:-1], x[:-1].flip(0))
        torch.testing.assert_close(perturb(x, "repeat_current"), x[-1:].expand_as(x))

    def test_continuation_and_interrupted_continuation_equal_uninterrupted(self):
        windows = torch.rand(12, 3, 1, generator=torch.Generator().manual_seed(33))
        bundle = DataBundle({"train": TensorDataset(windows)}, {}, {})
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            for variant in ("reference", "encoder_only"):
                p = dict(dataset="OPS-SAT", variant=variant, features=1, window_sz=3,
                         batch_sz=4, dim_feedforward=4, num_layers=1, epochs=3,
                         seed=2026, lr=.001, identifier=variant)
                parent, full, fork = (root / variant / k for k in ("parent", "full", "fork"))
                train({**p, "epochs": 6}, bundle, full, "full", "cpu")
                train(p, bundle, parent, "parent", "cpu")
                before = {name: hash_file(parent / name) for name in
                          ("checkpoints/final.ckpt", "checkpoints/resume.ckpt", "epochs.jsonl")}
                extended = initialize(parent, fork, p, "parent", "fork", 3)
                with patch.dict(os.environ, {"TOAST_NO_TQDM": "0"}), redirect_stderr(io.StringIO()):
                    with self.assertRaises(TrainingInterrupted), training_progress(fork, 3, 6, "OPS-SAT"):
                        train(extended, bundle, fork, "fork", "cpu", stop_after_epoch=4)
                    with training_progress(fork, 3, 6, "OPS-SAT"):
                        train(extended, bundle, fork, "fork", "cpu")
                self.assertEqual(json.loads((fork / "progress.json").read_text())["epoch"], 6)
                a, b = (torch.load(f / "checkpoints/final.ckpt", weights_only=True) for f in (full, fork))
                for k in a["model_state"]:
                    torch.testing.assert_close(a["model_state"][k], b["model_state"][k], rtol=0, atol=0)
                self.assertEqual(a["scheduler_state"], b["scheduler_state"])
                self.assertEqual(a["final_loss"], b["final_loss"])
                self.assertEqual(before, {name: hash_file(parent / name) for name in before})
                history = [json.loads(line)["epoch_one_based"] for line in (fork / "epochs.jsonl").read_text().splitlines()]
                self.assertEqual(history, list(range(1, 7)))

    def test_dense_raw_sweep_handles_ties(self):
        from sklearn.metrics import f1_score
        scores = np.array([.2, .2, .4, .4, .8, .9], dtype=np.float32)
        y = np.array([0, 1, 0, 1, 1, 0])
        result = raw_best_f1(y, scores)
        thresholds = np.r_[np.nextafter(scores.min(), np.float32(-np.inf)), np.unique(scores)]
        self.assertEqual(result["f1"], max(f1_score(y, scores > t) for t in thresholds))

    def test_default_plan_is_60_evaluations_then_30_continuations(self):
        jobs = plan(arguments([]))
        self.assertEqual(sum(s == "context" for s, _ in jobs), 60)
        self.assertEqual(sum(s == "continue" for s, _ in jobs), 30)
        self.assertEqual({p.parts[0] for s, p in jobs if s == "continue"}, {"OPS-SAT", "SMAP", "MSL"})

    def test_native_window_comparison_retains_same_real_target_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracks = root / "processed/tranad_plus/SMAP/Test/Tracks"
            tracks.mkdir(parents=True)
            for i, n in enumerate((7, 11)):
                np.save(tracks / f"{i}.npy", np.zeros((n, 25)))
            for w in (3, 5, 10):
                count = 18 + w-1  # two padded tracks minus initial w-1
                mask = real_point_mask({"dataset": "SMAP", "window_sz": w}, root, count)
                self.assertEqual(int(mask.sum()), 18)
                self.assertEqual(len(mask), count)

    def test_window_pa_diagnostic_matches_original_at_all_thresholds(self):
        from TranADPlus.src.omnianomaly import adjust_predicts
        from opssat_pipeline import adjust_per_fragment
        scores = np.array([.1, .6, .2, .9, .3, .8, .1, .7], dtype=np.float32)
        labels = np.array([1, 1, 0, 1, 1, 1, 0, 0], dtype=np.int8)
        thresholds = np.r_[np.nextafter(float(scores.min()), -np.inf), np.unique(scores)]
        for boundaries in (None, np.array([4])):
            values = pa_stats(scores, labels, thresholds, boundaries)
            for i, threshold in enumerate(thresholds):
                if boundaries is None:
                    pred = adjust_predicts(scores, labels, threshold).astype(bool)
                else:
                    groups = [{"values": np.zeros(4)}, {"values": np.zeros(4)}]
                    pred = adjust_per_fragment(scores, labels, groups, threshold).astype(bool)
                self.assertEqual(values["tp"][i], int((pred & labels.astype(bool)).sum()))
                self.assertEqual(values["fp"][i], int((pred & ~labels.astype(bool)).sum()))
            self.assertAlmostEqual(dense_pa_f1(scores, labels, boundaries)["f1"], max(values["f1"]))

    def test_real_runner_end_to_end_and_safe_rerun(self):
        from tranad_variants.tests.test_pipeline import ops_fixture
        from tranad_variants.runner import main as original_main
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            data, parents, out = root / "data", root / "parents", root / "diagnostics"
            ops_fixture(data)
            parent_plan = root / "parent_plan.json"
            parent_plan.write_text(json.dumps({"schema_version": 1, "name": "fixture", "datasets": ["OPS-SAT"],
                                              "variants": ["encoder_only"], "configurations": ["reference"], "seeds": [2026]}))
            with patch("opssat_pipeline._pot_threshold", return_value=(.5, None)):
                self.assertEqual(original_main(["--plan", str(parent_plan), "--dataset", "OPS-SAT", "--variant", "encoder_only",
                                               "--data-root", str(data), "--output-root", str(parents),
                                               "--device", "cpu", "--cpu-threads", "1"]), 0)
                args = ["--stage", "both", "--no-progress", "--dataset", "OPS-SAT", "--variant", "encoder_only", "--seed", "2026",
                        "--data-root", str(data), "--parent-root", str(parents / "v13/architectures"),
                        "--output-root", str(out), "--device", "cpu", "--cpu-threads", "1"]
                # Context includes both windows; the fixture has one parent.
                with patch("tranad_v14.run.plan", return_value=[("context", Path("OPS-SAT/encoder_only/reference/seed_2026")),
                                                                       ("continue", Path("OPS-SAT/encoder_only/reference/seed_2026"))]):
                    self.assertEqual(main(args), 0)
                    after = out / "continuation/OPS-SAT/encoder_only/reference/seed_2026"
                    result = json.loads((after / "metrics.json").read_text())
                    self.assertEqual(result["training"]["epochs_logged"], 100)
                    self.assertEqual(json.loads((after / "comparison.json").read_text())["final_epochs"], 100)
                    context = json.loads((out / "context/OPS-SAT/encoder_only/reference/seed_2026/metrics.json").read_text())
                    self.assertEqual(len(context["conditions"]), 3)
                    self.assertIn("fragment", context["conditions"][0])
                    before = hash_file(after / "checkpoints/final.ckpt")
                    self.assertEqual(main(args), 0)
                    self.assertEqual(before, hash_file(after / "checkpoints/final.ckpt"))


if __name__ == "__main__":
    unittest.main()
