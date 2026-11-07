from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from tranad_variants.data import DataBundle, load_data
from tranad_variants.training.engine import train, TrainingInterrupted
from tranad_variants.evaluation.metrics import evaluate_oracle
from tranad_variants.evaluation.selection import validation_selected
from tranad_variants.evaluation.scoring import collect_outputs
from tranad_variants.planning import load_campaign
from tranad_variants.runner import main
from tranad_variants.reports.plot_tradeoffs import main as plot_tradeoffs
from paper_metrics import paper_best_f1
from opssat_pipeline import prepare_data, FragmentWindows, original_metrics


def native_fixture(root):
    rng = np.random.default_rng(52)
    for split in ("Train", "Test"):
        for name in ("Tracks", "Labels"):
            (root / "processed/tranad_plus/SMAP" / split / name).mkdir(parents=True)
        for i in range(2):
            values = rng.uniform(.1, .9, (35, 25)).astype(np.float32)
            labels = np.zeros_like(values)
            if split == "Test":
                values[15:22] += 1
                labels[15:22, 0] = 1
            for kind, array in (("Tracks", values), ("Labels", labels)):
                np.save(root / "processed/tranad_plus/SMAP" / split / kind / f"{i}.npy", array)


def ops_fixture(root):
    rows = []
    rng = np.random.default_rng(42)
    for segment in range(10):
        train_split = int(segment < 6)
        anomaly = int(segment in (5, 8, 9))
        for t, value in enumerate(rng.uniform(.1, .9, 12) + anomaly):
            rows.append({"channel": "ch1", "segment": segment, "timestamp": str(t),
                         "value": value, "train": train_split, "anomaly": anomaly,
                         "label": "anomaly" if anomaly else "normal"})
    path = root / "raw/OPSSAT/segments.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_recovery_is_bitwise_uninterrupted_for_all_variants(self):
        rng = torch.Generator().manual_seed(99)
        windows = torch.rand(12, 3, 1, generator=rng)
        data = TensorDataset(windows, torch.zeros(12))
        bundle = DataBundle({"train": data}, {}, {})
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            for variant in ("reference", "shared_decoder", "single_pass", "encoder_only"):
                p = {"dataset": "OPS-SAT", "variant": variant, "features": 1, "window_sz": 3,
                     "batch_sz": 4, "dim_feedforward": 4, "num_layers": 1, "epochs": 3,
                     "seed": 2026, "lr": .001, "identifier": variant}
                full, resumed = root / variant / "full", root / variant / "resumed"
                train(p, bundle, full, "same_signature", "cpu")
                with self.assertRaises(TrainingInterrupted):
                    train(p, bundle, resumed, "same_signature", "cpu", stop_after_epoch=1)
                train(p, bundle, resumed, "same_signature", "cpu")
                a = torch.load(full / "checkpoints/final.ckpt", weights_only=True)
                b = torch.load(resumed / "checkpoints/final.ckpt", weights_only=True)
                self.assertEqual(a["final_loss"], b["final_loss"])
                for key in a["model_state"]:
                    torch.testing.assert_close(a["model_state"][key], b["model_state"][key], rtol=0, atol=0)
                with self.assertRaises(RuntimeError):
                    train(p, bundle, resumed, "different_signature", "cpu")

    def test_ops_split_and_windows_match_audited_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = ops_fixture(root)
            p = next(e.profile for e in load_campaign("pilot")[1] if e.profile["dataset"] == "OPS-SAT")
            bundle = load_data(p, root)
            tr, te, scalers, info, va = prepare_data({**p, "segments_csv": str(csv)}, return_validation=True)
            self.assertEqual(bundle.inventory["split"]["train_fragment_ids"], info["train_fragment_ids"])
            self.assertEqual(bundle.inventory["split"]["validation_fragment_ids"], info["validation_fragment_ids"])
            for split, groups in (("train", tr), ("test", te), ("validation", va)):
                expected = FragmentWindows(groups, scalers, p["window_sz"])
                torch.testing.assert_close(bundle.datasets[split].windows, expected.windows, rtol=0, atol=0)
            for seed in (2027, 2028):
                other = load_data({**p, "seed": seed}, root)
                self.assertEqual(other.inventory["split"]["validation_fragment_ids"], info["validation_fragment_ids"])

    def test_original_bf_and_single_alpha_metric_contract(self):
        scores = np.array([.1, .8, .2, .3, .1, .9, .8, .2], dtype=np.float32)
        labels = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int8)
        expected = paper_best_f1(scores, labels)
        settings = load_campaign("pilot")[2]
        with patch("tranad_variants.evaluation.metrics.bounded_pot", return_value={"pot_f1": None, "status": "not_available"}), redirect_stdout(io.StringIO()):
            actual = evaluate_oracle(scores[None], scores[None], labels, [None], .01, settings)
        self.assertEqual(actual["bf"]["bf_f1"], expected["bf_f1"])
        self.assertEqual(actual["bf"]["bf_ROC/AUC"], expected["bf_ROC/AUC"])
        self.assertFalse(actual["bf"]["alpha_applicable"])
        self.assertIsNone(actual["bf"]["alpha"])
        self.assertGreaterEqual(actual["bf_dense"]["bf_dense_f1"], actual["bf"]["bf_f1"])

    def test_fragment_pa_metrics_match_existing_ops_implementation(self):
        groups = [{"id": i, "values": np.zeros(4), "label": int(i < 2), "channel": "ch"} for i in range(3)]
        labels = np.repeat([1, 1, 0], 4).astype(np.int8)
        scores = np.array([.1, .8, .1, .1, .1, .1, .7, .1, .2, .3, .4, .1], dtype=np.float32)
        settings = load_campaign("pilot")[2]
        with patch("opssat_pipeline._pot_threshold", return_value=(.5, None)), redirect_stdout(io.StringIO()):
            expected_bf, expected_pot = original_metrics(scores, scores, labels, groups, .01)
            actual = evaluate_oracle(scores[None], scores[None], labels, [None], .01, settings, groups=groups)
        for metric in ("f1", "precision", "recall", "TP", "TN", "FP", "FN", "ROC/AUC"):
            self.assertEqual(actual["bf"][f"bf_{metric}"], expected_bf[f"bf_{metric}"])
            self.assertEqual(actual["pot"][f"pot_{metric}"], expected_pot[f"pot_{metric}"])
        self.assertEqual(actual["bf_dense"]["bf_dense_f1"], expected_bf["bf_dense_f1"])

    def test_nonfinite_model_outputs_create_diagnostic_and_fail(self):
        class BadModel(torch.nn.Module):
            output_count = 1
            def forward(self, x, target):
                return target * float("nan")
        dataset = TensorDataset(torch.ones(4, 3, 1), torch.zeros(4))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            diagnostic = Path(directory) / "diagnostic.json"
            with self.assertRaises(FloatingPointError):
                collect_outputs(BadModel(), dataset, "cpu", 2, "test", diagnostic)
            self.assertEqual(json.loads(diagnostic.read_text())["status"], "invalid_nonfinite_outputs")

    def test_full_runner_train_evaluate_skip_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            data, output = root / "data", root / "results"
            native_fixture(data)
            args = ["--campaign", "pilot", "--dataset", "SMAP", "--variant", "single_pass",
                    "--data-root", str(data), "--output-root", str(output), "--device", "cpu", "--cpu-threads", "1"]
            self.assertEqual(main(args), 0)
            folder = output / "v13/architectures/SMAP/single_pass/reference/seed_2026"
            metrics = json.loads((folder / "metrics.json").read_text())
            self.assertEqual(metrics["training"]["epochs_logged"], 50)
            self.assertFalse(metrics["paper_oracle"]["bf"]["alpha_applicable"])
            with np.load(folder / "scores.npz") as scores:
                self.assertEqual(scores["test_scores"].shape[0], 1)
                self.assertEqual(len(scores["alpha_values"]), 0)
            before = (folder / "checkpoints/final.ckpt").stat().st_mtime_ns
            self.assertEqual(main(args), 0)
            self.assertEqual((folder / "checkpoints/final.ckpt").stat().st_mtime_ns, before)
            with (folder / "scores.npz").open("ab") as stream:
                stream.write(b"corrupt")
            self.assertEqual(main(args), 1)

    def test_single_validation_selection_does_not_use_test_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ops_fixture(root)
            p = next(e.profile for e in load_campaign("pilot")[1] if e.profile["dataset"] == "OPS-SAT")
            bundle = load_data(p, root)
            outputs = {}
            for key, ds in bundle.datasets.items():
                target = ds.windows[:, -1, :].numpy()
                outputs[key] = {"outputs": [np.zeros_like(target)], "target": target}
            first = validation_selected(outputs, bundle, .995)
            bundle.labels["test"] = 1 - bundle.labels["test"]
            for group in bundle.groups["test"]:
                group["label"] = 1 - group["label"]
            second = validation_selected(outputs, bundle, .995)
            self.assertEqual(first["selection"], second["selection"])
            self.assertFalse(first["alpha_applicable"])
            self.assertIn("ch1", first["by_channel"])

    def test_campaign_continues_after_missing_dataset(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            data, output = root / "data", root / "results"
            native_fixture(data)
            result = main(["--campaign", "pilot", "--variant", "single_pass",
                           "--data-root", str(data), "--output-root", str(output),
                           "--device", "cpu", "--cpu-threads", "1"])
            self.assertEqual(result, 1)
            report = json.loads(next((output / "v13/architectures/_campaign_logs").glob("*.json")).read_text())
            self.assertEqual([r["status"] for r in report["runs"]], ["failed", "complete", "failed"])
            self.assertEqual(report["failures"], 2)

    def test_two_output_ops_runner_includes_validation_and_full_alpha_grid(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            data, output = root / "data", root / "results"
            ops_fixture(data)
            args = ["--campaign", "pilot", "--dataset", "OPS-SAT", "--variant", "reference",
                    "--data-root", str(data), "--output-root", str(output), "--device", "cpu", "--cpu-threads", "1"]
            self.assertEqual(main(args), 0)
            folder = output / "v13/architectures/OPS-SAT/reference/reference/seed_2026"
            metrics = json.loads((folder / "metrics.json").read_text())
            self.assertEqual(metrics["validation_selected"]["status"], "measured")
            self.assertTrue(metrics["validation_selected"]["alpha_applicable"])
            self.assertEqual(len(metrics["paper_oracle"]["alpha_grid"]), 21)
            self.assertEqual(metrics["paper_oracle"]["bf"]["point_adjustment"], "fragment_aware")
            with np.load(folder / "scores.npz") as scores:
                self.assertEqual(len(scores["alpha_values"]), 21)
                self.assertEqual(scores["validation_scores"].shape[0], 21)

    def test_plot_exports_standalone_figures(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            entry = {"dataset": "OPS-SAT", "variant": "single_pass", "configuration": "reference",
                     "execution_sha256": "test", "parameters": {"mean": 159, "sample_std": None},
                     "bf_dense_f1": {"mean": .8, "sample_std": .02, "n": 3}}
            (root / "aggregate.json").write_text(json.dumps({"groups": [entry]}))
            with patch("sys.argv", ["plot", "--results-root", str(root)]):
                plot_tradeoffs()
            self.assertTrue((root / "figures/OPS-SAT_bf_dense_f1_vs_parameters.png").is_file())
            self.assertTrue((root / "figures/OPS-SAT_bf_dense_f1_vs_parameters.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
