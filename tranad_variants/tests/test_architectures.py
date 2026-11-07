import unittest
import torch
from tranad_variants.models import build_model
from tranad_variants.training.objectives import reconstruction_loss
from tranad_variants.evaluation.scoring import reconstruction_scores
from tranad_variants.planning import load_campaign
from TranADPlus.src.models import TranAD


def profile(variant="reference"):
    return {"features": 3, "window_sz": 5, "batch_sz": 4, "dim_feedforward": 6,
            "num_layers": 1, "variant": variant}


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, variant):
        torch.manual_seed(2026)
        return build_model(profile(variant))

    def test_additional_campaign_completes_balanced_240_run_design(self):
        import itertools
        old = load_campaign("all")[1]
        new = load_campaign("additional")[1]
        self.assertEqual(len(old), 72)
        self.assertEqual(len(new), 168)
        self.assertFalse({e.identifier for e in old} & {e.identifier for e in new})
        actual = {(e.profile["dataset"], e.profile["variant"],
                   e.profile["configuration"], e.profile["seed"]) for e in old + new}
        expected = set(itertools.product(
            ("OPS-SAT", "SMAP", "MSL"),
            ("reference", "shared_decoder", "single_pass", "encoder_only"),
            ("reference", "compact", "window_only", "ff_only"),
            (2026, 2027, 2028, 2029, 2030)))
        self.assertEqual(actual, expected)
        self.assertEqual([e.profile["dataset"] for e in new],
                         ["OPS-SAT"] * 56 + ["SMAP"] * 56 + ["MSL"] * 56)

    def test_additional_controls_preserve_other_training_parameters(self):
        old = load_campaign("all")[1]
        new = load_campaign("additional")[1]
        baseline = {(e.profile["dataset"], e.profile["variant"]): e.profile for e in old
                    if e.profile["configuration"] == "reference" and e.profile["seed"] == 2026}
        for e in new:
            p = e.profile
            if p["configuration"] not in ("window_only", "ff_only"):
                continue
            ref = baseline[p["dataset"], p["variant"]]
            for key in ref:
                if key in ("identifier", "configuration", "seed", "window_sz", "dim_feedforward"):
                    continue
                self.assertEqual(p[key], ref[key], (p["identifier"], key))
            if p["configuration"] == "window_only":
                self.assertEqual(p["window_sz"] * 2, ref["window_sz"])
                self.assertEqual(p["dim_feedforward"], ref["dim_feedforward"])
            else:
                self.assertEqual(p["window_sz"], ref["window_sz"])
                self.assertEqual(p["dim_feedforward"] * 2, ref["dim_feedforward"])

    def test_reference_is_bitwise_original(self):
        p = profile()
        torch.manual_seed(2026)
        original = TranAD(n_feats=p["features"], window_sz=p["window_sz"], batch_sz=p["batch_sz"],
                         dim_feedforward=p["dim_feedforward"], num_encoder_layers=1, num_decoder_layers=1)
        reference = self.model("reference")
        for name, weight in original.state_dict().items():
            torch.testing.assert_close(reference.state_dict()[name], weight, rtol=0, atol=0)
        x = torch.rand(5, 4, 3)
        original.eval(); reference.eval()
        with torch.inference_mode():
            for expected, actual in zip(original(x, x[-1:]), reference(x, x[-1:])):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_single_pass_matches_reference_first_stage(self):
        reference, single = self.model("reference"), self.model("single_pass")
        reference.eval(); single.eval()
        x = torch.rand(5, 4, 3)
        with torch.inference_mode():
            torch.testing.assert_close(reference(x, x[-1:])[0], single(x, x[-1:]), rtol=0, atol=0)
        self.assertFalse(hasattr(single, "transformer_decoder2"))

    def test_shared_decoder_is_actually_tied_and_parameter_counts_drop(self):
        models = {name: self.model(name) for name in ("reference", "shared_decoder", "single_pass", "encoder_only")}
        shared = models["shared_decoder"]
        self.assertIs(shared.transformer_decoder1, shared.transformer_decoder2)
        counts = {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()}
        self.assertEqual(counts["shared_decoder"], counts["single_pass"])
        self.assertLess(counts["shared_decoder"], counts["reference"])
        self.assertLess(counts["encoder_only"], counts["single_pass"])
        for variant, model in models.items():
            x = torch.rand(5, 4, 3)
            out = model(x, x[-1:])
            loss = reconstruction_loss(out, x[-1:], 3)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None for p in model.parameters()), variant)

    def test_native_loss_kernel_and_gradients_match_original(self):
        a = torch.rand(1, 4, 3, requires_grad=True)
        b = torch.rand(1, 4, 3, requires_grad=True)
        target = torch.rand(1, 4, 3)
        loss_fn = torch.nn.MSELoss(reduction="none")
        expected = (1/3 * loss_fn(a, target) + (1-1/3) * loss_fn(b, target)).mean()
        expected.backward()
        gradients = [a.grad.clone(), b.grad.clone()]
        a.grad = b.grad = None
        actual = reconstruction_loss((a, b), target, 3, implementation="mse_loss")
        actual.backward()
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        for expected_grad, value in zip(gradients, (a, b)):
            torch.testing.assert_close(expected_grad, value.grad, rtol=0, atol=0)

    def test_single_output_rejects_meaningless_alpha(self):
        import numpy as np
        outputs = {"outputs": [np.ones((5, 1))], "target": np.zeros((5, 1))}
        with self.assertRaises(ValueError):
            reconstruction_scores(outputs, .5)
        self.assertEqual(reconstruction_scores(outputs).shape, (5,))

    def test_campaign_overlap_and_controls(self):
        campaigns = {name: load_campaign(name)[1] for name in ("pilot", "main", "compact", "encoder_only", "all")}
        self.assertEqual({k: len(v) for k, v in campaigns.items()},
                         {"pilot": 9, "main": 27, "compact": 27, "encoder_only": 18, "all": 72})
        pilot = {e.identifier: e.profile for e in campaigns["pilot"]}
        main = {e.identifier: e.profile for e in campaigns["main"]}
        for identifier, p in pilot.items():
            self.assertEqual(p, main[identifier])
        union = {e.identifier for name in ("main", "compact", "encoder_only") for e in campaigns[name]}
        self.assertEqual(union, {e.identifier for e in campaigns["all"]})
        for e in campaigns["all"]:
            p = e.profile
            self.assertEqual(p["epochs"], 50)
            if p["dataset"] == "OPS-SAT":
                self.assertEqual(p["split_seed"], 2026)
                self.assertEqual(p["validation_fraction"], .2)


if __name__ == "__main__":
    unittest.main()
