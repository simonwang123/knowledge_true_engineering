from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from geostudio_automation.kg_calibration import ResponseModel
from geostudio_automation.optimization import acquisition_scores, run_optimization
from geostudio_automation.optimization_config import Constraint, ExperimentConfig
from geostudio_automation.optimization_evaluation import (
    EvaluationResult,
    read_evaluation,
    write_evaluation,
)
from geostudio_automation.results import DISPLACEMENT_METRIC_FIELDS
from geostudio_automation.sampling import feasible_lhs, replace_in_strata, unit_design
from geostudio_automation.surrogate import SurrogateValidation

ROOT = Path(__file__).resolve().parents[1]


def configuration(**overrides):
    base = ExperimentConfig(
        template=ROOT / "templates/geo_studio_model_sample",
        constraints={name: Constraint(100.0, "<=") for name in DISPLACEMENT_METRIC_FIELDS},
        material_unit_costs={4: 100.0},
        initial_samples=4,
        adaptive_budget=1,
        total_budget=5,
        gp_restarts=0,
        candidate_count=8,
        candidate_attempts=64,
    )
    return replace(base, **overrides)


class BoxDomain:
    fingerprint = "box-domain-v1"

    def feasible(self, point):
        return True

    def cost(self, point):
        return float(10 + np.sum(point))

    def prepare(self, design):
        pass


class ToyEvaluator:
    fingerprint = "synthetic-test-v1"

    def __init__(self, *, fail_first=False, interrupt=False, always_fail=False):
        self.calls = 0
        self.fail_first, self.interrupt, self.always_fail = (
            fail_first,
            interrupt,
            always_fail,
        )

    def evaluate(self, design, directory, charge):
        self.calls += 1
        charge()
        if self.always_fail or (self.fail_first and self.calls == 1):
            return EvaluationResult(False, error="synthetic solver failure")
        x = unit_design(design)
        result = EvaluationResult(
            True, {name: float(np.sum(x) + j) for j, name in enumerate(DISPLACEMENT_METRIC_FIELDS)}
        )
        if self.interrupt and self.calls == 1:
            write_evaluation(design, directory, result)
            raise KeyboardInterrupt("simulated crash after solve before state commit")
        return result

    def recover(self, design, directory):
        return read_evaluation(design, directory)


def fake_validation(x, y, *, config, r2_threshold, progress=None, response_fields=None):
    models = [
        ResponseModel(
            None,
            float(np.mean(y[:, j])),
            float(np.std(y[:, j], ddof=1)),
            "test",
            "",
            (),
        )
        for j in range(len(DISPLACEMENT_METRIC_FIELDS))
    ]
    reports = [
        {
            "response": name,
            "samples": len(x),
            "r2": 0.9 if j == 0 else 0.2,
            "rmse": 0.1,
            "enabled": j == 0,
            "warnings": [],
        }
        for j, name in enumerate(DISPLACEMENT_METRIC_FIELDS)
    ]
    return SurrogateValidation(models, y.copy(), reports)


class SamplingTests(unittest.TestCase):
    def test_repaired_lhs_is_reproducible_and_strictly_latin(self):
        feasible = lambda p: p[0] + p[1] < 1.4
        a = feasible_lhs(20, feasible)
        b = feasible_lhs(20, feasible)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(all(feasible(p) for p in a))
        for dim in range(8):
            np.testing.assert_array_equal(
                np.sort(np.floor(a[:, dim] * 20)), np.arange(20)
            )

    def test_impossible_domain_stops_and_replacement_preserves_all_strata(self):
        with self.assertRaisesRegex(ValueError, "exhausted"):
            feasible_lhs(4, lambda p: False, restarts=2, repair_attempts=3)
        original = np.arange(8) / 10 + 0.01
        new = replace_in_strata(original, 20, lambda p: True, seed=4, excluded=set())
        np.testing.assert_array_equal(np.floor(new * 20), np.floor(original * 20))


class AcquisitionTests(unittest.TestCase):
    def test_log_probability_does_not_underflow_and_honors_lower_bounds(self):
        constraints = {name: Constraint(0, ">=") for name in DISPLACEMENT_METRIC_FIELDS}
        means = np.array([[-100.0] * 3, [-90.0] * 3])
        scores = acquisition_scores(
            means, np.ones_like(means), np.array([1.0, 2.0]), constraints, None
        )
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[1], scores[0])
        zero_noise = acquisition_scores(
            np.array([[1.0] * 3, [-1.0] * 3]),
            np.zeros((2, 3)),
            np.array([1.0, 2.0]),
            constraints,
            3,
        )
        self.assertAlmostEqual(zero_noise[0], np.log(2))
        self.assertEqual(zero_noise[1], -np.inf)


class ConfigurationTests(unittest.TestCase):
    def test_engineering_template_uses_three_strict_displacement_limits(self):
        config = ExperimentConfig.from_json(ROOT / "examples/optimization.json")
        self.assertEqual(set(config.constraints), set(DISPLACEMENT_METRIC_FIELDS))
        self.assertEqual([config.constraints[n].limit for n in DISPLACEMENT_METRIC_FIELDS], [0.080, 0.16, 0.075])
        self.assertTrue(all(c.direction == "<" for c in config.constraints.values()))

    def test_simulation_paths_are_config_relative(self):
        config = ExperimentConfig.from_json(
            ROOT / "examples/optimization.simulation.json"
        )
        self.assertEqual(config.template, configuration().template)


class OptimizationWorkflowTests(unittest.TestCase):
    @patch(
        "geostudio_automation.optimization.fit_and_validate",
        side_effect=fake_validation,
    )
    def test_unfinished_interrupted_attempt_is_not_refunded(self, fit):
        class InterruptedEvaluator(ToyEvaluator):
            def evaluate(self, design, directory, charge):
                if self.calls == 0:
                    self.calls += 1
                    charge()
                    raise KeyboardInterrupt("interrupted during meshing")
                return super().evaluate(design, directory, charge)

        with TemporaryDirectory() as temporary:
            out = Path(temporary) / "run"
            evaluator = InterruptedEvaluator()
            config = configuration(total_budget=6)
            with self.assertRaises(KeyboardInterrupt):
                run_optimization(config, out, domain=BoxDomain(), evaluator=evaluator)
            result = run_optimization(
                config, out, domain=BoxDomain(), evaluator=evaluator, resume=True
            )
            self.assertEqual(result.attempts_spent, 6)
            self.assertEqual(result.successful_samples, 5)
            state = json.loads((out / "checkpoint.json").read_text())
            self.assertTrue(state["attempts"][0]["charged"])
            self.assertEqual(state["attempts"][0]["status"], "failed")

    @patch(
        "geostudio_automation.optimization.fit_and_validate",
        side_effect=fake_validation,
    )
    def test_failed_derivative_inference_is_audited_and_falls_back(self, fit):
        from geostudio_automation.surrogate import SignConstrainedGP

        def fit_guide(response, observations, **kwargs):
            if observations:
                raise ValueError("synthetic Laplace failure")
            return SignConstrainedGP(response, [])

        with TemporaryDirectory() as temporary:
            out = Path(temporary) / "run"
            with (
                patch(
                    "geostudio_automation.optimization.select_virtual_derivatives",
                    return_value=([[{}], [], []], []),
                ),
                patch(
                    "geostudio_automation.optimization.SignConstrainedGP",
                    side_effect=fit_guide,
                ),
            ):
                result = run_optimization(
                    configuration(adaptive_budget=0, total_budget=4),
                    out,
                    domain=BoxDomain(),
                    evaluator=ToyEvaluator(),
                )
            audit = json.loads(
                (out / "iterations/samples_0004/virtual_derivatives.json").read_text()
            )
            self.assertEqual(result.successful_samples, 4)
            self.assertEqual(audit["inference"][0]["status"], "fallback_data_only")

    @patch(
        "geostudio_automation.optimization.fit_and_validate",
        side_effect=fake_validation,
    )
    def test_full_sixty_attempt_budget_and_periodic_exploration(self, fit):
        with TemporaryDirectory() as temporary:
            out = Path(temporary) / "run"
            config = configuration(
                initial_samples=20, adaptive_budget=40, total_budget=60
            )
            result = run_optimization(
                config,
                out,
                mode="plain",
                domain=BoxDomain(),
                evaluator=ToyEvaluator(fail_first=True),
            )
            self.assertEqual(result.attempts_spent, 60)
            self.assertEqual(result.successful_samples, 59)
            state = json.loads((out / "checkpoint.json").read_text())
            adaptive = [
                a for a in state["attempts"] if a["slot"] is None and a["charged"]
            ]
            self.assertEqual(
                len(adaptive), 39
            )  # One initial retry uses the shared budget.
            self.assertTrue(
                all(
                    adaptive[i]["phase"] == "uncertainty_exploration"
                    for i in range(4, len(adaptive), 5)
                )
            )

    @patch(
        "geostudio_automation.optimization.fit_and_validate",
        side_effect=fake_validation,
    )
    def test_failure_replacement_budget_resume_and_latin_structure(self, fit):
        with TemporaryDirectory() as temporary:
            out = Path(temporary) / "run"
            config = configuration(total_budget=6)
            evaluator = ToyEvaluator(fail_first=True)
            result = run_optimization(
                config, out, domain=BoxDomain(), evaluator=evaluator
            )
            self.assertEqual(result.attempts_spent, 6)
            self.assertEqual(result.successful_samples, 5)
            state = json.loads((out / "checkpoint.json").read_text())
            initial = [
                a
                for a in state["attempts"]
                if a["status"] == "succeeded" and a["slot"] is not None
            ]
            points = np.asarray([a["point"] for a in initial])
            for dim in range(8):
                np.testing.assert_array_equal(
                    np.sort(np.floor(points[:, dim] * 4)), np.arange(4)
                )
            self.assertNotEqual(
                state["attempts"][0]["point"], state["attempts"][1]["point"]
            )
            self.assertTrue((out / "convergence.png").is_file())
            previous_calls = evaluator.calls
            resumed = run_optimization(
                config, out, domain=BoxDomain(), evaluator=evaluator, resume=True
            )
            self.assertEqual(evaluator.calls, previous_calls)
            self.assertEqual(resumed.best_design, result.best_design)
            with self.assertRaisesRegex(ValueError, "changed"):
                run_optimization(
                    replace(config, random_seed=3),
                    out,
                    domain=BoxDomain(),
                    evaluator=evaluator,
                    resume=True,
                )

    @patch(
        "geostudio_automation.optimization.fit_and_validate",
        side_effect=fake_validation,
    )
    def test_crash_after_completed_evaluation_recovers_without_double_charge(self, fit):
        with TemporaryDirectory() as temporary:
            out = Path(temporary) / "run"
            evaluator = ToyEvaluator(interrupt=True)
            with self.assertRaises(KeyboardInterrupt):
                run_optimization(
                    configuration(), out, domain=BoxDomain(), evaluator=evaluator
                )
            state = json.loads((out / "checkpoint.json").read_text())
            self.assertTrue(state["attempts"][0]["charged"])
            result = run_optimization(
                configuration(),
                out,
                domain=BoxDomain(),
                evaluator=evaluator,
                resume=True,
            )
            self.assertEqual(result.attempts_spent, 5)
            self.assertEqual(result.successful_samples, 5)
            self.assertEqual(evaluator.calls, 5)

    @patch(
        "geostudio_automation.optimization.fit_and_validate",
        side_effect=fake_validation,
    )
    def test_no_feasible_design_and_plain_mode_never_calibrates(self, fit):
        with TemporaryDirectory() as temporary:
            config = configuration(
                constraints={n: Constraint(-100, "<=") for n in DISPLACEMENT_METRIC_FIELDS}
            )
            with patch(
                "geostudio_automation.optimization.calibrate_kg_edges"
            ) as calibration:
                result = run_optimization(
                    config,
                    Path(temporary) / "run",
                    mode="plain",
                    domain=BoxDomain(),
                    evaluator=ToyEvaluator(),
                )
            calibration.assert_not_called()
            self.assertIsNone(result.best_design)
            report = json.loads(result.report_path.read_text())
            self.assertIsNotNone(report["least_violating_design"])
            self.assertEqual(report["feasible_samples"], 0)

    def test_all_failed_evaluations_stop_at_budget_without_training(self):
        with TemporaryDirectory() as temporary:
            with patch("geostudio_automation.optimization.fit_and_validate") as fit:
                result = run_optimization(
                    configuration(),
                    Path(temporary) / "run",
                    domain=BoxDomain(),
                    evaluator=ToyEvaluator(always_fail=True),
                )
            fit.assert_not_called()
            self.assertEqual(result.attempts_spent, 5)
            self.assertEqual(result.successful_samples, 0)
            self.assertEqual(result.stop_reason, "initialization_budget_exhausted")

    @patch(
        "geostudio_automation.optimization.fit_and_validate",
        side_effect=fake_validation,
    )
    def test_preflight_failures_do_not_consume_budget_and_result_tampering_rejected(
        self, fit
    ):
        with TemporaryDirectory() as temporary:
            out = Path(temporary) / "run"
            domain = BoxDomain()
            calls = [0]

            def prepare(design):
                calls[0] += 1
                if calls[0] == 1:
                    raise ValueError("invalid XML reference")

            domain.prepare = prepare
            evaluator = ToyEvaluator()
            result = run_optimization(
                configuration(), out, domain=domain, evaluator=evaluator
            )
            self.assertEqual(result.attempts_spent, 5)
            self.assertEqual(evaluator.calls, 5)
            state = json.loads((out / "checkpoint.json").read_text())
            self.assertEqual(len(state["attempts"]), 6)
            self.assertFalse(state["attempts"][0]["charged"])
            cached_path = out / "runs" / "eval_0002" / "evaluation.json"
            cached = json.loads(cached_path.read_text())
            cached["result"]["metrics"][DISPLACEMENT_METRIC_FIELDS[0]] += 1
            cached_path.write_text(json.dumps(cached))
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                run_optimization(
                    configuration(),
                    out,
                    domain=domain,
                    evaluator=evaluator,
                    resume=True,
                )

    def test_accuracy_sampling_until_a_response_passes(self):
        def validation(x, y, **kwargs):
            value = fake_validation(x, y, **kwargs)
            if len(x) == 4:
                for report in value.reports:
                    report.update(enabled=False, r2=0.1)
            return value

        with (
            TemporaryDirectory() as temporary,
            patch(
                "geostudio_automation.optimization.fit_and_validate",
                side_effect=validation,
            ),
        ):
            out = Path(temporary) / "run"
            run_optimization(
                configuration(total_budget=6, adaptive_budget=2),
                out,
                domain=BoxDomain(),
                evaluator=ToyEvaluator(),
            )
            state = json.loads((out / "checkpoint.json").read_text())
            self.assertEqual(state["attempts"][4]["phase"], "accuracy_sampling")
            self.assertIn(
                state["attempts"][5]["phase"],
                {"constrained_improvement", "uncertainty_exploration"},
            )


if __name__ == "__main__":
    unittest.main()
