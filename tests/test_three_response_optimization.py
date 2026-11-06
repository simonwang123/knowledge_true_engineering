from __future__ import annotations

import csv
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from geostudio_automation.model import DesignVariables
from geostudio_automation.optimization import acquisition_scores, run_optimization
from geostudio_automation.optimization_config import Constraint, ExperimentConfig
from geostudio_automation.optimization_evaluation import EvaluationResult
from geostudio_automation.results import (
    DEFAULT_ANALYSIS_NAME,
    DEFAULT_ANALYSIS_STEP,
    DEFAULT_CVAR_ALPHA,
    DISPLACEMENT_METRIC_FIELDS,
)
from geostudio_automation.sampling import design_from_unit


class BoxDomain:
    fingerprint = "three-response-test-domain"

    def feasible(self, point):
        return True

    def prepare(self, design):
        return None

    def cost(self, point):
        return float(100 + np.sum(point))


class AdaptiveEvaluator:
    fingerprint = "three-response-test-evaluator"

    def __init__(self):
        self.calls = 0

    def evaluate(self, design, directory, charge):
        self.calls += 1
        charge()
        return EvaluationResult(True, {
            "crest_max_settlement_m": 0.07,
            "dam_max_settlement_m": 0.15,
            "dam_max_horizontal_displacement_m": 0.07,
            "dam_sigma1_effective_stress_cvar99p5_kpa": 1e8,
        })

    def recover(self, design, directory):
        raise AssertionError("no interrupted evaluations are expected")


class ThreeResponseOptimizationTests(unittest.TestCase):
    def test_strict_limits_and_zero_noise_acquisition(self):
        constraints = {
            name: Constraint(limit, "<")
            for name, limit in zip(DISPLACEMENT_METRIC_FIELDS, (0.080, 0.16, 0.075))
        }
        self.assertFalse(constraints[DISPLACEMENT_METRIC_FIELDS[0]].satisfied(0.080))
        self.assertTrue(constraints[DISPLACEMENT_METRIC_FIELDS[0]].satisfied(0.079))
        means = np.asarray([[0.080, 0.15, 0.07], [0.079, 0.15, 0.07]])
        scores = acquisition_scores(
            means, np.zeros_like(means), np.array([1., 1.]), constraints, None
        )
        self.assertEqual(scores[0], -np.inf)
        self.assertEqual(scores[1], 0.)

    def test_imported_lhs_is_reused_and_only_displacements_are_optimized(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "initial.csv"
            fields = (
                *DesignVariables.CSV_FIELDS,
                "parameter_hash", "analysis_name", "step", "cvar_alpha",
                *DISPLACEMENT_METRIC_FIELDS,
                "dam_sigma1_effective_stress_cvar99p5_kpa",
            )
            points = np.asarray([
                [(i + (j + 1) / 10) % 4 / 4 for j in range(8)]
                for i in range(4)
            ])
            with source.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for i, point in enumerate(points):
                    design = design_from_unit(point, f"prior_{i + 1:03d}")
                    writer.writerow({
                        **design.as_dict(),
                        "parameter_hash": design.parameter_hash(),
                        "analysis_name": DEFAULT_ANALYSIS_NAME,
                        "step": DEFAULT_ANALYSIS_STEP,
                        "cvar_alpha": DEFAULT_CVAR_ALPHA,
                        "crest_max_settlement_m": 0.080 if i == 0 else 0.07,
                        "dam_max_settlement_m": 0.15,
                        "dam_max_horizontal_displacement_m": 0.07,
                        "dam_sigma1_effective_stress_cvar99p5_kpa": 1e8,
                    })
            config = ExperimentConfig(
                template=root,
                constraints={
                    name: Constraint(limit, "<") for name, limit in zip(
                        DISPLACEMENT_METRIC_FIELDS, (0.080, 0.16, 0.075)
                    )
                },
                material_unit_costs={4: 1},
                initial_dataset=source,
                initial_samples=4,
                adaptive_budget=1,
                total_budget=5,
                gp_restarts=0,
                candidate_count=8,
                candidate_attempts=64,
            )
            evaluator = AdaptiveEvaluator()
            result = run_optimization(
                config, root / "out", mode="plain", domain=BoxDomain(), evaluator=evaluator
            )
            self.assertEqual(result.successful_samples, 5)
            self.assertEqual(evaluator.calls, 1)
            checkpoint = json.loads((root / "out/checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(sum(a["phase"] == "imported_initial" for a in checkpoint["attempts"]), 4)
            self.assertFalse(checkpoint["attempts"][0]["feasible"])
            self.assertTrue(checkpoint["attempts"][1]["feasible"])
            with result.dataset_path.open(encoding="utf-8-sig", newline="") as stream:
                columns = csv.DictReader(stream).fieldnames
            self.assertTrue(all(name in columns for name in DISPLACEMENT_METRIC_FIELDS))
            self.assertFalse(any("sigma" in name for name in columns))
            snapshot = json.loads((root / "out/iterations/samples_0004/validation.json").read_text(encoding="utf-8"))
            self.assertEqual(len(snapshot["responses"]), 3)
            resumed = run_optimization(
                config, root / "out", mode="plain", domain=BoxDomain(),
                evaluator=evaluator, resume=True,
            )
            self.assertEqual(resumed.successful_samples, 5)
            source.write_text(source.read_text(encoding="utf-8-sig") + "\n", encoding="utf-8-sig")
            with self.assertRaisesRegex(ValueError, "resume rejected"):
                run_optimization(
                    config, root / "out", mode="plain", domain=BoxDomain(),
                    evaluator=evaluator, resume=True,
                )


if __name__ == "__main__":
    unittest.main()
