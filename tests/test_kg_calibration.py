from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import geostudio_automation.kg_calibration as calibration
from geostudio_automation.kg_calibration import (
    CONSTRAINTS,
    DESIGN_VARS,
    PRIOR_SIGN,
    PRIOR_TIER,
    KGCalibrationConfig,
    calibrate_kg_edges,
)
from geostudio_automation.model import DesignVariables
from geostudio_automation.results import metric_fields


class _FunctionModel:
    def __init__(self, function, *, posterior_std: float = 0.0) -> None:
        self.function = function
        self.posterior_std = posterior_std

    def predict(self, points, *, return_std: bool = False):
        values = np.asarray(self.function(np.asarray(points)), dtype=float)
        if return_std:
            return values, np.full(len(values), self.posterior_std)
        return values


def _normalized_designs(count: int = 18) -> np.ndarray:
    rng = np.random.default_rng(421)
    values = np.empty((count, len(DESIGN_VARS)), dtype=float)
    levels = (np.arange(count) + 0.5) / count
    for index in range(len(DESIGN_VARS)):
        values[:, index] = levels[rng.permutation(count)]
    values[:, 5] = levels
    return values


def _physical_designs(count: int) -> np.ndarray:
    normalized = _normalized_designs(count)
    lower = np.asarray([DesignVariables.BOUNDS[name][0] for name in DESIGN_VARS])
    upper = np.asarray([DesignVariables.BOUNDS[name][1] for name in DESIGN_VARS])
    return lower + normalized * (upper - lower)


def _write_dataset(path: Path, count: int) -> None:
    x = _physical_designs(count)
    response_columns = metric_fields(0.98)
    fields = (
        "design_id",
        *DESIGN_VARS,
        "parameter_hash",
        "analysis_name",
        "step",
        "cvar_alpha",
        *response_columns,
    )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row_index, design in enumerate(x):
            normalized = _normalized_designs(count)[row_index]
            responses = (
                2.0 - 1.5 * normalized[0],
                1.0 + 2.0 * normalized[5],
                3.0 + 1.2 * normalized[2],
                4.0 - 1.8 * normalized[3],
                2.0 + 1.6 * normalized[7],
            )
            row = {
                "design_id": f"design-{row_index:03d}",
                **{
                    name: float(design[index]) for index, name in enumerate(DESIGN_VARS)
                },
                "parameter_hash": f"hash-{row_index:03d}",
                "analysis_name": "正常蓄水位应力分布",
                "step": 0,
                "cvar_alpha": 0.98,
                **dict(zip(response_columns, responses, strict=True)),
            }
            writer.writerow(row)


def _response_model(function, *, posterior_std: float = 0.0):
    return calibration._ResponseModel(
        _FunctionModel(function, posterior_std=posterior_std),
        0.0,
        1.0,
        "fitted",
        "test-kernel",
        (),
    )


def _edge_for(function, *, posterior_std: float = 0.0):
    x = _normalized_designs()
    config = KGCalibrationConfig(gp_restarts=0)
    return calibration._calibrate_edge(
        x,
        5,
        1,
        metric_fields(0.98)[1],
        _response_model(function, posterior_std=posterior_std),
        support_radius=10.0,
        required_comparisons=8,
        config=config,
    )


class LocalAleEdgeTests(unittest.TestCase):
    def test_strong_consistent_edge_is_active_and_bounded_by_prior(self) -> None:
        edge = _edge_for(lambda points: 2.0 * points[:, 5])

        self.assertEqual(edge.status, "active")
        self.assertEqual(edge.prior_direction_consistency, 1.0)
        self.assertGreaterEqual(edge.ale_effect_strength, 0.5)
        self.assertGreater(edge.effective_weight, 0.0)
        self.assertLessEqual(edge.effective_weight, edge.prior_tier)

    def test_stable_opposite_direction_is_a_conflict_without_flipping(self) -> None:
        edge = _edge_for(lambda points: -2.0 * points[:, 5])

        self.assertEqual(edge.status, "sign_conflict")
        self.assertEqual(edge.prior_sign, PRIOR_SIGN[5][1])
        self.assertEqual(edge.effective_weight, 0.0)

    def test_strong_non_monotonic_edge_is_unstable(self) -> None:
        edge = _edge_for(lambda points: 4.0 * (points[:, 5] - 0.5) ** 2)

        self.assertEqual(edge.status, "unstable_direction")
        self.assertEqual(edge.effective_weight, 0.0)

    def test_stable_but_small_effect_is_weak(self) -> None:
        edge = _edge_for(lambda points: 0.2 * points[:, 5])

        self.assertEqual(edge.status, "weak_effect")
        self.assertEqual(edge.prior_direction_consistency, 1.0)
        self.assertLess(edge.ale_effect_strength, 0.5)

    def test_high_posterior_uncertainty_removes_local_comparisons(self) -> None:
        edge = _edge_for(lambda points: 2.0 * points[:, 5], posterior_std=1.1)

        self.assertEqual(edge.status, "insufficient_support")
        self.assertEqual(edge.valid_comparisons, 0)


class CalibrationWorkflowTests(unittest.TestCase):
    def test_reuses_data_only_models_and_applies_per_response_gate(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.csv"
            _write_dataset(dataset_path, 16)
            dataset = calibration._load_dataset(dataset_path)
            x = calibration._normalize_designs(dataset.x)
            config = KGCalibrationConfig(gp_restarts=0)
            models = [calibration.fit_response_model(x, dataset.y[:, j], config, response_index=j) for j in range(5)]
            with patch.object(calibration, "_fit_response_model") as fit:
                outputs = calibrate_kg_edges(
                    dataset_path, root / "calibration", response_models=models,
                    response_enabled=[True, False, True, False, False], config=config,
                )
            fit.assert_not_called()
            for edge in outputs.edges:
                if edge.constraint_index in (2, 4, 5):
                    self.assertEqual(edge.status, "accuracy_not_met")
                    self.assertEqual(edge.effective_weight, 0)
            with self.assertRaisesRegex(ValueError, "scaling"):
                wrong = [models[1], *models[1:]]
                calibrate_kg_edges(dataset_path, root / "bad", response_models=wrong)

    def test_writes_auditable_outputs_and_fits_each_response_once(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.csv"
            output = root / "calibration"
            _write_dataset(dataset, 16)
            model = _response_model(lambda points: 2.0 * points[:, 5])

            with patch.object(
                calibration, "_fit_response_model", return_value=model
            ) as fit:
                result = calibrate_kg_edges(dataset, output)

            self.assertEqual(fit.call_count, len(CONSTRAINTS))
            self.assertEqual(len(result.edges), 40)
            self.assertTrue(result.weights_path.is_file())
            self.assertTrue(result.edge_calibration_path.is_file())
            self.assertTrue(result.report_path.is_file())
            payload = json.loads(result.weights_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["prior_sign"], [list(row) for row in PRIOR_SIGN])
            self.assertEqual(payload["prior_tier"], [list(row) for row in PRIOR_TIER])
            self.assertEqual(np.shape(payload["effective_weight"]), (8, 5))
            self.assertEqual(np.shape(payload["active_mask"]), (8, 5))

            with self.assertRaisesRegex(ValueError, "already exists"):
                calibrate_kg_edges(dataset, output)

    def test_too_few_samples_writes_zero_weight_results_without_fitting(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.csv"
            _write_dataset(dataset, 15)

            with patch.object(calibration, "_fit_response_model") as fit:
                result = calibrate_kg_edges(dataset, root / "calibration")

            fit.assert_not_called()
            self.assertTrue(
                all(edge.status == "insufficient_samples" for edge in result.edges)
            )
            self.assertEqual(result.active_edges, 0)
            self.assertTrue(
                all(value == 0.0 for row in result.effective_weights for value in row)
            )

    def test_real_gp_run_is_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.csv"
            _write_dataset(dataset, 16)
            config = KGCalibrationConfig(gp_restarts=0)

            first = calibrate_kg_edges(dataset, root / "first", config=config)
            second = calibrate_kg_edges(dataset, root / "second", config=config)

            self.assertEqual(first.effective_weights, second.effective_weights)
            self.assertEqual(
                [edge.status for edge in first.edges],
                [edge.status for edge in second.edges],
            )


if __name__ == "__main__":
    unittest.main()
