from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from geostudio_automation.kg_calibration import KGCalibrationConfig, ResponseModel
from geostudio_automation.optimization_config import Constraint
from geostudio_automation.results import METRIC_FIELDS
from geostudio_automation.surrogate import (
    SignConstrainedGP,
    fit_and_validate,
    matern52_blocks,
    predict_response,
    select_virtual_derivatives,
)


def fixed_response():
    x = np.full((3, 8), 0.5)
    x[:, 0] = [0.1, 0.5, 0.9]
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1.7) * Matern(length_scale=0.7, nu=2.5)
        + WhiteKernel(0.05),
        optimizer=None,
        alpha=1e-8,
    ).fit(x, np.zeros(3))
    return ResponseModel(gp, 0.0, 1.0, "fitted", str(gp.kernel_), ())


class DerivativeKernelTests(unittest.TestCase):
    def test_derivative_blocks_match_finite_differences_including_coincident_points(
        self,
    ):
        rng = np.random.default_rng(4)
        x, z = rng.random((3, 8)), rng.random((4, 8))
        dims = np.array([0, 3, 0, 7])
        lengths = np.linspace(0.4, 1.2, 8)
        kernel = ConstantKernel(1.7) * Matern(length_scale=lengths, nu=2.5)
        cross, dd = matern52_blocks(x, z, dims, 1.7, lengths)
        h = 1e-5
        for j, dim in enumerate(dims):
            plus, minus = z[j : j + 1].copy(), z[j : j + 1].copy()
            plus[0, dim] += h
            minus[0, dim] -= h
            np.testing.assert_allclose(
                cross[:, j],
                ((kernel(x, plus) - kernel(x, minus)) / (2 * h))[:, 0],
                atol=1e-8,
            )
        for i, dim in enumerate(dims):
            plus, minus = z[i : i + 1].copy(), z[i : i + 1].copy()
            plus[0, dim] += h
            minus[0, dim] -= h
            fd = (
                matern52_blocks(plus, z, dims, 1.7, lengths)[0]
                - matern52_blocks(minus, z, dims, 1.7, lengths)[0]
            ) / (2 * h)
            np.testing.assert_allclose(dd[i], fd[0], atol=1e-7)
        np.testing.assert_allclose(dd, dd.T, atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(dd).min(), 0)

    def test_positive_and_negative_signs_change_gradient_without_fake_magnitudes(self):
        response = fixed_response()
        anchor = np.full(8, 0.5)
        query = np.vstack([anchor.copy(), anchor.copy()])
        query[0, 0] -= 1e-4
        query[1, 0] += 1e-4
        gradients = []
        for sign in (-1, 1):
            guide = SignConstrainedGP(
                response,
                [
                    {
                        "point": anchor.tolist(),
                        "variable_index": 0,
                        "sign": sign,
                        "weight": 1.0,
                    }
                ],
            )
            means, stds = guide.predict(query)
            gradients.append((means[1] - means[0]) / 2e-4)
            self.assertTrue(np.isfinite(stds).all())
            self.assertTrue(
                (stds <= predict_response(response, query)[1] + 1e-10).all()
            )
        self.assertLess(gradients[0], -0.1)
        self.assertGreater(gradients[1], 0.1)

    def test_zero_weights_are_exactly_the_data_only_model(self):
        response = fixed_response()
        guide = SignConstrainedGP(
            response,
            [{"point": [0.5] * 8, "variable_index": 0, "sign": 1, "weight": 0.0}],
        )
        points = np.random.default_rng(2).random((5, 8))
        for actual, expected in zip(
            guide.predict(points), predict_response(response, points)
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_guided_derivative_matches_finite_difference_of_guided_mean(self):
        response = fixed_response()
        anchor = np.full(8, 0.5)
        guide = SignConstrainedGP(response, [{
            "point": anchor.tolist(), "variable_index": 0,
            "sign": -1, "weight": 1.0,
        }])
        mean, variance = guide.derivative_marginals(anchor[None, :], np.array([0]))
        plus, minus = anchor.copy(), anchor.copy()
        plus[0] += 1e-5
        minus[0] -= 1e-5
        numerical = (guide.predict(plus[None, :])[0][0] - guide.predict(minus[None, :])[0][0]) / 2e-5
        self.assertAlmostEqual(mean[0], numerical, places=5)
        self.assertGreaterEqual(variance[0], 0)


class ValidationTests(unittest.TestCase):
    def test_loo_refits_each_fold_and_computes_pooled_r2(self):
        x = np.random.default_rng(3).random((6, 8))
        y = np.column_stack([np.arange(6, dtype=float) + j for j in range(5)])
        seen = []

        def fit(train_x, train_y, config, *, response_index):
            seen.append((train_x.copy(), train_y.copy(), response_index))
            return ResponseModel(None, float(np.mean(train_y)), 0.0, "test", "", ())

        with patch(
            "geostudio_automation.surrogate.kg._fit_response_model", side_effect=fit
        ):
            validation = fit_and_validate(
                x, y, config=KGCalibrationConfig(gp_restarts=0)
            )
        self.assertEqual(len(seen), 5 * 7)
        for j in range(5):
            for held in range(6):
                train_x, train_y, index = seen[j * 7 + held]
                np.testing.assert_array_equal(train_x, np.delete(x, held, axis=0))
                np.testing.assert_array_equal(train_y, np.delete(y[:, j], held))
                self.assertEqual(index, j)
                self.assertAlmostEqual(
                    validation.predictions[held, j], np.mean(np.delete(y[:, j], held))
                )
            expected = 1 - np.sum(
                (y[:, j] - validation.predictions[:, j]) ** 2
            ) / np.sum((y[:, j] - np.mean(y[:, j])) ** 2)
            self.assertAlmostEqual(validation.reports[j]["r2"], expected)

    def test_constant_responses_never_enable_knowledge(self):
        result = fit_and_validate(
            np.random.default_rng(2).random((4, 8)),
            np.ones((4, 5)),
            config=KGCalibrationConfig(gp_restarts=0),
        )
        self.assertEqual(result.enabled, [False] * 5)
        self.assertTrue(all(r["r2"] is None for r in result.reports))


class VirtualSelectionTests(unittest.TestCase):
    def test_iterative_selection_uses_candidate_posterior_and_cap(self):
        x = np.full((3, 8), 0.5)
        x[:, 0] = [0.1, 0.5, 0.9]
        candidates = x.copy()
        candidates[:, 0] = [0.2, 0.4, 0.8]
        weights = np.zeros((8, 5))
        weights[0, 0], weights[1, 0], weights[2, 1], weights[3, 1] = 0.9, 0.8, 0.9, 0.7
        chosen, audit = select_virtual_derivatives(
            x, candidates, [fixed_response()] * 5, weights,
            [True, True, False, False, False],
            {name: Constraint(0, "<=") for name in METRIC_FIELDS},
            acquisition_relevant=np.ones(len(candidates), dtype=bool), maximum=3,
        )
        self.assertTrue(chosen[0])
        self.assertTrue(chosen[1])
        self.assertTrue(all(len(v) <= 3 for v in chosen))
        self.assertEqual(chosen[2:], [[], [], []])
        self.assertTrue(all(o["status"] == "selected" for o in chosen[0] + chosen[1]))
        self.assertTrue(all(0.1 <= o["opposite_probability"] <= 0.5 for o in chosen[0] + chosen[1]))
        self.assertTrue(all(o["point"] in candidates.tolist() for o in chosen[0] + chosen[1]))

    def test_conflicting_direction_is_audited_not_forced(self):
        weights = np.zeros((8, 5))
        weights[0, 0] = 1
        candidates = np.full((1, 8), 0.5)
        with patch(
            "geostudio_automation.surrogate.derivative_marginals",
            return_value=(np.array([10.0]), np.array([1.0]), None),
        ):
            chosen, audit = select_virtual_derivatives(
                np.full((3, 8), 0.5), candidates,
                [fixed_response()] * 5,
                weights,
                [True] * 5,
                {name: Constraint(1, ">=") for name in METRIC_FIELDS},
                acquisition_relevant=np.ones(1, dtype=bool),
            )
        self.assertEqual(chosen[0], [])
        self.assertTrue(any(o["status"] == "local_sign_conflict" for o in audit))


if __name__ == "__main__":
    unittest.main()
