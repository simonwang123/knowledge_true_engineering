"""Data-only validation and soft derivative-sign Gaussian processes.

The sign likelihood is Phi(sign * normalized_derivative / softness)**weight.
Kernel hyperparameters are fitted ONLY to real observations. Laplace inference
then operates on the conditional derivative distribution, whitened by Cholesky.
No fabricated derivative magnitudes or response values enter the dataset.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr
from threadpoolctl import threadpool_limits

from . import kg_calibration as kg
from .optimization_config import Constraint
from .results import METRIC_FIELDS


def predict_response(response: kg.ResponseModel, points: np.ndarray):
    if response.model is None:
        return np.full(len(points), response.mean), np.zeros(len(points))
    means, stds = response.model.predict(points, return_std=True)
    return (
        response.mean + response.standard_deviation * means,
        response.standard_deviation * stds,
    )


@dataclass
class SurrogateValidation:
    models: list[kg.ResponseModel]
    predictions: np.ndarray
    reports: list[dict]

    @property
    def enabled(self) -> list[bool]:
        return [report["enabled"] for report in self.reports]


def fit_and_validate(
    x: np.ndarray,
    y: np.ndarray,
    *,
    config: kg.KGCalibrationConfig,
    r2_threshold: float = 0.75,
    progress: Callable[[str], None] | None = None,
    response_fields: Sequence[str] = METRIC_FIELDS,
) -> SurrogateValidation:
    if len(x) < 4 or y.shape != (len(x), len(response_fields)):
        raise ValueError(
            "validation requires at least four complete observations per response"
        )
    predictions = np.empty_like(y, dtype=float)
    models, reports = [], []
    # Small dense GP fits are faster and reproducible without BLAS oversubscription.
    with threadpool_limits(limits=1):
        for j, name in enumerate(response_fields):
            fold_warnings = []
            failed = False
            for held in range(len(x)):
                mask = np.arange(len(x)) != held
                try:
                    fold = kg.fit_response_model(
                        x[mask], y[mask, j], config, response_index=j
                    )
                    predictions[held, j] = predict_response(fold, x[held : held + 1])[
                        0
                    ][0]
                    fold_warnings.extend(fold.warnings)
                except (ValueError, np.linalg.LinAlgError) as error:
                    failed = True
                    predictions[held, j] = np.nan
                    fold_warnings.append(f"fold {held}: {error}")
                if progress and (held + 1) % 10 == 0:
                    progress(f"LOO {name}: {held + 1}/{len(x)}")
            model = kg.fit_response_model(x, y[:, j], config, response_index=j)
            models.append(model)
            sst = float(np.sum((y[:, j] - np.mean(y[:, j])) ** 2))
            failed = failed or not np.isfinite(predictions[:, j]).all()
            constant = np.ptp(y[:, j]) <= 1e-12 * max(
                1.0, float(np.max(np.abs(y[:, j])))
            )
            residual = y[:, j] - predictions[:, j]
            rmse = None if failed else float(np.sqrt(np.mean(residual**2)))
            r2 = None if failed or constant else float(1 - np.sum(residual**2) / sst)
            enabled = r2 is not None and np.isfinite(r2) and r2 >= r2_threshold
            reports.append(
                {
                    "response": name,
                    "samples": len(x),
                    "r2": r2,
                    "rmse": rmse,
                    "enabled": bool(enabled),
                    "constant_response": bool(constant),
                    "fitted_kernel": model.fitted_kernel,
                    "warnings": list(dict.fromkeys([*fold_warnings, *model.warnings])),
                }
            )
            if progress:
                progress(
                    f"{name}: LOO R2={r2}, accuracy={'passed' if enabled else 'failed'}"
                )
    return SurrogateValidation(models, predictions, reports)


def matern52_blocks(x, z, dims, amplitude: float, length_scale):
    """Return K(f(x), d(z)) and K(d(z), d(z)); derivatives use unit inputs.

    WhiteKernel is deliberately excluded: independent observation noise has no
    differentiable covariance. Coincident-point limits are handled analytically.
    """
    x, z, dims = np.asarray(x), np.asarray(z), np.asarray(dims, dtype=int)
    lengths = np.broadcast_to(np.asarray(length_scale, dtype=float), (x.shape[1],))
    delta = x[:, None, :] - z[None, :, :]
    r = np.sqrt(np.sum((delta / lengths) ** 2, axis=2))
    c = amplitude * (5.0 / 3.0) * (1 + np.sqrt(5) * r) * np.exp(-np.sqrt(5) * r)
    cross = (
        c
        * np.take_along_axis(delta, dims[None, :, None], axis=2)[:, :, 0]
        / lengths[dims] ** 2
    )
    delta = z[:, None, :] - z[None, :, :]
    r = np.sqrt(np.sum((delta / lengths) ** 2, axis=2))
    exp = np.exp(-np.sqrt(5) * r)
    c = amplitude * (5.0 / 3.0) * (1 + np.sqrt(5) * r) * exp
    row_delta = np.take_along_axis(delta, dims[:, None, None], axis=2)[:, :, 0]
    col_delta = np.take_along_axis(delta, dims[None, :, None], axis=2)[:, :, 0]
    derivative = c * (dims[:, None] == dims[None, :]) / lengths[dims][:, None] ** 2
    derivative -= (
        amplitude
        * (25.0 / 3.0)
        * exp
        * row_delta
        * col_delta
        / (lengths[dims][:, None] ** 2 * lengths[dims][None, :] ** 2)
    )
    return cross, derivative


def _kernel_parameters(response: kg.ResponseModel):
    kernel = response.model.kernel_.k1
    return float(kernel.k1.constant_value), kernel.k2.length_scale


def derivative_posterior(response: kg.ResponseModel, points, dims):
    model = response.model
    amplitude, lengths = _kernel_parameters(response)
    cross, prior = matern52_blocks(model.X_train_, points, dims, amplitude, lengths)
    solved = solve_triangular(model.L_, cross, lower=True)
    return cross.T @ model.alpha_, prior - solved.T @ solved, cross, prior


def _cholesky(matrix: np.ndarray) -> np.ndarray:
    matrix = (matrix + matrix.T) / 2
    for jitter in (1e-12, 1e-10, 1e-8, 1e-6):
        try:
            return np.linalg.cholesky(matrix + jitter * np.eye(len(matrix)))
        except np.linalg.LinAlgError:
            pass
    raise np.linalg.LinAlgError(
        "conditional derivative covariance is not positive definite"
    )


class SignConstrainedGP:
    def __init__(
        self,
        response: kg.ResponseModel,
        observations: Sequence[dict],
        *,
        softness: float = 0.1,
    ):
        self.response = response
        self.observations = [o for o in observations if o["weight"] > 0]
        self.active = bool(self.observations) and response.model is not None
        if not self.active:
            return
        if not np.isfinite(softness) or softness <= 0:
            raise ValueError("derivative softness must be positive")
        self.points = np.asarray([o["point"] for o in self.observations])
        self.dims = np.asarray(
            [o["variable_index"] for o in self.observations], dtype=int
        )
        signs = np.asarray([o["sign"] for o in self.observations])
        weights = np.asarray([o["weight"] for o in self.observations])
        if not np.isin(signs, [-1, 1]).all() or not np.isfinite(weights).all():
            raise ValueError("invalid derivative signs or weights")
        mean, cov, cross, prior = derivative_posterior(response, self.points, self.dims)
        self.scale = np.sqrt(np.diag(prior))
        mean = mean / self.scale
        cov = cov / self.scale[:, None] / self.scale[None, :]
        self.factor = _cholesky(cov)
        self.training_cross = cross / self.scale
        a = signs / softness

        def terms(t):
            z = a * (mean + self.factor @ t)
            log_cdf = log_ndtr(z)
            mills = np.exp(-0.5 * z**2 - 0.5 * np.log(2 * np.pi) - log_cdf)
            curvature = weights * a**2 * np.maximum(mills * (mills + z), 0)
            value = 0.5 * np.dot(t, t) - np.dot(weights, log_cdf)
            gradient = t - self.factor.T @ (weights * a * mills)
            hessian = np.eye(len(t)) + self.factor.T @ (
                curvature[:, None] * self.factor
            )
            return value, gradient, hessian

        result = minimize(
            lambda t: terms(t)[0],
            np.zeros(len(mean)),
            jac=lambda t: terms(t)[1],
            hess=lambda t: terms(t)[2],
            method="trust-exact",
            options={"maxiter": 100, "gtol": 1e-7},
        )
        if (
            not np.isfinite(result.x).all()
            or np.linalg.norm(terms(result.x)[1], ord=np.inf) > 1e-5
        ):
            raise ValueError(
                f"derivative Laplace inference did not converge: {result.message}"
            )
        self.mode = result.x
        self.hessian_factor = np.linalg.cholesky(terms(result.x)[2])
        self.iterations = int(result.nit)

    def predict(self, points):
        if not self.active:
            return predict_response(self.response, points)
        model = self.response.model
        base_mean, base_std = model.predict(points, return_std=True)
        amplitude, lengths = _kernel_parameters(self.response)
        cross, _ = matern52_blocks(points, self.points, self.dims, amplitude, lengths)
        covariance = cross / self.scale - model.kernel_(
            points, model.X_train_
        ) @ cho_solve((model.L_, True), self.training_cross)
        b = solve_triangular(self.factor, covariance.T, lower=True).T
        corrected = solve_triangular(self.hessian_factor, b.T, lower=True)
        variance = base_std**2 - np.sum(b**2, axis=1) + np.sum(corrected**2, axis=0)
        if not np.isfinite(variance).all() or np.min(variance) < -1e-7:
            raise ValueError("invalid sign-constrained predictive variance")
        mean = base_mean + b @ self.mode
        if not np.isfinite(mean).all():
            raise ValueError("invalid sign-constrained predictive mean")
        return (
            self.response.mean + self.response.standard_deviation * mean,
            self.response.standard_deviation * np.sqrt(np.maximum(variance, 0)),
        )


def select_virtual_derivatives(
    x,
    y,
    design_ids,
    models,
    weights,
    enabled,
    constraints: dict[str, Constraint],
    *,
    maximum: int = 20,
    conflict_probability: float = 0.8,
    response_fields: Sequence[str] = METRIC_FIELDS,
):
    """Select up to two active variables independently for each violated response."""
    selected: list[list[dict]] = [[] for _ in response_fields]
    audit = []
    for j, name in enumerate(response_fields):
        if not enabled[j] or models[j].model is None:
            continue
        variables = sorted(
            (i for i in range(8) if weights[i][j] > 0),
            key=lambda i: (-weights[i][j], i),
        )[:2]
        if not variables:
            continue
        scale = max(models[j].standard_deviation, 1e-12)
        residuals = constraints[name].residual(y[:, j]) / scale
        rows = sorted(
            (i for i in range(len(x)) if residuals[i] > 0),
            key=lambda i: (-residuals[i], design_ids[i]),
        )
        seen = set()
        for row in rows:
            for variable in variables:
                key = (tuple(x[row]), variable)
                if key in seen:
                    continue
                seen.add(key)
                sign = kg.PRIOR_SIGN[variable][j]
                observation = {
                    "design_id": design_ids[row],
                    "response": name,
                    "response_index": j,
                    "variable": kg.DESIGN_VARS[variable],
                    "variable_index": variable,
                    "point": x[row].tolist(),
                    "sign": sign,
                    "weight": float(weights[variable][j]),
                    "standardized_violation": float(residuals[row]),
                }
                try:
                    mean, cov, _, _ = derivative_posterior(
                        models[j], x[row : row + 1], [variable]
                    )
                    std = np.sqrt(max(float(cov[0, 0]), 0))
                    opposite = (
                        float(ndtr(-sign * mean[0] / std))
                        if std > 1e-12
                        else float(sign * mean[0] < 0)
                    )
                    observation["opposite_probability"] = opposite
                    observation["status"] = (
                        "local_sign_conflict"
                        if opposite >= conflict_probability
                        else "capacity_limit"
                        if len(selected[j]) >= maximum
                        else "selected"
                    )
                except (ValueError, np.linalg.LinAlgError) as error:
                    observation.update(
                        status="derivative_check_failed", error=str(error)
                    )
                audit.append(observation)
                if observation["status"] == "selected":
                    selected[j].append(observation)
    return selected, audit
