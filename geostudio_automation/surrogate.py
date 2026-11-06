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


def _derivative_cross(points, dims, sites, site_dims, amplitude, lengths):
    """Covariance between two sets of Matern-5/2 partial derivatives."""
    points, sites = np.asarray(points), np.asarray(sites)
    dims, site_dims = np.asarray(dims, dtype=int), np.asarray(site_dims, dtype=int)
    lengths = np.broadcast_to(np.asarray(lengths, dtype=float), (points.shape[1],))
    delta = points[:, None, :] - sites[None, :, :]
    r = np.sqrt(np.sum((delta / lengths) ** 2, axis=2))
    exponential = np.exp(-np.sqrt(5) * r)
    row = np.take_along_axis(delta, dims[:, None, None], axis=2)[:, :, 0]
    col = np.take_along_axis(delta, site_dims[None, :, None], axis=2)[:, :, 0]
    return (
        amplitude * (5 / 3) * (1 + np.sqrt(5) * r) * exponential
        * (dims[:, None] == site_dims[None, :]) / lengths[dims][:, None] ** 2
        - amplitude * (25 / 3) * exponential * row * col
        / (lengths[dims][:, None] ** 2 * lengths[site_dims][None, :] ** 2)
    )


def derivative_marginals(response: kg.ResponseModel, points, dims):
    """Data-only derivative means and marginal variances without a dense candidate matrix."""
    model = response.model
    amplitude, lengths = _kernel_parameters(response)
    points, dims = np.asarray(points), np.asarray(dims, dtype=int)
    lengths = np.broadcast_to(np.asarray(lengths, dtype=float), (points.shape[1],))
    delta = model.X_train_[:, None, :] - points[None, :, :]
    r = np.sqrt(np.sum((delta / lengths) ** 2, axis=2))
    coordinate = np.take_along_axis(delta, dims[None, :, None], axis=2)[:, :, 0]
    cross = (
        amplitude * (5 / 3) * (1 + np.sqrt(5) * r)
        * np.exp(-np.sqrt(5) * r) * coordinate / lengths[dims] ** 2
    )
    solved = solve_triangular(model.L_, cross, lower=True)
    variance = amplitude * (5 / 3) / lengths[np.asarray(dims, dtype=int)] ** 2
    return cross.T @ model.alpha_, np.maximum(variance - np.sum(solved**2, axis=0), 0), cross


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

    def derivative_marginals(self, points, dims):
        """Derivative posterior after the currently selected soft sign observations."""
        mean, variance, cross = derivative_marginals(self.response, points, dims)
        if not self.active:
            return mean, variance
        amplitude, lengths = _kernel_parameters(self.response)
        prior = _derivative_cross(
            points, dims, self.points, self.dims, amplitude, lengths
        ) / self.scale
        covariance = prior - cross.T @ cho_solve(
            (self.response.model.L_, True), self.training_cross
        )
        b = solve_triangular(self.factor, covariance.T, lower=True).T
        corrected = solve_triangular(self.hessian_factor, b.T, lower=True)
        variance = variance - np.sum(b**2, axis=1) + np.sum(corrected**2, axis=0)
        if not np.isfinite(variance).all() or np.min(variance) < -1e-7:
            raise ValueError("invalid guided derivative variance")
        return mean + b @ self.mode, np.maximum(variance, 0)


def select_virtual_derivatives(
    x,
    candidates,
    models,
    weights,
    enabled,
    constraints: dict[str, Constraint],
    *,
    acquisition_relevant,
    softness: float = 0.1,
    maximum: int = 20,
    minimum_probability: float = 0.1,
    maximum_probability: float = 0.5,
    conflict_probability: float = 0.8,
    boundary_std: float = 2.0,
    minimum_distance: float = 0.15,
    support_factor: float = 1.5,
    response_fields: Sequence[str] = METRIC_FIELDS,
):
    """Greedily place signs where the current derivative posterior is informative.

    Eligibility and relevance use only the real-data GPs. After each placement the
    derivative probabilities are updated with the guided posterior. Conflicting
    locations are audited, not coerced into the prior direction.
    """
    x, candidates = np.asarray(x), np.asarray(candidates)
    acquisition_relevant = np.asarray(acquisition_relevant, dtype=bool)
    if candidates.ndim != 2 or candidates.shape[1] != x.shape[1]:
        raise ValueError("candidate points must match the observed design dimension")
    if acquisition_relevant.shape != (len(candidates),):
        raise ValueError("acquisition relevance must match the candidate count")
    selected: list[list[dict]] = [[] for _ in response_fields]
    audit = []
    # Compare candidate coverage with the observed design's own nearest-neighbour
    # spacing. This excludes extrapolation without assuming a fixed 8-D radius.
    distances = np.linalg.norm(candidates[:, None, :] - x[None, :, :], axis=2)
    observed_distances = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    np.fill_diagonal(observed_distances, np.inf)
    support_radius = support_factor * float(np.quantile(np.min(observed_distances, axis=1), 0.9))
    supported = np.min(distances, axis=1) <= support_radius
    for j, name in enumerate(response_fields):
        if not enabled[j] or models[j].model is None:
            continue
        variables = sorted(
            (i for i in range(8) if weights[i][j] > 0),
            key=lambda i: (-weights[i][j], i),
        )[:2]
        if not variables:
            continue
        means, stds = predict_response(models[j], candidates)
        boundary = np.abs(constraints[name].residual(means)) <= boundary_std * stds
        relevant = supported & (boundary | acquisition_relevant)
        points = np.concatenate([candidates] * len(variables))
        dims = np.repeat(variables, len(candidates))
        eligible = np.tile(relevant, len(variables))
        for step in range(maximum):
            try:
                guide = SignConstrainedGP(models[j], selected[j], softness=softness)
                derivative_mean, derivative_variance = guide.derivative_marginals(points, dims)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                audit.append({"response": name, "status": "derivative_check_failed", "error": str(error)})
                break
            signed = np.asarray([kg.PRIOR_SIGN[i][j] for i in dims]) * derivative_mean
            derivative_std = np.sqrt(derivative_variance)
            probability = np.where(
                derivative_std > 1e-12,
                ndtr(-signed / np.maximum(derivative_std, 1e-12)),
                (signed < 0).astype(float),
            )
            scores = probability * np.asarray([weights[i][j] for i in dims])
            scores[~eligible | (probability < minimum_probability) | (probability > maximum_probability)] = -np.inf
            if not np.isfinite(scores).any():
                break
            index = int(np.argmax(scores))
            candidate_index = index % len(candidates)
            variable = int(dims[index])
            observation = {
                "candidate_index": candidate_index,
                "response": name,
                "response_index": j,
                "variable": kg.DESIGN_VARS[variable],
                "variable_index": variable,
                "point": candidates[candidate_index].tolist(),
                "sign": kg.PRIOR_SIGN[variable][j],
                "weight": float(weights[variable][j]),
                "opposite_probability": float(probability[index]),
                "near_constraint_boundary": bool(boundary[candidate_index]),
                "acquisition_relevant": bool(acquisition_relevant[candidate_index]),
                "selection_step": step + 1,
                "status": "selected",
            }
            selected[j].append(observation)
            audit.append(observation)
            # The selected site's sign and its near duplicates carry the same
            # information. Other variables remain eligible at this location.
            start = variables.index(variable) * len(candidates)
            nearby = np.linalg.norm(candidates - candidates[candidate_index], axis=1) < minimum_distance
            eligible[start : start + len(candidates)] &= ~nearby
        # Record unresolved data-prior conflicts in the decision region.
        informative_count = 0
        disagreement_count = 0
        for variable in variables:
            dim = np.full(len(candidates), variable)
            mean, variance, _ = derivative_marginals(models[j], candidates, dim)
            signed = kg.PRIOR_SIGN[variable][j] * mean
            std = np.sqrt(variance)
            probability = ndtr(-signed / np.maximum(std, 1e-12))
            informative_count += int(np.sum(
                relevant & (probability >= minimum_probability)
                & (probability <= maximum_probability)
            ))
            conflict = relevant & (probability > maximum_probability)
            disagreement_count += int(np.sum(conflict))
            if np.any(conflict):
                index = int(np.argmax(np.where(conflict, probability, -np.inf)))
                audit.append({
                    "response": name, "variable": kg.DESIGN_VARS[variable],
                    "candidate_index": index, "point": candidates[index].tolist(),
                    "opposite_probability": float(probability[index]),
                    "conflicting_candidate_count": int(np.sum(conflict)),
                    "status": (
                        "local_sign_conflict"
                        if probability[index] >= conflict_probability
                        else "direction_disagreement"
                    ),
                })
        audit.append({
            "response": name, "status": "selection_summary",
            "candidate_count": len(candidates),
            "supported_candidate_count": int(np.sum(supported)),
            "relevant_candidate_count": int(np.sum(relevant)),
            "informative_variable_candidate_count": informative_count,
            "disagreeing_variable_candidate_count": disagreement_count,
            "selected_count": len(selected[j]),
        })
    return selected, audit
