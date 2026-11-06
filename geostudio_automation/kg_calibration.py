from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import warnings
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .model import DesignVariables
from .results import metric_fields

KG_CALIBRATION_SCHEMA_VERSION = 1

DESIGN_VARS: tuple[str, ...] = DesignVariables.CSV_FIELDS[1:]
CONSTRAINTS: tuple[str, ...] = (
    "crest_max_settlement",
    "body_max_settlement",
    "body_max_horizontal_displacement",
    "major_principal_stress",
    "minor_principal_stress",
)

PRIOR_SIGN: tuple[tuple[int, ...], ...] = (
    (-1, -1, -1, -1, +1),
    (-1, -1, -1, -1, +1),
    (+1, +1, +1, +1, +1),
    (-1, -1, +1, +1, +1),
    (+1, +1, +1, +1, -1),
    (+1, +1, +1, +1, -1),
    (+1, +1, +1, +1, -1),
    (+1, +1, -1, +1, -1),
)

PRIOR_TIER: tuple[tuple[float, ...], ...] = (
    (1.0, 0.5, 1.0, 0.5, 1.0),
    (1.0, 1.0, 1.0, 0.5, 1.0),
    (0.1, 0.5, 0.5, 1.0, 0.5),
    (1.0, 0.5, 0.1, 0.1, 0.1),
    (1.0, 0.5, 0.5, 0.1, 0.1),
    (0.5, 1.0, 1.0, 0.5, 0.5),
    (1.0, 1.0, 0.1, 0.5, 0.1),
    (0.5, 1.0, 0.1, 0.1, 0.1),
)


@dataclass(frozen=True, slots=True)
class KGCalibrationConfig:
    min_samples: int = 16
    ale_bins: int = 3
    min_samples_per_bin: int = 4
    min_valid_backgrounds: int = 8
    min_valid_fraction: float = 0.5
    direction_threshold: float = 0.8
    strength_threshold: float = 0.5
    max_posterior_std: float = 1.0
    support_distance_quantile: float = 0.95
    zero_effect_tolerance: float = 1.0e-8
    random_seed: int = 2026
    gp_restarts: int = 3

    def validate(self) -> None:
        if self.min_samples <= 0:
            raise ValueError("min_samples must be positive")
        if self.ale_bins < 2:
            raise ValueError("ale_bins must be at least 2")
        if self.min_samples_per_bin <= 0:
            raise ValueError("min_samples_per_bin must be positive")
        if self.min_valid_backgrounds <= 0:
            raise ValueError("min_valid_backgrounds must be positive")
        if not 0 < self.min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must be in (0, 1]")
        if not 0.5 < self.direction_threshold <= 1:
            raise ValueError("direction_threshold must be in (0.5, 1]")
        if self.strength_threshold < 0:
            raise ValueError("strength_threshold must be non-negative")
        if self.max_posterior_std <= 0:
            raise ValueError("max_posterior_std must be positive")
        if not 0 < self.support_distance_quantile <= 1:
            raise ValueError("support_distance_quantile must be in (0, 1]")
        if self.zero_effect_tolerance < 0:
            raise ValueError("zero_effect_tolerance must be non-negative")
        if self.gp_restarts < 0:
            raise ValueError("gp_restarts must be non-negative")


@dataclass(frozen=True, slots=True)
class EdgeCalibration:
    design_variable_index: int
    design_variable: str
    constraint_index: int
    constraint: str
    response_column: str
    prior_sign: int
    prior_tier: float
    valid_comparisons: int
    required_comparisons: int
    positive_ratio: float
    negative_ratio: float
    prior_direction_consistency: float
    ale_effect_strength: float
    status: str
    effective_weight: float

    def as_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class KGCalibrationOutputs:
    output_directory: Path
    weights_path: Path
    edge_calibration_path: Path
    report_path: Path
    effective_weights: tuple[tuple[float, ...], ...]
    active_mask: tuple[tuple[bool, ...], ...]
    edges: tuple[EdgeCalibration, ...]

    @property
    def active_edges(self) -> int:
        return sum(edge.status == "active" for edge in self.edges)


@dataclass(frozen=True, slots=True)
class _Dataset:
    design_ids: tuple[str, ...]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    response_columns: tuple[str, ...]
    analysis_name: str
    step: int
    cvar_alpha: float
    sha256: str


@dataclass(frozen=True, slots=True)
class _ResponseModel:
    model: GaussianProcessRegressor | None
    mean: float
    standard_deviation: float
    status: str
    fitted_kernel: str
    warnings: tuple[str, ...]


# Public name for sharing a data-only fit with the optimizer. Keep the private
# alias for compatibility with existing callers and tests.
ResponseModel = _ResponseModel


def calibrate_kg_edges(
    dataset_path: Path,
    output_directory: Path,
    *,
    config: KGCalibrationConfig | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
    response_models: Sequence[ResponseModel] | None = None,
    response_enabled: Sequence[bool] | None = None,
    response_fields: Sequence[str] | None = None,
) -> KGCalibrationOutputs:
    """Calibrate selected knowledge edges from an Initial DoE dataset."""
    selected = config or KGCalibrationConfig()
    selected.validate()
    dataset = _load_dataset(dataset_path, response_fields=response_fields)
    _validate_priors()
    full_fields = metric_fields(dataset.cvar_alpha)
    indices = [full_fields.index(name) for name in dataset.response_columns]
    constraints = [CONSTRAINTS[index] for index in indices]

    output_directory = output_directory.resolve()
    weights_path = output_directory / "kg_weights.json"
    edge_path = output_directory / "kg_edge_calibration.csv"
    report_path = output_directory / "calibration_report.json"
    targets = (weights_path, edge_path, report_path)
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise ValueError(
            "calibration output already exists; pass force=True to replace: " + names
        )

    x_normalized = _normalize_designs(dataset.x)
    enabled = list(response_enabled) if response_enabled is not None else [True] * len(indices)
    if len(enabled) != len(indices) or any(type(v) is not bool for v in enabled):
        raise ValueError("response_enabled must match selected response count")
    supplied_models = response_models
    if supplied_models is not None:
        if len(supplied_models) != len(indices):
            raise ValueError("response_models must match selected response count")
        for index, response in enumerate(supplied_models):
            y = dataset.y[:, index]
            if not np.isclose(response.mean, np.mean(y), rtol=1e-10, atol=1e-12) or not np.isclose(
                response.standard_deviation, np.std(y, ddof=1), rtol=1e-10, atol=1e-12
            ):
                raise ValueError("supplied response model scaling does not match dataset")
            if response.model is not None:
                if not isinstance(response.model, GaussianProcessRegressor):
                    raise ValueError("calibration requires data-only GaussianProcessRegressor fits")
                if response.model.X_train_.shape != x_normalized.shape or not np.allclose(
                    response.model.X_train_, x_normalized, rtol=0, atol=1e-12
                ) or not np.allclose(
                    response.model.y_train_, (y - response.mean) / response.standard_deviation
                ):
                    raise ValueError("supplied response model training data does not match dataset")
            elif len(y) >= selected.min_samples and response.standard_deviation > np.finfo(float).eps:
                raise ValueError("a nonconstant response requires a fitted data-only GP")
    support_radius = _support_radius(x_normalized, selected.support_distance_quantile)
    required_comparisons = max(
        selected.min_valid_backgrounds,
        math.ceil(len(dataset.design_ids) * selected.min_valid_fraction),
    )

    response_models = []
    global_warnings: list[str] = []
    if len(dataset.design_ids) < selected.min_samples:
        global_warnings.append(
            f"only {len(dataset.design_ids)} samples are available; "
            f"at least {selected.min_samples} are required"
        )
        response_models = [
            _ResponseModel(
                None,
                float(np.mean(dataset.y[:, index])),
                0.0,
                "not_fitted_insufficient_samples",
                "",
                (),
            )
            for index in range(len(indices))
        ]
    else:
        for response_index, response_column in enumerate(dataset.response_columns):
            response = supplied_models[response_index] if supplied_models is not None else _fit_response_model(
                x_normalized,
                dataset.y[:, response_index],
                selected,
                response_index=response_index,
            )
            response_models.append(response)
            if progress is not None:
                progress(
                    f"[{response_index + 1}/{len(indices)}] "
                    f"{response_column}: {response.status}"
                )
            global_warnings.extend(response.warnings)

    edges: list[EdgeCalibration] = []
    effective = np.zeros((len(DESIGN_VARS), len(indices)), dtype=float)
    active = np.zeros((len(DESIGN_VARS), len(indices)), dtype=bool)
    for variable_index, variable in enumerate(DESIGN_VARS):
        for response_index, constraint in enumerate(constraints):
            if len(dataset.design_ids) < selected.min_samples:
                edge = _empty_edge(
                    variable_index,
                    response_index,
                    dataset.response_columns[response_index],
                    required_comparisons,
                    "insufficient_samples",
                )
            elif not enabled[response_index]:
                edge = _empty_edge(
                    variable_index, response_index,
                    dataset.response_columns[response_index], required_comparisons,
                    "accuracy_not_met",
                )
            else:
                edge = _calibrate_edge(
                    x_normalized,
                    variable_index,
                    response_index,
                    dataset.response_columns[response_index],
                    response_models[response_index],
                    support_radius,
                    required_comparisons,
                    selected,
                )
            edges.append(edge)
            effective[variable_index, response_index] = edge.effective_weight
            active[variable_index, response_index] = edge.status == "active"

    created_at = datetime.now(UTC).isoformat()
    dataset_metadata = {
        "path": str(dataset_path.resolve()),
        "sha256": dataset.sha256,
        "samples": len(dataset.design_ids),
        "analysis_name": dataset.analysis_name,
        "step": dataset.step,
        "cvar_alpha": dataset.cvar_alpha,
    }
    weights_payload = {
        "schema_version": KG_CALIBRATION_SCHEMA_VERSION,
        "dataset": dataset_metadata,
        "design_variables": list(DESIGN_VARS),
        "constraints": constraints,
        "response_columns": list(dataset.response_columns),
        "prior_sign": [[row[index] for index in indices] for row in PRIOR_SIGN],
        "prior_tier": [[row[index] for index in indices] for row in PRIOR_TIER],
        "effective_weight": effective.tolist(),
        "active_mask": active.tolist(),
        "config": asdict(selected),
        "response_enabled": enabled,
        "created_at": created_at,
    }
    status_counts = Counter(edge.status for edge in edges)
    report_payload = {
        "schema_version": KG_CALIBRATION_SCHEMA_VERSION,
        "dataset": dataset_metadata,
        "response_models": [
            {
                "constraint": constraints[index],
                "response_column": dataset.response_columns[index],
                "status": response.status,
                "response_mean": response.mean,
                "response_standard_deviation": response.standard_deviation,
                "fitted_kernel": response.fitted_kernel,
                "warnings": list(response.warnings),
            }
            for index, response in enumerate(response_models)
        ],
        "required_valid_comparisons_per_edge": required_comparisons,
        "support_radius_normalized": support_radius,
        "active_edges": int(np.sum(active)),
        "status_counts": dict(sorted(status_counts.items())),
        "warnings": list(dict.fromkeys(global_warnings)),
        "created_at": created_at,
    }

    _write_json_atomic(weights_path, weights_payload)
    _write_csv_atomic(edge_path, _edge_fields(), [edge.as_row() for edge in edges])
    _write_json_atomic(report_path, report_payload)
    return KGCalibrationOutputs(
        output_directory,
        weights_path,
        edge_path,
        report_path,
        tuple(tuple(float(value) for value in row) for row in effective),
        tuple(tuple(bool(value) for value in row) for row in active),
        tuple(edges),
    )


def _load_dataset(path: Path, *, response_fields: Sequence[str] | None = None) -> _Dataset:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"Initial DoE dataset does not exist: {path}")
    raw = path.read_bytes()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("Initial DoE dataset has no CSV header")
        header = set(reader.fieldnames)
        base_required = {
            "design_id",
            *DESIGN_VARS,
            "analysis_name",
            "step",
            "cvar_alpha",
        }
        missing = sorted(base_required - header)
        if missing:
            raise ValueError(
                "Initial DoE dataset is missing columns: " + ", ".join(missing)
            )
        rows = list(reader)
    if not rows:
        raise ValueError("Initial DoE dataset contains no successful samples")

    design_ids = tuple(row["design_id"].strip() for row in rows)
    if any(not value for value in design_ids):
        raise ValueError("Initial DoE dataset contains an empty design_id")
    duplicates = sorted(
        value for value, count in Counter(design_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError("duplicate design_id values: " + ", ".join(duplicates))

    analysis_names = {row["analysis_name"].strip() for row in rows}
    if "" in analysis_names or len(analysis_names) != 1:
        raise ValueError(
            "analysis_name must be non-empty and identical for all samples"
        )
    steps = {_non_negative_integer(row["step"], "step") for row in rows}
    if len(steps) != 1:
        raise ValueError("step must be identical for all samples")
    alphas = {_finite_float(row["cvar_alpha"], "cvar_alpha") for row in rows}
    if len(alphas) != 1:
        raise ValueError("cvar_alpha must be identical for all samples")
    alpha = next(iter(alphas))
    if not 0 < alpha < 1:
        raise ValueError("cvar_alpha must be strictly between 0 and 1")
    all_fields = metric_fields(alpha)
    response_columns = tuple(response_fields) if response_fields is not None else all_fields
    if response_columns not in (all_fields[:3], all_fields):
        raise ValueError("response_fields must select three displacements or all responses")
    missing_responses = sorted(set(response_columns) - header)
    if missing_responses:
        raise ValueError(
            "Initial DoE dataset is missing response columns: "
            + ", ".join(missing_responses)
        )

    x = np.asarray(
        [[_finite_float(row[name], name) for name in DESIGN_VARS] for row in rows],
        dtype=float,
    )
    y = np.asarray(
        [[_finite_float(row[name], name) for name in response_columns] for row in rows],
        dtype=float,
    )
    for index, name in enumerate(DESIGN_VARS):
        lower, upper = DesignVariables.BOUNDS[name]
        invalid = np.flatnonzero((x[:, index] < lower) | (x[:, index] > upper))
        if len(invalid):
            raise ValueError(f"{name} contains values outside [{lower:g}, {upper:g}]")
    unique_designs = np.unique(x, axis=0)
    if len(unique_designs) != len(x):
        raise ValueError("Initial DoE dataset contains duplicate design vectors")

    return _Dataset(
        design_ids,
        x,
        y,
        response_columns,
        next(iter(analysis_names)),
        next(iter(steps)),
        alpha,
        hashlib.sha256(raw).hexdigest(),
    )


def _normalize_designs(x: NDArray[np.float64]) -> NDArray[np.float64]:
    lower = np.asarray([DesignVariables.BOUNDS[name][0] for name in DESIGN_VARS])
    upper = np.asarray([DesignVariables.BOUNDS[name][1] for name in DESIGN_VARS])
    return (x - lower) / (upper - lower)


def _fit_response_model(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    config: KGCalibrationConfig,
    *,
    response_index: int,
) -> _ResponseModel:
    mean = float(np.mean(y))
    standard_deviation = float(np.std(y, ddof=1))
    if standard_deviation <= np.finfo(float).eps:
        return _ResponseModel(
            None,
            mean,
            standard_deviation,
            "constant_response",
            "",
            ("response is constant; its knowledge edges cannot be activated",),
        )
    standardized = (y - mean) / standard_deviation
    kernel = ConstantKernel(1.0, (1.0e-2, 1.0e2)) * Matern(
        length_scale=0.5, length_scale_bounds=(0.05, 5.0), nu=2.5
    ) + WhiteKernel(noise_level=1.0e-6, noise_level_bounds=(1.0e-8, 1.0e-1))
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1.0e-8,
        normalize_y=False,
        n_restarts_optimizer=config.gp_restarts,
        random_state=config.random_seed + response_index,
    )
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model.fit(x, standardized)
    except Exception as error:
        raise ValueError(
            f"could not fit GP for response {response_index + 1}: {error}"
        ) from error
    warning_messages = tuple(
        f"response {response_index + 1}: {item.message}" for item in captured
    )
    return _ResponseModel(
        model,
        mean,
        standard_deviation,
        "fitted",
        str(model.kernel_),
        warning_messages,
    )


def fit_response_model(
    x: NDArray[np.float64], y: NDArray[np.float64], config: KGCalibrationConfig,
    *, response_index: int,
) -> ResponseModel:
    """Fit one data-only GP on unit-box inputs, standardizing this training set only."""
    return _fit_response_model(x, y, config, response_index=response_index)


def _calibrate_edge(
    x: NDArray[np.float64],
    variable_index: int,
    response_index: int,
    response_column: str,
    response: _ResponseModel,
    support_radius: float,
    required_comparisons: int,
    config: KGCalibrationConfig,
) -> EdgeCalibration:
    if response.model is None:
        return _empty_edge(
            variable_index,
            response_index,
            response_column,
            required_comparisons,
            "unstable_direction",
        )
    bins = _quantile_bins(x[:, variable_index], config.ale_bins)
    if bins is None:
        return _empty_edge(
            variable_index,
            response_index,
            response_column,
            required_comparisons,
            "insufficient_support",
        )
    bin_indices = np.digitize(x[:, variable_index], bins[1:-1], right=True)
    bin_counts = np.bincount(bin_indices, minlength=config.ale_bins)
    if np.any(bin_counts < config.min_samples_per_bin):
        return _empty_edge(
            variable_index,
            response_index,
            response_column,
            required_comparisons,
            "insufficient_support",
        )

    valid_deltas: list[float] = []
    deltas_by_bin: list[list[float]] = [[] for _ in range(config.ale_bins)]
    for sample_index, background in enumerate(x):
        bin_index = int(bin_indices[sample_index])
        lower_query = background.copy()
        upper_query = background.copy()
        lower_query[variable_index] = bins[bin_index]
        upper_query[variable_index] = bins[bin_index + 1]
        if not _queries_are_supported(
            lower_query,
            upper_query,
            x,
            support_radius,
        ):
            continue
        predictions, deviations = response.model.predict(
            np.vstack((lower_query, upper_query)), return_std=True
        )
        if np.any(deviations > config.max_posterior_std):
            continue
        delta = float(predictions[1] - predictions[0])
        valid_deltas.append(delta)
        deltas_by_bin[bin_index].append(delta)

    if len(valid_deltas) < required_comparisons or any(
        not values for values in deltas_by_bin
    ):
        return _edge_from_metrics(
            variable_index,
            response_index,
            response_column,
            required_comparisons,
            valid_deltas,
            0.0,
            "insufficient_support",
            config,
        )

    mean_bin_effects = np.asarray(
        [float(np.mean(values)) for values in deltas_by_bin], dtype=float
    )
    accumulated = np.concatenate(([0.0], np.cumsum(mean_bin_effects)))
    strength = float(np.max(accumulated) - np.min(accumulated))
    prior_sign = PRIOR_SIGN[variable_index][response_index]
    positive_ratio, negative_ratio = _direction_ratios(valid_deltas, config)
    consistency = positive_ratio if prior_sign > 0 else negative_ratio
    opposite = negative_ratio if prior_sign > 0 else positive_ratio
    if opposite >= config.direction_threshold:
        status = "sign_conflict"
    elif consistency < config.direction_threshold:
        status = "unstable_direction"
    elif strength < config.strength_threshold:
        status = "weak_effect"
    else:
        status = "active"
    return _edge_from_metrics(
        variable_index,
        response_index,
        response_column,
        required_comparisons,
        valid_deltas,
        strength,
        status,
        config,
    )


def _edge_from_metrics(
    variable_index: int,
    response_index: int,
    response_column: str,
    required_comparisons: int,
    deltas: Sequence[float],
    strength: float,
    status: str,
    config: KGCalibrationConfig,
) -> EdgeCalibration:
    positive_ratio, negative_ratio = _direction_ratios(deltas, config)
    prior_sign = PRIOR_SIGN[variable_index][response_index]
    consistency = positive_ratio if prior_sign > 0 else negative_ratio
    prior_tier = PRIOR_TIER[variable_index][response_index]
    weight = (
        prior_tier * consistency * min(1.0, strength) if status == "active" else 0.0
    )
    return EdgeCalibration(
        variable_index + 1,
        DESIGN_VARS[variable_index],
        response_index + 1,
        CONSTRAINTS[response_index],
        response_column,
        prior_sign,
        prior_tier,
        len(deltas),
        required_comparisons,
        positive_ratio,
        negative_ratio,
        consistency,
        strength,
        status,
        weight,
    )


def _empty_edge(
    variable_index: int,
    response_index: int,
    response_column: str,
    required_comparisons: int,
    status: str,
) -> EdgeCalibration:
    return EdgeCalibration(
        variable_index + 1,
        DESIGN_VARS[variable_index],
        response_index + 1,
        CONSTRAINTS[response_index],
        response_column,
        PRIOR_SIGN[variable_index][response_index],
        PRIOR_TIER[variable_index][response_index],
        0,
        required_comparisons,
        0.0,
        0.0,
        0.0,
        0.0,
        status,
        0.0,
    )


def _quantile_bins(
    values: NDArray[np.float64], count: int
) -> NDArray[np.float64] | None:
    edges = np.quantile(values, np.linspace(0.0, 1.0, count + 1))
    if len(np.unique(edges)) != count + 1:
        return None
    return np.asarray(edges, dtype=float)


def _direction_ratios(
    deltas: Sequence[float], config: KGCalibrationConfig
) -> tuple[float, float]:
    if not deltas:
        return 0.0, 0.0
    values = np.asarray(deltas)
    total = len(values)
    positive = float(np.sum(values > config.zero_effect_tolerance)) / total
    negative = float(np.sum(values < -config.zero_effect_tolerance)) / total
    return positive, negative


def _support_radius(x: NDArray[np.float64], quantile: float) -> float:
    if len(x) < 2:
        return 0.0
    distances = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    return float(np.quantile(nearest, quantile))


def _queries_are_supported(
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    x: NDArray[np.float64],
    radius: float,
) -> bool:
    if np.any(lower < 0.0) or np.any(lower > 1.0):
        return False
    if np.any(upper < 0.0) or np.any(upper > 1.0):
        return False
    lower_distance = float(np.min(np.linalg.norm(x - lower, axis=1)))
    upper_distance = float(np.min(np.linalg.norm(x - upper, axis=1)))
    tolerance = max(1.0e-12, radius * 1.0e-12)
    return lower_distance <= radius + tolerance and upper_distance <= radius + tolerance


def _validate_priors() -> None:
    expected = (len(DESIGN_VARS), len(CONSTRAINTS))
    if np.shape(PRIOR_SIGN) != expected or np.shape(PRIOR_TIER) != expected:
        raise ValueError(f"prior matrices must both have shape {expected}")
    if any(value not in (-1, 1) for row in PRIOR_SIGN for value in row):
        raise ValueError("PRIOR_SIGN values must be -1 or +1")
    if any(value not in (0.1, 0.5, 1.0) for row in PRIOR_TIER for value in row):
        raise ValueError("PRIOR_TIER values must be 0.1, 0.5, or 1.0")


def _edge_fields() -> tuple[str, ...]:
    return tuple(EdgeCalibration.__dataclass_fields__)


def _finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _positive_integer(value: object, label: str) -> int:
    numeric = _finite_float(value, label)
    result = int(numeric)
    if result <= 0 or not math.isclose(numeric, result):
        raise ValueError(f"{label} must be a positive integer")
    return result


def _non_negative_integer(value: object, label: str) -> int:
    numeric = _finite_float(value, label)
    result = int(numeric)
    if result < 0 or not math.isclose(numeric, result):
        raise ValueError(f"{label} must be a non-negative integer")
    return result


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_csv_atomic(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


__all__ = [
    "CONSTRAINTS",
    "DESIGN_VARS",
    "KG_CALIBRATION_SCHEMA_VERSION",
    "PRIOR_SIGN",
    "PRIOR_TIER",
    "KGCalibrationConfig",
    "KGCalibrationOutputs",
    "ResponseModel",
    "calibrate_kg_edges",
    "fit_response_model",
]
