"""Sequential constrained BO with auditable, data-gated monotonicity knowledge."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr

from .kg_calibration import (
    PRIOR_SIGN,
    PRIOR_TIER,
    KGCalibrationConfig,
    _write_csv_atomic,
    calibrate_kg_edges,
)
from .model import DesignVariables
from .optimization_config import ExperimentConfig
from .optimization_evaluation import (
    EvaluationResult,
    Evaluator,
    GeometryDomain,
    GeoStudioEvaluator,
    SimulatedEvaluator,
    read_evaluation,
    write_evaluation,
)
from .results import (
    DEFAULT_ANALYSIS_NAME,
    DEFAULT_ANALYSIS_STEP,
    DEFAULT_CVAR_ALPHA,
    DISPLACEMENT_METRIC_FIELDS,
)
from .sampling import (
    design_from_unit,
    feasible_lhs,
    point_key,
    replace_in_strata,
    sobol_candidates,
    unit_design,
)
from .surrogate import (
    SignConstrainedGP,
    fit_and_validate,
    predict_response,
    select_virtual_derivatives,
)

OPTIMIZATION_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class OptimizationOutputs:
    output_directory: Path
    report_path: Path
    dataset_path: Path
    best_design: dict | None
    attempts_spent: int
    successful_samples: int
    stop_reason: str


def _json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


@contextmanager
def _experiment_lock(directory: Path):
    """OS-owned lock is released even if the Python process is interrupted."""
    stream = (directory / ".optimizer.lock").open("a+b")
    if stream.tell() == 0:
        stream.write(b"\0")
        stream.flush()
    stream.seek(0)
    try:
        if __import__("os").name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise ValueError(
            "another optimizer is already using this output directory"
        ) from error
    try:
        yield
    finally:
        stream.close()


def _successful(state):
    return [a for a in state["attempts"] if a["status"] == "succeeded"]


def _load_initial_dataset(config: ExperimentConfig, domain):
    path = config.initial_dataset
    if path is None:
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != config.initial_samples:
        raise ValueError(
            f"initial_dataset has {len(rows)} rows; expected {config.initial_samples}"
        )
    designs, metrics = [], []
    for row in rows:
        design = DesignVariables.from_mapping(row)
        if design.validate_bounds():
            raise ValueError(f"initial_dataset contains out-of-bounds design {design.design_id}")
        if row.get("parameter_hash") != design.parameter_hash():
            raise ValueError(f"initial_dataset parameter hash mismatch: {design.design_id}")
        if (row.get("analysis_name") != DEFAULT_ANALYSIS_NAME
                or int(row.get("step", -1)) != DEFAULT_ANALYSIS_STEP
                or float(row.get("cvar_alpha", -1)) != DEFAULT_CVAR_ALPHA):
            raise ValueError(f"initial_dataset result selection mismatch: {design.design_id}")
        values = {name: float(row[name]) for name in DISPLACEMENT_METRIC_FIELDS}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError(f"initial_dataset has nonfinite displacement: {design.design_id}")
        designs.append(design)
        metrics.append(values)
    if len({design.design_id for design in designs}) != len(designs):
        raise ValueError("initial_dataset has duplicate design IDs")
    points = np.asarray([unit_design(design) for design in designs])
    strata = np.floor(points * len(points)).astype(int)
    expected = np.tile(np.arange(len(points))[:, None], (1, points.shape[1]))
    if not np.array_equal(np.sort(strata, axis=0), expected):
        raise ValueError("initial_dataset does not preserve Latin strata")
    for design in designs:
        domain.prepare(design)
    return list(zip(designs, points, metrics, strict=True))


def _spent(state, *, adaptive=False):
    return sum(
        bool(a["charged"]) and (not adaptive or a["slot"] is None)
        for a in state["attempts"]
    )


def _dataset_rows(state):
    return [
        {
            **a["design"],
            "parameter_hash": DesignVariables(**a["design"]).parameter_hash(),
            "analysis_name": DEFAULT_ANALYSIS_NAME,
            "step": DEFAULT_ANALYSIS_STEP,
            "cvar_alpha": DEFAULT_CVAR_ALPHA,
            **{name: a["metrics"][name] for name in DISPLACEMENT_METRIC_FIELDS},
            "cost_per_m": a["cost_per_m"],
        }
        for a in _successful(state)
    ]


def _export_data(directory, state):
    fields = (
        *DesignVariables.CSV_FIELDS,
        "parameter_hash",
        "analysis_name",
        "step",
        "cvar_alpha",
        *DISPLACEMENT_METRIC_FIELDS,
        "cost_per_m",
    )
    _write_csv_atomic(directory / "dataset.csv", fields, _dataset_rows(state))
    history = []
    best_cost = None
    for a in state["attempts"]:
        if a.get("feasible"):
            best_cost = (
                a["cost_per_m"]
                if best_cost is None
                else min(best_cost, a["cost_per_m"])
            )
        history.append(
            {
                "attempt": a["index"],
                "design_id": a["design"]["design_id"],
                "phase": a["phase"],
                "initial_slot": a["slot"],
                "status": a["status"],
                "charged": a["charged"],
                "cost_per_m": a["cost_per_m"],
                "feasible": a.get("feasible", ""),
                "best_feasible_cost_per_m": best_cost,
                "error": a.get("error", ""),
                **{name: a.get("metrics", {}).get(name, "") for name in DISPLACEMENT_METRIC_FIELDS},
            }
        )
    history_fields = (
        "attempt",
        "design_id",
        "phase",
        "initial_slot",
        "status",
        "charged",
        "cost_per_m",
        "feasible",
        "best_feasible_cost_per_m",
        "error",
        *DISPLACEMENT_METRIC_FIELDS,
    )
    _write_csv_atomic(directory / "optimization_history.csv", history_fields, history)
    initial = sorted(
        (a for a in _successful(state) if a["slot"] is not None),
        key=lambda a: a["slot"],
    )
    _write_csv_atomic(
        directory / "initial_designs_final.csv",
        DesignVariables.CSV_FIELDS,
        [a["design"] for a in initial],
    )


def _model_snapshot(config, directory, state, mode, progress):
    rows = _successful(state)
    x = np.asarray([a["point"] for a in rows])
    y = np.asarray([[a["metrics"][name] for name in DISPLACEMENT_METRIC_FIELDS] for a in rows])
    ids = [a["design"]["design_id"] for a in rows]
    kg_config = KGCalibrationConfig(
        random_seed=config.random_seed, gp_restarts=config.gp_restarts
    )
    validation = fit_and_validate(
        x, y, config=kg_config, r2_threshold=config.r2_threshold, progress=progress,
        response_fields=DISPLACEMENT_METRIC_FIELDS,
    )
    snapshot = directory / "iterations" / f"samples_{len(rows):04d}"
    _json(
        snapshot / "validation.json",
        {"method": "leave_one_out_refit", "responses": validation.reports},
    )
    loo_rows = [
        {
            "design_id": ids[i],
            "response": name,
            "observed": float(y[i, j]),
            "loo_prediction": float(validation.predictions[i, j])
            if np.isfinite(validation.predictions[i, j])
            else None,
        }
        for i in range(len(rows))
        for j, name in enumerate(DISPLACEMENT_METRIC_FIELDS)
    ]
    _write_csv_atomic(
        snapshot / "loo_predictions.csv",
        ("design_id", "response", "observed", "loo_prediction"),
        loo_rows,
    )
    weights = np.zeros((8, len(DISPLACEMENT_METRIC_FIELDS)))
    if mode == "knowledge":
        calibrated = calibrate_kg_edges(
            directory / "dataset.csv",
            snapshot / "calibration",
            config=kg_config,
            response_models=validation.models,
            response_enabled=validation.enabled,
            response_fields=DISPLACEMENT_METRIC_FIELDS,
            force=True,
        )
        weights = np.asarray(calibrated.effective_weights)
    else:
        _json(
            snapshot / "calibration" / "kg_weights.json",
            {
                "mode": "plain",
                "prior_sign": [list(row[:len(DISPLACEMENT_METRIC_FIELDS)]) for row in PRIOR_SIGN],
                "prior_tier": [list(row[:len(DISPLACEMENT_METRIC_FIELDS)]) for row in PRIOR_TIER],
                "effective_weight": weights.tolist(),
                "active_mask": (weights > 0).tolist(),
                "response_enabled": [False] * len(DISPLACEMENT_METRIC_FIELDS),
            },
        )
    _json(
        snapshot / "virtual_derivatives.json",
        {"observations": [], "inference": [], "status": "awaiting_candidate_selection"},
    )
    return validation, weights, snapshot


def acquisition_scores(means, stds, costs, constraints, best_cost):
    """Log constrained improvement for an exactly evaluated geometric objective."""
    log_probability = np.zeros(len(costs))
    for j, name in enumerate(constraints):
        residual = constraints[name].residual(means[:, j])
        positive = stds[:, j] > 1e-14
        component = np.full(len(costs), -np.inf)
        component[positive] = log_ndtr(-residual[positive] / stds[positive, j])
        component[~positive & constraints[name].satisfied(means[:, j])] = 0.0
        log_probability += component
    if best_cost is None:
        return log_probability
    improvement = np.maximum(best_cost - costs, 0)
    scores = np.full(len(costs), -np.inf)
    positive = improvement > 0
    scores[positive] = np.log(improvement[positive]) + log_probability[positive]
    return scores


def _choose_candidate(
    config, directory, state, domain, validation, weights, snapshot, progress
):
    excluded = {point_key(a["point"]) for a in state["attempts"]}
    candidates = sobol_candidates(
        config.candidate_count,
        domain.feasible,
        seed=config.random_seed + 10000 + len(state["attempts"]),
        excluded=excluded,
        max_attempts=config.candidate_attempts,
        progress=progress,
    )
    costs = np.asarray([domain.cost(point) for point in candidates])
    response_count = len(DISPLACEMENT_METRIC_FIELDS)
    data_means, data_stds = np.column_stack(
        [predict_response(model, candidates)[0] for model in validation.models]
    ), np.column_stack(
        [predict_response(model, candidates)[1] for model in validation.models]
    )
    feasible_costs = [a["cost_per_m"] for a in _successful(state) if a.get("feasible")]
    best = min(feasible_costs) if feasible_costs else None
    data_scores = acquisition_scores(data_means, data_stds, costs, config.constraints, best)
    relevant = np.zeros(len(candidates), dtype=bool)
    finite = np.flatnonzero(np.isfinite(data_scores))
    if len(finite):
        count = max(1, int(np.ceil(0.25 * len(finite))))
        relevant[finite[np.argsort(-data_scores[finite], kind="stable")[:count]]] = True
    observed = np.asarray([a["point"] for a in _successful(state)])
    observations, audit = select_virtual_derivatives(
        observed, candidates, validation.models, weights, validation.enabled,
        config.constraints, acquisition_relevant=relevant,
        softness=config.derivative_softness,
        maximum=config.max_virtual_derivatives,
        conflict_probability=config.local_conflict_probability,
        response_fields=DISPLACEMENT_METRIC_FIELDS,
    )
    predictors, inference = [], []
    for j, response in enumerate(validation.models):
        try:
            predictor = SignConstrainedGP(
                response, observations[j], softness=config.derivative_softness
            )
            predictor.predict(candidates[:1])
            inference.append({
                "response": DISPLACEMENT_METRIC_FIELDS[j],
                "virtual_count": len(observations[j]),
                "status": "guided" if predictor.active else "data_only",
            })
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            predictor = SignConstrainedGP(response, [])
            inference.append({
                "response": DISPLACEMENT_METRIC_FIELDS[j],
                "virtual_count": len(observations[j]),
                "status": "fallback_data_only", "error": str(error),
            })
        predictors.append(predictor)
    audit_payload = {"observations": audit, "inference": inference,
                     "candidate_count": len(candidates)}
    _json(snapshot / f"virtual_derivatives_{len(state['attempts']) + 1:04d}.json", audit_payload)
    _json(snapshot / "virtual_derivatives.json", audit_payload)
    means, stds = np.empty((len(candidates), response_count)), np.empty((len(candidates), response_count))
    failures = []
    for j, predictor in enumerate(predictors):
        try:
            means[:, j], stds[:, j] = predictor.predict(candidates)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            means[:, j], stds[:, j] = predict_response(validation.models[j], candidates)
            failures.append({"response": DISPLACEMENT_METRIC_FIELDS[j], "error": str(error)})
    scales = np.asarray([max(m.standard_deviation, 1e-12) for m in validation.models])
    eligible = (
        ~np.asarray(validation.enabled)
        if not any(validation.enabled)
        else np.ones(response_count, dtype=bool)
    )
    # Exploration always uses data-only uncertainty, never artificially tightened knowledge variance.
    uncertainty = np.max((data_stds / scales)[:, eligible], axis=1)
    if np.max(uncertainty) <= 1e-14:
        observed = np.asarray([a["point"] for a in _successful(state)])
        uncertainty = np.min(
            np.linalg.norm(candidates[:, None] - observed[None, :], axis=2), axis=1
        )
    if not any(validation.enabled):
        phase, scores = "accuracy_sampling", uncertainty
    elif (_spent(state, adaptive=True) + 1) % config.exploration_interval == 0:
        phase, scores = "uncertainty_exploration", uncertainty
    else:
        scores = acquisition_scores(means, stds, costs, config.constraints, best)
        phase = "feasibility_search" if best is None else "constrained_improvement"
        if not np.isfinite(scores).any():
            phase, scores = "uncertainty_exploration", uncertainty
    chosen = int(np.argmax(scores))
    selection = {
        "phase": phase,
        "point": candidates[chosen].tolist(),
        "cost_per_m": float(costs[chosen]),
        "score": float(scores[chosen]),
        "predicted_means": means[chosen].tolist(),
        "predicted_stds": stds[chosen].tolist(),
        "prediction_fallbacks": failures,
    }
    _json(snapshot / f"selection_{len(state['attempts']) + 1:04d}.json", selection)
    return candidates[chosen], phase, selection


def run_optimization(
    config: ExperimentConfig,
    output_directory: Path,
    *,
    mode: str = "knowledge",
    resume: bool = False,
    simulate: bool = False,
    evaluator: Evaluator | None = None,
    domain=None,
    progress: Callable[[str], None] | None = None,
) -> OptimizationOutputs:
    config.validate()
    if simulate and config.initial_dataset is not None:
        raise ValueError("cannot mix imported real responses with simulated evaluations")
    if config.simulation_only and not simulate and evaluator is None:
        raise ValueError("this configuration is simulation_only; supply --simulate")
    if mode not in {"knowledge", "plain"}:
        raise ValueError("mode must be knowledge or plain")
    domain = domain or GeometryDomain(config)
    evaluator = evaluator or (
        SimulatedEvaluator() if simulate else GeoStudioEvaluator(config, domain)
    )
    directory = output_directory.resolve()
    if directory.exists() and any(directory.iterdir()) and not resume:
        raise ValueError(
            "optimization output already exists; use --resume or a new directory"
        )
    if resume and not (directory / "checkpoint.json").is_file():
        raise ValueError("--resume requires an existing checkpoint.json")
    directory.mkdir(parents=True, exist_ok=True)
    with _experiment_lock(directory):
        if not resume and (directory / "checkpoint.json").exists():
            raise ValueError("another run initialized this directory; use --resume")
        return _run_locked(config, directory, mode, resume, evaluator, domain, progress)


def _run_locked(config, directory, mode, resume, evaluator, domain, progress):
    imported = _load_initial_dataset(config, domain) if config.initial_dataset else None
    identity = {
        "schema_version": OPTIMIZATION_SCHEMA_VERSION,
        "config": config.as_dict(),
        "mode": mode,
        "template_fingerprint": domain.fingerprint,
        "evaluator": evaluator.fingerprint,
        "bounds": DesignVariables.BOUNDS,
        "prior_sign": [list(row[:len(DISPLACEMENT_METRIC_FIELDS)]) for row in PRIOR_SIGN],
        "prior_tier": [list(row[:len(DISPLACEMENT_METRIC_FIELDS)]) for row in PRIOR_TIER],
        "initial_dataset_sha256": (
            hashlib.sha256(config.initial_dataset.read_bytes()).hexdigest()
            if config.initial_dataset else None
        ),
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    checkpoint = directory / "checkpoint.json"
    if resume:
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("fingerprint") != fingerprint:
            raise ValueError(
                "resume rejected: configuration, mode, template, bounds, expert prior or evaluator changed"
            )
        for attempt in _successful(state):
            design = DesignVariables(**attempt["design"])
            cached = read_evaluation(design, directory / "runs" / design.design_id)
            if (
                cached is None
                or not cached.succeeded
                or cached.metrics != attempt["metrics"]
            ):
                raise ValueError(
                    f"resume rejected: result evidence changed for {design.design_id}"
                )
    else:
        state = {
            "fingerprint": fingerprint,
            "identity": identity,
            "initial_points": None,
            "attempts": [],
            "status": "initializing",
        }
        _json(directory / "resolved_config.json", identity)
        _json(checkpoint, state)

    if imported is not None and state["initial_points"] is None:
        state["initial_points"] = [point.tolist() for _, point, _ in imported]
        for index, (design, point, metrics) in enumerate(imported, 1):
            result = EvaluationResult(True, metrics, artifacts=[str(config.initial_dataset.resolve())])
            write_evaluation(design, directory / "runs" / design.design_id, result)
            state["attempts"].append({
                "index": index, "slot": index - 1, "phase": "imported_initial",
                "point": point.tolist(), "design": design.as_dict(),
                "cost_per_m": domain.cost(point), "status": "succeeded",
                "charged": True, "metrics": metrics, "error": "",
                "feasible": all(config.constraints[name].satisfied(metrics[name])
                                for name in DISPLACEMENT_METRIC_FIELDS),
            })
        _json(checkpoint, state)

    if state["initial_points"] is None:
        points = feasible_lhs(
            config.initial_samples,
            domain.feasible,
            seed=config.random_seed,
            restarts=config.lhs_restarts,
            repair_attempts=config.lhs_repair_attempts,
            progress=progress,
        )
        state["initial_points"] = points.tolist()
        _json(checkpoint, state)
    _write_csv_atomic(
        directory / "initial_designs.csv",
        DesignVariables.CSV_FIELDS,
        [
            (
                next(a["design"] for a in state["attempts"] if a["slot"] == i)
                if imported is not None
                else design_from_unit(np.asarray(p), f"lhs_{i + 1:03d}").as_dict()
            )
            for i, p in enumerate(state["initial_points"])
        ],
    )

    def commit(attempt, result):
        result.validate()
        design = DesignVariables(**attempt["design"])
        write_evaluation(design, directory / "runs" / design.design_id, result)
        attempt.update(
            status="succeeded" if result.succeeded else "failed",
            metrics=result.metrics,
            error=result.error,
        )
        if result.succeeded:
            if not attempt["charged"]:
                raise ValueError(
                    "evaluator returned success without committing its evaluation budget"
                )
            attempt["feasible"] = all(
                config.constraints[n].satisfied(result.metrics[n])
                for n in DISPLACEMENT_METRIC_FIELDS
            )
        _json(checkpoint, state)
        _export_data(directory, state)

    for attempt in state["attempts"]:
        if attempt["status"] == "running":
            design = DesignVariables(**attempt["design"])
            recovered = evaluator.recover(design, directory / "runs" / design.design_id)
            if recovered is not None and recovered.succeeded and attempt["charged"]:
                commit(attempt, recovered)
            else:
                commit(
                    attempt,
                    EvaluationResult(
                        False,
                        error="interrupted evaluation; budget remains charged if meshing had started",
                    ),
                )

    _export_data(directory, state)
    validation = weights = snapshot = None
    fitted_count = -1
    stop_reason = "budget_exhausted"
    while True:
        successes = _successful(state)
        completed_slots = {a["slot"] for a in successes if a["slot"] is not None}
        initial_complete = len(completed_slots) == config.initial_samples
        if initial_complete and fitted_count != len(successes):
            if progress:
                progress(
                    f"fitting {len(successes)} observed samples; evaluations spent {_spent(state)}/{config.total_budget}"
                )
            validation, weights, snapshot = _model_snapshot(
                config, directory, state, mode, progress
            )
            fitted_count = len(successes)
        if _spent(state) >= config.total_budget:
            stop_reason = (
                "budget_exhausted"
                if initial_complete
                else "initialization_budget_exhausted"
            )
            break
        if initial_complete and _spent(state, adaptive=True) >= config.adaptive_budget:
            stop_reason = "adaptive_budget_exhausted"
            break
        if (
            sum(not a["charged"] for a in state["attempts"])
            >= config.replacement_attempts
        ):
            stop_reason = "preflight_failure_limit"
            break
        excluded = {point_key(a["point"]) for a in state["attempts"]}
        slot, selection = None, None
        try:
            if not initial_complete:
                slot = next(
                    i for i in range(config.initial_samples) if i not in completed_slots
                )
                original = np.asarray(state["initial_points"][slot])
                if point_key(original) in excluded:
                    point = replace_in_strata(
                        original,
                        config.initial_samples,
                        domain.feasible,
                        seed=config.random_seed + 20000 + len(state["attempts"]),
                        excluded=excluded,
                        attempts=config.replacement_attempts,
                    )
                else:
                    point = original
                phase = "initial"
            else:
                point, phase, selection = _choose_candidate(
                    config,
                    directory,
                    state,
                    domain,
                    validation,
                    weights,
                    snapshot,
                    progress,
                )
        except ValueError as error:
            stop_reason = "sampling_exhausted"
            state["stop_detail"] = str(error)
            break
        index = len(state["attempts"]) + 1
        design = design_from_unit(point, f"eval_{index:04d}")
        attempt = {
            "index": index,
            "slot": slot,
            "phase": phase,
            "point": point.tolist(),
            "design": design.as_dict(),
            "cost_per_m": domain.cost(point),
            "status": "running",
            "charged": False,
            "selection": selection,
        }
        state["attempts"].append(attempt)
        state["status"] = "running"
        _json(checkpoint, state)
        if progress:
            progress(
                f"{design.design_id}: {phase}; budget {_spent(state)}/{config.total_budget}"
            )

        def charge(current=attempt):
            if current["charged"]:
                raise ValueError("an evaluation may only charge its budget once")
            if _spent(state) >= config.total_budget:
                raise ValueError("evaluation budget exhausted")
            current["charged"] = True
            _json(checkpoint, state)

        try:
            domain.prepare(design)
            result = evaluator.evaluate(
                design, directory / "runs" / design.design_id, charge
            )
            result.validate()
            if result.succeeded and not attempt["charged"]:
                raise ValueError(
                    "evaluator returned success without charging its budget"
                )
        except Exception as error:  # isolate one failed real evaluation; interrupts preserve the pending state
            result = EvaluationResult(False, error=f"{type(error).__name__}: {error}")
        commit(attempt, result)
        if progress:
            progress(
                f"{design.design_id}: {'succeeded' if result.succeeded else result.error}"
            )

    state.update(status="finished", stop_reason=stop_reason)
    _json(checkpoint, state)
    _export_data(directory, state)
    return _finish(config, directory, state, validation, stop_reason)


def _finish(config, directory, state, validation, stop_reason):
    successes = _successful(state)
    feasible = [a for a in successes if a.get("feasible")]
    best = (
        min(feasible, key=lambda a: (a["cost_per_m"], a["index"])) if feasible else None
    )
    closest = None
    if successes and best is None:
        y = np.asarray([[a["metrics"][n] for n in DISPLACEMENT_METRIC_FIELDS] for a in successes])
        scales = np.maximum(
            np.std(y, axis=0),
            1e-8
            * np.maximum(
                np.abs([config.constraints[n].limit for n in DISPLACEMENT_METRIC_FIELDS]), 1
            ),
        )
        violations = np.column_stack(
            [
                np.maximum(config.constraints[n].residual(y[:, j]), 0) / scales[j]
                for j, n in enumerate(DISPLACEMENT_METRIC_FIELDS)
            ]
        )
        closest = successes[int(np.argmin(np.sum(violations, axis=1)))]
    summary = {
        "schema_version": OPTIMIZATION_SCHEMA_VERSION,
        "stop_reason": stop_reason,
        "stop_detail": state.get("stop_detail"),
        "evaluator": state["identity"]["evaluator"],
        "mode": state["identity"]["mode"],
        "attempts_spent": _spent(state),
        "adaptive_attempts_spent": _spent(state, adaptive=True),
        "successful_samples": len(successes),
        "feasible_samples": len(feasible),
        "initial_complete": len([a for a in successes if a["slot"] is not None])
        == config.initial_samples,
        "best_design": best,
        "least_violating_design": closest,
        "validation": validation.reports if validation else [],
        "cost_units": "CNY per metre of dam length",
        "response_origin": "synthetic"
        if state["identity"]["evaluator"].startswith("synthetic")
        else "evaluator",
    }
    _json(directory / "optimization_report.json", summary)
    _json(
        directory / "best_design.json",
        {
            "feasible": best is not None,
            "design": best,
            "least_violating_design": closest,
        },
    )
    _plot_convergence(directory, state)
    return OptimizationOutputs(
        directory,
        directory / "optimization_report.json",
        directory / "dataset.csv",
        best,
        _spent(state),
        len(successes),
        stop_reason,
    )


def _plot_convergence(directory, state):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 7), constrained_layout=True)
    FigureCanvasAgg(figure)
    cost_ax, r2_ax = figure.subplots(2, 1)
    spent, best, xs, ys = 0, np.inf, [], []
    for attempt in state["attempts"]:
        spent += int(attempt["charged"])
        if attempt.get("feasible"):
            best = min(best, attempt["cost_per_m"])
        if attempt["charged"]:
            xs.append(spent)
            ys.append(best if np.isfinite(best) else np.nan)
    cost_ax.plot(xs, ys, marker=".")
    cost_ax.set(xlabel="Charged evaluations", ylabel="Best feasible cost (CNY/m)")
    cost_ax.set_title(
        "Synthetic validation"
        if state["identity"]["evaluator"].startswith("synthetic")
        else "Optimization history"
    )
    if not np.isfinite(ys).any():
        cost_ax.text(
            0.5,
            0.5,
            "No observed feasible design",
            ha="center",
            transform=cost_ax.transAxes,
        )
    snapshots = sorted((directory / "iterations").glob("samples_*/validation.json"))
    for j, name in enumerate(DISPLACEMENT_METRIC_FIELDS):
        values = [
            json.loads(p.read_text(encoding="utf-8"))["responses"][j] for p in snapshots
        ]
        r2_ax.plot(
            [v["samples"] for v in values],
            [v["r2"] if v["r2"] is not None else np.nan for v in values],
            label=name,
        )
    r2_ax.axhline(
        state["identity"]["config"]["r2_threshold"], color="black", linestyle="--"
    )
    r2_ax.set(xlabel="Successful real/synthetic evaluations", ylabel="Leave-one-out R2")
    if snapshots:
        r2_ax.legend(fontsize=6, loc="best")
    figure.savefig(directory / "convergence.png", dpi=160)
