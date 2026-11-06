from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .model import DesignVariables
from .results import (
    DEFAULT_ANALYSIS_NAME,
    DEFAULT_ANALYSIS_STEP,
    DEFAULT_CVAR_ALPHA,
    EXCLUDED_MATERIAL_NAME_KEYWORDS,
    MATERIAL_FILTER_VERSION,
    RESULTS_SCHEMA_VERSION,
    extract_project_responses,
    metric_fields,
    project_provenance,
)


BATCH_EXTRACTION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ExtractionOutcome:
    design_id: str
    project_path: str
    response_path: str
    succeeded: bool
    error_summary: str
    cached: bool = False


@dataclass(frozen=True, slots=True)
class BatchExtractionOutputs:
    output_directory: Path
    dataset_path: Path
    audit_path: Path
    report_path: Path
    outcomes: tuple[ExtractionOutcome, ...]

    @property
    def successful_projects(self) -> int:
        return sum(outcome.succeeded for outcome in self.outcomes)

    @property
    def failed_projects(self) -> int:
        return sum(not outcome.succeeded for outcome in self.outcomes)


def extract_projects(
    project_paths: Sequence[Path],
    output_directory: Path,
    *,
    analysis_name: str = DEFAULT_ANALYSIS_NAME,
    step: int = DEFAULT_ANALYSIS_STEP,
    cvar_alpha: float = DEFAULT_CVAR_ALPHA,
    force: bool = False,
    fail_fast: bool = False,
    progress: Callable[[str], None] | None = None,
) -> BatchExtractionOutputs:
    """Extract five responses and write one training dataset for solved projects."""
    projects = _validated_projects(project_paths)
    if not analysis_name.strip():
        raise ValueError("analysis name must not be empty")
    if step < 0:
        raise ValueError("analysis step must be a non-negative integer")
    if not math.isfinite(cvar_alpha) or cvar_alpha <= 0 or cvar_alpha >= 1:
        raise ValueError("CVaR alpha must be finite and strictly between 0 and 1")
    _reject_duplicate_design_ids(projects)

    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    per_run_directory = output_directory / "per_run"
    per_run_directory.mkdir(parents=True, exist_ok=True)
    fields = metric_fields(cvar_alpha)

    outcomes: list[ExtractionOutcome] = []
    dataset_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for index, project_path in enumerate(projects, start=1):
        fallback_id = project_path.stem
        design_id = fallback_id
        parameter_hash = ""
        response_path = per_run_directory / fallback_id / "response_metrics.json"
        try:
            parameters_path = project_path.parent / "parameters.json"
            parameters_payload = _read_json_object(parameters_path, "parameters")
            parameters, parameter_hash, design_id = _validated_parameters(
                parameters_payload
            )
            response_path = per_run_directory / design_id / "response_metrics.json"
            solve_report_path = _validated_solve_report(project_path)

            response_payload = None if force else _read_optional_json(response_path)
            cached = response_payload is not None and _response_is_current(
                response_payload,
                project_path,
                parameter_hash=parameter_hash,
                analysis_name=analysis_name,
                step=step,
                cvar_alpha=cvar_alpha,
                required_metrics=fields,
            )
            if not cached:
                response = extract_project_responses(
                    project_path,
                    parameters_payload,
                    analysis_name=analysis_name,
                    step=step,
                    cvar_alpha=cvar_alpha,
                )
                response_payload = response.as_dict()
                _write_json_atomic(response_path, response_payload)
            assert response_payload is not None

            metrics = _mapping(response_payload.get("metrics"), "response metrics")
            row = {field: parameters[field] for field in DesignVariables.CSV_FIELDS}
            row.update(
                {
                    "parameter_hash": parameter_hash,
                    "analysis_name": analysis_name,
                    "step": step,
                    "cvar_alpha": cvar_alpha,
                    **{
                        field: _finite_float(metrics[field], field)
                        for field in fields
                    },
                }
            )
            dataset_rows.append(row)
            audit_rows.append(
                _successful_audit_row(
                    design_id,
                    project_path,
                    solve_report_path,
                    response_path,
                    response_payload,
                    cached,
                )
            )
            outcome = ExtractionOutcome(
                design_id,
                str(project_path),
                str(response_path),
                True,
                "",
                cached,
            )
            note = "cached" if cached else "extracted"
        except Exception as error:  # isolate one failed design from the batch
            error_summary = _one_line(str(error))
            failure_payload = {
                "schema_version": RESULTS_SCHEMA_VERSION,
                "status": "failed",
                "design_id": design_id,
                "parameter_hash": parameter_hash,
                "analysis_name": analysis_name,
                "step": step,
                "cvar_alpha": cvar_alpha,
                "project": (
                    project_provenance(project_path)
                    if project_path.is_file()
                    else {"path": str(project_path)}
                ),
                "error_summary": error_summary,
                "extracted_at": datetime.now(UTC).isoformat(),
            }
            _write_json_atomic(response_path, failure_payload)
            audit_rows.append(
                {
                    "design_id": design_id,
                    "parameter_hash": parameter_hash,
                    "project_path": str(project_path),
                    "solve_report_path": "",
                    "response_path": str(response_path),
                    "cached": False,
                    "status": "failed",
                    "error_summary": error_summary,
                }
            )
            outcome = ExtractionOutcome(
                design_id,
                str(project_path),
                str(response_path),
                False,
                error_summary,
            )
            note = f"failed: {error_summary}"
        outcomes.append(outcome)
        if progress is not None:
            progress(f"[{index}/{len(projects)}] {design_id}: {note}")
        if fail_fast and not outcome.succeeded:
            break

    dataset_path = output_directory / "dataset.csv"
    audit_path = output_directory / "extraction_audit.csv"
    report_path = output_directory / "extraction_report.json"
    dataset_fields = (
        *DesignVariables.CSV_FIELDS,
        "parameter_hash",
        "analysis_name",
        "step",
        "cvar_alpha",
        *fields,
    )
    audit_fields = (
        "design_id",
        "parameter_hash",
        "project_path",
        "solve_report_path",
        "response_path",
        "cached",
        "status",
        "error_summary",
        "queried_nodes",
        "queried_elements",
        "queried_gauss_points",
        "crest_material_nodes",
        "dam_body_nodes",
        "eligible_stress_elements",
        "eligible_stress_gauss_points",
        "eligible_area_m2",
        "sigma1_var_kpa",
        "sigma1_raw_max_kpa",
        "sigma3_var_kpa",
        "sigma3_raw_max_kpa",
    )
    _write_csv_atomic(dataset_path, dataset_fields, dataset_rows)
    _write_csv_atomic(audit_path, audit_fields, audit_rows)
    report = {
        "schema_version": BATCH_EXTRACTION_SCHEMA_VERSION,
        "study_type": "five_response_training_dataset",
        "dataset_path": str(dataset_path),
        "audit_path": str(audit_path),
        "per_run_directory": str(per_run_directory),
        "analysis_name": analysis_name,
        "step": step,
        "cvar_alpha": cvar_alpha,
        "tail_area_fraction": 1.0 - cvar_alpha,
        "response_fields": list(fields),
        "excluded_material_name_keywords": list(
            EXCLUDED_MATERIAL_NAME_KEYWORDS
        ),
        "requested_projects": len(projects),
        "processed_projects": len(outcomes),
        "successful_projects": sum(outcome.succeeded for outcome in outcomes),
        "failed_projects": sum(not outcome.succeeded for outcome in outcomes),
        "failures": [
            {
                "design_id": outcome.design_id,
                "project_path": outcome.project_path,
                "error_summary": outcome.error_summary,
            }
            for outcome in outcomes
            if not outcome.succeeded
        ],
        "created_at": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(report_path, report)
    return BatchExtractionOutputs(
        output_directory,
        dataset_path,
        audit_path,
        report_path,
        tuple(outcomes),
    )


def _validated_projects(project_paths: Sequence[Path]) -> list[Path]:
    if not project_paths:
        raise ValueError("at least one solved GSZ project is required")
    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in project_paths:
        project = candidate.resolve()
        if project.suffix.casefold() != ".gsz":
            raise ValueError(f"extraction input must be a .gsz file: {project}")
        if not project.is_file():
            raise ValueError(f"GeoStudio project was not found: {project}")
        if project not in seen:
            seen.add(project)
            result.append(project)
    return sorted(result, key=lambda path: str(path).casefold())


def _reject_duplicate_design_ids(projects: Sequence[Path]) -> None:
    discovered: list[str] = []
    for project in projects:
        payload = _read_optional_json(project.parent / "parameters.json")
        if not isinstance(payload, Mapping):
            continue
        parameters = payload.get("parameters")
        if not isinstance(parameters, Mapping):
            continue
        design_id = parameters.get("design_id")
        if isinstance(design_id, str) and design_id:
            discovered.append(design_id)
    duplicates = sorted(
        design_id
        for design_id, count in Counter(discovered).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(
            "multiple projects use the same design_id: " + ", ".join(duplicates)
        )


def _validated_parameters(
    payload: Mapping[str, object],
) -> tuple[Mapping[str, object], str, str]:
    parameters = _mapping(payload.get("parameters"), "parameters")
    for field in DesignVariables.CSV_FIELDS:
        if field not in parameters:
            raise ValueError(f"parameters are missing {field}")
        if field != "design_id":
            _finite_float(parameters[field], f"parameter {field}")
    design = DesignVariables.from_mapping(
        {field: str(parameters[field]) for field in DesignVariables.CSV_FIELDS}
    )
    bound_errors = design.validate_bounds()
    if bound_errors:
        raise ValueError("invalid parameters: " + "; ".join(bound_errors))
    design_id = design.design_id
    parameter_hash = payload.get("parameter_hash")
    if not isinstance(parameter_hash, str) or not parameter_hash:
        raise ValueError("parameters JSON has no parameter_hash")
    if not isinstance(payload.get("derived_coordinates"), Mapping):
        raise ValueError("parameters JSON has no derived_coordinates object")
    return parameters, parameter_hash, design_id


def _validated_solve_report(project_path: Path) -> Path:
    report_path = (
        project_path.parent / "solve_report.json"
        if project_path.parent.name == project_path.stem
        else project_path.parent / f"{project_path.stem}.solve_report.json"
    )
    payload = _read_json_object(report_path, "solve report")
    if payload.get("status") != "solved":
        raise ValueError(f"solve report does not have solved status: {report_path}")
    if report_path.stat().st_mtime_ns < project_path.stat().st_mtime_ns:
        raise ValueError(
            "GeoStudio project is newer than its solve report; re-solve before "
            f"extracting: {project_path}"
        )
    return report_path


def _response_is_current(
    payload: Mapping[str, object],
    project_path: Path,
    *,
    parameter_hash: str,
    analysis_name: str,
    step: int,
    cvar_alpha: float,
    required_metrics: Sequence[str],
) -> bool:
    project = payload.get("project")
    metrics = payload.get("metrics")
    material_filter = payload.get("material_filter")
    if (
        not isinstance(project, Mapping)
        or not isinstance(metrics, Mapping)
        or not isinstance(material_filter, Mapping)
    ):
        return False
    stat = project_path.stat()
    forbidden = {"horizontal_displacement_gradient", "settlement_gradient"}
    try:
        alpha_matches = math.isclose(
            float(payload.get("cvar_alpha", -1.0)), cvar_alpha, abs_tol=1.0e-12
        )
    except (TypeError, ValueError):
        return False
    return (
        payload.get("schema_version") == RESULTS_SCHEMA_VERSION
        and payload.get("status") == "extracted"
        and payload.get("parameter_hash") == parameter_hash
        and payload.get("analysis_name") == analysis_name
        and payload.get("step") == step
        and alpha_matches
        and project.get("size_bytes") == stat.st_size
        and project.get("modified_time_ns") == stat.st_mtime_ns
        and material_filter.get("version") == MATERIAL_FILTER_VERSION
        and material_filter.get("excluded_material_name_keywords")
        == list(EXCLUDED_MATERIAL_NAME_KEYWORDS)
        and all(field in metrics for field in required_metrics)
        and forbidden.isdisjoint(metrics)
    )


def _successful_audit_row(
    design_id: str,
    project_path: Path,
    solve_report_path: Path,
    response_path: Path,
    payload: Mapping[str, object],
    cached: bool,
) -> dict[str, object]:
    counts = _mapping(payload.get("counts"), "response counts")
    scope = _mapping(payload.get("scope"), "response scope")
    stress_scope = _mapping(scope.get("dam_body_stresses"), "stress scope")
    tails = _mapping(payload.get("stress_tail_details"), "stress tail details")
    sigma1 = _mapping(tails.get("sigma1_max_effective_stress"), "sigma1 tail")
    sigma3 = _mapping(tails.get("sigma3_min_effective_stress"), "sigma3 tail")
    return {
        "design_id": design_id,
        "parameter_hash": payload.get("parameter_hash", ""),
        "project_path": str(project_path),
        "solve_report_path": str(solve_report_path),
        "response_path": str(response_path),
        "cached": cached,
        "status": "extracted",
        "error_summary": "",
        "queried_nodes": counts["queried_nodes"],
        "queried_elements": counts["queried_elements"],
        "queried_gauss_points": counts["queried_gauss_points"],
        "crest_material_nodes": counts["crest_material_nodes"],
        "dam_body_nodes": counts["dam_body_nodes"],
        "eligible_stress_elements": counts["eligible_stress_elements"],
        "eligible_stress_gauss_points": counts[
            "eligible_stress_gauss_points"
        ],
        "eligible_area_m2": stress_scope["eligible_area_m2"],
        "sigma1_var_kpa": sigma1["var_kpa"],
        "sigma1_raw_max_kpa": sigma1["maximum_kpa"],
        "sigma3_var_kpa": sigma3["var_kpa"],
        "sigma3_raw_max_kpa": sigma3["maximum_kpa"],
    }


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is missing or invalid")
    return value


def _finite_float(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    value = _read_optional_json(path)
    if value is None:
        raise ValueError(f"{label} JSON is missing or invalid: {path}")
    return value


def _read_optional_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


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


def _one_line(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ")[:500]


__all__ = [
    "BATCH_EXTRACTION_SCHEMA_VERSION",
    "BatchExtractionOutputs",
    "ExtractionOutcome",
    "extract_projects",
]
