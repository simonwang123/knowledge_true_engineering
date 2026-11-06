from __future__ import annotations

import csv
import json
import platform
import shutil
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import shapely

from .geometry import GeometryBuild, cell_report
from .mesh import MeshValidation, validate_gsz_mesh
from .model import DesignVariables
from .plotting import write_model_picture
from .runtime import (
    AnalysisSolve,
    GeoStudioRuntimeError,
    RuntimeInfo,
    analysis_names,
    discover_runtime,
    force_native_remesh_and_validate,
    solve_analyses,
)
from .template import TemplateArchive
from .xml_model import GeoStudioXmlModel


@dataclass(frozen=True, slots=True)
class BuildResult:
    design_id: str
    project_path: str
    succeeded: bool
    error_summary: str
    report_path: str
    preview_path: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MeshResult:
    project_path: str
    succeeded: bool
    error_summary: str
    report_path: str
    validation: MeshValidation | None = None

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["validation"] = self.validation.as_dict() if self.validation else None
        return value


@dataclass(frozen=True, slots=True)
class SolveResult:
    project_path: str
    succeeded: bool
    error_summary: str
    report_path: str
    analyses: tuple[AnalysisSolve, ...] = ()

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["analyses"] = [item.as_dict() for item in self.analyses]
        return value


@dataclass(frozen=True, slots=True)
class PipelineResult:
    design_id: str
    project_path: str
    build_status: str
    mesh_status: str
    solve_status: str
    error_summary: str

    @property
    def succeeded(self) -> bool:
        statuses = (self.build_status, self.mesh_status, self.solve_status)
        return "failed" not in statuses and "mesh_mismatch" not in statuses

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def read_designs(path: Path) -> list[DesignVariables]:
    path = path.resolve()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("design CSV has no header")
        missing = [
            field
            for field in DesignVariables.CSV_FIELDS
            if field not in reader.fieldnames
        ]
        if missing:
            if (
                "secondary_right_slope_distance" in missing
                and "secondary_bottom_clearance" in reader.fieldnames
            ):
                raise ValueError(
                    "design CSV uses the obsolete secondary_bottom_clearance "
                    "geometry; remove secondary_left_bottom_angle and replace "
                    "secondary_bottom_clearance with "
                    "secondary_right_slope_distance"
                )
            raise ValueError(f"design CSV is missing columns: {', '.join(missing)}")

        designs: list[DesignVariables] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                design = DesignVariables.from_mapping(row)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid design CSV row {line_number}: {error}"
                ) from error
            errors = design.validate_bounds()
            if errors:
                raise ValueError(
                    f"invalid design CSV row {line_number}: {'; '.join(errors)}"
                )
            designs.append(design)

    if not designs:
        raise ValueError("design CSV contains no designs")
    counts = Counter(design.design_id for design in designs)
    duplicates = sorted(design_id for design_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate design_id values: {', '.join(duplicates)}")
    return designs


def select_designs(
    designs: Sequence[DesignVariables], design_ids: Sequence[str] | None
) -> list[DesignVariables]:
    if not design_ids:
        return list(designs)
    duplicates = sorted(
        design_id for design_id, count in Counter(design_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(
            f"duplicate --design-id values: {', '.join(duplicates)}"
        )
    requested = set(design_ids)
    available = {design.design_id for design in designs}
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(f"unknown design_id values: {', '.join(unknown)}")
    return [design for design in designs if design.design_id in requested]


def resolve_projects(
    projects: Sequence[Path] | None = None,
    *,
    project_dir: Path | None = None,
    glob_pattern: str = "*.gsz",
) -> list[Path]:
    if projects and project_dir is not None:
        raise ValueError("use either --project or --project-dir, not both")
    if not projects and project_dir is None:
        raise ValueError("at least one --project or --project-dir is required")
    if not glob_pattern:
        raise ValueError("--glob must not be empty")
    if Path(glob_pattern).is_absolute():
        raise ValueError("--glob must be relative to --project-dir")

    candidates: list[Path]
    if project_dir is not None:
        directory = project_dir.resolve()
        if not directory.is_dir():
            raise ValueError(f"project directory was not found: {directory}")
        candidates = sorted(
            path for path in directory.glob(glob_pattern) if path.is_file()
        )
    else:
        candidates = [path.resolve() for path in projects or ()]

    resolved: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        value = path.resolve()
        if value.suffix.lower() != ".gsz":
            raise ValueError(f"GeoStudio project must use the .gsz suffix: {value}")
        if not value.is_file():
            raise ValueError(f"GeoStudio project was not found: {value}")
        if value not in seen:
            seen.add(value)
            resolved.append(value)
    if not resolved:
        source = project_dir.resolve() if project_dir else "explicit inputs"
        raise ValueError(f"no GSZ projects matched: {source}")
    return resolved


def plan_project_destinations(
    sources: Sequence[Path],
    *,
    output_project: Path | None = None,
    output_dir: Path | None = None,
    overwrite_output: bool = False,
) -> list[Path]:
    if output_project is not None and output_dir is not None:
        raise ValueError("use either --output-project or --output-dir, not both")
    if output_project is not None:
        if len(sources) != 1:
            raise ValueError("--output-project requires exactly one input project")
        destinations = [output_project.resolve()]
    elif output_dir is not None:
        root = output_dir.resolve()
        destinations = [root / source.name for source in sources]
    else:
        destinations = [source.resolve() for source in sources]

    invalid = [path for path in destinations if path.suffix.lower() != ".gsz"]
    if invalid:
        raise ValueError(f"output project must use the .gsz suffix: {invalid[0]}")
    duplicate_names = sorted(
        str(path)
        for path, count in Counter(path for path in destinations).items()
        if count > 1
    )
    if duplicate_names:
        raise ValueError(
            "multiple input projects map to the same destination: "
            + ", ".join(duplicate_names)
        )
    existing = [
        destination
        for source, destination in zip(sources, destinations, strict=True)
        if source.resolve() != destination.resolve()
        and destination.exists()
        and not overwrite_output
    ]
    if existing:
        raise ValueError(
            "output project already exists; pass --overwrite-output to replace it: "
            f"{existing[0]}"
        )
    return destinations


def build_design(
    template: Path | TemplateArchive,
    design: DesignVariables,
    output_project: Path,
    *,
    force: bool = False,
    preview_path: Path | None = None,
    model: GeoStudioXmlModel | None = None,
) -> BuildResult:
    start = time.perf_counter()
    archive = template if isinstance(template, TemplateArchive) else TemplateArchive.load(template)
    xml_model = model or GeoStudioXmlModel(
        archive.root(), reference_design=archive.reference_design
    )
    output_project = output_project.resolve()
    preview = (
        preview_path.resolve()
        if preview_path is not None
        else output_project.parent / f"{output_project.stem}.png"
    )
    report_path = _sidecar_path(output_project, "build", "json")
    log_path = _sidecar_path(output_project, "build", "log")
    build: GeometryBuild | None = None
    error_summary = ""
    succeeded = False
    payload: dict[str, object] = {
        "design_id": design.design_id,
        "status": "failed",
        "project_path": str(output_project),
        "template": str(archive.source),
        "parameters": design.as_dict(),
        "parameter_hash": design.parameter_hash(),
        "started_at": _timestamp(),
    }
    _append_log(log_path, f"starting build for {design.design_id}")
    try:
        if output_project.exists() and not force:
            raise FileExistsError(
                f"project already exists; pass --force to replace it: {output_project}"
            )
        result = xml_model.rebuild(design)
        build = result.geometry
        output_project.parent.mkdir(parents=True, exist_ok=True)
        archive.write_gsz(output_project, result.xml)
        geometry_report = cell_report(result.geometry.cells)
        geometry_report.update(
            {
                "point_count": result.point_count,
                "line_count": result.line_count,
                "region_count": result.region_count,
                "boundary_condition_reference_count": result.boundary_reference_count,
                "derived_coordinates": result.geometry.derived.as_dict(),
                "xml_reference_integrity": True,
                "shared_edge_consistency": True,
                "mesh_state": "not_generated",
            }
        )
        payload.update(
            {
                "status": "built",
                "geometry": geometry_report,
                "template_app_version": archive.root().attrib.get("AppVersion"),
            }
        )
        _write_json(
            output_project.parent / "parameters.json",
            {
                "parameters": design.as_dict(),
                "parameter_hash": design.parameter_hash(),
                "units": {
                    "length": "m",
                    "upstream_slope": "H/V",
                    "downstream_slope": "H/V",
                    "secondary_right_slope_distance": "m (perpendicular)",
                    "derived.secondary_left_bottom_angle": "degree",
                },
                "derived_coordinates": result.geometry.derived.as_dict(),
                "software": _software_info(),
            },
        )
        succeeded = True
        _append_log(
            log_path,
            f"built {result.point_count} points, {result.line_count} lines, "
            f"and {result.region_count} regions",
        )
    except Exception as error:
        error_summary = str(error)
        payload["error_summary"] = error_summary
        _append_log(log_path, f"build failed: {error_summary}")
    finally:
        if build is not None or not output_project.exists():
            try:
                write_model_picture(
                    preview,
                    xml_model.original_regions,
                    design,
                    build=build,
                    status="built" if succeeded else "build_failed",
                    error=error_summary,
                    reference_design=xml_model.reference_design,
                )
                payload["preview_path"] = str(preview)
            except Exception as preview_error:
                payload["preview_error"] = str(preview_error)
                _append_log(log_path, f"preview failed: {preview_error}")
        payload["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        payload["finished_at"] = _timestamp()
        _write_json(report_path, payload)

    return BuildResult(
        design_id=design.design_id,
        project_path=str(output_project),
        succeeded=succeeded,
        error_summary=error_summary,
        report_path=str(report_path),
        preview_path=str(preview) if preview.is_file() else "",
    )


def build_designs(
    template_path: Path,
    designs_path: Path,
    output_root: Path,
    *,
    design_ids: Sequence[str] | None = None,
    force: bool = False,
    fail_fast: bool = False,
) -> list[BuildResult]:
    archive = TemplateArchive.load(template_path)
    model = GeoStudioXmlModel(
        archive.root(), reference_design=archive.reference_design
    )
    designs = select_designs(read_designs(designs_path), design_ids)
    output_root = output_root.resolve()
    results: list[BuildResult] = []
    for design in designs:
        run_dir = output_root / design.design_id
        result = build_design(
            archive,
            design,
            run_dir / f"{design.design_id}.gsz",
            force=force,
            preview_path=output_root / "model_picture" / f"{design.design_id}.png",
            model=model,
        )
        results.append(result)
        if fail_fast and not result.succeeded:
            break
    return results


def remesh_project(
    project_path: Path,
    *,
    output_project: Path | None = None,
    overwrite_output: bool = False,
    runtime: RuntimeInfo | None = None,
    mesh_tolerance: float = 1.0e-5,
) -> MeshResult:
    if mesh_tolerance <= 0:
        raise ValueError("mesh tolerance must be positive")
    source = project_path.resolve()
    destination = (output_project or source).resolve()
    report_path = _sidecar_path(destination, "mesh", "json")
    log_path = _sidecar_path(destination, "mesh", "log")
    validation: MeshValidation | None = None
    error_summary = ""
    succeeded = False
    info = runtime or discover_runtime()
    payload: dict[str, object] = {
        "status": "failed",
        "source_project": str(source),
        "project_path": str(destination),
        "runtime": info.as_dict(),
        "started_at": _timestamp(),
    }
    _append_log(log_path, f"starting native remesh for {source}")
    try:
        _prepare_project_copy(source, destination, overwrite_output)
        validation = force_native_remesh_and_validate(
            destination,
            info,
            tolerance=mesh_tolerance,
        )
        succeeded = validation.matched
        payload["status"] = "meshed" if succeeded else "mesh_mismatch"
        payload["validation"] = validation.as_dict()
        if not succeeded:
            error_summary = (
                f"{len(validation.mismatches)} XML/PLY boundary line mismatches"
            )
        _append_log(
            log_path,
            f"mesh validation {'passed' if succeeded else 'failed'}: "
            f"{validation.xml_line_count} XML lines, {validation.node_count} nodes",
        )
    except Exception as error:
        error_summary = str(error)
        _append_log(log_path, f"remesh failed: {error_summary}")
    payload["error_summary"] = error_summary
    payload["finished_at"] = _timestamp()
    _write_json(report_path, payload)
    return MeshResult(
        project_path=str(destination),
        succeeded=succeeded,
        error_summary=error_summary,
        report_path=str(report_path),
        validation=validation,
    )


def solve_project(
    project_path: Path,
    *,
    output_project: Path | None = None,
    overwrite_output: bool = False,
    runtime: RuntimeInfo | None = None,
    mesh_tolerance: float = 1.0e-5,
) -> SolveResult:
    if mesh_tolerance <= 0:
        raise ValueError("mesh tolerance must be positive")
    source = project_path.resolve()
    destination = (output_project or source).resolve()
    report_path = _sidecar_path(destination, "solve", "json")
    log_path = _sidecar_path(destination, "solve", "log")
    info = runtime or discover_runtime()
    analyses: tuple[AnalysisSolve, ...] = ()
    progress: list[AnalysisSolve] = []
    error_summary = ""
    succeeded = False
    payload: dict[str, object] = {
        "status": "failed",
        "source_project": str(source),
        "project_path": str(destination),
        "runtime": info.as_dict(),
        "started_at": _timestamp(),
    }
    _append_log(log_path, f"starting solve for {source}")
    try:
        _prepare_project_copy(source, destination, overwrite_output)
        validation = validate_gsz_mesh(destination, tolerance=mesh_tolerance)
        payload["mesh_validation"] = validation.as_dict()
        if not validation.matched:
            raise GeoStudioRuntimeError(
                f"mesh validation failed with {len(validation.mismatches)} "
                "XML/PLY boundary line mismatches; run mesh before solve"
            )
        names = analysis_names(destination)

        def record_progress(item: AnalysisSolve) -> None:
            progress.append(item)
            payload["status"] = "solving"
            payload["analyses"] = [value.as_dict() for value in progress]
            payload["updated_at"] = _timestamp()
            _write_json(report_path, payload)
            _append_log(
                log_path,
                f"analysis {len(progress)}/10 {item.analysis}: "
                f"{'succeeded' if item.succeeded else item.error_message}",
            )

        solved = solve_analyses(destination, names, on_result=record_progress)
        analyses = tuple(solved)
        succeeded = len(solved) == len(names) and all(
            item.succeeded for item in solved
        )
        if not succeeded:
            failed = next((item for item in solved if not item.succeeded), None)
            error_summary = (
                failed.error_message if failed else "not all ten analyses completed"
            )
        payload["status"] = "solved" if succeeded else "failed"
        payload["analyses"] = [item.as_dict() for item in solved]
    except Exception as error:
        error_summary = str(error)
        analyses = tuple(progress)
        _append_log(log_path, f"solve failed: {error_summary}")
    payload["status"] = "solved" if succeeded else "failed"
    payload["analyses"] = [item.as_dict() for item in analyses]
    payload["error_summary"] = error_summary
    payload["finished_at"] = _timestamp()
    _write_json(report_path, payload)
    return SolveResult(
        project_path=str(destination),
        succeeded=succeeded,
        error_summary=error_summary,
        report_path=str(report_path),
        analyses=analyses,
    )


def remesh_projects(
    sources: Sequence[Path],
    destinations: Sequence[Path],
    *,
    runtime: RuntimeInfo,
    mesh_tolerance: float = 1.0e-5,
    overwrite_output: bool = False,
    fail_fast: bool = False,
) -> list[MeshResult]:
    if len(sources) != len(destinations):
        raise ValueError("source and destination project counts differ")
    results: list[MeshResult] = []
    for source, destination in zip(sources, destinations, strict=True):
        result = remesh_project(
            source,
            output_project=destination,
            overwrite_output=overwrite_output,
            runtime=runtime,
            mesh_tolerance=mesh_tolerance,
        )
        results.append(result)
        if fail_fast and not result.succeeded:
            break
    return results


def solve_projects(
    sources: Sequence[Path],
    destinations: Sequence[Path],
    *,
    runtime: RuntimeInfo,
    mesh_tolerance: float = 1.0e-5,
    overwrite_output: bool = False,
    fail_fast: bool = False,
) -> list[SolveResult]:
    if len(sources) != len(destinations):
        raise ValueError("source and destination project counts differ")
    results: list[SolveResult] = []
    for source, destination in zip(sources, destinations, strict=True):
        result = solve_project(
            source,
            output_project=destination,
            overwrite_output=overwrite_output,
            runtime=runtime,
            mesh_tolerance=mesh_tolerance,
        )
        results.append(result)
        if fail_fast and not result.succeeded:
            break
    return results


def run_pipeline(
    template_path: Path,
    designs_path: Path,
    output_root: Path,
    *,
    design_ids: Sequence[str] | None = None,
    stop_after: str = "solve",
    force: bool = False,
    fail_fast: bool = False,
    geostudio_directory: Path | None = None,
    mesh_tolerance: float = 1.0e-5,
) -> list[PipelineResult]:
    if stop_after not in {"build", "mesh", "solve"}:
        raise ValueError("stop_after must be build, mesh, or solve")
    if mesh_tolerance <= 0:
        raise ValueError("mesh tolerance must be positive")

    archive = TemplateArchive.load(template_path)
    model = GeoStudioXmlModel(
        archive.root(), reference_design=archive.reference_design
    )
    designs = select_designs(read_designs(designs_path), design_ids)
    runtime = (
        discover_runtime(geostudio_directory) if stop_after != "build" else None
    )
    if stop_after == "solve" and runtime is not None and runtime.gsi_version is None:
        raise GeoStudioRuntimeError(
            "the official gsi package is not installed; run "
            "scripts/install_geostudio_api.ps1"
        )
    output_root = output_root.resolve()
    results: list[PipelineResult] = []
    for design in designs:
        run_dir = output_root / design.design_id
        project = run_dir / f"{design.design_id}.gsz"
        build = build_design(
            archive,
            design,
            project,
            force=force,
            preview_path=output_root / "model_picture" / f"{design.design_id}.png",
            model=model,
        )
        build_status = "built" if build.succeeded else "failed"
        mesh_status = "not_run"
        solve_status = "not_run"
        error_summary = build.error_summary

        if build.succeeded and stop_after in {"mesh", "solve"}:
            assert runtime is not None
            mesh = remesh_project(
                project,
                runtime=runtime,
                mesh_tolerance=mesh_tolerance,
            )
            mesh_status = (
                "meshed"
                if mesh.succeeded
                else (
                    "mesh_mismatch"
                    if mesh.validation is not None
                    else "failed"
                )
            )
            error_summary = mesh.error_summary
            if mesh.succeeded and stop_after == "solve":
                solve = solve_project(
                    project,
                    runtime=runtime,
                    mesh_tolerance=mesh_tolerance,
                )
                solve_status = "solved" if solve.succeeded else "failed"
                error_summary = solve.error_summary

        result = PipelineResult(
            design_id=design.design_id,
            project_path=str(project.resolve()),
            build_status=build_status,
            mesh_status=mesh_status,
            solve_status=solve_status,
            error_summary=error_summary,
        )
        results.append(result)
        _write_pipeline_manifest(output_root / "manifest.csv", results)
        if fail_fast and not result.succeeded:
            break
    return results


def _prepare_project_copy(
    source: Path, destination: Path, overwrite_output: bool
) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"GeoStudio project was not found: {source}")
    if source == destination:
        return
    if destination.exists() and not overwrite_output:
        raise FileExistsError(
            "output project already exists; pass --overwrite-output to replace it: "
            f"{destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _sidecar_path(project: Path, stage: str, suffix: str) -> Path:
    if project.parent.name == project.stem:
        return project.parent / f"{stage}_report.{suffix}"
    return project.parent / f"{project.stem}.{stage}_report.{suffix}"


def _write_pipeline_manifest(path: Path, results: Iterable[PipelineResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "design_id",
                "project_path",
                "build_status",
                "mesh_status",
                "solve_status",
                "error_summary",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.as_dict())


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{_timestamp()} {message}\n")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _software_info() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "shapely": shapely.__version__,
        "package": "knowledge-true-engineering 0.1.0",
        "platform": sys.platform,
    }
