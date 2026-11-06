"""Geometry/cost and replaceable serial evaluation backends."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from shapely.errors import GEOSException

from .extraction import extract_projects
from .geometry import build_geometry
from .kg_calibration import _write_json_atomic
from .model import DesignVariables, GeometryError
from .optimization_config import ExperimentConfig
from .results import DISPLACEMENT_METRIC_FIELDS, METRIC_FIELDS
from .runtime import discover_runtime
from .sampling import design_from_unit, point_key, unit_design
from .template import TemplateArchive
from .workflow import build_design, remesh_project, solve_project
from .xml_model import GeoStudioXmlModel


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GeometryDomain:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.archive = TemplateArchive.load(config.template)
        self.model = GeoStudioXmlModel(
            self.archive.root(), reference_design=self.archive.reference_design
        )
        materials = {
            int(m.findtext("ID"))
            for m in self.archive.root().iter("Material")
            if m.find("ID") is not None
        }
        unknown = set(config.material_unit_costs) - materials
        if unknown:
            raise ValueError(f"priced material IDs not in template: {sorted(unknown)}")
        digest = hashlib.sha256(self.archive.xml_bytes)
        for name, content in sorted(self.archive.assets.items()):
            digest.update(name.encode("utf-8"))
            digest.update(content)
        digest.update(
            json.dumps(
                self.archive.reference_design.as_dict()
                if self.archive.reference_design
                else None,
                sort_keys=True,
            ).encode()
        )
        self.fingerprint = digest.hexdigest()
        self.cache: OrderedDict = OrderedDict()
        self.last_error = ""

    def areas(self, point: np.ndarray) -> dict[int, float]:
        key = point_key(point)
        if key in self.cache:
            self.cache.move_to_end(key)
            result = self.cache[key]
            if isinstance(result, str):
                raise GeometryError(result)
            return result
        try:
            design = design_from_unit(point, "geometry_probe")
            errors = design.validate_bounds()
            if errors:
                raise GeometryError("; ".join(errors))
            geometry = build_geometry(
                self.model.original_regions,
                design,
                reference_design=self.model.reference_design,
            )
            areas: dict[int, float] = {}
            for cell in geometry.cells:
                areas[cell.material_id] = areas.get(cell.material_id, 0.0) + float(
                    cell.polygon.area
                )
            result = areas
        except (GeometryError, ValueError, GEOSException) as error:
            result = str(error)
        self.cache[key] = result
        if len(self.cache) > 4096:
            self.cache.popitem(last=False)
        if isinstance(result, str):
            raise GeometryError(result)
        return result

    def feasible(self, point: np.ndarray) -> bool:
        try:
            self.areas(point)
            return True
        except GeometryError as error:
            self.last_error = str(error)
            return False

    def cost(self, point: np.ndarray) -> float:
        areas = self.areas(point)
        return float(
            sum(
                areas.get(material, 0.0) * price
                for material, price in self.config.material_unit_costs.items()
            )
        )

    def prepare(self, design: DesignVariables) -> None:
        # Includes XML references and boundary conditions, before any costly call.
        self.model.rebuild(design)


@dataclass
class EvaluationResult:
    succeeded: bool
    metrics: dict[str, float] = field(default_factory=dict)
    error: str = ""
    artifacts: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.succeeded and (
            not set(DISPLACEMENT_METRIC_FIELDS).issubset(self.metrics)
            or not all(np.isfinite(value) for value in self.metrics.values())
        ):
            raise ValueError(
                "evaluator must return three finite displacement responses on success"
            )


class Evaluator(Protocol):
    fingerprint: str

    def evaluate(
        self, design: DesignVariables, directory: Path, charge: Callable[[], None]
    ) -> EvaluationResult: ...

    def recover(
        self, design: DesignVariables, directory: Path
    ) -> EvaluationResult | None: ...


class GeoStudioEvaluator:
    def __init__(self, config: ExperimentConfig, domain: GeometryDomain):
        self.config, self.domain = config, domain
        self.runtime = discover_runtime(config.geostudio_directory)
        if self.runtime.gsi_version is None:
            raise ValueError(
                "official gsi package is not installed; run scripts/install_geostudio_api.ps1"
            )
        self.fingerprint = "geostudio:" + json.dumps(
            self.runtime.as_dict(), sort_keys=True
        )

    def evaluate(self, design, directory, charge):
        project = directory / f"{design.design_id}.gsz"
        built = build_design(
            self.domain.archive, design, project, model=self.domain.model
        )
        if not built.succeeded:
            return EvaluationResult(False, error=built.error_summary)
        # Budget is committed immediately before starting native meshing.
        charge()
        meshed = remesh_project(
            project, runtime=self.runtime, mesh_tolerance=self.config.mesh_tolerance
        )
        if not meshed.succeeded:
            return EvaluationResult(False, error=meshed.error_summary)
        solved = solve_project(
            project, runtime=self.runtime, mesh_tolerance=self.config.mesh_tolerance
        )
        if not solved.succeeded:
            return EvaluationResult(False, error=solved.error_summary)
        return self._extract(design, directory)

    def _extract(self, design, directory):
        project = directory / f"{design.design_id}.gsz"
        output = extract_projects([project], directory / "extracted")
        if not output.successful_projects:
            return EvaluationResult(False, error=output.outcomes[0].error_summary)
        response_path = Path(output.outcomes[0].response_path)
        payload = json.loads(response_path.read_text(encoding="utf-8"))
        return EvaluationResult(
            True,
            payload["metrics"],
            artifacts=[
                str(project.resolve()),
                str((directory / "parameters.json").resolve()),
                str((directory / "build_report.json").resolve()),
                str((directory / "mesh_report.json").resolve()),
                str((directory / "solve_report.json").resolve()),
                str(response_path.resolve()),
            ],
        )

    def recover(self, design, directory):
        cached = read_evaluation(design, directory)
        if (directory / "evaluation.json").exists():
            # An invalid committed cache must not be bypassed by re-extracting
            # potentially changed project files as though they were original.
            return cached
        # Recover a solve that completed before the optimizer committed its row.
        project = directory / f"{design.design_id}.gsz"
        report_path = directory / "solve_report.json"
        parameters_path = directory / "parameters.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
            if (
                parameters["parameter_hash"] != design.parameter_hash()
                or report.get("status") != "solved"
                or report_path.stat().st_mtime_ns < project.stat().st_mtime_ns
            ):
                return None
            result = self._extract(design, directory)
            return result if result.succeeded else None
        except (OSError, ValueError, KeyError):
            return None


class SimulatedEvaluator:
    """Deterministic synthetic responses, explicitly not engineering predictions."""

    fingerprint = "synthetic-responses-v1"

    def evaluate(self, design, directory, charge):
        charge()
        x = unit_design(design)
        values = np.array(
            [
                0.10
                - 0.030 * x[0]
                - 0.015 * x[1]
                + 0.020 * x[2]
                - 0.025 * x[3]
                + 0.010 * x[5],
                0.24
                - 0.050 * x[0]
                - 0.040 * x[1]
                + 0.035 * x[2]
                - 0.050 * x[3]
                + 0.070 * x[5],
                0.07
                - 0.020 * x[0]
                - 0.015 * x[1]
                - 0.010 * x[2]
                - 0.015 * x[3]
                + 0.025 * x[5],
                1450 - 180 * x[0] - 100 * x[1] + 160 * x[2] + 140 * x[3] + 60 * x[5],
                480 + 90 * x[0] + 70 * x[1] + 40 * x[2] + 20 * x[3] - 60 * x[5],
            ]
        )
        values += np.array([0.002, 0.005, 0.002, 12, 8]) * np.sin(np.pi * x[0] * x[5])
        return EvaluationResult(
            True, dict(zip(METRIC_FIELDS, map(float, values), strict=True))
        )

    def recover(self, design, directory):
        return read_evaluation(design, directory)


def write_evaluation(
    design: DesignVariables, directory: Path, result: EvaluationResult
) -> dict[str, str]:
    result.validate()
    evidence = {
        str(Path(path).resolve()): file_digest(Path(path)) for path in result.artifacts
    }
    _write_json_atomic(
        directory / "evaluation.json",
        {
            "parameter_hash": design.parameter_hash(),
            "result": asdict(result),
            "evidence": evidence,
        },
    )
    return evidence


def read_evaluation(
    design: DesignVariables, directory: Path
) -> EvaluationResult | None:
    try:
        payload = json.loads(
            (directory / "evaluation.json").read_text(encoding="utf-8")
        )
        if payload["parameter_hash"] != design.parameter_hash():
            return None
        if any(
            file_digest(Path(path)) != digest
            for path, digest in payload["evidence"].items()
        ):
            return None
        result = EvaluationResult(**payload["result"])
        result.validate()
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None
