"""Flexible GeoStudio modelling, remeshing, solving, and result extraction."""

from .extraction import BatchExtractionOutputs, ExtractionOutcome, extract_projects
from .kg_calibration import (
    KGCalibrationConfig,
    KGCalibrationOutputs,
    calibrate_kg_edges,
)
from .model import DesignVariables, GeometryError
from .runtime import AnalysisSolve, GeoStudioRuntimeError, RuntimeInfo
from .workflow import (
    BuildResult,
    MeshResult,
    PipelineResult,
    SolveResult,
    build_design,
    remesh_project,
    run_pipeline,
    solve_project,
)

__all__ = [
    "AnalysisSolve",
    "BatchExtractionOutputs",
    "BuildResult",
    "DesignVariables",
    "ExtractionOutcome",
    "GeoStudioRuntimeError",
    "GeometryError",
    "KGCalibrationConfig",
    "KGCalibrationOutputs",
    "MeshResult",
    "PipelineResult",
    "RuntimeInfo",
    "SolveResult",
    "build_design",
    "calibrate_kg_edges",
    "extract_projects",
    "remesh_project",
    "run_pipeline",
    "solve_project",
]
