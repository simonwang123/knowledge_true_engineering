from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .extraction import extract_projects
from .kg_calibration import KGCalibrationConfig, calibrate_kg_edges
from .results import (
    DEFAULT_ANALYSIS_NAME,
    DEFAULT_ANALYSIS_STEP,
    DEFAULT_CVAR_ALPHA,
)
from .runtime import GeoStudioRuntimeError, discover_runtime
from .workflow import (
    build_designs,
    plan_project_destinations,
    remesh_projects,
    resolve_projects,
    run_pipeline,
    solve_projects,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geostudio-automation",
        description=(
            "Build, natively remesh, and solve parameterized GeoStudio models "
            "through independently callable stages."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    sample = commands.add_parser("sample", help="generate geometry-valid Latin hypercube designs without solving")
    sample.add_argument("--template", required=True, type=Path)
    sample.add_argument("--output", required=True, type=Path, help="new design CSV path")
    sample.add_argument("--count", type=int, default=20)
    sample.add_argument("--seed", type=int, default=2026)

    optimize = commands.add_parser("optimize", help="run constrained Bayesian optimization with optional knowledge guidance")
    optimize.add_argument("--config", required=True, type=Path, help="experiment JSON; paths are relative to this file")
    optimize.add_argument("--output", required=True, type=Path)
    optimize.add_argument("--resume", action="store_true", help="resume the identical experiment from its checkpoint")
    optimize.add_argument("--mode", choices=("knowledge", "plain"), default="knowledge")
    optimize.add_argument("--simulate", action="store_true", help="use explicitly synthetic responses; never starts GeoStudio")

    build = commands.add_parser("build", help="build GSZ models from a design CSV")
    _add_design_arguments(build)
    build.add_argument(
        "--force",
        action="store_true",
        help="replace an existing generated project",
    )
    build.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first failed design",
    )

    mesh = commands.add_parser("mesh", help="natively remesh existing GSZ projects")
    _add_project_arguments(mesh)
    _add_destination_arguments(mesh)
    _add_runtime_arguments(mesh)

    solve = commands.add_parser(
        "solve",
        help="validate current meshes and solve all ten analyses",
    )
    _add_project_arguments(solve)
    _add_destination_arguments(solve)
    _add_runtime_arguments(solve)

    extract = commands.add_parser(
        "extract",
        help="extract five training responses from solved GSZ projects",
    )
    _add_project_arguments(extract)
    extract.add_argument(
        "--output",
        required=True,
        type=Path,
        help="directory for dataset, audit, report, and per-run JSON files",
    )
    extract.add_argument(
        "--analysis-name",
        default=DEFAULT_ANALYSIS_NAME,
        help=f"analysis to query (default: {DEFAULT_ANALYSIS_NAME})",
    )
    extract.add_argument(
        "--step",
        type=int,
        default=DEFAULT_ANALYSIS_STEP,
        help=f"analysis result step (default: {DEFAULT_ANALYSIS_STEP})",
    )
    extract.add_argument(
        "--cvar-alpha",
        type=float,
        default=DEFAULT_CVAR_ALPHA,
        help=f"upper-tail CVaR confidence level (default: {DEFAULT_CVAR_ALPHA})",
    )
    extract.add_argument(
        "--force",
        action="store_true",
        help="ignore valid per-run extraction caches",
    )
    extract.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first failed project",
    )

    calibrate = commands.add_parser(
        "calibrate-kg",
        help="calibrate static knowledge-graph edges from an Initial DoE dataset",
    )
    calibration_defaults = KGCalibrationConfig()
    calibrate.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Initial DoE dataset.csv produced by the extract stage",
    )
    calibrate.add_argument(
        "--output",
        required=True,
        type=Path,
        help="directory for calibrated weights and audit reports",
    )
    calibrate.add_argument(
        "--min-samples",
        type=int,
        default=calibration_defaults.min_samples,
        help="minimum successful Initial DoE samples (default: 16)",
    )
    calibrate.add_argument(
        "--direction-threshold",
        type=float,
        default=calibration_defaults.direction_threshold,
        help="minimum local direction consistency (default: 0.8)",
    )
    calibrate.add_argument(
        "--strength-threshold",
        type=float,
        default=calibration_defaults.strength_threshold,
        help="minimum standardized ALE range (default: 0.5)",
    )
    calibrate.add_argument(
        "--random-seed",
        type=int,
        default=calibration_defaults.random_seed,
        help="deterministic GP random seed (default: 2026)",
    )
    calibrate.add_argument(
        "--force",
        action="store_true",
        help="replace existing calibration output files",
    )

    pipeline = commands.add_parser(
        "pipeline",
        help="run build, mesh, and solve as a transparent convenience chain",
    )
    _add_design_arguments(pipeline)
    pipeline.add_argument(
        "--stop-after",
        choices=("build", "mesh", "solve"),
        default="solve",
        help="last stage to run (default: solve)",
    )
    pipeline.add_argument(
        "--force",
        action="store_true",
        help="replace an existing generated project during the build stage",
    )
    pipeline.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first failed design",
    )
    pipeline.add_argument(
        "--geostudio-dir",
        type=Path,
        help="GeoStudio 2025.2 installation directory",
    )
    pipeline.add_argument(
        "--mesh-tolerance",
        type=float,
        default=1.0e-5,
        help="XML/PLY endpoint tolerance in metres",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "sample":
            from .sampling import write_initial_designs

            path = write_initial_designs(
                arguments.template, arguments.output,
                count=arguments.count, seed=arguments.seed, progress=print,
            )
            print(f"designs: {path}")
            print(f"sampling report: {path.with_suffix('.sampling.json')}")
            return 0

        if arguments.command == "optimize":
            from .optimization import run_optimization
            from .optimization_config import ExperimentConfig

            outputs = run_optimization(
                ExperimentConfig.from_json(arguments.config), arguments.output,
                mode=arguments.mode, resume=arguments.resume,
                simulate=arguments.simulate, progress=print,
            )
            print(f"report: {outputs.report_path}")
            print(f"dataset: {outputs.dataset_path}")
            print(f"evaluations: {outputs.attempts_spent}; successful samples: {outputs.successful_samples}")
            print(f"stopped: {outputs.stop_reason}")
            return 0 if outputs.best_design is not None and outputs.stop_reason in {
                "budget_exhausted", "adaptive_budget_exhausted"
            } else 1

        if arguments.command == "build":
            results = build_designs(
                arguments.template,
                arguments.designs,
                arguments.output,
                design_ids=arguments.design_ids,
                force=arguments.force,
                fail_fast=arguments.fail_fast,
            )
            for result in results:
                _print_result(
                    result.design_id,
                    result.succeeded,
                    "built",
                    result.error_summary,
                )
            return 0 if all(result.succeeded for result in results) else 1

        if arguments.command == "pipeline":
            results = run_pipeline(
                arguments.template,
                arguments.designs,
                arguments.output,
                design_ids=arguments.design_ids,
                stop_after=arguments.stop_after,
                force=arguments.force,
                fail_fast=arguments.fail_fast,
                geostudio_directory=arguments.geostudio_dir,
                mesh_tolerance=arguments.mesh_tolerance,
            )
            for result in results:
                final_status = {
                    "build": result.build_status,
                    "mesh": result.mesh_status,
                    "solve": result.solve_status,
                }[arguments.stop_after]
                _print_result(
                    result.design_id,
                    result.succeeded,
                    final_status,
                    result.error_summary,
                )
            print(f"manifest: {(arguments.output / 'manifest.csv').resolve()}")
            return 0 if all(result.succeeded for result in results) else 1

        if arguments.command == "calibrate-kg":
            config = KGCalibrationConfig(
                min_samples=arguments.min_samples,
                direction_threshold=arguments.direction_threshold,
                strength_threshold=arguments.strength_threshold,
                random_seed=arguments.random_seed,
            )
            outputs = calibrate_kg_edges(
                arguments.dataset,
                arguments.output,
                config=config,
                force=arguments.force,
                progress=print,
            )
            print(f"weights: {outputs.weights_path}")
            print(f"edge calibration: {outputs.edge_calibration_path}")
            print(f"report: {outputs.report_path}")
            print(f"active edges: {outputs.active_edges}/{len(outputs.edges)}")
            return 0

        sources = resolve_projects(
            arguments.projects,
            project_dir=arguments.project_dir,
            glob_pattern=arguments.glob_pattern,
        )
        if arguments.command == "extract":
            outputs = extract_projects(
                sources,
                arguments.output,
                analysis_name=arguments.analysis_name,
                step=arguments.step,
                cvar_alpha=arguments.cvar_alpha,
                force=arguments.force,
                fail_fast=arguments.fail_fast,
                progress=print,
            )
            print(f"dataset: {outputs.dataset_path}")
            print(f"audit: {outputs.audit_path}")
            print(f"report: {outputs.report_path}")
            return 0 if outputs.failed_projects == 0 else 1

        destinations = plan_project_destinations(
            sources,
            output_project=arguments.output_project,
            output_dir=arguments.output_dir,
            overwrite_output=arguments.overwrite_output,
        )
        runtime = discover_runtime(arguments.geostudio_dir)
        if arguments.command == "solve" and runtime.gsi_version is None:
            raise GeoStudioRuntimeError(
                "the official gsi package is not installed; run "
                "scripts/install_geostudio_api.ps1"
            )

        if arguments.command == "mesh":
            mesh_results = remesh_projects(
                sources,
                destinations,
                runtime=runtime,
                mesh_tolerance=arguments.mesh_tolerance,
                overwrite_output=arguments.overwrite_output,
                fail_fast=arguments.fail_fast,
            )
            for result in mesh_results:
                _print_result(
                    result.project_path,
                    result.succeeded,
                    "meshed",
                    result.error_summary,
                )
            return 0 if all(result.succeeded for result in mesh_results) else 1

        solve_results = solve_projects(
            sources,
            destinations,
            runtime=runtime,
            mesh_tolerance=arguments.mesh_tolerance,
            overwrite_output=arguments.overwrite_output,
            fail_fast=arguments.fail_fast,
        )
        for result in solve_results:
            _print_result(
                result.project_path,
                result.succeeded,
                "solved",
                result.error_summary,
            )
        return 0 if all(result.succeeded for result in solve_results) else 1
    except (OSError, TypeError, ValueError, GeoStudioRuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _add_design_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--template",
        required=True,
        type=Path,
        help="template .gsz file or extracted template directory",
    )
    parser.add_argument(
        "--designs",
        required=True,
        type=Path,
        help="CSV containing design_id and the eight design variables",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="output root containing one directory per design",
    )
    parser.add_argument(
        "--design-id",
        action="append",
        dest="design_ids",
        help="process only this design ID (repeat for multiple designs)",
    )


def _add_project_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--project",
        action="append",
        type=Path,
        dest="projects",
        help="GSZ project to process (repeat for multiple projects)",
    )
    source.add_argument(
        "--project-dir",
        type=Path,
        help="directory in which projects are discovered",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default="*.gsz",
        help="project pattern relative to --project-dir (default: *.gsz)",
    )


def _add_destination_arguments(parser: argparse.ArgumentParser) -> None:
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--output-project",
        type=Path,
        help="copy one input to this GSZ before processing",
    )
    destination.add_argument(
        "--output-dir",
        type=Path,
        help="copy all inputs to this directory before processing",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="replace an existing copied destination project",
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--geostudio-dir",
        type=Path,
        help="GeoStudio 2025.2 installation directory",
    )
    parser.add_argument(
        "--mesh-tolerance",
        type=float,
        default=1.0e-5,
        help="XML/PLY endpoint tolerance in metres",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first failed project",
    )


def _print_result(label: str, succeeded: bool, status: str, error: str) -> None:
    if succeeded:
        print(f"{label}: {status}")
    else:
        print(f"{label}: failed: {error}")


if __name__ == "__main__":
    raise SystemExit(main())
