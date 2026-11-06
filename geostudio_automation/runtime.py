from __future__ import annotations

import os
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

from .mesh import MeshValidation, validate_gsz_mesh

MINIMUM_GEOSTUDIO_VERSION = (25, 2, 0, 0)


class GeoStudioRuntimeError(RuntimeError):
    """Raised when GeoStudio or its official Python API cannot be used."""


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    install_directory: str
    executable: str
    geostudio_version: str
    gsi_version: str | None

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnalysisSolve:
    analysis: str
    succeeded: bool
    error_message: str

    def as_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def discover_runtime(install_directory: Path | None = None) -> RuntimeInfo:
    if install_directory is None:
        candidates = sorted(
            Path(r"C:\Program Files\Seequent").glob("GeoStudio 2025.2*"),
            reverse=True,
        )
        if not candidates:
            raise GeoStudioRuntimeError("GeoStudio 2025.2 installation was not found")
        install_directory = candidates[0]

    install_directory = install_directory.resolve()
    executable = install_directory / "Bin" / "GeoStudio.exe"
    geocmd = install_directory / "Bin" / "GeoCmd.exe"
    if not executable.is_file():
        raise GeoStudioRuntimeError(f"GeoStudio executable was not found: {executable}")
    if not geocmd.is_file():
        raise GeoStudioRuntimeError(f"GeoCmd executable was not found: {geocmd}")

    version = _windows_file_version(executable)
    if _version_tuple(version) < MINIMUM_GEOSTUDIO_VERSION:
        minimum = ".".join(str(value) for value in MINIMUM_GEOSTUDIO_VERSION[:3])
        raise GeoStudioRuntimeError(
            f"GeoStudio {version} is unsupported; version {minimum} or later is required"
        )
    try:
        gsi_version = metadata.version("gsi")
    except metadata.PackageNotFoundError:
        gsi_version = None
    return RuntimeInfo(str(install_directory), str(executable), version, gsi_version)


def force_native_remesh_and_validate(
    project_path: Path,
    runtime: RuntimeInfo,
    *,
    tolerance: float = 1.0e-5,
    timeout_seconds: int = 120,
) -> MeshValidation:
    """Force GeoStudio to rebuild a mesh, then verify XML/PLY consistency."""
    project_path = project_path.resolve()
    geocmd = Path(runtime.install_directory) / "Bin" / "GeoCmd.exe"
    if not project_path.is_file():
        raise GeoStudioRuntimeError(f"GeoStudio project was not found: {project_path}")
    if not geocmd.is_file():
        raise GeoStudioRuntimeError(f"GeoCmd executable was not found: {geocmd}")

    try:
        completed = subprocess.run(
            [str(geocmd), str(project_path), "/clean"],
            cwd=project_path.parent,
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise GeoStudioRuntimeError(f"GeoCmd native remesh failed: {error}") from error

    if completed.returncode != 0:
        output = _decode_process_output(completed.stdout + completed.stderr).strip()
        detail = f": {output}" if output else ""
        raise GeoStudioRuntimeError(
            f"GeoCmd /clean returned exit code {completed.returncode}{detail}"
        )

    try:
        return validate_gsz_mesh(project_path, tolerance=tolerance)
    except Exception as error:
        raise GeoStudioRuntimeError(
            f"GeoCmd completed, but the rebuilt mesh could not be validated: {error}"
        ) from error


def analysis_names(project_path: Path) -> list[str]:
    with zipfile.ZipFile(project_path) as archive:
        xml_files = [
            name
            for name in archive.namelist()
            if "/" not in name.rstrip("/") and name.lower().endswith(".xml")
        ]
        if len(xml_files) != 1:
            raise ValueError("GSZ must contain exactly one root XML file")
        root = ET.fromstring(archive.read(xml_files[0]))
    names = [
        analysis.findtext("Name", "")
        for analysis in root.findall("./Analyses/Analysis")
    ]
    if len(names) != 10 or any(not name for name in names):
        raise ValueError(f"expected ten named analyses, found {len(names)}")
    return names


def solve_analyses(
    project_path: Path,
    names: list[str],
    *,
    on_result: Callable[[AnalysisSolve], None] | None = None,
) -> list[AnalysisSolve]:
    if len(names) != 10:
        raise ValueError(f"expected ten analyses, received {len(names)}")
    project_path = project_path.resolve()
    with _working_directory(project_path.parent):
        try:
            import gsi  # type: ignore[import-not-found]
        except ImportError as error:
            raise GeoStudioRuntimeError(
                "the official gsi package is not installed; run "
                "scripts/install_geostudio_api.ps1"
            ) from error

        results: list[AnalysisSolve] = []
        project = None
        try:
            project = gsi.OpenProject(str(project_path))
            for name in names:
                try:
                    response = project.SolveAnalyses(
                        gsi.SolveAnalysesRequest(
                            analyses=[name],
                            solve_dependencies=False,
                        )
                    )
                    completion = response.completion_status.get(name)
                    if completion is None and len(response.completion_status) == 1:
                        completion = next(iter(response.completion_status.values()))
                    succeeded = bool(
                        completion and completion.succeeded and response.all_succeeded
                    )
                    error_message = (
                        ""
                        if succeeded
                        else (
                            completion.error_message
                            if completion is not None
                            else "GeoStudio returned no completion status"
                        )
                    )
                except Exception as error:
                    succeeded = False
                    error_message = str(error)
                result = AnalysisSolve(name, succeeded, error_message)
                results.append(result)
                if on_result is not None:
                    on_result(result)
                if not succeeded:
                    break
            project.Save()
        except Exception as error:
            raise GeoStudioRuntimeError(f"GeoStudio solve failed: {error}") from error
        finally:
            if project is not None:
                project.Close()
        return results


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    """Keep GSI's process-level log beside the project, then restore the cwd."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _decode_process_output(data: bytes) -> str:
    for encoding in ("utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode(errors="replace")


def _windows_file_version(executable: Path) -> str:
    escaped_path = str(executable).replace("'", "''")
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"(Get-Item -LiteralPath '{escaped_path}').VersionInfo.FileVersion",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip().replace(",", ".").replace(" ", "")
    if not value:
        raise GeoStudioRuntimeError(
            f"could not read GeoStudio version from {executable}"
        )
    return value


def _version_tuple(version: str) -> tuple[int, int, int, int]:
    values = [int(value) for value in re.findall(r"\d+", version)[:4]]
    return tuple((values + [0, 0, 0, 0])[:4])  # type: ignore[return-value]
