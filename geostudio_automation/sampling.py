"""Bounded, reproducible sampling; geometry is checked without solving."""

from __future__ import annotations

from collections.abc import Callable, Collection
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from .model import DesignVariables

NAMES = DesignVariables.CSV_FIELDS[1:]
BOUNDS = np.asarray([DesignVariables.BOUNDS[name] for name in NAMES], dtype=float)


def write_initial_designs(
    template: Path,
    output: Path,
    *,
    count: int = 20,
    seed: int = 2026,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Write a strict LHS whose designs pass geometry and XML preflight.

    This stage never starts GeoStudio and needs no response limits or prices.
    Existing sample files are preserved so an experiment cannot be resampled
    accidentally after solving has started.
    """
    import csv
    import hashlib
    import json

    import scipy
    from shapely.errors import GEOSException

    from .model import GeometryError
    from .template import TemplateArchive
    from .xml_model import GeoStudioXmlModel

    if type(count) is not int or count < 2:
        raise ValueError("sample count must be an integer >= 2")
    if type(seed) is not int or seed < 0:
        raise ValueError("sample seed must be a non-negative integer")
    output = output.resolve()
    report_path = output.with_suffix(".sampling.json")
    for path in (output, report_path):
        if path.exists():
            raise FileExistsError(f"sampling output already exists: {path}; use a new path")

    archive = TemplateArchive.load(template)
    model = GeoStudioXmlModel(archive.root(), reference_design=archive.reference_design)
    checked: dict[tuple[float, ...], bool] = {}

    def feasible(point: np.ndarray) -> bool:
        key = point_key(point)
        if key not in checked:
            try:
                model.rebuild(design_from_unit(point, "sampling_probe"))
                checked[key] = True
            except (GeometryError, GEOSException):
                checked[key] = False
        return checked[key]

    points = feasible_lhs(count, feasible, seed=seed, progress=progress)
    strata = np.floor(points * count).astype(int)
    expected = np.tile(np.arange(count)[:, None], (1, len(NAMES)))
    if not np.array_equal(np.sort(strata, axis=0), expected):
        raise ValueError("sampling failed the Latin stratification check")
    designs = [design_from_unit(p, f"lhs_{i:03d}") for i, p in enumerate(points, 1)]
    report = {
        "method": "geometry-constrained Latin hypercube",
        "count": count,
        "seed": seed,
        "bounds": DesignVariables.BOUNDS,
        "scipy_version": scipy.__version__,
        "template": str(template.resolve()),
        "template_xml_sha256": hashlib.sha256(archive.xml_bytes).hexdigest(),
        "template_reference_design": (
            archive.reference_design.as_dict() if archive.reference_design else None
        ),
        "latin_strata_valid": True,
        "geometry_and_xml_valid": True,
        "geostudio_solved": False,
        "design_ids": [design.design_id for design in designs],
        "unit_points": points.tolist(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=DesignVariables.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(design.as_dict() for design in designs)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return output


def design_from_unit(point: np.ndarray, design_id: str) -> DesignVariables:
    values = BOUNDS[:, 0] + np.asarray(point) * (BOUNDS[:, 1] - BOUNDS[:, 0])
    return DesignVariables(
        design_id=design_id, **dict(zip(NAMES, map(float, values), strict=True))
    )


def unit_design(design: DesignVariables) -> np.ndarray:
    return (np.asarray([getattr(design, n) for n in NAMES]) - BOUNDS[:, 0]) / (
        BOUNDS[:, 1] - BOUNDS[:, 0]
    )


def point_key(point: np.ndarray) -> tuple[float, ...]:
    # Independent of design_id; protects against effectively repeated evaluations.
    return tuple(np.round(point, 12).tolist())


def feasible_lhs(
    count: int,
    feasible: Callable[[np.ndarray], bool],
    *,
    seed: int = 2026,
    restarts: int = 10,
    repair_attempts: int = 2000,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    best = 0
    for restart in range(restarts):
        points = qmc.LatinHypercube(d=8, seed=seed + restart).random(count)
        rng = np.random.default_rng(seed + restart)
        valid = np.asarray([feasible(p) for p in points])
        for attempt in range(repair_attempts + 1):
            best = max(best, int(valid.sum()))
            if valid.all():
                return points
            if attempt == repair_attempts:
                break
            a = int(rng.choice(np.flatnonzero(~valid)))
            if rng.random() < 0.45:
                trial = (np.floor(points[a] * count) + rng.random(8)) / count
                points[a], valid[a] = trial, feasible(trial)
            else:
                b = int(rng.choice([i for i in range(count) if i != a]))
                dims = rng.choice(8, size=int(rng.integers(1, 4)), replace=False)
                trial = points[[a, b]].copy()
                trial[0, dims], trial[1, dims] = (
                    trial[1, dims].copy(),
                    trial[0, dims].copy(),
                )
                flags = np.asarray([feasible(p) for p in trial])
                if flags.sum() >= valid[[a, b]].sum():
                    points[[a, b]], valid[[a, b]] = trial, flags
            if progress and attempt % 100 == 0:
                progress(
                    f"LHS repair {restart + 1}/{restarts}: {int(valid.sum())}/{count} feasible"
                )
    raise ValueError(
        f"LHS repair exhausted: best {best}/{count} geometrically feasible points"
    )


def replace_in_strata(
    original: np.ndarray,
    count: int,
    feasible: Callable[[np.ndarray], bool],
    *,
    seed: int,
    excluded: Collection[tuple[float, ...]],
    attempts: int = 2000,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    strata = np.floor(np.asarray(original) * count)
    for _ in range(attempts):
        point = (strata + rng.random(8)) / count
        if point_key(point) not in excluded and feasible(point):
            return point
    raise ValueError(
        "could not replace failed initial sample inside its original Latin strata"
    )


def sobol_candidates(
    count: int,
    feasible: Callable[[np.ndarray], bool],
    *,
    seed: int,
    excluded: Collection[tuple[float, ...]],
    max_attempts: int = 50000,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    sampler = qmc.Sobol(d=8, scramble=True, seed=seed)
    pool = sampler.random_base2(int(np.ceil(np.log2(max_attempts))))[:max_attempts]
    selected = []
    seen = set(excluded)
    for index, point in enumerate(pool):
        key = point_key(point)
        if key not in seen and feasible(point):
            selected.append(point)
            seen.add(key)
            if len(selected) == count:
                return np.asarray(selected)
        if progress and index % 256 == 0:
            progress(f"candidate geometry: {len(selected)}/{count} feasible")
    raise ValueError(
        f"candidate generation exhausted: {len(selected)}/{count} feasible new points"
    )
