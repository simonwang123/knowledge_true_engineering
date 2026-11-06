from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar


class RunStatus(StrEnum):
    VALIDATED = "validated"
    SOLVED = "solved"
    INVALID_GEOMETRY = "invalid_geometry"
    MESH_MISMATCH = "mesh_mismatch"
    SOLVE_FAILED = "solve_failed"


@dataclass(frozen=True, slots=True)
class DesignVariables:
    design_id: str
    upstream_slope: float
    downstream_slope: float
    crest_width: float
    modulus_zone_height: float
    secondary_top_width: float
    secondary_height: float
    secondary_bottom_width: float
    secondary_right_slope_distance: float

    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        "upstream_slope": (1.3, 1.7),
        "downstream_slope": (1.3, 1.7),
        "crest_width": (6, 12),
        "modulus_zone_height": (5.0, 20),
        "secondary_top_width": (10.0, 30),
        "secondary_height": (20, 40),
        "secondary_bottom_width": (25, 70),
        "secondary_right_slope_distance": (2, 15),
    }

    CSV_FIELDS: ClassVar[tuple[str, ...]] = (
        "design_id",
        "upstream_slope",
        "downstream_slope",
        "crest_width",
        "modulus_zone_height",
        "secondary_top_width",
        "secondary_height",
        "secondary_bottom_width",
        "secondary_right_slope_distance",
    )

    @classmethod
    def from_mapping(cls, row: dict[str, str]) -> DesignVariables:
        missing = [field for field in cls.CSV_FIELDS if not row.get(field, "").strip()]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")
        design_id = row["design_id"].strip()
        if not design_id:
            raise ValueError("design_id must not be empty")
        if any(char in design_id for char in '<>:"/\\|?*') or design_id in {".", ".."}:
            raise ValueError(f"design_id contains a Windows-invalid character: {design_id!r}")
        if design_id.endswith((".", " ")):
            raise ValueError("design_id must not end with a dot or space")
        reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
        if design_id.split(".", 1)[0].upper() in reserved:
            raise ValueError(f"design_id is a reserved Windows name: {design_id!r}")
        values = {field: float(row[field]) for field in cls.CSV_FIELDS[1:]}
        return cls(design_id=design_id, **values)

    def validate_bounds(self) -> list[str]:
        errors: list[str] = []
        for field, (lower, upper) in self.BOUNDS.items():
            value = getattr(self, field)
            if not math.isfinite(value) or not lower <= value <= upper:
                errors.append(f"{field}={value:g} is outside [{lower:g}, {upper:g}]")
        return errors

    def as_dict(self) -> dict[str, str | float]:
        return asdict(self)

    def parameter_hash(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelConstants:
    dam_axis_x: float = 238.47177
    crest_elevation: float = 478.0
    dam_base_elevation: float = 404.0
    modulus_zone_top: float = 475.0
    baseline_crest_width: float = 10.14354
    baseline_upstream_slope: float = 1.400427837837838
    baseline_downstream_slope: float = 1.582275810810811
    secondary_left_angle_bounds: tuple[float, float] = (45.0, 150.0)
    coordinate_precision: int = 12

    # The six construction lifts used by analyses 2 through 7.
    lift_boundaries: tuple[float, ...] = (
        404.0,
        425.16493,
        439.88568,
        448.7449,
        455.56978,
        467.1658,
        478.0,
    )


CONSTANTS = ModelConstants()


class GeometryError(ValueError):
    """Raised when a parameter combination cannot form a valid model."""
