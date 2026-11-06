"""Explicit engineering inputs and reproducible optimization settings."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .results import DISPLACEMENT_METRIC_FIELDS


@dataclass(frozen=True)
class Constraint:
    limit: float
    direction: str

    def residual(self, value):
        """Positive means violated; derivative priors remain in raw-response units."""
        return value - self.limit if self.direction in {"<", "<="} else self.limit - value

    def satisfied(self, value):
        if self.direction == "<":
            return value < self.limit
        if self.direction == "<=":
            return value <= self.limit
        if self.direction == ">":
            return value > self.limit
        return value >= self.limit


@dataclass(frozen=True)
class ExperimentConfig:
    template: Path
    constraints: dict[str, Constraint]
    material_unit_costs: dict[int, float]
    random_seed: int = 2026
    initial_samples: int = 20
    adaptive_budget: int = 40
    total_budget: int = 60
    r2_threshold: float = 0.70
    gp_restarts: int = 3
    lhs_restarts: int = 10
    lhs_repair_attempts: int = 2000
    replacement_attempts: int = 2000
    candidate_count: int = 512
    candidate_attempts: int = 50000
    exploration_interval: int = 5
    max_virtual_derivatives: int = 20
    derivative_softness: float = 0.1
    local_conflict_probability: float = 0.8
    mesh_tolerance: float = 1.0e-5
    geostudio_directory: Path | None = None
    simulation_only: bool = False
    initial_dataset: Path | None = None

    def validate(self) -> None:
        if not isinstance(self.template, Path):
            raise TypeError("template must be a path")
        if type(self.simulation_only) is not bool:
            raise TypeError("simulation_only must be boolean")
        if self.initial_dataset is not None and not isinstance(self.initial_dataset, Path):
            raise TypeError("initial_dataset must be a path")
        if set(self.constraints) != set(DISPLACEMENT_METRIC_FIELDS):
            raise ValueError(
                "constraints must configure exactly the three displacement responses"
            )
        for name, constraint in self.constraints.items():
            if constraint.direction not in {"<", "<=", ">", ">="}:
                raise ValueError(f"{name}: direction must be <, <=, > or >=")
            if isinstance(constraint.limit, bool) or not math.isfinite(
                constraint.limit
            ):
                raise ValueError(f"{name}: a finite engineering limit is required")
        if not self.material_unit_costs:
            raise ValueError(
                "material_unit_costs must explicitly select priced material IDs"
            )
        for material, cost in self.material_unit_costs.items():
            if type(material) is not int or material <= 0:
                raise ValueError("material IDs must be positive integers")
            if isinstance(cost, bool) or not math.isfinite(cost) or cost <= 0:
                raise ValueError(
                    f"material {material}: unit cost must be finite and positive"
                )
        for name in (
            "initial_samples",
            "total_budget",
            "lhs_restarts",
            "lhs_repair_attempts",
            "replacement_attempts",
            "candidate_count",
            "candidate_attempts",
            "exploration_interval",
            "max_virtual_derivatives",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("adaptive_budget", "gp_restarts", "random_seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.initial_samples < 4 or self.total_budget < self.initial_samples:
            raise ValueError(
                "initial_samples must be >= 4 and total_budget >= initial_samples"
            )
        if self.candidate_attempts < self.candidate_count:
            raise ValueError("candidate_attempts must be >= candidate_count")
        if not math.isfinite(self.r2_threshold) or not 0 <= self.r2_threshold <= 1:
            raise ValueError("r2_threshold must be in [0, 1]")
        if not 0.5 < self.local_conflict_probability < 1:
            raise ValueError("local_conflict_probability must be in (0.5, 1)")
        for name in ("derivative_softness", "mesh_tolerance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["template"] = str(self.template.resolve())
        payload["geostudio_directory"] = (
            str(self.geostudio_directory.resolve())
            if self.geostudio_directory
            else None
        )
        payload["initial_dataset"] = (
            str(self.initial_dataset.resolve()) if self.initial_dataset else None
        )
        payload["material_unit_costs"] = {
            str(k): v for k, v in self.material_unit_costs.items()
        }
        return payload

    @classmethod
    def from_json(cls, path: Path) -> ExperimentConfig:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise TypeError("experiment config must be a JSON object")
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"unknown configuration fields: {', '.join(sorted(unknown))}"
            )
        try:
            constraints = {
                name: Constraint(
                    limit=_number(value["limit"], name), direction=value["direction"]
                )
                for name, value in raw["constraints"].items()
            }
            costs = {
                int(key): _number(value, f"material {key}")
                for key, value in raw["material_unit_costs"].items()
            }
            if len(costs) != len(raw["material_unit_costs"]):
                raise ValueError("duplicate numeric material IDs")
            values = {**raw, "constraints": constraints, "material_unit_costs": costs}
            for name in ("template", "geostudio_directory", "initial_dataset"):
                if values.get(name) is not None:
                    candidate = Path(values[name])
                    values[name] = (path.resolve().parent / candidate).resolve()
            result = cls(**values)
            result.validate()
            return result
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError(
                "config requires template, three displacement limits/directions and material prices"
            ) from error


def _number(value, label: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{label}: fill in a numeric value before running")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: a numeric value is required") from error
    if not math.isfinite(result):
        raise ValueError(f"{label}: value must be finite")
    return result
