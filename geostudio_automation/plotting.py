from __future__ import annotations

import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from shapely import box
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from .geometry import (
    CORE_MATERIALS,
    GeometryBuild,
    SourceRegion,
    _right_intersection,
    design_limits,
    secondary_zone_layout,
    transform_template_regions,
)
from .model import CONSTANTS, DesignVariables

MATERIAL_COLORS: dict[int, str] = {
    1: "#4C78A8",
    2: "#F2CF5B",
    3: "#F28E2B",
    4: "#18C9D2",
    5: "#E83EBA",
    6: "#242424",
    7: "#9C755F",
    8: "#314E89",
    9: "#D9D9D9",
    10: "#B5CF6B",
    11: "#8CD17D",
    12: "#A020F0",
    13: "#FAF513",
}


@dataclass(frozen=True, slots=True)
class AttemptedShape:
    label: str
    geometry: Polygon | MultiPolygon
    color: str
    alpha: float
    hatch: str | None = None


def write_model_picture(
    path: Path,
    original_regions: list[SourceRegion],
    design: DesignVariables,
    *,
    build: GeometryBuild | None,
    status: str,
    error: str = "",
    reference_design: DesignVariables | None = None,
) -> None:
    """Write a deterministic PNG for both valid and rejected designs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    legend_materials: set[int] = set()

    if build is not None:
        base_geometries: list[Polygon | MultiPolygon] = []
        for cell in build.cells:
            _draw_geometry(
                axis,
                cell.polygon,
                facecolor=MATERIAL_COLORS.get(cell.material_id, "#BBBBBB"),
                edgecolor="#333333",
                linewidth=0.45,
                alpha=0.92,
            )
            base_geometries.append(cell.polygon)
            legend_materials.add(cell.material_id)
    else:
        transformed = _transformed_template(
            original_regions,
            design,
            reference_design=reference_design,
        )
        base_geometries = [region.polygon for region in transformed]
        for region in transformed:
            _draw_geometry(
                axis,
                region.polygon,
                facecolor=MATERIAL_COLORS.get(region.material_id, "#BBBBBB"),
                edgecolor="#777777",
                linewidth=0.35,
                alpha=0.34,
            )
            legend_materials.add(region.material_id)
        for attempted in _attempted_shapes(transformed, design):
            _draw_geometry(
                axis,
                attempted.geometry,
                facecolor=attempted.color,
                edgecolor="#D62728",
                linewidth=1.5,
                alpha=attempted.alpha,
                hatch=attempted.hatch,
            )

    _draw_design_envelope(axis, design)
    _set_extent(axis, base_geometries)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Elevation (m)")
    axis.grid(True, color="#E6E6E6", linewidth=0.5, zorder=0)
    axis.set_title(f"{design.design_id} | {status}", color=_status_color(status), fontsize=13)

    handles = [
        Patch(facecolor=MATERIAL_COLORS[material], edgecolor="#333333", label=f"Material {material}")
        for material in sorted(legend_materials)
    ]
    if build is None:
        handles.extend(
            [
                Patch(facecolor="#E83EBA", edgecolor="#D62728", label="Attempted secondary zone"),
                Patch(facecolor="#242424", edgecolor="#D62728", label="Attempted drainage zone"),
            ]
        )
    axis.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    if error:
        axis.text(
            0.01,
            0.02,
            "\n".join(textwrap.wrap(error, width=95)),
            transform=axis.transAxes,
            fontsize=8,
            color="#8B0000",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#FFF2F2", "edgecolor": "#D62728"},
        )
    figure.savefig(path, dpi=180, format="png", facecolor="white")
    plt.close(figure)


def _transformed_template(
    original_regions: list[SourceRegion],
    design: DesignVariables,
    *,
    reference_design: DesignVariables | None = None,
) -> list[SourceRegion]:
    try:
        return transform_template_regions(
            original_regions,
            design,
            reference_design=reference_design,
        )
    except Exception:
        return original_regions


def _attempted_shapes(
    transformed: list[SourceRegion], design: DesignVariables
) -> list[AttemptedShape]:
    shapes: list[AttemptedShape] = []
    try:
        core = unary_union(
            [region.polygon for region in transformed if region.material_id in CORE_MATERIALS]
        )
        y_top = CONSTANTS.modulus_zone_top - design.modulus_zone_height
        y_bottom = y_top - design.secondary_height
        min_x, _, max_x, _ = core.bounds
        modulus = core.intersection(box(min_x - 1.0, y_top, max_x + 1.0, CONSTANTS.modulus_zone_top))
        if not modulus.is_empty:
            shapes.append(AttemptedShape("modulus", modulus, MATERIAL_COLORS[12], 0.55, "//"))

        try:
            contact_bottom = _right_intersection(core, y_bottom)
        except Exception:
            contact_bottom = design_limits(y_bottom, design)[1]
        try:
            contact_base = _right_intersection(core, CONSTANTS.dam_base_elevation)
        except Exception:
            contact_base = design_limits(CONSTANTS.dam_base_elevation, design)[1]

        layout = secondary_zone_layout(
            design,
            downstream_contact_x_at_bottom=contact_bottom,
        )
        top_left = layout.top_left
        top_right = layout.top_right
        bottom_left = layout.bottom_left
        bottom_right = layout.bottom_right
        dx_per_vertical = (
            top_left[0] - bottom_left[0]
        ) / design.secondary_height
        secondary = Polygon([top_left, top_right, bottom_right, bottom_left])
        drainage_left_base = (
            bottom_left[0]
            - dx_per_vertical * (y_bottom - CONSTANTS.dam_base_elevation),
            CONSTANTS.dam_base_elevation,
        )
        drainage = Polygon(
            [
                bottom_left,
                bottom_right,
                (contact_bottom, y_bottom),
                (contact_base, CONSTANTS.dam_base_elevation),
                drainage_left_base,
            ]
        )
        if not secondary.is_empty:
            shapes.append(AttemptedShape("secondary", secondary, MATERIAL_COLORS[5], 0.60, "xx"))
        if not drainage.is_empty:
            shapes.append(AttemptedShape("drainage", drainage, MATERIAL_COLORS[6], 0.45, ".."))
    except Exception:
        return shapes
    return shapes


def _draw_design_envelope(axis: Axes, design: DesignVariables) -> None:
    try:
        upstream_toe, downstream_toe = design_limits(CONSTANTS.dam_base_elevation, design)
        crest_left, crest_right = design_limits(CONSTANTS.crest_elevation, design)
        axis.plot(
            [upstream_toe, crest_left, crest_right, downstream_toe],
            [
                CONSTANTS.dam_base_elevation,
                CONSTANTS.crest_elevation,
                CONSTANTS.crest_elevation,
                CONSTANTS.dam_base_elevation,
            ],
            color="#0B3C5D",
            linestyle="--",
            linewidth=1.2,
            label="Design envelope",
            zorder=8,
        )
    except Exception:
        return


def _draw_geometry(
    axis: Axes,
    geometry: Polygon | MultiPolygon,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float,
    alpha: float,
    hatch: str | None = None,
) -> None:
    for polygon in _polygon_parts(geometry):
        x, y = polygon.exterior.xy
        axis.fill(
            x,
            y,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            hatch=hatch,
            zorder=2,
        )


def _polygon_parts(geometry: Polygon | MultiPolygon) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:  # type: ignore[union-attr]
            if isinstance(part, Polygon):
                yield part


def _set_extent(axis: Axes, geometries: list[Polygon | MultiPolygon]) -> None:
    usable = [geometry for geometry in geometries if not geometry.is_empty]
    if not usable:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        return
    # Invalid attempted polygons are intentionally drawable in failure reports,
    # but GEOS cannot union every self-intersecting shape.  Extents only need the
    # individual bounds and therefore must not depend on a successful topology
    # operation.
    bounds = [geometry.bounds for geometry in usable]
    min_x = min(value[0] for value in bounds)
    min_y = min(value[1] for value in bounds)
    max_x = max(value[2] for value in bounds)
    max_y = max(value[3] for value in bounds)
    margin_x = max((max_x - min_x) * 0.035, 1.0)
    margin_y = max((max_y - min_y) * 0.06, 1.0)
    axis.set_xlim(min_x - margin_x, max_x + margin_x)
    axis.set_ylim(min_y - margin_y, max_y + margin_y)


def _status_color(status: str) -> str:
    if status in {"built", "meshed", "solved", "validated"}:
        return "#2E7D32"
    if status == "mesh_mismatch":
        return "#EF6C00"
    return "#B71C1C"
