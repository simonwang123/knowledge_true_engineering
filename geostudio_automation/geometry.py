from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import pairwise
from math import atan2, degrees, hypot, sqrt

from shapely import box, set_precision
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import polygonize, unary_union
from shapely.validation import explain_validity

from .model import CONSTANTS, DesignVariables, GeometryError, ModelConstants

CORE_MATERIALS = frozenset({4, 5, 6, 12})
GEOMETRY_AREA_EPSILON = 1.0e-7
MIN_AREA = 1.0e-1
MIN_EDGE_LENGTH = 2.0e-1
GEOMETRY_TOLERANCE = 1.0e-1
UPSTREAM_SLOPE_MATERIALS = frozenset({2, 3})
UPSTREAM_WEATHERED_MATERIALS = frozenset({10, 11})
UPSTREAM_OUTER_FOLLOW_DISTANCE = 20.0
TOPOLOGY_SNAP_TOLERANCE = 1.0e-3


@dataclass(frozen=True, slots=True)
class SourceRegion:
    polygon: Polygon
    material_id: int
    activation_stage: int
    source: str


@dataclass(frozen=True, slots=True)
class RegionCell:
    polygon: Polygon
    material_id: int
    activation_stage: int
    source: str


@dataclass(frozen=True, slots=True)
class DerivedGeometry:
    crest_left: tuple[float, float]
    crest_right: tuple[float, float]
    upstream_toe: tuple[float, float]
    downstream_toe: tuple[float, float]
    modulus_bottom_elevation: float
    secondary_top_left: tuple[float, float]
    secondary_top_right: tuple[float, float]
    secondary_bottom_left: tuple[float, float]
    secondary_bottom_right: tuple[float, float]
    secondary_left_bottom_angle: float
    secondary_right_slope_distance: float
    secondary_right_horizontal_offset: float
    secondary_downstream_reference_bottom: tuple[float, float]
    drainage_bottom_left: tuple[float, float]
    drainage_top_right: tuple[float, float]
    drainage_bottom_right: tuple[float, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "crest_left": self.crest_left,
            "crest_right": self.crest_right,
            "upstream_toe": self.upstream_toe,
            "downstream_toe": self.downstream_toe,
            "modulus_bottom_elevation": self.modulus_bottom_elevation,
            "secondary_top_left": self.secondary_top_left,
            "secondary_top_right": self.secondary_top_right,
            "secondary_bottom_left": self.secondary_bottom_left,
            "secondary_bottom_right": self.secondary_bottom_right,
            "secondary_left_bottom_angle": self.secondary_left_bottom_angle,
            "secondary_right_slope_distance": self.secondary_right_slope_distance,
            "secondary_right_horizontal_offset": self.secondary_right_horizontal_offset,
            "secondary_downstream_reference_bottom": self.secondary_downstream_reference_bottom,
            "drainage_bottom_left": self.drainage_bottom_left,
            "drainage_top_right": self.drainage_top_right,
            "drainage_bottom_right": self.drainage_bottom_right,
        }


@dataclass(slots=True)
class GeometryBuild:
    cells: list[RegionCell]
    derived: DerivedGeometry
    core_domain: Polygon | MultiPolygon


@dataclass(frozen=True, slots=True)
class SecondaryZoneLayout:
    top_left: tuple[float, float]
    top_right: tuple[float, float]
    bottom_left: tuple[float, float]
    bottom_right: tuple[float, float]
    left_bottom_angle: float
    right_horizontal_offset: float
    downstream_reference_bottom: tuple[float, float]


@dataclass(frozen=True, slots=True)
class TemplateWarpControls:
    upstream: frozenset[tuple[float, float]]
    downstream: frozenset[tuple[float, float]]
    toe_corrections: dict[tuple[float, float], tuple[float, float]] = field(default_factory=dict)


def baseline_limits(
    y: float,
    constants: ModelConstants = CONSTANTS,
    *,
    reference_design: DesignVariables | None = None,
) -> tuple[float, float]:
    if reference_design is not None:
        return design_limits(y, reference_design, constants)
    vertical = constants.crest_elevation - y
    return (
        constants.dam_axis_x - constants.baseline_crest_width / 2 - constants.baseline_upstream_slope * vertical,
        constants.dam_axis_x + constants.baseline_crest_width / 2 + constants.baseline_downstream_slope * vertical,
    )


def design_limits(
    y: float, design: DesignVariables, constants: ModelConstants = CONSTANTS
) -> tuple[float, float]:
    vertical = constants.crest_elevation - y
    return (
        constants.dam_axis_x - design.crest_width / 2 - design.upstream_slope * vertical,
        constants.dam_axis_x + design.crest_width / 2 + design.downstream_slope * vertical,
    )


def secondary_zone_layout(
    design: DesignVariables,
    constants: ModelConstants = CONSTANTS,
    *,
    downstream_contact_x_at_bottom: float | None = None,
) -> SecondaryZoneLayout:
    """Construct a trapezoid whose right side is parallel to the downstream face.

    ``secondary_right_slope_distance`` is the shortest perpendicular distance
    from the right side to a line through the real downstream contact at the
    secondary-zone bottom elevation. That reference line is parallel to the
    ideal downstream slope. Top and bottom widths directly place the left
    vertices; the left-bottom angle is derived.
    """
    y_top = constants.modulus_zone_top - design.modulus_zone_height
    y_bottom = y_top - design.secondary_height
    horizontal_offset = design.secondary_right_slope_distance * sqrt(
        1.0 + design.downstream_slope**2
    )
    reference_bottom_x = (
        design_limits(y_bottom, design, constants)[1]
        if downstream_contact_x_at_bottom is None
        else downstream_contact_x_at_bottom
    )
    reference_top_x = reference_bottom_x - (
        design.downstream_slope * design.secondary_height
    )
    top_right = (reference_top_x - horizontal_offset, y_top)
    bottom_right = (reference_bottom_x - horizontal_offset, y_bottom)
    top_left = (top_right[0] - design.secondary_top_width, y_top)
    bottom_left = (bottom_right[0] - design.secondary_bottom_width, y_bottom)
    left_bottom_angle = degrees(
        atan2(
            top_left[1] - bottom_left[1],
            top_left[0] - bottom_left[0],
        )
    )
    return SecondaryZoneLayout(
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        left_bottom_angle=left_bottom_angle,
        right_horizontal_offset=horizontal_offset,
        downstream_reference_bottom=(reference_bottom_x, y_bottom),
    )


def warp_coordinate(
    x: float,
    y: float,
    design: DesignVariables,
    constants: ModelConstants = CONSTANTS,
    *,
    slope_anchor: str | None = None,
    reference_design: DesignVariables | None = None,
) -> tuple[float, float]:
    """Warp the existing dam envelope while leaving the distant foundation fixed.

    The interior is mapped by its normalized cross-section coordinate. Thin zones
    immediately outside either face are translated with that face, which preserves
    their horizontal thickness instead of scaling it with the dam width.
    """
    if y < constants.dam_base_elevation - GEOMETRY_TOLERANCE or y > constants.crest_elevation + GEOMETRY_TOLERANCE:
        return x, y
    old_left, old_right = baseline_limits(
        y, constants, reference_design=reference_design
    )
    new_left, new_right = design_limits(y, design, constants)
    if slope_anchor == "upstream":
        return x + new_left - old_left, y
    if slope_anchor == "downstream":
        return x + new_right - old_right, y
    if old_left - UPSTREAM_OUTER_FOLLOW_DISTANCE <= x < old_left:
        return x + new_left - old_left, y
    if old_left <= x <= old_right:
        fraction = (x - old_left) / (old_right - old_left)
        return new_left + fraction * (new_right - new_left), y
    if old_right < x <= old_right + 2.0:
        return x + new_right - old_right, y
    return x, y


def warp_polygon(
    polygon: Polygon,
    design: DesignVariables,
    *,
    controls: TemplateWarpControls | None = None,
    reference_design: DesignVariables | None = None,
) -> Polygon:
    def warp(value: tuple[float, float]) -> tuple[float, float]:
        coordinate = (float(value[0]), float(value[1]))
        if controls is None:
            return warp_coordinate(
                *coordinate, design, reference_design=reference_design
            )
        return warp_template_coordinate(
            *coordinate,
            design,
            controls,
            reference_design=reference_design,
        )

    exterior = [warp(value) for value in polygon.exterior.coords]
    holes = [[warp(value) for value in ring.coords] for ring in polygon.interiors]
    return Polygon(exterior, holes)


def transform_template_regions(
    original_regions: list[SourceRegion],
    design: DesignVariables,
    *,
    reference_design: DesignVariables | None = None,
) -> list[SourceRegion]:
    """Move template regions while keeping all slope-linked points coherent.

    The face slab, cushion and transition zones share points with the rockfill.
    A coordinate is therefore classified once and the same slope-anchored motion
    is reused everywhere that coordinate occurs.  This prevents a shared point
    from following the face in one region while being width-scaled in another.
    """
    regions = _merge_upstream_regions(original_regions)
    controls = template_warp_controls(regions, reference_design=reference_design)
    return [
        SourceRegion(
            warp_polygon(
                region.polygon,
                design,
                controls=controls,
                reference_design=reference_design,
            ),
            region.material_id,
            region.activation_stage,
            region.source,
        )
        for region in regions
    ]


def _merge_upstream_regions(regions: list[SourceRegion]) -> list[SourceRegion]:
    """Dissolve cushion subdivisions with the same material and activation stage.

    The template's small toe cushion and its upper strip are one material.
    Keeping their artificial interface would retain a hooked seam even after
    straightening the cushion/transition contact.
    """
    groups: dict[tuple[int, int], list[RegionCell]] = {}
    result = []
    for region in regions:
        if region.material_id not in UPSTREAM_SLOPE_MATERIALS:
            result.append(region)
            continue
        key = (region.material_id, region.activation_stage)
        groups.setdefault(key, []).append(
            RegionCell(region.polygon, region.material_id, region.activation_stage, region.source)
        )
    for group in groups.values():
        source = min(region.source for region in group)
        cells = [
            RegionCell(region.polygon, region.material_id, region.activation_stage, source)
            for region in group
        ]
        result.extend(
            SourceRegion(cell.polygon, cell.material_id, cell.activation_stage, cell.source)
            for cell in _merge_compatible_cells(cells)
        )
    return result


def _upstream_toe_corrections(
    regions: list[SourceRegion], constants: ModelConstants
) -> dict[tuple[float, float], tuple[float, float]]:
    """Continue upstream interfaces to the base instead of turning inward.

    Derive corrections from the template's actual upper edges, not point IDs
    or an assumed layer thickness. Interpolate intervening base vertices so
    the shared foundation edge remains ordered after the toe is shortened.
    """
    base_y = constants.dam_base_elevation
    corrections: dict[tuple[float, float], tuple[float, float]] = {}
    base_points: set[tuple[float, float]] = set()
    for region in _merge_upstream_regions(regions):
        if region.material_id not in UPSTREAM_SLOPE_MATERIALS:
            continue
        ring = list(region.polygon.exterior.coords)[:-1]
        base_points.update(point for point in ring if abs(point[1] - base_y) <= 1.0e-8)
        for index, corner in enumerate(ring):
            neighbors = (ring[index - 1], ring[(index + 1) % len(ring)])
            toe, upper = sorted(neighbors, key=lambda point: point[1])
            if (
                abs(toe[1] - base_y) > 1.0e-8
                or corner[1] <= base_y + GEOMETRY_TOLERANCE
                or upper[1] <= corner[1] + GEOMETRY_TOLERANCE
                or toe[0] <= corner[0]
                or upper[0] <= corner[0]
            ):
                continue
            slope = (upper[0] - corner[0]) / (upper[1] - corner[1])
            corrected = (corner[0] - slope * (corner[1] - base_y), toe[1])
            if toe in corrections and hypot(
                corrected[0] - corrections[toe][0], corrected[1] - corrections[toe][1]
            ) > TOPOLOGY_SNAP_TOLERANCE:
                raise GeometryError("upstream toe has conflicting interface continuations")
            corrections[toe] = corrected

    if not corrections:
        return corrections
    anchors = {point: point for point in (min(base_points), max(base_points))}
    anchors.update(corrections)
    ordered = sorted(anchors)
    for left, right in pairwise(ordered):
        if anchors[right][0] <= anchors[left][0]:
            raise GeometryError("straightened upstream interfaces cross at the dam base")
    for region in regions:
        for point in region.polygon.exterior.coords:
            if point in corrections or abs(point[1] - base_y) > 1.0e-8:
                continue
            for left, right in pairwise(ordered):
                if left[0] < point[0] < right[0]:
                    fraction = (point[0] - left[0]) / (right[0] - left[0])
                    corrections[point] = (
                        anchors[left][0] + fraction * (anchors[right][0] - anchors[left][0]),
                        point[1],
                    )
                    break
    return corrections


def template_warp_controls(
    regions: list[SourceRegion],
    constants: ModelConstants = CONSTANTS,
    *,
    reference_design: DesignVariables | None = None,
) -> TemplateWarpControls:
    upstream: set[tuple[float, float]] = set()
    downstream: set[tuple[float, float]] = set()
    for region in regions:
        coordinates = {
            (float(x), float(y))
            for x, y in list(region.polygon.exterior.coords)[:-1]
        }
        if region.material_id in UPSTREAM_SLOPE_MATERIALS:
            upstream.update(coordinates)
            continue
        # The upstream weathered layers share their inner boundary with the
        # face system and their outer vertices with the foundation. Move each
        # complete layer with that face: a distance cutoff can leave its outer
        # corner fixed while the adjacent corner moves past it at large slopes.
        # The same coordinate controls also move shared foundation/BC vertices;
        # distant foundation vertices retain their existing fixed mapping.
        if (
            region.material_id in UPSTREAM_WEATHERED_MATERIALS
            and region.polygon.bounds[2] < constants.dam_axis_x
        ):
            upstream.update(coordinates)
            continue
        if region.material_id != 1:
            continue

        min_x, _, max_x, _ = region.polygon.bounds
        if max_x < constants.dam_axis_x - GEOMETRY_TOLERANCE:
            upstream.update(coordinates)
        elif min_x > constants.dam_axis_x + GEOMETRY_TOLERANCE:
            downstream.update(coordinates)
        else:
            for coordinate in coordinates:
                old_left, old_right = baseline_limits(
                    coordinate[1],
                    constants,
                    reference_design=reference_design,
                )
                target = (
                    upstream
                    if abs(coordinate[0] - old_left) <= abs(coordinate[0] - old_right)
                    else downstream
                )
                target.add(coordinate)
    # Cushion/transition membership is authoritative at the narrow crest where
    # a point can be geometrically closer to the downstream face while still
    # belonging to the upstream construction system.
    downstream.difference_update(upstream)
    return TemplateWarpControls(
        frozenset(upstream),
        frozenset(downstream),
        _upstream_toe_corrections(regions, constants),
    )


def warp_template_coordinate(
    x: float,
    y: float,
    design: DesignVariables,
    controls: TemplateWarpControls,
    *,
    reference_design: DesignVariables | None = None,
) -> tuple[float, float]:
    """Apply the same template-point mapping used by rebuilt regions."""
    coordinate = (float(x), float(y))
    corrected = controls.toe_corrections.get(coordinate, coordinate)
    if coordinate in controls.upstream:
        return warp_coordinate(
            *corrected,
            design,
            slope_anchor="upstream",
            reference_design=reference_design,
        )
    if coordinate in controls.downstream:
        return warp_coordinate(
            *corrected,
            design,
            slope_anchor="downstream",
            reference_design=reference_design,
        )
    return warp_coordinate(*corrected, design, reference_design=reference_design)


def _polygon_parts(geometry: object) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        if geometry.area > GEOMETRY_AREA_EPSILON:
            yield geometry
    elif isinstance(geometry, MultiPolygon):
        for polygon in geometry.geoms:
            if polygon.area > GEOMETRY_AREA_EPSILON:
                yield polygon
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:  # type: ignore[union-attr]
            yield from _polygon_parts(part)


def _right_intersection(core: Polygon | MultiPolygon, y: float) -> float:
    min_x, _, max_x, _ = core.bounds
    cross_section = core.intersection(LineString([(min_x - 10.0, y), (max_x + 10.0, y)]))
    coordinates: list[tuple[float, float]] = []
    if isinstance(cross_section, LineString):
        coordinates.extend(cross_section.coords)
    elif hasattr(cross_section, "geoms"):
        for part in cross_section.geoms:  # type: ignore[union-attr]
            if isinstance(part, LineString):
                coordinates.extend(part.coords)
            elif isinstance(part, Point):
                coordinates.append((part.x, part.y))
    if not coordinates:
        raise GeometryError(f"dam/pressure contact boundary is absent at elevation {y:g} m")
    return max(x for x, _ in coordinates)


def _contained(name: str, geometry: BaseGeometry, core: Polygon | MultiPolygon) -> None:
    outside = geometry.difference(core.buffer(1.0e-4))
    if outside.area > GEOMETRY_AREA_EPSILON:
        raise GeometryError(f"{name} extends outside the parameterized dam domain by {outside.area:.6g} m2")


def _stage_for_band(lower: float, upper: float, constants: ModelConstants = CONSTANTS) -> int:
    midpoint = (lower + upper) / 2
    for index, (low, high) in enumerate(zip(constants.lift_boundaries, constants.lift_boundaries[1:])):
        if low - GEOMETRY_TOLERANCE <= midpoint <= high + GEOMETRY_TOLERANCE:
            return index + 2
    raise GeometryError(f"no construction stage for elevation interval [{lower:g}, {upper:g}]")


def _split_core_material(
    geometry: Polygon | MultiPolygon,
    material_id: int,
    source: str,
    constants: ModelConstants = CONSTANTS,
) -> list[SourceRegion]:
    result: list[SourceRegion] = []
    min_x, _, max_x, _ = geometry.bounds
    for lower, upper in zip(constants.lift_boundaries, constants.lift_boundaries[1:]):
        clipped = geometry.intersection(box(min_x - 1.0, lower, max_x + 1.0, upper))
        for part in _polygon_parts(clipped):
            result.append(SourceRegion(part, material_id, _stage_for_band(lower, upper, constants), source))
    return result


def _node_and_classify(sources: list[SourceRegion]) -> list[RegionCell]:
    sources = _snap_source_vertices(sources, TOPOLOGY_SNAP_TOLERANCE)
    noded_boundaries = unary_union([source.polygon.boundary for source in sources])
    raw_cells = list(polygonize(noded_boundaries))
    result: list[RegionCell] = []
    for raw_polygon in raw_cells:
        normalized = set_precision(raw_polygon, grid_size=1.0e-9, mode="valid_output")
        for polygon in _polygon_parts(normalized):
            if polygon.area <= GEOMETRY_AREA_EPSILON:
                continue
            point = polygon.representative_point()
            candidates = [
                source
                for source in sources
                if source.polygon.buffer(TOPOLOGY_SNAP_TOLERANCE * 2).covers(point)
            ]
            if not candidates:
                continue
            # Boundaries may produce microscopic ambiguous cells. Select the source
            # with the greatest actual coverage and then use stable metadata ordering.
            source = max(
                candidates,
                key=lambda item: (
                    polygon.intersection(item.polygon).area,
                    -item.activation_stage,
                    -item.material_id,
                    item.source,
                ),
            )
            result.append(
                RegionCell(orient(polygon, sign=1.0), source.material_id, source.activation_stage, source.source)
            )
    result.sort(key=_cell_sort_key)
    return result


def _cell_sort_key(cell: RegionCell) -> tuple[int, int, float, float, float]:
    return (
        cell.activation_stage,
        cell.material_id,
        round(cell.polygon.centroid.y, 8),
        round(cell.polygon.centroid.x, 8),
        round(cell.polygon.area, 8),
    )


def _merge_compatible_cells(cells: list[RegionCell]) -> list[RegionCell]:
    """Dissolve adjacent cells that carry identical model metadata."""
    grouped: dict[tuple[int, int, str], list[RegionCell]] = {}
    for cell in cells:
        key = (cell.material_id, cell.activation_stage, cell.source)
        grouped.setdefault(key, []).append(cell)

    merged_cells: list[RegionCell] = []
    for (material_id, activation_stage, source), group in grouped.items():
        merged = unary_union([cell.polygon for cell in group])
        for polygon in _polygon_parts(merged):
            # GeoStudio regions cannot contain holes. If dissolving a connected
            # component would create one, retain its original atomic cells.
            if polygon.interiors:
                merged_cells.extend(
                    cell
                    for cell in group
                    if polygon.covers(cell.polygon.representative_point())
                )
                continue
            merged_cells.append(
                RegionCell(
                    orient(polygon, sign=1.0),
                    material_id,
                    activation_stage,
                    source,
                )
            )

    merged_cells.sort(key=_cell_sort_key)
    return merged_cells


def _snap_source_vertices(sources: list[SourceRegion], tolerance: float) -> list[SourceRegion]:
    priorities: dict[tuple[float, float], int] = {}
    for source in sources:
        priority = 0 if source.source.startswith("template-region-") else 1
        for coordinate in source.polygon.exterior.coords:
            value = (float(coordinate[0]), float(coordinate[1]))
            priorities[value] = min(priority, priorities.get(value, priority))
        for ring in source.polygon.interiors:
            for coordinate in ring.coords:
                value = (float(coordinate[0]), float(coordinate[1]))
                priorities[value] = min(priority, priorities.get(value, priority))

    coordinates = sorted(priorities)
    parent = list(range(len(coordinates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, first in enumerate(coordinates):
        for right in range(left + 1, len(coordinates)):
            second = coordinates[right]
            if second[0] - first[0] > tolerance:
                break
            if hypot(second[0] - first[0], second[1] - first[1]) <= tolerance:
                union(left, right)

    groups: dict[int, list[tuple[float, float]]] = {}
    for index, coordinate in enumerate(coordinates):
        groups.setdefault(find(index), []).append(coordinate)
    replacement: dict[tuple[float, float], tuple[float, float]] = {}
    for group in groups.values():
        representative = min(group, key=lambda coordinate: (priorities[coordinate], coordinate[1], coordinate[0]))
        for coordinate in group:
            replacement[coordinate] = representative

    snapped: list[SourceRegion] = []
    for source in sources:
        exterior = [replacement[(float(x), float(y))] for x, y in source.polygon.exterior.coords]
        holes = [
            [replacement[(float(x), float(y))] for x, y in ring.coords]
            for ring in source.polygon.interiors
        ]
        polygon = Polygon(exterior, holes)
        if polygon.is_valid and polygon.area > GEOMETRY_AREA_EPSILON:
            snapped.append(SourceRegion(polygon, source.material_id, source.activation_stage, source.source))
    return snapped


def build_geometry(
    original_regions: list[SourceRegion],
    design: DesignVariables,
    constants: ModelConstants = CONSTANTS,
    *,
    reference_design: DesignVariables | None = None,
) -> GeometryBuild:
    errors = design.validate_bounds()
    if errors:
        raise GeometryError("; ".join(errors))

    transformed = transform_template_regions(
        original_regions,
        design,
        reference_design=reference_design,
    )
    invalid = [
        region.source
        for region in transformed
        if not region.polygon.is_valid
        or region.polygon.area <= GEOMETRY_AREA_EPSILON
    ]
    if invalid:
        details = ", ".join(invalid[:5])
        raise GeometryError(f"template warp produced invalid regions: {details}")

    core_domain = unary_union([region.polygon for region in transformed if region.material_id in CORE_MATERIALS])
    if core_domain.is_empty or not core_domain.is_valid:
        raise GeometryError(f"invalid primary dam domain: {explain_validity(core_domain)}")

    y_top = constants.modulus_zone_top - design.modulus_zone_height
    y_bottom = y_top - design.secondary_height
    if y_bottom <= constants.dam_base_elevation + GEOMETRY_TOLERANCE:
        raise GeometryError(
            f"secondary zone bottom elevation {y_bottom:g} m must be above dam base {constants.dam_base_elevation:g} m"
        )

    contact_bottom = _right_intersection(core_domain, y_bottom)
    layout = secondary_zone_layout(
        design,
        constants,
        downstream_contact_x_at_bottom=contact_bottom,
    )
    top_left = layout.top_left
    top_right = layout.top_right
    bottom_left = layout.bottom_left
    bottom_right = layout.bottom_right
    lower_angle, upper_angle = constants.secondary_left_angle_bounds
    if not lower_angle < layout.left_bottom_angle < upper_angle:
        raise GeometryError(
            "derived secondary left-bottom angle "
            f"{layout.left_bottom_angle:.6g} degrees must be strictly between "
            f"{lower_angle:g} and {upper_angle:g} degrees"
        )
    left_side_dx_per_vertical = (
        top_left[0] - bottom_left[0]
    ) / design.secondary_height
    contact_base = _right_intersection(core_domain, constants.dam_base_elevation)
    min_x, _, max_x, _ = core_domain.bounds
    drainage_left_base = (
        bottom_left[0] - left_side_dx_per_vertical * (y_bottom - constants.dam_base_elevation),
        constants.dam_base_elevation,
    )

    secondary = Polygon([top_left, top_right, bottom_right, bottom_left])
    # Extend the drainage outline beyond the core and then clip it back to the
    # exact core boundary. Joining contact_bottom directly to contact_base makes
    # a straight chord, while the transformed downstream face can be a nearly
    # straight polyline. The microscopic gap between the two otherwise becomes
    # an isolated primary-material sliver.
    outside_x = max_x + 1.0
    drainage_outline = Polygon(
        [
            bottom_left,
            bottom_right,
            (outside_x, y_bottom),
            (outside_x, constants.dam_base_elevation),
            drainage_left_base,
        ]
    )
    drainage = drainage_outline.intersection(core_domain)
    for name, polygon in (("secondary rockfill zone", secondary), ("drainage zone", drainage)):
        if polygon.is_empty:
            raise GeometryError(f"{name} is empty")
        if not polygon.is_valid:
            raise GeometryError(f"{name} is invalid: {explain_validity(polygon)}")
        if polygon.area < MIN_AREA:
            raise GeometryError(
                f"{name} area {polygon.area:.6g} m2 is below the "
                f"{MIN_AREA:.6g} m2 minimum"
            )
        _contained(name, polygon, core_domain)

    modulus = core_domain.intersection(box(min_x - 1.0, y_top, max_x + 1.0, constants.modulus_zone_top))
    _contained("modulus-increased zone", modulus, core_domain)  # type: ignore[arg-type]
    overlap = secondary.intersection(drainage).area
    if overlap > GEOMETRY_AREA_EPSILON:
        raise GeometryError(f"secondary and drainage zones overlap by {overlap:.6g} m2")
    primary = core_domain.difference(unary_union([modulus, secondary, drainage]))

    dynamic: list[SourceRegion] = []
    for geometry, material_id, name in (
        (primary, 4, "primary"),
        (secondary, 5, "secondary"),
        (drainage, 6, "drainage"),
        (modulus, 12, "modulus"),
    ):
        if geometry.is_empty:
            raise GeometryError(f"{name} material region is empty")
        dynamic.extend(_split_core_material(geometry, material_id, name, constants))

    non_core = [region for region in transformed if region.material_id not in CORE_MATERIALS]
    sources = non_core + dynamic
    cells = _merge_compatible_cells(_node_and_classify(sources))
    if not cells:
        raise GeometryError("polygon reconstruction produced no regions")

    for index, left in enumerate(cells):
        if not left.polygon.is_valid:
            raise GeometryError(f"rebuilt region {index + 1} is invalid")
        if left.polygon.area < MIN_AREA:
            raise GeometryError(
                f"rebuilt region {index + 1} area {left.polygon.area:.6g} m2 "
                f"is below the {MIN_AREA:.6g} m2 minimum"
            )
        for right in cells[index + 1 :]:
            if left.polygon.intersection(right.polygon).area > GEOMETRY_AREA_EPSILON:
                raise GeometryError("rebuilt material regions overlap")
    _validate_shared_edges(cells)
    shortest = min(
        hypot(x2 - x1, y2 - y1)
        for cell in cells
        for (x1, y1), (x2, y2) in zip(cell.polygon.exterior.coords, cell.polygon.exterior.coords[1:])
    )
    if shortest < MIN_EDGE_LENGTH:
        raise GeometryError(f"rebuilt geometry contains a {shortest:.6g} m short edge")

    crest_left, crest_right = design_limits(constants.crest_elevation, design, constants)
    upstream_toe, downstream_toe = design_limits(constants.dam_base_elevation, design, constants)
    derived = DerivedGeometry(
        crest_left=(crest_left, constants.crest_elevation),
        crest_right=(crest_right, constants.crest_elevation),
        upstream_toe=(upstream_toe, constants.dam_base_elevation),
        downstream_toe=(downstream_toe, constants.dam_base_elevation),
        modulus_bottom_elevation=y_top,
        secondary_top_left=top_left,
        secondary_top_right=top_right,
        secondary_bottom_left=bottom_left,
        secondary_bottom_right=bottom_right,
        secondary_left_bottom_angle=layout.left_bottom_angle,
        secondary_right_slope_distance=design.secondary_right_slope_distance,
        secondary_right_horizontal_offset=layout.right_horizontal_offset,
        secondary_downstream_reference_bottom=layout.downstream_reference_bottom,
        drainage_bottom_left=drainage_left_base,
        drainage_top_right=(contact_bottom, y_bottom),
        drainage_bottom_right=(contact_base, constants.dam_base_elevation),
    )
    return GeometryBuild(cells=cells, derived=derived, core_domain=core_domain)


def _validate_shared_edges(cells: list[RegionCell]) -> None:
    coordinates = {
        (float(x), float(y))
        for cell in cells
        for x, y in list(cell.polygon.exterior.coords)[:-1]
    }
    counts: dict[tuple[tuple[float, float], tuple[float, float]], int] = {}
    for cell in cells:
        for first, second in zip(cell.polygon.exterior.coords, cell.polygon.exterior.coords[1:]):
            first_value = (float(first[0]), float(first[1]))
            second_value = (float(second[0]), float(second[1]))
            segment = LineString([first_value, second_value])
            interior = [
                coordinate
                for coordinate in coordinates
                if coordinate not in {first_value, second_value}
                and segment.distance(Point(coordinate)) <= 1.0e-8
                and 0 < segment.project(Point(coordinate), normalized=True) < 1
            ]
            ordered = [first_value, *sorted(interior, key=lambda value: segment.project(Point(value))), second_value]
            for left, right in pairwise(ordered):
                edge = tuple(sorted((left, right)))
                counts[edge] = counts.get(edge, 0) + 1

    domain_boundary = unary_union([cell.polygon for cell in cells]).boundary
    for (first, second), count in counts.items():
        if count > 2:
            raise GeometryError("more than two regions share a rebuilt edge")
        if count == 1:
            midpoint = Point((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
            if domain_boundary.distance(midpoint) > GEOMETRY_TOLERANCE:
                raise GeometryError("rebuilt regions contain a non-conforming internal edge")


def cell_report(cells: list[RegionCell]) -> dict[str, object]:
    by_material: dict[str, float] = {}
    by_stage: dict[str, int] = {}
    for cell in cells:
        material = str(cell.material_id)
        stage = str(cell.activation_stage)
        by_material[material] = by_material.get(material, 0.0) + cell.polygon.area
        by_stage[stage] = by_stage.get(stage, 0) + 1
    minimum_edge = min(
        hypot(x2 - x1, y2 - y1)
        for cell in cells
        for (x1, y1), (x2, y2) in zip(cell.polygon.exterior.coords, cell.polygon.exterior.coords[1:])
    )
    return {
        "region_count": len(cells),
        "area_by_material": {key: round(value, 8) for key, value in sorted(by_material.items(), key=lambda x: int(x[0]))},
        "region_count_by_activation_stage": dict(sorted(by_stage.items(), key=lambda x: int(x[0]))),
        "minimum_edge_length": minimum_edge,
        "all_regions_valid": all(cell.polygon.is_valid for cell in cells),
        "overlap_area": 0.0,
    }
