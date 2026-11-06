from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from shapely.geometry import Point, Polygon

from .runtime import GeoStudioRuntimeError


RESULTS_SCHEMA_VERSION = 1
DEFAULT_ANALYSIS_NAME = "正常蓄水位应力分布"
DEFAULT_ANALYSIS_STEP = 3
DEFAULT_CVAR_ALPHA = 0.995
DEFAULT_COORDINATE_TOLERANCE = 1.0e-5
CREST_MATERIAL_NAME = "坝顶"
EXCLUDED_MATERIAL_NAME_KEYWORDS: tuple[str, ...] = (
    "面板",
    "坝顶",
    "地基",
    "垫层",
    "过渡",
    "风化",
)
MATERIAL_FILTER_VERSION = 1

DISPLACEMENT_METRIC_FIELDS: tuple[str, ...] = (
    "crest_max_settlement_m",
    "dam_max_settlement_m",
    "dam_max_horizontal_displacement_m",
)

METRIC_FIELDS: tuple[str, ...] = (
    *DISPLACEMENT_METRIC_FIELDS,
    "dam_sigma1_effective_stress_cvar98_kpa",
    "dam_sigma3_effective_stress_cvar98_kpa",
)


def metric_fields(alpha: float = DEFAULT_CVAR_ALPHA) -> tuple[str, ...]:
    """Return the five response column names for a selected CVaR level."""
    _validate_alpha(alpha)
    token = _alpha_token(alpha)
    return (
        "crest_max_settlement_m",
        "dam_max_settlement_m",
        "dam_max_horizontal_displacement_m",
        f"dam_sigma1_effective_stress_cvar{token}_kpa",
        f"dam_sigma3_effective_stress_cvar{token}_kpa",
    )


class ResponseExtractionError(GeoStudioRuntimeError):
    """Raised when a solved GeoStudio model cannot yield valid responses."""


@dataclass(frozen=True, slots=True)
class NodeResult:
    node_id: int
    x: float
    y: float
    x_displacement: float
    y_displacement: float


@dataclass(frozen=True, slots=True)
class ElementNodeMembership:
    element_id: int
    node_id: int


@dataclass(frozen=True, slots=True)
class GaussPointResult:
    gauss_point_id: int
    element_id: int
    x: float
    y: float
    max_effective_stress: float
    min_effective_stress: float


@dataclass(frozen=True, slots=True)
class ResponseMeshData:
    nodes: tuple[NodeResult, ...]
    element_nodes: tuple[ElementNodeMembership, ...]
    gauss_points: tuple[GaussPointResult, ...]
    queried_element_node_rows: int
    ignored_element_node_rows: int


@dataclass(frozen=True, slots=True)
class MaterialRegion:
    region_id: int
    material_id: int
    material_name: str
    polygon: Polygon


@dataclass(frozen=True, slots=True)
class MaterialMetadata:
    regions: tuple[MaterialRegion, ...]
    names: Mapping[int, str]
    excluded_ids: frozenset[int]
    crest_id: int


@dataclass(frozen=True, slots=True)
class MaterialClassification:
    node_material_ids: Mapping[int, tuple[int, ...]]
    material_names: Mapping[int, str]
    excluded_material_ids: frozenset[int]
    excluded_node_ids: frozenset[int]
    element_material_ids: Mapping[int, int]
    crest_material_id: int


@dataclass(frozen=True, slots=True)
class WeightedStressPoint:
    gauss_point_id: int
    element_id: int
    x: float
    y: float
    sigma1_kpa: float
    sigma3_kpa: float
    area_weight_m2: float
    material_id: int
    material_name: str


@dataclass(frozen=True, slots=True)
class WeightedTailResult:
    alpha: float
    var_kpa: float
    cvar_kpa: float
    maximum_kpa: float
    total_area_m2: float
    target_tail_area_m2: float
    selected_tail_area_m2: float
    contributing_gauss_points: int
    contributing_elements: int
    weighted_centroid_x_m: float
    weighted_centroid_y_m: float
    points: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class ProjectResponses:
    design_id: str
    parameter_hash: str
    analysis_name: str
    step: int
    cvar_alpha: float
    project: dict[str, object]
    material_filter: dict[str, object]
    scope: dict[str, object]
    counts: dict[str, int]
    metrics: dict[str, float]
    units: dict[str, str]
    locations: dict[str, object]
    stress_tail_details: dict[str, object]
    extracted_at: str
    schema_version: int = RESULTS_SCHEMA_VERSION
    status: str = "extracted"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_project_responses(
    project_path: Path,
    parameters_payload: Mapping[str, object],
    *,
    analysis_name: str = DEFAULT_ANALYSIS_NAME,
    step: int = DEFAULT_ANALYSIS_STEP,
    cvar_alpha: float = DEFAULT_CVAR_ALPHA,
    coordinate_tolerance: float = DEFAULT_COORDINATE_TOLERANCE,
) -> ProjectResponses:
    """Extract the three displacement and two stress-CVaR responses."""
    _validate_alpha(cvar_alpha)
    if step < 0:
        raise ResponseExtractionError(
            "analysis step must be a non-negative integer"
        )
    if not math.isfinite(coordinate_tolerance) or coordinate_tolerance <= 0:
        raise ResponseExtractionError(
            "coordinate tolerance must be a positive finite number"
        )

    project_path = project_path.resolve()
    if not project_path.is_file():
        raise ResponseExtractionError(
            f"GeoStudio project was not found: {project_path}"
        )
    design_id, parameter_hash, derived = _parameter_metadata(parameters_payload)
    metadata = load_material_metadata(project_path, analysis_name)
    mesh = query_response_mesh(
        project_path,
        analysis_name=analysis_name,
        step=step,
    )
    classification = classify_materials(mesh, metadata)

    displacement_metrics, displacement_counts, displacement_scope, locations = (
        calculate_displacement_metrics(
            mesh.nodes,
            derived,
            classification,
            coordinate_tolerance=coordinate_tolerance,
        )
    )
    stress_points, stress_counts, stress_scope = build_weighted_stress_points(
        mesh,
        derived,
        classification,
    )
    sigma1_tail = weighted_upper_tail(
        stress_points,
        value=lambda point: point.sigma1_kpa,
        alpha=cvar_alpha,
    )
    sigma3_tail = weighted_upper_tail(
        stress_points,
        value=lambda point: point.sigma3_kpa,
        alpha=cvar_alpha,
    )
    token = _alpha_token(cvar_alpha)
    metrics = {
        **displacement_metrics,
        f"dam_sigma1_effective_stress_cvar{token}_kpa": sigma1_tail.cvar_kpa,
        f"dam_sigma3_effective_stress_cvar{token}_kpa": sigma3_tail.cvar_kpa,
    }
    _require_finite_values(metrics.values(), "calculated metrics")

    counts = {
        **displacement_counts,
        **stress_counts,
        "material_mapped_elements": len(classification.element_material_ids),
        "queried_element_node_rows": mesh.queried_element_node_rows,
        "ignored_element_node_placeholder_rows": mesh.ignored_element_node_rows,
        "sigma1_tail_gauss_points": sigma1_tail.contributing_gauss_points,
        "sigma1_tail_elements": sigma1_tail.contributing_elements,
        "sigma3_tail_gauss_points": sigma3_tail.contributing_gauss_points,
        "sigma3_tail_elements": sigma3_tail.contributing_elements,
    }
    material_filter = _material_filter_payload(classification)
    return ProjectResponses(
        design_id=design_id,
        parameter_hash=parameter_hash,
        analysis_name=analysis_name,
        step=step,
        cvar_alpha=cvar_alpha,
        project=project_provenance(project_path),
        material_filter=material_filter,
        scope={
            "dam_body_displacements": displacement_scope,
            "dam_body_stresses": stress_scope,
        },
        counts=counts,
        metrics=metrics,
        units={field: "m" if field.endswith("_m") else "kPa" for field in metrics},
        locations=locations,
        stress_tail_details={
            "sigma1_max_effective_stress": asdict(sigma1_tail),
            "sigma3_min_effective_stress": asdict(sigma3_tail),
            "mapping": {
                "sigma1": "GeoStudio Gauss field eMaxEffectiveStress",
                "sigma3": "GeoStudio Gauss field eMinEffectiveStress",
                "tail_direction": (
                    "upper numeric tail for both fields; values are not converted "
                    "to absolute magnitudes"
                ),
            },
        },
        extracted_at=datetime.now(UTC).isoformat(),
    )


def query_response_mesh(
    project_path: Path,
    *,
    analysis_name: str = DEFAULT_ANALYSIS_NAME,
    step: int = DEFAULT_ANALYSIS_STEP,
) -> ResponseMeshData:
    """Read nodes, active element connectivity, and principal stresses via GSI."""
    project_path = project_path.resolve()
    _assert_project_not_open(project_path)
    try:
        import gsi  # type: ignore[import-not-found]
    except ImportError as error:
        raise ResponseExtractionError(
            "the official gsi package is not installed; run "
            "scripts/install_geostudio_api.ps1"
        ) from error

    project = None
    close_error: Exception | None = None
    try:
        project = gsi.OpenProject(str(project_path))
        project.LoadResults(gsi.LoadResultsRequest(analysis=analysis_name))

        node_params = {
            "node_id": gsi.DataParamType.eNodeNum,
            "x": gsi.DataParamType.eXCoord,
            "y": gsi.DataParamType.eYCoord,
            "x_displacement": gsi.DataParamType.eXDisplacement,
            "y_displacement": gsi.DataParamType.eYDisplacement,
        }
        _validate_table_metadata(
            project,
            gsi,
            analysis_name,
            gsi.ResultType.Nodes,
            node_params,
            {
                "x": "m",
                "y": "m",
                "x_displacement": "m",
                "y_displacement": "m",
            },
            "Nodes",
        )
        nodes = nodes_from_columns(
            _query_columns(
                project,
                gsi,
                analysis_name,
                step,
                gsi.ResultType.Nodes,
                node_params,
                "Nodes",
            )
        )

        element_params = {
            "element_id": gsi.DataParamType.eElemNum,
            "node_id": gsi.DataParamType.eNodeNum,
        }
        _validate_table_metadata(
            project,
            gsi,
            analysis_name,
            gsi.ResultType.ElementNodes,
            element_params,
            {},
            "ElementNodes",
        )
        element_columns = _query_columns(
            project,
            gsi,
            analysis_name,
            step,
            gsi.ResultType.ElementNodes,
            element_params,
            "ElementNodes",
        )
        element_nodes, ignored_rows = element_nodes_from_columns(element_columns)

        gauss_params = {
            "element_id": gsi.DataParamType.eElemNum,
            "gauss_point_id": gsi.DataParamType.eGaussPointNum,
            "x": gsi.DataParamType.eXCoord,
            "y": gsi.DataParamType.eYCoord,
            "max_effective_stress": gsi.DataParamType.eMaxEffectiveStress,
            "min_effective_stress": gsi.DataParamType.eMinEffectiveStress,
        }
        _validate_table_metadata(
            project,
            gsi,
            analysis_name,
            gsi.ResultType.Gauss,
            gauss_params,
            {
                "x": "m",
                "y": "m",
                "max_effective_stress": "kPa",
                "min_effective_stress": "kPa",
            },
            "Gauss",
        )
        gauss_points = gauss_points_from_columns(
            _query_columns(
                project,
                gsi,
                analysis_name,
                step,
                gsi.ResultType.Gauss,
                gauss_params,
                "Gauss",
            )
        )
    except ResponseExtractionError:
        raise
    except Exception as error:
        raise ResponseExtractionError(
            f"GeoStudio result extraction failed: {error}"
        ) from error
    finally:
        if project is not None:
            try:
                project.Close()
            except Exception as error:  # GSI exposes implementation-specific errors
                close_error = error
    if close_error is not None:
        raise ResponseExtractionError(
            f"GeoStudio result session could not be closed cleanly: {close_error}"
        ) from close_error
    return ResponseMeshData(
        tuple(nodes),
        tuple(element_nodes),
        tuple(gauss_points),
        len(element_columns["element_id"]),
        ignored_rows,
    )


def load_material_metadata(
    project_path: Path,
    analysis_name: str,
) -> MaterialMetadata:
    """Read the active material polygons for one analysis from the GSZ XML."""
    try:
        with zipfile.ZipFile(project_path) as archive:
            names = [
                name
                for name in archive.namelist()
                if "/" not in name.rstrip("/") and name.lower().endswith(".xml")
            ]
            if len(names) != 1:
                raise ResponseExtractionError(
                    "GeoStudio project must contain exactly one root XML model"
                )
            root = ET.fromstring(archive.read(names[0]))
    except ResponseExtractionError:
        raise
    except (OSError, ET.ParseError, zipfile.BadZipFile) as error:
        raise ResponseExtractionError(
            f"GeoStudio project material metadata could not be read: {error}"
        ) from error

    analyses = [
        item
        for item in root.findall("./Analyses/Analysis")
        if item.findtext("Name") == analysis_name
    ]
    if len(analyses) != 1:
        raise ResponseExtractionError(
            f"project XML does not contain exactly one analysis named {analysis_name!r}"
        )
    analysis_id = _xml_positive_int(analyses[0].findtext("ID"), "analysis ID")
    contexts = [
        item
        for item in root.findall("./Contexts/Context")
        if item.findtext("AnalysisID") == str(analysis_id)
    ]
    if len(contexts) != 1:
        raise ResponseExtractionError(
            f"project XML has no unique material context for analysis {analysis_name!r}"
        )
    material_uses = contexts[0].find("GeometryUsesMaterials")
    if material_uses is None:
        raise ResponseExtractionError(
            f"analysis {analysis_name!r} has no material assignments"
        )

    material_names: dict[int, str] = {}
    for material in root.findall("./Materials/Material"):
        material_id = _xml_positive_int(material.findtext("ID"), "material ID")
        material_name = (material.findtext("Name") or "").strip()
        if not material_name:
            raise ResponseExtractionError(f"material ID {material_id} has no name")
        if material_id in material_names:
            raise ResponseExtractionError(
                f"project XML contains duplicate material ID {material_id}"
            )
        material_names[material_id] = material_name

    crest_ids = [
        material_id
        for material_id, name in material_names.items()
        if name == CREST_MATERIAL_NAME
    ]
    if len(crest_ids) != 1:
        raise ResponseExtractionError(
            f"project must contain exactly one material named {CREST_MATERIAL_NAME!r}"
        )
    crest_id = crest_ids[0]
    excluded_ids = frozenset(
        material_id
        for material_id, name in material_names.items()
        if any(keyword in name for keyword in EXCLUDED_MATERIAL_NAME_KEYWORDS)
    )

    assignments: dict[int, int] = {}
    for material_use in material_uses:
        reference = material_use.attrib.get("ID", "")
        if not reference.startswith("Regions-"):
            continue
        try:
            region_id = int(reference.removeprefix("Regions-"))
            material_id = int(material_use.attrib["Entry"])
        except (KeyError, ValueError) as error:
            raise ResponseExtractionError(
                f"invalid material assignment {material_use.attrib!r}"
            ) from error
        if region_id <= 0 or material_id <= 0:
            raise ResponseExtractionError(
                f"invalid material assignment {material_use.attrib!r}"
            )
        assignments[region_id] = material_id
    if crest_id not in assignments.values():
        raise ResponseExtractionError(
            f"material {CREST_MATERIAL_NAME!r} is not assigned to an active region "
            f"in analysis {analysis_name!r}"
        )

    point_coordinates: dict[int, tuple[float, float]] = {}
    for point in root.findall("./Geometries/Geometry/Points/Point"):
        try:
            point_id = int(point.attrib["ID"])
            coordinate = (float(point.attrib["X"]), float(point.attrib["Y"]))
        except (KeyError, ValueError) as error:
            raise ResponseExtractionError(
                f"invalid geometry point {point.attrib!r}"
            ) from error
        _require_finite_values(coordinate, f"geometry point {point_id}")
        point_coordinates[point_id] = coordinate

    regions: list[MaterialRegion] = []
    for region in root.findall("./Geometries/Geometry/Regions/Region"):
        region_id = _xml_positive_int(region.findtext("ID"), "region ID")
        if region_id not in assignments:
            continue
        try:
            point_ids = [
                int(value)
                for value in (region.findtext("PointIDs") or "").split(",")
                if value
            ]
            polygon = Polygon([point_coordinates[point_id] for point_id in point_ids])
        except (KeyError, ValueError) as error:
            raise ResponseExtractionError(
                f"region {region_id} has invalid point references"
            ) from error
        if not polygon.is_valid or polygon.area <= 0:
            raise ResponseExtractionError(
                f"region {region_id} does not form a valid material polygon"
            )
        material_id = assignments[region_id]
        if material_id not in material_names:
            raise ResponseExtractionError(
                f"region {region_id} references unknown material ID {material_id}"
            )
        regions.append(
            MaterialRegion(
                region_id,
                material_id,
                material_names[material_id],
                polygon,
            )
        )
    if not regions:
        raise ResponseExtractionError(
            f"analysis {analysis_name!r} has no active polygonal material regions"
        )
    return MaterialMetadata(
        tuple(regions), material_names, excluded_ids, crest_id
    )


def classify_materials(
    mesh: ResponseMeshData,
    metadata: MaterialMetadata,
) -> MaterialClassification:
    """Map active elements to material regions and propagate membership to nodes."""
    node_lookup = {node.node_id: node for node in mesh.nodes}
    if len(node_lookup) != len(mesh.nodes):
        raise ResponseExtractionError("Nodes results contain duplicate node IDs")
    element_nodes = element_node_sequences(mesh.element_nodes)
    node_material_sets: dict[int, set[int]] = defaultdict(set)
    element_material_ids: dict[int, int] = {}

    for element_id, node_ids in sorted(element_nodes.items()):
        missing = [node_id for node_id in node_ids if node_id not in node_lookup]
        if missing:
            raise ResponseExtractionError(
                f"element {element_id} references missing result node {missing[0]}"
            )
        centroid = Point(
            sum(node_lookup[node_id].x for node_id in node_ids) / len(node_ids),
            sum(node_lookup[node_id].y for node_id in node_ids) / len(node_ids),
        )
        matches = [
            region for region in metadata.regions if region.polygon.covers(centroid)
        ]
        if len(matches) != 1:
            matched = ", ".join(str(region.region_id) for region in matches)
            detail = f"matched regions {matched}" if matched else "matched no region"
            raise ResponseExtractionError(
                f"element {element_id} centroid ({centroid.x:.12g}, "
                f"{centroid.y:.12g}) {detail}"
            )
        material_id = matches[0].material_id
        element_material_ids[element_id] = material_id
        for node_id in node_ids:
            node_material_sets[node_id].add(material_id)

    node_material_ids = {
        node_id: tuple(sorted(material_ids))
        for node_id, material_ids in node_material_sets.items()
    }
    excluded_node_ids = frozenset(
        node_id
        for node_id, material_ids in node_material_ids.items()
        if metadata.excluded_ids.intersection(material_ids)
    )
    return MaterialClassification(
        node_material_ids=node_material_ids,
        material_names=dict(metadata.names),
        excluded_material_ids=metadata.excluded_ids,
        excluded_node_ids=excluded_node_ids,
        element_material_ids=element_material_ids,
        crest_material_id=metadata.crest_id,
    )


def calculate_displacement_metrics(
    nodes: Sequence[NodeResult],
    derived_coordinates: Mapping[str, object],
    classification: MaterialClassification,
    *,
    coordinate_tolerance: float = DEFAULT_COORDINATE_TOLERANCE,
) -> tuple[dict[str, float], dict[str, int], dict[str, object], dict[str, object]]:
    body_polygon, polygon_points = _dam_body_polygon(derived_coordinates)
    selection_polygon = body_polygon.buffer(coordinate_tolerance)
    body_before_filter = sorted(
        (node for node in nodes if selection_polygon.covers(Point(node.x, node.y))),
        key=lambda node: node.node_id,
    )
    if not body_before_filter:
        raise ResponseExtractionError("no result nodes fall inside the dam envelope")
    missing = [
        node.node_id
        for node in body_before_filter
        if node.node_id not in classification.node_material_ids
    ]
    if missing:
        preview = ", ".join(str(value) for value in missing[:10])
        raise ResponseExtractionError(
            "dam-envelope nodes have no incident-element material mapping: " + preview
        )
    body_nodes = [
        node
        for node in body_before_filter
        if node.node_id not in classification.excluded_node_ids
    ]
    if not body_nodes:
        raise ResponseExtractionError(
            "no dam-body nodes remain after excluded-material filtering"
        )
    crest_nodes = [
        node
        for node in nodes
        if classification.crest_material_id
        in classification.node_material_ids.get(node.node_id, ())
    ]
    if not crest_nodes:
        raise ResponseExtractionError(
            f"active material {CREST_MATERIAL_NAME!r} has no result nodes"
        )

    crest_node = min(crest_nodes, key=lambda node: (node.y_displacement, node.node_id))
    settlement_node = min(
        body_nodes, key=lambda node: (node.y_displacement, node.node_id)
    )
    horizontal_node = min(
        body_nodes, key=lambda node: (-abs(node.x_displacement), node.node_id)
    )
    metrics = {
        "crest_max_settlement_m": max(0.0, -crest_node.y_displacement),
        "dam_max_settlement_m": max(0.0, -settlement_node.y_displacement),
        "dam_max_horizontal_displacement_m": abs(horizontal_node.x_displacement),
    }
    counts = {
        "queried_nodes": len(nodes),
        "crest_material_nodes": len(crest_nodes),
        "dam_body_nodes_before_material_filter": len(body_before_filter),
        "excluded_material_nodes_in_dam_envelope": len(body_before_filter)
        - len(body_nodes),
        "dam_body_nodes": len(body_nodes),
    }
    scope = {
        "definition": "main_dam_envelope_with_material_exclusions",
        "polygon": [[x, y] for x, y in polygon_points],
        "coordinate_tolerance_m": coordinate_tolerance,
        "crest_definition": "all nodes incident to the exact 坝顶 material",
        "body_node_filter": (
            "exclude node if any incident element uses an excluded material"
        ),
    }
    locations = {
        "crest_max_settlement": _node_location(crest_node, classification),
        "dam_max_settlement": _node_location(settlement_node, classification),
        "dam_max_horizontal_displacement": _node_location(
            horizontal_node, classification
        ),
    }
    return metrics, counts, scope, locations


def build_weighted_stress_points(
    mesh: ResponseMeshData,
    derived_coordinates: Mapping[str, object],
    classification: MaterialClassification,
) -> tuple[list[WeightedStressPoint], dict[str, int], dict[str, object]]:
    """Allocate in-scope element area equally among its in-scope Gauss points."""
    scope_polygon, scope_coordinates = _dam_body_polygon(derived_coordinates)
    tolerance = max(scope_polygon.length * 1.0e-10, 1.0e-7)
    buffered_scope = scope_polygon.buffer(tolerance)
    node_lookup = {node.node_id: node for node in mesh.nodes}
    element_node_ids = element_node_sequences(mesh.element_nodes)
    gauss_by_element: dict[int, list[GaussPointResult]] = defaultdict(list)
    for point in mesh.gauss_points:
        gauss_by_element[point.element_id].append(point)

    weighted: list[WeightedStressPoint] = []
    eligible_elements = 0
    clipped_elements = 0
    for element_id, gauss_points in sorted(gauss_by_element.items()):
        material_id = classification.element_material_ids.get(element_id)
        if material_id is None:
            raise ResponseExtractionError(
                f"Gauss results reference unmapped element {element_id}"
            )
        if material_id in classification.excluded_material_ids:
            continue
        node_ids = element_node_ids.get(element_id)
        if node_ids is None:
            raise ResponseExtractionError(
                f"Gauss results reference element {element_id} without connectivity"
            )
        try:
            polygon = Polygon(
                [(node_lookup[node_id].x, node_lookup[node_id].y) for node_id in node_ids]
            )
        except KeyError as error:
            raise ResponseExtractionError(
                f"element {element_id} references missing node {error.args[0]}"
            ) from error
        if not polygon.is_valid or polygon.area <= 0:
            raise ResponseExtractionError(
                f"element {element_id} does not form a valid positive-area polygon"
            )
        area_in_scope = float(polygon.intersection(scope_polygon).area)
        if area_in_scope <= 0:
            continue
        selected = [
            point
            for point in gauss_points
            if buffered_scope.covers(Point(point.x, point.y))
        ]
        if not selected:
            continue
        eligible_elements += 1
        if area_in_scope < polygon.area * (1.0 - 1.0e-10):
            clipped_elements += 1
        area_per_point = area_in_scope / len(selected)
        material_name = classification.material_names[material_id]
        for point in selected:
            weighted.append(
                WeightedStressPoint(
                    point.gauss_point_id,
                    element_id,
                    point.x,
                    point.y,
                    point.max_effective_stress,
                    point.min_effective_stress,
                    area_per_point,
                    material_id,
                    material_name,
                )
            )
    if not weighted:
        raise ResponseExtractionError(
            "no material-filtered Gauss stress points fall inside the dam envelope"
        )
    total_area = sum(point.area_weight_m2 for point in weighted)
    if not math.isfinite(total_area) or total_area <= 0:
        raise ResponseExtractionError("eligible Gauss-point area is not positive")
    counts = {
        "queried_elements": len(element_node_ids),
        "queried_gauss_points": len(mesh.gauss_points),
        "eligible_stress_elements": eligible_elements,
        "eligible_stress_gauss_points": len(weighted),
        "scope_clipped_stress_elements": clipped_elements,
    }
    scope = {
        "definition": "main_dam_envelope_with_element_material_exclusions",
        "polygon": [[x, y] for x, y in scope_coordinates],
        "eligible_area_m2": total_area,
        "stress_basis": "raw_gauss_point",
        "tail": "upper_numeric_tail_by_physical_area",
        "weighting_basis": (
            "element intersection area divided equally among in-scope Gauss points"
        ),
        "weighting_limitation": (
            "GSI exposes no quadrature weight or Jacobian result; within-element "
            "equal allocation conserves area but approximates exact integration weights"
        ),
    }
    return weighted, counts, scope


def weighted_upper_tail(
    points: Sequence[WeightedStressPoint],
    *,
    value: Callable[[WeightedStressPoint], float],
    alpha: float = DEFAULT_CVAR_ALPHA,
) -> WeightedTailResult:
    """Calculate exact-area weighted VaR and CVaR for an upper numeric tail."""
    _validate_alpha(alpha)
    if not points:
        raise ValueError("at least one weighted stress point is required")
    valued: list[tuple[float, WeightedStressPoint]] = []
    for point in points:
        stress = float(value(point))
        if not math.isfinite(stress):
            raise ValueError("stress values must be finite")
        if not math.isfinite(point.area_weight_m2) or point.area_weight_m2 <= 0:
            raise ValueError("Gauss-point area weights must be positive and finite")
        valued.append((stress, point))
    valued.sort(key=lambda item: (-item[0], item[1].gauss_point_id))
    total_area = sum(point.area_weight_m2 for _, point in valued)
    target_area = total_area * (1.0 - alpha)
    remaining = target_area
    weighted_sum = 0.0
    centroid_x_sum = 0.0
    centroid_y_sum = 0.0
    selected: list[dict[str, object]] = []
    contributing_elements: set[int] = set()
    var_value = valued[-1][0]
    index = 0
    tolerance = max(total_area * 1.0e-13, 1.0e-15)
    while index < len(valued) and remaining > tolerance:
        group_value = valued[index][0]
        end = index + 1
        while end < len(valued) and math.isclose(
            valued[end][0], group_value, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            end += 1
        group = valued[index:end]
        group_area = sum(point.area_weight_m2 for _, point in group)
        used_group_area = min(group_area, remaining)
        fraction = used_group_area / group_area
        var_value = group_value
        for stress, point in group:
            used_area = point.area_weight_m2 * fraction
            if used_area <= 0:
                continue
            weighted_sum += stress * used_area
            centroid_x_sum += point.x * used_area
            centroid_y_sum += point.y * used_area
            contributing_elements.add(point.element_id)
            selected.append(
                {
                    "gauss_point_id": point.gauss_point_id,
                    "element_id": point.element_id,
                    "x_m": point.x,
                    "y_m": point.y,
                    "stress_kpa": stress,
                    "available_area_weight_m2": point.area_weight_m2,
                    "selected_area_weight_m2": used_area,
                    "cutoff_fraction": fraction,
                    "material_id": point.material_id,
                    "material_name": point.material_name,
                }
            )
        remaining -= used_group_area
        index = end
    selected_area = target_area - max(remaining, 0.0)
    if selected_area <= 0 or not math.isclose(
        selected_area, target_area, rel_tol=1.0e-10, abs_tol=tolerance
    ):
        raise ValueError("could not fill the requested CVaR tail area")
    return WeightedTailResult(
        alpha=alpha,
        var_kpa=var_value,
        cvar_kpa=weighted_sum / selected_area,
        maximum_kpa=valued[0][0],
        total_area_m2=total_area,
        target_tail_area_m2=target_area,
        selected_tail_area_m2=selected_area,
        contributing_gauss_points=len(selected),
        contributing_elements=len(contributing_elements),
        weighted_centroid_x_m=centroid_x_sum / selected_area,
        weighted_centroid_y_m=centroid_y_sum / selected_area,
        points=tuple(selected),
    )


def nodes_from_columns(columns: Mapping[str, Sequence[float]]) -> list[NodeResult]:
    required = ("node_id", "x", "y", "x_displacement", "y_displacement")
    count = _validate_columns(columns, required, "Nodes")
    return [
        NodeResult(
            _result_id(columns["node_id"][index], "node_id"),
            float(columns["x"][index]),
            float(columns["y"][index]),
            float(columns["x_displacement"][index]),
            float(columns["y_displacement"][index]),
        )
        for index in range(count)
    ]


def element_nodes_from_columns(
    columns: Mapping[str, Sequence[float]],
) -> tuple[list[ElementNodeMembership], int]:
    count = _validate_columns(columns, ("element_id", "node_id"), "ElementNodes")
    memberships: list[ElementNodeMembership] = []
    ignored = 0
    for index in range(count):
        raw_element_id = float(columns["element_id"][index])
        if raw_element_id <= 0:
            ignored += 1
            continue
        memberships.append(
            ElementNodeMembership(
                _result_id(raw_element_id, "element_id"),
                _result_id(columns["node_id"][index], "node_id"),
            )
        )
    if not memberships:
        raise ResponseExtractionError(
            "ElementNodes results contain no active connectivity rows"
        )
    return memberships, ignored


def gauss_points_from_columns(
    columns: Mapping[str, Sequence[float]],
) -> list[GaussPointResult]:
    required = (
        "element_id",
        "gauss_point_id",
        "x",
        "y",
        "max_effective_stress",
        "min_effective_stress",
    )
    count = _validate_columns(columns, required, "Gauss")
    points: list[GaussPointResult] = []
    for index in range(count):
        raw_element_id = float(columns["element_id"][index])
        raw_gauss_id = float(columns["gauss_point_id"][index])
        if raw_element_id <= 0 or raw_gauss_id <= 0:
            continue
        points.append(
            GaussPointResult(
                _result_id(raw_gauss_id, "gauss_point_id"),
                _result_id(raw_element_id, "element_id"),
                float(columns["x"][index]),
                float(columns["y"][index]),
                float(columns["max_effective_stress"][index]),
                float(columns["min_effective_stress"][index]),
            )
        )
    if not points:
        raise ResponseExtractionError("Gauss results contain no active rows")
    if len({point.gauss_point_id for point in points}) != len(points):
        raise ResponseExtractionError("Gauss results contain duplicate point IDs")
    return points


def element_node_sequences(
    memberships: Sequence[ElementNodeMembership],
) -> dict[int, tuple[int, ...]]:
    sequences: dict[int, list[int]] = defaultdict(list)
    seen: dict[int, set[int]] = defaultdict(set)
    for membership in memberships:
        if membership.node_id in seen[membership.element_id]:
            raise ResponseExtractionError(
                f"element {membership.element_id} contains duplicate node "
                f"{membership.node_id}"
            )
        seen[membership.element_id].add(membership.node_id)
        sequences[membership.element_id].append(membership.node_id)
    for element_id, node_ids in sequences.items():
        if len(node_ids) < 3:
            raise ResponseExtractionError(
                f"element {element_id} has fewer than three unique nodes"
            )
    return {element_id: tuple(node_ids) for element_id, node_ids in sequences.items()}


def project_provenance(project_path: Path) -> dict[str, object]:
    stat = project_path.stat()
    return {
        "path": str(project_path.resolve()),
        "size_bytes": stat.st_size,
        "modified_time_ns": stat.st_mtime_ns,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
    }


def _query_columns(
    project: object,
    gsi: object,
    analysis_name: str,
    step: int,
    table: int,
    parameters: Mapping[str, int],
    table_name: str,
) -> dict[str, list[float]]:
    response = project.QueryResults(  # type: ignore[attr-defined]
        gsi.QueryResultsRequest(  # type: ignore[attr-defined]
            analysis=analysis_name,
            step=step,
            table=table,
            dataparams=list(parameters.values()),
        )
    )
    columns: dict[str, list[float]] = {}
    for name, parameter in parameters.items():
        if parameter not in response.results:
            raise ResponseExtractionError(f"{table_name} results are missing {name}")
        columns[name] = list(response.results[parameter].values)
    return columns


def _validate_table_metadata(
    project: object,
    gsi: object,
    analysis_name: str,
    table: int,
    parameters: Mapping[str, int],
    expected_units: Mapping[str, str],
    table_name: str,
) -> None:
    response = project.QueryTableParamsInfo(  # type: ignore[attr-defined]
        gsi.QueryTableParamsInfoRequest(analysis=analysis_name, table=table)  # type: ignore[attr-defined]
    )
    metadata = {item.dataparam: item for item in response.params_info}
    missing = [
        name for name, parameter in parameters.items() if parameter not in metadata
    ]
    if missing:
        raise ResponseExtractionError(
            f"{table_name} table is missing required parameters: {', '.join(missing)}"
        )
    for name, expected in expected_units.items():
        actual = str(metadata[parameters[name]].units).strip()
        if actual.casefold() != expected.casefold():
            raise ResponseExtractionError(
                f"{table_name}.{name} uses units {actual!r}; expected {expected!r}"
            )


def _validate_columns(
    columns: Mapping[str, Sequence[float]],
    required: Sequence[str],
    table_name: str,
) -> int:
    missing = [name for name in required if name not in columns]
    if missing:
        raise ResponseExtractionError(
            f"{table_name} results are missing columns: {', '.join(missing)}"
        )
    lengths = {name: len(columns[name]) for name in required}
    if not lengths or next(iter(lengths.values())) == 0:
        raise ResponseExtractionError(f"{table_name} results contain no rows")
    if len(set(lengths.values())) != 1:
        detail = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ResponseExtractionError(
            f"{table_name} result column lengths differ: {detail}"
        )
    for name in required:
        _require_finite_values(columns[name], f"{table_name}.{name}")
    return next(iter(lengths.values()))


def _dam_body_polygon(
    derived: Mapping[str, object],
) -> tuple[Polygon, list[tuple[float, float]]]:
    coordinates = [
        _coordinate(derived, "upstream_toe"),
        _coordinate(derived, "crest_left"),
        _coordinate(derived, "crest_right"),
        _coordinate(derived, "downstream_toe"),
    ]
    polygon = Polygon(coordinates)
    if not polygon.is_valid or polygon.area <= 0:
        raise ResponseExtractionError(
            "derived coordinates do not form a valid main-dam envelope"
        )
    return polygon, coordinates


def _parameter_metadata(
    payload: Mapping[str, object],
) -> tuple[str, str, Mapping[str, object]]:
    parameters = payload.get("parameters")
    derived = payload.get("derived_coordinates")
    parameter_hash = payload.get("parameter_hash")
    if not isinstance(parameters, Mapping):
        raise ResponseExtractionError("parameters.json has no parameters object")
    design_id = parameters.get("design_id")
    if not isinstance(design_id, str) or not design_id:
        raise ResponseExtractionError("parameters.json has no design_id")
    if not isinstance(parameter_hash, str) or not parameter_hash:
        raise ResponseExtractionError("parameters.json has no parameter_hash")
    if not isinstance(derived, Mapping):
        raise ResponseExtractionError(
            "parameters.json has no derived_coordinates object"
        )
    return design_id, parameter_hash, derived


def _coordinate(derived: Mapping[str, object], name: str) -> tuple[float, float]:
    value = derived.get(name)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ResponseExtractionError(
            f"derived coordinate {name!r} is missing or invalid"
        )
    x, y = float(value[0]), float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ResponseExtractionError(f"derived coordinate {name!r} is not finite")
    return x, y


def _node_location(
    node: NodeResult,
    classification: MaterialClassification,
) -> dict[str, object]:
    material_ids = classification.node_material_ids.get(node.node_id, ())
    return {
        "node_id": node.node_id,
        "x_m": node.x,
        "y_m": node.y,
        "x_displacement_m": node.x_displacement,
        "y_displacement_m": node.y_displacement,
        "incident_materials": [
            {
                "material_id": material_id,
                "material_name": classification.material_names[material_id],
            }
            for material_id in material_ids
        ],
    }


def _material_filter_payload(
    classification: MaterialClassification,
) -> dict[str, object]:
    return {
        "version": MATERIAL_FILTER_VERSION,
        "strategy": {
            "nodes": "exclude if any incident element uses an excluded material",
            "elements": "exclude if the element uses an excluded material",
        },
        "excluded_material_name_keywords": list(EXCLUDED_MATERIAL_NAME_KEYWORDS),
        "excluded_materials": [
            {
                "material_id": material_id,
                "material_name": classification.material_names[material_id],
            }
            for material_id in sorted(classification.excluded_material_ids)
        ],
        "crest_material": {
            "material_id": classification.crest_material_id,
            "material_name": classification.material_names[
                classification.crest_material_id
            ],
        },
    }


def _validate_alpha(alpha: float) -> None:
    if not math.isfinite(alpha) or alpha <= 0 or alpha >= 1:
        raise ValueError("CVaR alpha must be finite and strictly between 0 and 1")


def _alpha_token(alpha: float) -> str:
    percent = alpha * 100.0
    rounded = round(percent)
    if math.isclose(percent, rounded, abs_tol=1.0e-10):
        return str(rounded)
    return f"{percent:.8f}".rstrip("0").rstrip(".").replace(".", "p")


def _result_id(value: float, name: str) -> int:
    number = float(value)
    result = int(round(number))
    if result <= 0 or not math.isclose(number, result, abs_tol=1.0e-9):
        raise ResponseExtractionError(
            f"{name} contains an invalid identifier: {value!r}"
        )
    return result


def _xml_positive_int(value: str | None, name: str) -> int:
    try:
        result = int(value or "")
    except ValueError as error:
        raise ResponseExtractionError(
            f"project XML has an invalid {name}: {value!r}"
        ) from error
    if result <= 0:
        raise ResponseExtractionError(f"project XML has an invalid {name}: {value!r}")
    return result


def _require_finite_values(values: Iterable[float], label: str) -> None:
    for index, value in enumerate(values):
        if not math.isfinite(float(value)):
            raise ResponseExtractionError(
                f"{label} contains a non-finite value at row {index + 1}"
            )


def _assert_project_not_open(project_path: Path) -> None:
    if os.name != "nt":
        try:
            with project_path.open("rb") as stream:
                stream.read(1)
        except PermissionError as error:
            raise ResponseExtractionError(
                "GeoStudio project is in use; close it before extracting results: "
                f"{project_path}"
            ) from error
        return

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    handle = create_file(str(project_path), 0x80000000, 0, None, 3, 0x80, None)
    invalid_handle = wintypes.HANDLE(-1).value
    if handle == invalid_handle:
        error_code = ctypes.get_last_error()
        if error_code in {32, 33}:
            raise ResponseExtractionError(
                "GeoStudio project is open or locked; close it before extracting "
                f"results: {project_path}"
            )
        raise ResponseExtractionError(
            "could not open GeoStudio project for result extraction "
            f"(Windows error {error_code}): {project_path}"
        )
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    close_handle(handle)


__all__ = [
    "CREST_MATERIAL_NAME",
    "DEFAULT_ANALYSIS_NAME",
    "DEFAULT_ANALYSIS_STEP",
    "DEFAULT_CVAR_ALPHA",
    "EXCLUDED_MATERIAL_NAME_KEYWORDS",
    "GaussPointResult",
    "MaterialClassification",
    "MaterialMetadata",
    "MaterialRegion",
    "METRIC_FIELDS",
    "NodeResult",
    "ElementNodeMembership",
    "ProjectResponses",
    "ResponseExtractionError",
    "ResponseMeshData",
    "WeightedStressPoint",
    "WeightedTailResult",
    "build_weighted_stress_points",
    "calculate_displacement_metrics",
    "classify_materials",
    "element_node_sequences",
    "extract_project_responses",
    "load_material_metadata",
    "metric_fields",
    "project_provenance",
    "query_response_mesh",
    "weighted_upper_tail",
]
