from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass

from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient

from .geometry import (
    GEOMETRY_TOLERANCE,
    TOPOLOGY_SNAP_TOLERANCE,
    GeometryBuild,
    RegionCell,
    SourceRegion,
    build_geometry,
    template_warp_controls,
    warp_template_coordinate,
)
from .model import CONSTANTS, DesignVariables, GeometryError
from .template import xml_bytes


@dataclass(frozen=True, slots=True)
class BoundaryReference:
    context_index: int
    container_tag: str
    item_tag: str
    attributes: dict[str, str]
    segment: LineString


@dataclass(slots=True)
class XmlBuildResult:
    root: ET.Element
    xml: bytes
    geometry: GeometryBuild
    point_count: int
    line_count: int
    region_count: int
    boundary_reference_count: int


class GeoStudioXmlModel:
    def __init__(
        self,
        root: ET.Element,
        *,
        reference_design: DesignVariables | None = None,
    ):
        self.root = root
        self.reference_design = reference_design
        self.geometry_element = self._required("./Geometries/Geometry")
        self.point_element = self._required("./Geometries/Geometry/Points")
        self.line_element = self._required("./Geometries/Geometry/Lines")
        self.region_element = self._required("./Geometries/Geometry/Regions")
        self.contexts = self.root.findall("./Contexts/Context")
        if len(self.contexts) != 10:
            raise ValueError(f"template must contain 10 analysis contexts; found {len(self.contexts)}")

        self.points = {
            int(point.attrib["ID"]): (float(point.attrib["X"]), float(point.attrib["Y"]))
            for point in self.point_element.findall("Point")
        }
        self.lines = {
            int(line.findtext("ID", "0")): (
                int(line.findtext("PointID1", "0")),
                int(line.findtext("PointID2", "0")),
            )
            for line in self.line_element.findall("Line")
        }
        self.pinned_line_ids = {
            int(line.findtext("ID", "0"))
            for line in self.line_element.findall("Line")
            if line.findtext("Pinned", "false").lower() == "true"
        }
        self.pinned_point_ids = {
            int(point.attrib["ID"])
            for point in self.point_element.findall("Point")
            if point.attrib.get("Pinned", "false").lower() == "true"
        }
        self.original_regions = self._read_regions()
        self.warp_controls = template_warp_controls(
            self.original_regions,
            reference_design=self.reference_design,
        )
        self.boundary_references = self._read_boundary_references()

    def _required(self, path: str) -> ET.Element:
        element = self.root.find(path)
        if element is None:
            raise ValueError(f"template is missing {path}")
        return element

    def _read_regions(self) -> list[SourceRegion]:
        final_materials: dict[int, int] = {}
        final_container = self.contexts[-1].find("GeometryUsesMaterials")
        if final_container is None:
            raise ValueError("final context has no material assignments")
        for use in final_container:
            final_materials[_reference_number(use.attrib["ID"], "Regions")] = int(use.attrib["Entry"])

        activation: dict[int, int] = {}
        for context in self.contexts:
            stage = int(context.findtext("AnalysisID", "0"))
            container = context.find("GeometryUsesMaterials")
            if container is None:
                continue
            for use in container:
                region_id = _reference_number(use.attrib["ID"], "Regions")
                activation.setdefault(region_id, stage)

        regions: list[SourceRegion] = []
        for element in self.region_element.findall("Region"):
            region_id = int(element.findtext("ID", "0"))
            point_ids = [int(value) for value in element.findtext("PointIDs", "").split(",") if value]
            try:
                polygon = Polygon([self.points[point_id] for point_id in point_ids])
                material = final_materials[region_id]
                stage = activation[region_id]
            except KeyError as error:
                raise ValueError(f"incomplete template reference for region {region_id}: {error}") from error
            if not polygon.is_valid or polygon.area <= 0:
                raise ValueError(f"template region {region_id} is invalid")
            regions.append(SourceRegion(polygon, material, stage, f"template-region-{region_id}"))
        return regions

    def _read_boundary_references(self) -> list[BoundaryReference]:
        references: list[BoundaryReference] = []
        for index, context in enumerate(self.contexts):
            for container in context:
                if not container.tag.startswith("GeometryUses") or container.tag == "GeometryUsesMaterials":
                    continue
                for item in container:
                    reference = item.attrib.get("ID", "")
                    if not reference.startswith("Lines-"):
                        continue
                    line_id = _reference_number(reference, "Lines")
                    point1, point2 = self.lines[line_id]
                    references.append(
                        BoundaryReference(
                            index,
                            container.tag,
                            item.tag,
                            dict(item.attrib),
                            LineString([self.points[point1], self.points[point2]]),
                        )
                    )
        return references

    def rebuild(self, design: DesignVariables) -> XmlBuildResult:
        geometry = build_geometry(
            self.original_regions,
            design,
            reference_design=self.reference_design,
        )
        root = deepcopy(self.root)
        model = GeoStudioXmlModel.__new__(GeoStudioXmlModel)
        model.root = root
        model.reference_design = self.reference_design
        model.geometry_element = _required_from(root, "./Geometries/Geometry")
        model.point_element = _required_from(root, "./Geometries/Geometry/Points")
        model.line_element = _required_from(root, "./Geometries/Geometry/Lines")
        model.region_element = _required_from(root, "./Geometries/Geometry/Regions")
        model.contexts = root.findall("./Contexts/Context")
        model.points = self.points
        model.lines = self.lines
        model.pinned_line_ids = self.pinned_line_ids
        model.pinned_point_ids = self.pinned_point_ids
        model.original_regions = self.original_regions
        model.warp_controls = self.warp_controls
        model.boundary_references = self.boundary_references

        topology = _Topology.from_cells(geometry.cells)
        model._replace_geometry(topology, design)
        count = model._replace_contexts(topology, geometry.cells, design)
        model._clear_obsolete_point_labels()
        model._validate_references(topology)
        output = xml_bytes(root)
        return XmlBuildResult(
            root=root,
            xml=output,
            geometry=geometry,
            point_count=len(topology.coordinates),
            line_count=len(topology.edges),
            region_count=len(geometry.cells),
            boundary_reference_count=count,
        )

    def _replace_geometry(self, topology: _Topology, design: DesignVariables) -> None:
        self.point_element.clear()
        self.point_element.attrib["Len"] = str(len(topology.coordinates))
        pinned_points = {
            _round_coordinate(
                warp_template_coordinate(
                    *self.points[point_id],
                    design,
                    self.warp_controls,
                    reference_design=self.reference_design,
                )
            )
            for point_id in self.pinned_point_ids
        }
        for point_id, (x, y) in enumerate(topology.coordinates, start=1):
            attributes = {"ID": str(point_id), "X": _number(x), "Y": _number(y)}
            # Noding snaps and rounds vertices; exact equality can lose a
            # template pin even when it follows the correct rebuilt point.
            if any(math.hypot(x - px, y - py) <= TOPOLOGY_SNAP_TOLERANCE for px, py in pinned_points):
                attributes["Pinned"] = "true"
            ET.SubElement(self.point_element, "Point", attributes)

        transformed_pinned = []
        for line_id in self.pinned_line_ids:
            point1, point2 = self.lines[line_id]
            transformed_pinned.append(
                LineString(
                    [
                        warp_template_coordinate(
                            *self.points[point1],
                            design,
                            self.warp_controls,
                            reference_design=self.reference_design,
                        ),
                        warp_template_coordinate(
                            *self.points[point2],
                            design,
                            self.warp_controls,
                            reference_design=self.reference_design,
                        ),
                    ]
                )
            )

        self.line_element.clear()
        self.line_element.attrib["Len"] = str(len(topology.edges))
        for line_id, (point1, point2) in enumerate(topology.edges, start=1):
            element = ET.SubElement(self.line_element, "Line")
            ET.SubElement(element, "ID").text = str(line_id)
            ET.SubElement(element, "PointID1").text = str(point1)
            ET.SubElement(element, "PointID2").text = str(point2)
            segment = topology.line_geometry(line_id)
            if any(_line_is_covered(segment, reference) for reference in transformed_pinned):
                ET.SubElement(element, "Pinned").text = "true"

        self.region_element.clear()
        self.region_element.attrib["Len"] = str(len(topology.region_point_ids))
        for region_id, point_ids in enumerate(topology.region_point_ids, start=1):
            element = ET.SubElement(self.region_element, "Region")
            ET.SubElement(element, "ID").text = str(region_id)
            ET.SubElement(element, "PointIDs").text = ",".join(str(value) for value in point_ids)
            ET.SubElement(element, "Mesh", {"Pattern": "Unstructured Mixed Elements"})

    def _replace_contexts(
        self, topology: _Topology, cells: list[RegionCell], design: DesignVariables
    ) -> int:
        for context_index, context in enumerate(self.contexts):
            stage = int(context.findtext("AnalysisID", "0"))
            materials = context.find("GeometryUsesMaterials")
            if materials is None:
                insert_at = 1 if context.find("AnalysisID") is not None else 0
                materials = ET.Element("GeometryUsesMaterials")
                context.insert(insert_at, materials)
            materials.clear()
            active = [
                (region_id, cell)
                for region_id, cell in enumerate(cells, start=1)
                if cell.activation_stage <= stage
            ]
            materials.attrib["Len"] = str(len(active))
            for region_id, cell in active:
                ET.SubElement(
                    materials,
                    "GeometryUsesMaterial",
                    {"ID": f"Regions-{region_id}", "Entry": str(cell.material_id)},
                )

            for container in context:
                if container.tag.startswith("GeometryUses") and container.tag != "GeometryUsesMaterials":
                    container.clear()
                    container.attrib["Len"] = "0"

        grouped: dict[tuple[int, str, str, tuple[tuple[str, str], ...]], list[LineString]] = {}
        for reference in self.boundary_references:
            attributes = tuple(sorted((key, value) for key, value in reference.attributes.items() if key != "ID"))
            key = (reference.context_index, reference.container_tag, reference.item_tag, attributes)
            transformed = LineString(
                [
                    warp_template_coordinate(
                        x,
                        y,
                        design,
                        self.warp_controls,
                        reference_design=self.reference_design,
                    )
                    for x, y in reference.segment.coords
                ]
            )
            grouped.setdefault(key, []).append(transformed)

        total = 0
        for (context_index, container_tag, item_tag, attributes), target_segments in grouped.items():
            context = self.contexts[context_index]
            container = context.find(container_tag)
            if container is None:
                container = ET.SubElement(context, container_tag, {"Len": "0"})
            matched = [
                line_id
                for line_id in range(1, len(topology.edges) + 1)
                if any(_line_is_covered(topology.line_geometry(line_id), target) for target in target_segments)
            ]
            if not matched:
                raise GeometryError(
                    f"boundary condition {container_tag}/{item_tag} in analysis {context_index + 1} matched no rebuilt line"
                )
            for line_id in matched:
                item_attributes = dict(attributes)
                item_attributes["ID"] = f"Lines-{line_id}"
                ET.SubElement(container, item_tag, item_attributes)
                total += 1

        for context in self.contexts:
            for container in context:
                if container.tag.startswith("GeometryUses"):
                    container.attrib["Len"] = str(len(container))
        return total

    def _clear_obsolete_point_labels(self) -> None:
        labels = self.geometry_element.find("PointLabel")
        if labels is not None:
            labels.clear()
            labels.attrib["Len"] = "0"

    def _validate_references(self, topology: _Topology) -> None:
        point_ids = set(range(1, len(topology.coordinates) + 1))
        line_ids = set(range(1, len(topology.edges) + 1))
        region_ids = set(range(1, len(topology.region_point_ids) + 1))
        for point1, point2 in topology.edges:
            if point1 not in point_ids or point2 not in point_ids:
                raise GeometryError("a rebuilt line contains a dangling point reference")
        for points in topology.region_point_ids:
            if not set(points) <= point_ids:
                raise GeometryError("a rebuilt region contains a dangling point reference")
        for context in self.contexts:
            for container in context:
                if not container.tag.startswith("GeometryUses"):
                    continue
                for item in container:
                    reference = item.attrib.get("ID", "")
                    if reference.startswith("Lines-") and _reference_number(reference, "Lines") not in line_ids:
                        raise GeometryError(f"dangling line reference: {reference}")
                    if reference.startswith("Regions-") and _reference_number(reference, "Regions") not in region_ids:
                        raise GeometryError(f"dangling region reference: {reference}")


@dataclass(slots=True)
class _Topology:
    coordinates: list[tuple[float, float]]
    edges: list[tuple[int, int]]
    region_point_ids: list[list[int]]

    @classmethod
    def from_cells(cls, cells: list[RegionCell]) -> _Topology:
        base_rings: list[list[tuple[float, float]]] = []
        all_coordinates: set[tuple[float, float]] = set()
        for cell in cells:
            polygon = orient(cell.polygon, sign=1.0)
            if polygon.interiors:
                raise GeometryError("rebuilt regions with holes are not supported by GeoStudio")
            coordinates = [_round_coordinate(value) for value in list(polygon.exterior.coords)[:-1]]
            coordinates = _remove_repeated(coordinates)
            if len(coordinates) < 3:
                raise GeometryError("coordinate rounding collapsed a rebuilt region")
            base_rings.append(coordinates)
            all_coordinates.update(coordinates)

        # Insert every global boundary vertex that lies on a longer region edge.
        # This converts A--C / A--B--C T-junction representations into identical
        # shared A--B and B--C edges before IDs are generated.
        normalized: list[list[tuple[float, float]]] = []
        for ring in base_rings:
            expanded: list[tuple[float, float]] = []
            for first, second in zip(ring, ring[1:] + ring[:1]):
                segment = LineString([first, second])
                candidates = [
                    coordinate
                    for coordinate in all_coordinates
                    if coordinate not in {first, second}
                    and segment.distance(Point(coordinate)) <= 1.0e-8
                    and -1.0e-10 <= segment.project(Point(coordinate), normalized=True) <= 1.0 + 1.0e-10
                ]
                ordered = [first, *sorted(candidates, key=lambda value: segment.project(Point(value)))]
                expanded.extend(ordered)
            normalized.append(_remove_repeated(expanded))

        coordinate_list = sorted(all_coordinates, key=lambda value: (value[1], value[0]))
        point_id = {value: index for index, value in enumerate(coordinate_list, start=1)}
        regions = [[point_id[value] for value in coordinates] for coordinates in normalized]
        edge_set: set[tuple[int, int]] = set()
        for region in regions:
            for point1, point2 in zip(region, region[1:] + region[:1]):
                edge_set.add(tuple(sorted((point1, point2))))
        edges = sorted(edge_set)
        return cls(coordinate_list, edges, regions)

    def line_geometry(self, line_id: int) -> LineString:
        point1, point2 = self.edges[line_id - 1]
        return LineString([self.coordinates[point1 - 1], self.coordinates[point2 - 1]])


def _required_from(root: ET.Element, path: str) -> ET.Element:
    element = root.find(path)
    if element is None:
        raise ValueError(f"template is missing {path}")
    return element


def _reference_number(reference: str, kind: str) -> int:
    prefix = f"{kind}-"
    if not reference.startswith(prefix):
        raise ValueError(f"expected {prefix} reference, got {reference!r}")
    return int(reference[len(prefix) :])


def _round_coordinate(value: tuple[float, float]) -> tuple[float, float]:
    return round(value[0], CONSTANTS.coordinate_precision), round(value[1], CONSTANTS.coordinate_precision)


def _remove_repeated(values: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for value in values:
        if not result or value != result[-1]:
            result.append(value)
    if len(result) > 1 and result[0] == result[-1]:
        result.pop()
    return result


def _number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1.0e-10):
        return str(round(value))
    return f"{value:.12f}".rstrip("0").rstrip(".")


def _line_is_covered(line: LineString, target: LineString) -> bool:
    tolerance = max(2.0e-3, line.length * 1.0e-8)
    uncovered = line.difference(target.buffer(tolerance, cap_style=2)).length
    # A short line can be shorter than the absolute tolerance while lying
    # nowhere near the target. Require nearly all of the line to be covered.
    return uncovered <= tolerance and uncovered < line.length * 0.05
