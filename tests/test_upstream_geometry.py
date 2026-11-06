from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

from shapely.geometry import LineString
from shapely.ops import unary_union

from geostudio_automation.geometry import (
    transform_template_regions,
    warp_polygon,
    warp_template_coordinate,
)
from geostudio_automation.model import CONSTANTS, DesignVariables
from geostudio_automation.xml_model import GeoStudioXmlModel


TEMPLATE_XML = (
    Path(__file__).resolve().parents[1]
    / "templates/geo_studio_model_sample/original_sample.xml"
)


class UpstreamGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = GeoStudioXmlModel(ET.parse(TEMPLATE_XML).getroot())
        self.design = DesignVariables("upstream_check", 1.46, 1.5, 12, 18, 20, 40, 55, 10)

    def test_layers_continue_to_base_without_hooks_and_keep_their_widths(self) -> None:
        for slope, width in ((1.46, 12), (1.3, 4), (1.3, 20), (1.6, 4)):
            with self.subTest(slope=slope, width=width):
                design = replace(self.design, upstream_slope=slope, crest_width=width)
                rebuilt = self.model.rebuild(design)
                layers = [cell for cell in rebuilt.geometry.cells if cell.material_id in {2, 3}]
                self.assertEqual(len(layers), 2)
                for cell in layers:
                    self.assertEqual(cell.activation_stage, 8)
                    coordinates = list(cell.polygon.exterior.coords)
                    for first, second in zip(coordinates, coordinates[1:]):
                        if first[1] != second[1]:
                            # Every inclined interface rises toward the crest;
                            # a toe pointing back into the dam has the opposite sign.
                            self.assertGreater((second[0] - first[0]) * (second[1] - first[1]), 0)
                    expected_width = {2: 3.0, 3: 4.0}[cell.material_id]
                    for elevation in (404.0, 404.5, 406.0, 407.75, 410.0):
                        section = LineString([(-100, elevation), (600, elevation)])
                        self.assertAlmostEqual(cell.polygon.intersection(section).length, expected_width, places=5)

    def test_straightening_preserves_outer_domain_and_foundation(self) -> None:
        original_controls = replace(self.model.warp_controls, toe_corrections={})
        before = [
            warp_polygon(region.polygon, self.design, controls=original_controls)
            for region in self.model.original_regions
        ]
        after = transform_template_regions(self.model.original_regions, self.design)
        self.assertLess(
            unary_union(before).symmetric_difference(unary_union([region.polygon for region in after])).area,
            1.0e-7,
        )
        for original, warped in zip(self.model.original_regions, before):
            if original.material_id == 9:
                foundation = next(region.polygon for region in after if region.material_id == 9)
                self.assertLess(warped.symmetric_difference(foundation).area, 1.0e-7)

    def test_corrected_template_can_be_reused_without_changing_geometry(self) -> None:
        first = self.model.rebuild(self.design)
        second = GeoStudioXmlModel(first.root, reference_design=self.design).rebuild(self.design)
        self.assertEqual(first.point_count, second.point_count)
        self.assertEqual(first.line_count, second.line_count)
        self.assertEqual(first.region_count, second.region_count)
        for material in {cell.material_id for cell in first.geometry.cells}:
            before = unary_union([cell.polygon for cell in first.geometry.cells if cell.material_id == material])
            after = unary_union([cell.polygon for cell in second.geometry.cells if cell.material_id == material])
            self.assertLess(before.symmetric_difference(after).area, 1.0e-7)

    def test_toe_line_pins_and_boundary_references_follow_corrected_contact(self) -> None:
        root = ET.parse(TEMPLATE_XML).getroot()
        toe_line = next(
            line for line in root.findall("./Geometries/Geometry/Lines/Line")
            if {line.findtext("PointID1"), line.findtext("PointID2")} == {"9", "14"}
        )
        ET.SubElement(toe_line, "Pinned").text = "true"
        point = root.find("./Geometries/Geometry/Points/Point[@ID='14']")
        point.set("Pinned", "true")
        container = root.findall("./Contexts/Context")[-1].find("GeometryUsesXDispBCs")
        ET.SubElement(container, container[0].tag, {"ID": f"Lines-{toe_line.findtext('ID')}", "Entry": "1"})
        container.set("Len", str(len(container)))
        model = GeoStudioXmlModel(root)
        result = model.rebuild(self.design)
        parsed = GeoStudioXmlModel(result.root, reference_design=self.design)
        target = LineString([
            warp_template_coordinate(*model.points[point_id], self.design, model.warp_controls)
            for point_id in (9, 14)
        ])
        self.assertTrue(any(
            LineString([parsed.points[a], parsed.points[b]]).hausdorff_distance(target) < 1.0e-7
            for line_id in parsed.pinned_line_ids for a, b in [parsed.lines[line_id]]
        ))
        self.assertTrue(any(
            reference.segment.hausdorff_distance(target) < 1.0e-7
            for reference in parsed.boundary_references
            if reference.context_index == 9 and reference.attributes["Entry"] == "1"
        ))
        self.assertTrue(any(
            abs(parsed.points[point_id][0] - target.coords[1][0]) < 1.0e-7
            and parsed.points[point_id][1] == CONSTANTS.dam_base_elevation
            for point_id in parsed.pinned_point_ids
        ))

    def test_foundation_constraints_do_not_spread_to_short_toe_lines(self) -> None:
        result = self.model.rebuild(self.design)
        points = {
            int(point.get("ID")): (float(point.get("X")), float(point.get("Y")))
            for point in result.root.findall("./Geometries/Geometry/Points/Point")
        }
        lines = {
            int(line.findtext("ID")): (
                int(line.findtext("PointID1")), int(line.findtext("PointID2"))
            )
            for line in result.root.findall("./Geometries/Geometry/Lines/Line")
        }
        for context in result.root.findall("./Contexts/Context"):
            stage = int(context.findtext("AnalysisID"))
            for tag in ("GeometryUsesXDispBCs", "GeometryUsesYDispBCs"):
                for use in context.find(tag):
                    line_id = int(use.get("ID").split("-", 1)[1])
                    endpoints = [points[point_id] for point_id in lines[line_id]]
                    if use.get("Entry") in {"1", "3"}:
                        self.assertTrue(
                            all(abs(x) < 1.0e-6 for x, _ in endpoints)
                            or all(abs(x - 558.25) < 1.0e-6 for x, _ in endpoints)
                            or all(abs(y - 272.0) < 1.0e-6 for _, y in endpoints),
                            (stage, tag, line_id, endpoints),
                        )
                    if stage == 10 and use.get("Entry") == "17":
                        self.assertTrue(
                            all(y > 411.9 for _, y in endpoints),
                            (stage, tag, line_id, endpoints),
                        )


if __name__ == "__main__":
    unittest.main()
