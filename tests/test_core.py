from __future__ import annotations

import json
import math
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from shapely.geometry import Polygon

from geostudio_automation.mesh import validate_gsz_mesh
from geostudio_automation.model import DesignVariables, GeometryError
from geostudio_automation.template import TemplateArchive
from geostudio_automation.workflow import build_design
from geostudio_automation.xml_model import GeoStudioXmlModel

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "geo_studio_model_sample"
MESH_FIXTURE = ROOT / "tests" / "fixtures" / "mesh_1.ply"


def feasible_design(design_id: str = "feasible") -> DesignVariables:
    return DesignVariables(
        design_id,
        1.46,
        1.5,
        12.0,
        18.0,
        20.0,
        40.0,
        55.0,
        16.641005886756876,
    )


class CoreModelTests(unittest.TestCase):
    def test_right_slope_distance_uses_inclusive_bounds(self) -> None:
        base = feasible_design().as_dict()
        for value in (0.1, 20.0):
            values = dict(base)
            values["secondary_right_slope_distance"] = value
            self.assertEqual(DesignVariables(**values).validate_bounds(), [])
        for value in (0.099, 20.001):
            values = dict(base)
            values["secondary_right_slope_distance"] = value
            self.assertTrue(DesignVariables(**values).validate_bounds())

    def test_right_side_is_parallel_and_distance_is_perpendicular(self) -> None:
        archive = TemplateArchive.load(TEMPLATE)
        design = feasible_design()
        result = GeoStudioXmlModel(archive.root()).rebuild(design)
        derived = result.geometry.derived
        right_dx = derived.secondary_bottom_right[0] - derived.secondary_top_right[0]
        right_dy = derived.secondary_top_right[1] - derived.secondary_bottom_right[1]
        self.assertAlmostEqual(right_dx / right_dy, design.downstream_slope)
        downstream_top = derived.secondary_downstream_reference_bottom[0] - (
            design.downstream_slope * design.secondary_height
        )
        perpendicular = (
            downstream_top - derived.secondary_top_right[0]
        ) / math.sqrt(1.0 + design.downstream_slope**2)
        self.assertAlmostEqual(
            perpendicular,
            design.secondary_right_slope_distance,
        )
        measured_angle = math.degrees(
            math.atan2(
                derived.secondary_top_left[1] - derived.secondary_bottom_left[1],
                derived.secondary_top_left[0] - derived.secondary_bottom_left[0],
            )
        )
        self.assertAlmostEqual(
            measured_angle,
            derived.secondary_left_bottom_angle,
        )

    def test_out_of_range_derived_left_angle_is_rejected(self) -> None:
        archive = TemplateArchive.load(TEMPLATE)
        model = GeoStudioXmlModel(archive.root())
        design = DesignVariables(
            "invalid-derived-angle",
            1.4,
            1.6,
            10.0,
            5.0,
            35.0,
            20.0,
            20.0,
            10.0,
        )
        self.assertEqual(design.validate_bounds(), [])
        with self.assertRaisesRegex(GeometryError, "derived secondary left-bottom angle"):
            model.rebuild(design)

    def test_rebuild_is_deterministic_and_keeps_ten_contexts(self) -> None:
        archive = TemplateArchive.load(TEMPLATE)
        model = GeoStudioXmlModel(archive.root())
        first = model.rebuild(feasible_design())
        second = model.rebuild(feasible_design())
        self.assertEqual(first.xml, second.xml)
        self.assertEqual(first.point_count, 70)
        self.assertEqual(first.line_count, 97)
        self.assertEqual(first.region_count, 28)
        contexts = first.root.findall("./Contexts/Context")
        self.assertEqual(len(contexts), 10)
        lengths = [len(context.find("GeometryUsesMaterials")) for context in contexts]
        self.assertEqual(lengths, sorted(lengths))
        self.assertEqual(lengths[-1], first.region_count)

        points = {
            int(point.attrib["ID"]): (
                float(point.attrib["X"]),
                float(point.attrib["Y"]),
            )
            for point in first.root.findall("./Geometries/Geometry/Points/Point")
        }
        for region in first.root.findall("./Geometries/Geometry/Regions/Region"):
            ids = [int(value) for value in region.findtext("PointIDs", "").split(",")]
            self.assertTrue(Polygon([points[value] for value in ids]).exterior.is_ccw)

        lines = {
            int(line.findtext("ID", "0")): (
                int(line.findtext("PointID1", "0")),
                int(line.findtext("PointID2", "0")),
            )
            for line in first.root.findall("./Geometries/Geometry/Lines/Line")
        }
        for context in first.root.findall("./Contexts/Context"):
            for container in context:
                if not container.tag.startswith("GeometryUses"):
                    continue
                for item in container:
                    reference = item.attrib.get("ID", "")
                    if reference.startswith("Lines-"):
                        self.assertIn(int(reference.split("-", 1)[1]), lines)

    def test_original_xml_matches_self_contained_mesh_fixture(self) -> None:
        with TemporaryDirectory() as directory:
            project = Path(directory) / "original.gsz"
            with zipfile.ZipFile(project, "w") as archive:
                archive.write(TEMPLATE / "original_sample.xml", "original.xml")
                archive.write(MESH_FIXTURE, "mesh_1.ply")
            report = validate_gsz_mesh(project)
        self.assertTrue(report.matched)
        self.assertEqual(report.node_count, 5999)
        self.assertEqual(report.xml_line_count, 103)

    def test_rebuilt_xml_is_rejected_with_stale_mesh(self) -> None:
        template = TemplateArchive.load(TEMPLATE)
        rebuilt = GeoStudioXmlModel(template.root()).rebuild(feasible_design())
        with TemporaryDirectory() as directory:
            project = Path(directory) / "stale.gsz"
            with zipfile.ZipFile(project, "w") as archive:
                archive.writestr("stale.xml", rebuilt.xml)
                archive.write(MESH_FIXTURE, "mesh_1.ply")
            report = validate_gsz_mesh(project)
        self.assertFalse(report.matched)
        self.assertGreater(len(report.mismatches), 0)

    def test_build_strips_generated_mesh_and_writes_audit_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template"
            template.mkdir()
            (template / "sample.xml").write_bytes(
                (TEMPLATE / "original_sample.xml").read_bytes()
            )
            (template / "mesh_1.ply").write_bytes(MESH_FIXTURE.read_bytes())
            (template / "ThumbNail.bmp").write_bytes(
                (TEMPLATE / "ThumbNail.bmp").read_bytes()
            )
            project = root / "built" / "built.gsz"
            result = build_design(template, feasible_design("built"), project)
            self.assertTrue(result.succeeded, result.error_summary)
            with zipfile.ZipFile(project) as archive:
                names = archive.namelist()
            self.assertFalse(any(name.lower().endswith(".ply") for name in names))
            self.assertIn("ThumbNail.bmp", names)
            report = json.loads(
                (project.parent / "build_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "built")
            self.assertEqual(report["geometry"]["mesh_state"], "not_generated")
            self.assertTrue((project.parent / "parameters.json").is_file())
            self.assertTrue(Path(result.preview_path).is_file())
            with self.assertRaisesRegex(ValueError, "mesh_.*absent"):
                validate_gsz_mesh(project)


if __name__ == "__main__":
    unittest.main()
