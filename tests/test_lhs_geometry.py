from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from shapely.ops import unary_union

from geostudio_automation.cli import main
from geostudio_automation.geometry import transform_template_regions
from geostudio_automation.model import DesignVariables
from geostudio_automation.sampling import write_initial_designs
from geostudio_automation.template import TemplateArchive
from geostudio_automation.xml_model import GeoStudioXmlModel


TEMPLATE = Path(__file__).resolve().parents[1] / "templates/geo_studio_model_sample"


class HighUpstreamSlopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        archive = TemplateArchive.load(TEMPLATE)
        cls.model = GeoStudioXmlModel(archive.root())
        cls.design = DesignVariables("high_slope", 1.7, 1.5, 10, 15, 20, 30, 60, 5)

    def test_weathered_layers_keep_area_and_shared_foundation_contact(self) -> None:
        for slope in (1.3, 1.46, 1.66, 1.68, 1.7):
            for width in (6, 12):
                with self.subTest(slope=slope, width=width):
                    design = replace(self.design, upstream_slope=slope, crest_width=width)
                    regions = transform_template_regions(self.model.original_regions, design)
                    self.assertTrue(all(r.polygon.is_valid for r in regions))
                    polygons = [r.polygon for r in regions]
                    self.assertAlmostEqual(
                        sum(p.area for p in polygons), unary_union(polygons).area, places=6
                    )
                    foundation = next(r.polygon for r in regions if r.material_id == 9)
                    self.assertEqual(foundation.bounds, (0.0, 272.0, 558.25, 412.0))
                    for region in regions:
                        if region.material_id not in (10, 11):
                            continue
                        original = next(
                            r for r in self.model.original_regions if r.source == region.source
                        )
                        self.assertAlmostEqual(region.polygon.area, original.polygon.area, places=7)
                        self.assertGreater(
                            region.polygon.boundary.intersection(foundation.boundary).length, 0
                        )

    def test_high_slopes_rebuild_across_crest_width_bounds(self) -> None:
        for slope in (1.66, 1.68, 1.7):
            for width in (6, 9, 12):
                with self.subTest(slope=slope, width=width):
                    design = replace(self.design, upstream_slope=slope, crest_width=width)
                    result = self.model.rebuild(design)
                    self.assertEqual(len(result.root.findall("./Contexts/Context")), 10)
                    self.assertTrue(all(c.polygon.is_valid for c in result.geometry.cells))

    def test_high_slope_template_reuse_and_remote_boundary_conditions(self) -> None:
        first = self.model.rebuild(self.design)
        parsed = GeoStudioXmlModel(first.root, reference_design=self.design)
        second = parsed.rebuild(self.design)
        for material in {c.material_id for c in first.geometry.cells}:
            before = unary_union([c.polygon for c in first.geometry.cells if c.material_id == material])
            after = unary_union([c.polygon for c in second.geometry.cells if c.material_id == material])
            self.assertLess(before.symmetric_difference(after).area, 1.0e-7)
        for reference in parsed.boundary_references:
            if (
                reference.container_tag in ("GeometryUsesXDispBCs", "GeometryUsesYDispBCs")
                and reference.attributes["Entry"] in {"1", "3"}
            ):
                endpoints = list(reference.segment.coords)
                self.assertTrue(
                    all(abs(x) < 1.0e-6 for x, _ in endpoints)
                    or all(abs(x - 558.25) < 1.0e-6 for x, _ in endpoints)
                    or all(abs(y - 272) < 1.0e-6 for _, y in endpoints),
                    endpoints,
                )


class InitialSampleSafetyTests(unittest.TestCase):
    def test_existing_samples_are_not_overwritten(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "designs.csv"
            output.write_text("existing experiment", encoding="utf-8")
            with redirect_stderr(io.StringIO()):
                code = main([
                    "sample", "--template", str(TEMPLATE), "--output", str(output)
                ])
            self.assertEqual(code, 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "existing experiment")

    def test_invalid_count_fails_without_writing_files(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "designs.csv"
            with self.assertRaisesRegex(ValueError, ">= 2"):
                write_initial_designs(TEMPLATE, output, count=1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
