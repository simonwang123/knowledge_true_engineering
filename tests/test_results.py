from __future__ import annotations

import math
import unittest
from pathlib import Path

from geostudio_automation.results import (
    DEFAULT_ANALYSIS_STEP,
    ElementNodeMembership,
    GaussPointResult,
    MaterialClassification,
    NodeResult,
    ResponseMeshData,
    WeightedStressPoint,
    build_weighted_stress_points,
    calculate_displacement_metrics,
    metric_fields,
    weighted_upper_tail,
)
from geostudio_automation.template import TemplateArchive
from geostudio_automation.workflow import read_designs
from geostudio_automation.xml_model import GeoStudioXmlModel

ROOT = Path(__file__).resolve().parents[1]


def weighted_point(
    point_id: int,
    sigma1: float,
    sigma3: float,
    area: float,
    *,
    x: float = 0.0,
) -> WeightedStressPoint:
    return WeightedStressPoint(
        gauss_point_id=point_id,
        element_id=point_id,
        x=x,
        y=0.0,
        sigma1_kpa=sigma1,
        sigma3_kpa=sigma3,
        area_weight_m2=area,
        material_id=4,
        material_name="主堆石区",
    )


class DisplacementMetricTests(unittest.TestCase):
    def test_default_result_selection_uses_final_saved_step(self) -> None:
        self.assertEqual(DEFAULT_ANALYSIS_STEP, 3)

    def test_crest_uses_all_crest_material_nodes_but_body_excludes_them(self) -> None:
        nodes = [
            NodeResult(1, 1.0, 10.0, 0.1, -0.3),
            NodeResult(2, 2.0, 9.0, 0.2, -0.5),
            NodeResult(3, 4.0, 8.0, 0.3, -0.2),
            NodeResult(4, 5.0, 7.0, 9.0, -9.0),
            NodeResult(5, 6.0, 6.0, -0.6, -0.1),
        ]
        classification = MaterialClassification(
            node_material_ids={
                1: (13,),
                2: (1, 13),
                3: (4,),
                4: (1, 4),
                5: (4,),
            },
            material_names={1: "面板", 4: "主堆石区", 13: "坝顶"},
            excluded_material_ids=frozenset({1, 13}),
            excluded_node_ids=frozenset({1, 2, 4}),
            element_material_ids={},
            crest_material_id=13,
        )
        derived = {
            "upstream_toe": (0.0, 0.0),
            "crest_left": (0.0, 10.0),
            "crest_right": (10.0, 10.0),
            "downstream_toe": (10.0, 0.0),
        }

        metrics, counts, _, locations = calculate_displacement_metrics(
            nodes, derived, classification
        )

        self.assertEqual(metrics["crest_max_settlement_m"], 0.5)
        self.assertEqual(metrics["dam_max_settlement_m"], 0.2)
        self.assertEqual(metrics["dam_max_horizontal_displacement_m"], 0.6)
        self.assertEqual(counts["crest_material_nodes"], 2)
        self.assertEqual(counts["dam_body_nodes"], 2)
        self.assertEqual(locations["crest_max_settlement"]["node_id"], 2)
        self.assertEqual(locations["dam_max_settlement"]["node_id"], 3)
        self.assertNotIn("settlement_gradient", metrics)
        self.assertNotIn("horizontal_displacement_gradient", metrics)

    def test_metric_fields_contain_only_five_responses(self) -> None:
        fields = metric_fields(0.98)
        self.assertEqual(len(fields), 5)
        self.assertNotIn("settlement_gradient", fields)
        self.assertNotIn("horizontal_displacement_gradient", fields)


class StressCvarTests(unittest.TestCase):
    def test_weighted_tail_uses_exact_area_and_shares_cutoff_ties(self) -> None:
        points = [
            weighted_point(1, 100.0, -100.0, 0.2, x=0.0),
            weighted_point(2, 50.0, -10.0, 0.4, x=1.0),
            weighted_point(3, 50.0, -10.0, 0.4, x=2.0),
        ]

        sigma1 = weighted_upper_tail(
            points, value=lambda point: point.sigma1_kpa, alpha=0.5
        )
        sigma3 = weighted_upper_tail(
            points, value=lambda point: point.sigma3_kpa, alpha=0.5
        )

        self.assertTrue(math.isclose(sigma1.selected_tail_area_m2, 0.5))
        self.assertTrue(math.isclose(sigma1.cvar_kpa, 70.0))
        cutoff = [item for item in sigma1.points if item["stress_kpa"] == 50.0]
        self.assertEqual(len(cutoff), 2)
        self.assertTrue(
            all(math.isclose(float(item["cutoff_fraction"]), 0.375) for item in cutoff)
        )
        self.assertEqual(sigma3.maximum_kpa, -10.0)
        self.assertEqual(sigma3.cvar_kpa, -10.0)

    def test_weighted_points_exclude_crest_elements_and_conserve_area(self) -> None:
        nodes = tuple(
            NodeResult(node_id, x, y, 0.0, 0.0)
            for node_id, x, y in (
                (1, 0.0, 0.0),
                (2, 1.0, 0.0),
                (3, 1.0, 1.0),
                (4, 0.0, 1.0),
                (5, 2.0, 0.0),
                (6, 2.0, 1.0),
            )
        )
        memberships = tuple(
            ElementNodeMembership(element_id, node_id)
            for element_id, node_ids in ((10, (1, 2, 3, 4)), (11, (2, 5, 6, 3)))
            for node_id in node_ids
        )
        gauss = tuple(
            GaussPointResult(point_id, element_id, x, y, stress, stress / 2)
            for point_id, element_id, x, y, stress in (
                (101, 10, 0.25, 0.25, 10.0),
                (102, 10, 0.75, 0.75, 20.0),
                (103, 11, 1.25, 0.25, 1000.0),
                (104, 11, 1.75, 0.75, 1000.0),
            )
        )
        mesh = ResponseMeshData(nodes, memberships, gauss, 8, 0)
        classification = MaterialClassification(
            node_material_ids={},
            material_names={4: "主堆石区", 13: "坝顶"},
            excluded_material_ids=frozenset({13}),
            excluded_node_ids=frozenset(),
            element_material_ids={10: 4, 11: 13},
            crest_material_id=13,
        )
        derived = {
            "upstream_toe": (0.0, 0.0),
            "crest_left": (0.0, 1.0),
            "crest_right": (2.0, 1.0),
            "downstream_toe": (2.0, 0.0),
        }

        points, counts, scope = build_weighted_stress_points(
            mesh, derived, classification
        )

        self.assertEqual({point.element_id for point in points}, {10})
        self.assertEqual(counts["eligible_stress_elements"], 1)
        self.assertEqual(counts["eligible_stress_gauss_points"], 2)
        self.assertTrue(math.isclose(float(scope["eligible_area_m2"]), 1.0))
        self.assertTrue(
            all(math.isclose(point.area_weight_m2, 0.5) for point in points)
        )


class CrestTemplateTests(unittest.TestCase):
    def test_template_and_rebuild_keep_stage_eight_crest_material(self) -> None:
        archive = TemplateArchive.load(ROOT / "templates" / "geo_studio_model_sample")
        model = GeoStudioXmlModel(
            archive.root(), reference_design=archive.reference_design
        )
        source = [region for region in model.original_regions if region.material_id == 13]
        self.assertEqual(len(source), 1)
        self.assertEqual(source[0].activation_stage, 8)

        design = read_designs(ROOT / "examples" / "designs.csv")[0]
        rebuilt = model.rebuild(design)
        crest_cells = [
            cell for cell in rebuilt.geometry.cells if cell.material_id == 13
        ]
        self.assertEqual(len(crest_cells), 1)
        self.assertEqual(crest_cells[0].activation_stage, 8)


if __name__ == "__main__":
    unittest.main()
