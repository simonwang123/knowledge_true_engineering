from __future__ import annotations

import json
import math
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from geostudio_automation.extraction import extract_projects
from geostudio_automation.model import DesignVariables
from geostudio_automation.optimization_config import Constraint, ExperimentConfig
from geostudio_automation.optimization_evaluation import (
    GeometryDomain,
    GeoStudioEvaluator,
    write_evaluation,
)
from geostudio_automation.results import (
    EXCLUDED_MATERIAL_NAME_KEYWORDS,
    METRIC_FIELDS,
    metric_fields,
)
from geostudio_automation.workflow import run_pipeline

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    os.environ.get("GEOSTUDIO_INTEGRATION") == "1",
    "set GEOSTUDIO_INTEGRATION=1 to use GeoStudio",
)
class GeoStudioIntegrationTests(unittest.TestCase):
    def test_optimization_evaluator_charges_once_and_recovers_a_solved_design(self) -> None:
        # Integration-only sentinel limits/prices exercise the adapter, not a design study.
        config = ExperimentConfig(
            template=ROOT / "templates/geo_studio_model_sample",
            constraints={name: Constraint(1e9, "<=") for name in METRIC_FIELDS},
            material_unit_costs={4: 1.0, 5: 1.0, 6: 1.0, 12: 1.0},
        )
        domain = GeometryDomain(config)
        evaluator = GeoStudioEvaluator(config, domain)
        design = DesignVariables("optimizer_integration", 1.46, 1.5, 12, 18, 20, 40, 55, 10)
        charged = []
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / design.design_id
            domain.prepare(design)
            result = evaluator.evaluate(design, directory, lambda: charged.append(True))
            self.assertTrue(result.succeeded, result.error)
            self.assertEqual(charged, [True])
            result.validate()
            write_evaluation(design, directory, result)
            recovered = evaluator.recover(design, directory)
            self.assertIsNotNone(recovered)
            self.assertEqual(recovered.metrics, result.metrics)

    def test_baseline_builds_solves_and_extracts_five_responses(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "runs"
            results = run_pipeline(
                ROOT / "templates" / "geo_studio_model_sample",
                ROOT / "examples" / "designs.csv",
                output,
                design_ids=["baseline_feasible"],
            )
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].succeeded, results[0].error_summary)
            run_dir = output / "baseline_feasible"
            mesh = json.loads(
                (run_dir / "mesh_report.json").read_text(encoding="utf-8")
            )
            solve = json.loads(
                (run_dir / "solve_report.json").read_text(encoding="utf-8")
            )
            self.assertTrue(mesh["validation"]["matched"])
            self.assertEqual(solve["status"], "solved")
            self.assertEqual(len(solve["analyses"]), 10)
            self.assertTrue(all(item["succeeded"] for item in solve["analyses"]))

            extraction = extract_projects(
                [run_dir / "baseline_feasible.gsz"], output / "extracted"
            )
            self.assertEqual(extraction.successful_projects, 1)
            response = json.loads(
                (
                    output
                    / "extracted"
                    / "per_run"
                    / "baseline_feasible"
                    / "response_metrics.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(tuple(response["metrics"]), metric_fields(0.98))
            self.assertTrue(
                all(math.isfinite(float(value)) for value in response["metrics"].values())
            )
            self.assertGreater(response["counts"]["crest_material_nodes"], 0)
            for key in ("dam_max_settlement", "dam_max_horizontal_displacement"):
                names = [
                    item["material_name"]
                    for item in response["locations"][key]["incident_materials"]
                ]
                self.assertFalse(
                    any(
                        keyword in name
                        for keyword in EXCLUDED_MATERIAL_NAME_KEYWORDS
                        for name in names
                    )
                )
            eligible_area = response["scope"]["dam_body_stresses"][
                "eligible_area_m2"
            ]
            for tail_name in (
                "sigma1_max_effective_stress",
                "sigma3_min_effective_stress",
            ):
                selected_area = response["stress_tail_details"][tail_name][
                    "selected_tail_area_m2"
                ]
                self.assertTrue(
                    math.isclose(selected_area, eligible_area * 0.02, rel_tol=1.0e-10)
                )


if __name__ == "__main__":
    unittest.main()
