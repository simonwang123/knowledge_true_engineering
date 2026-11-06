from __future__ import annotations

import csv
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from geostudio_automation.extraction import extract_projects
from geostudio_automation.results import (
    DEFAULT_ANALYSIS_NAME,
    EXCLUDED_MATERIAL_NAME_KEYWORDS,
    MATERIAL_FILTER_VERSION,
    RESULTS_SCHEMA_VERSION,
    metric_fields,
    project_provenance,
)


def parameters_payload(design_id: str = "design") -> dict[str, object]:
    return {
        "parameters": {
            "design_id": design_id,
            "upstream_slope": 1.46,
            "downstream_slope": 1.5,
            "crest_width": 12.0,
            "modulus_zone_height": 18.0,
            "secondary_top_width": 20.0,
            "secondary_height": 40.0,
            "secondary_bottom_width": 55.0,
            "secondary_right_slope_distance": 10.0,
        },
        "parameter_hash": "hash-001",
        "derived_coordinates": {
            "upstream_toe": [0.0, 0.0],
            "crest_left": [0.0, 10.0],
            "crest_right": [10.0, 10.0],
            "downstream_toe": [10.0, 0.0],
        },
    }


def response_payload(project: Path) -> dict[str, object]:
    fields = metric_fields(0.98)
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "status": "extracted",
        "design_id": "design",
        "parameter_hash": "hash-001",
        "analysis_name": DEFAULT_ANALYSIS_NAME,
        "step": 3,
        "cvar_alpha": 0.98,
        "project": project_provenance(project),
        "material_filter": {
            "version": MATERIAL_FILTER_VERSION,
            "excluded_material_name_keywords": list(
                EXCLUDED_MATERIAL_NAME_KEYWORDS
            ),
        },
        "scope": {
            "dam_body_stresses": {"eligible_area_m2": 100.0},
        },
        "counts": {
            "queried_nodes": 100,
            "queried_elements": 80,
            "queried_gauss_points": 320,
            "crest_material_nodes": 10,
            "dam_body_nodes": 50,
            "eligible_stress_elements": 40,
            "eligible_stress_gauss_points": 160,
        },
        "metrics": {field: float(index + 1) for index, field in enumerate(fields)},
        "stress_tail_details": {
            "sigma1_max_effective_stress": {
                "var_kpa": 1000.0,
                "maximum_kpa": 1200.0,
            },
            "sigma3_min_effective_stress": {
                "var_kpa": 400.0,
                "maximum_kpa": 500.0,
            },
        },
    }


class BatchExtractionTests(unittest.TestCase):
    def test_writes_five_response_dataset_and_reuses_cache(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "design"
            run.mkdir()
            project = run / "design.gsz"
            project.write_bytes(b"solved-project")
            (run / "parameters.json").write_text(
                json.dumps(parameters_payload()), encoding="utf-8"
            )
            (run / "solve_report.json").write_text(
                json.dumps({"status": "solved"}), encoding="utf-8"
            )
            output = root / "extracted"
            payload = response_payload(project)

            with patch(
                "geostudio_automation.extraction.extract_project_responses",
                return_value=SimpleNamespace(as_dict=lambda: payload),
            ) as extractor:
                first = extract_projects([project], output, cvar_alpha=0.98)
                second = extract_projects([project], output, cvar_alpha=0.98)

            self.assertEqual(first.successful_projects, 1)
            self.assertEqual(second.successful_projects, 1)
            self.assertTrue(second.outcomes[0].cached)
            extractor.assert_called_once()
            with first.dataset_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["crest_max_settlement_m"], "1.0")
            self.assertEqual(
                rows[0]["dam_sigma3_effective_stress_cvar98_kpa"], "5.0"
            )
            self.assertNotIn("settlement_gradient", rows[0])
            self.assertNotIn("horizontal_displacement_gradient", rows[0])
            self.assertTrue(first.audit_path.is_file())
            self.assertTrue(first.report_path.is_file())

    def test_project_newer_than_solve_report_is_a_failed_sample(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "design"
            run.mkdir()
            project = run / "design.gsz"
            project.write_bytes(b"solved-project")
            (run / "parameters.json").write_text(
                json.dumps(parameters_payload()), encoding="utf-8"
            )
            report = run / "solve_report.json"
            report.write_text(json.dumps({"status": "solved"}), encoding="utf-8")
            newer = report.stat().st_mtime_ns + 2_000_000_000
            os.utime(project, ns=(newer, newer))

            with patch(
                "geostudio_automation.extraction.extract_project_responses"
            ) as extractor:
                outputs = extract_projects([project], root / "extracted")

            self.assertEqual(outputs.failed_projects, 1)
            self.assertIn("newer than its solve report", outputs.outcomes[0].error_summary)
            extractor.assert_not_called()
            with outputs.dataset_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])


if __name__ == "__main__":
    unittest.main()
