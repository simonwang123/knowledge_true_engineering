from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from geostudio_automation.mesh import MeshValidation
from geostudio_automation.runtime import AnalysisSolve, RuntimeInfo
from geostudio_automation.workflow import (
    build_designs,
    plan_project_destinations,
    read_designs,
    remesh_project,
    remesh_projects,
    resolve_projects,
    select_designs,
    solve_project,
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "geo_studio_model_sample"
HEADER = [
    "design_id",
    "upstream_slope",
    "downstream_slope",
    "crest_width",
    "modulus_zone_height",
    "secondary_top_width",
    "secondary_height",
    "secondary_bottom_width",
    "secondary_right_slope_distance",
]


def write_designs(path: Path, ids: tuple[str, ...] = ("first", "second")) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        for design_id in ids:
            writer.writerow(
                [
                    design_id,
                    1.46,
                    1.5,
                    12,
                    18,
                    20,
                    40,
                    55,
                    16.641005886756876,
                ]
            )


def fake_runtime() -> RuntimeInfo:
    return RuntimeInfo("fake", "fake.exe", "25.2.0.700", "2025.2.0")


def matching_mesh() -> MeshValidation:
    return MeshValidation(2, "mesh_2.ply", 100, 98, 98, 1.0e-5, True, ())


class DesignInputTests(unittest.TestCase):
    def test_selects_by_id_in_csv_order_and_rejects_unknown(self) -> None:
        with TemporaryDirectory() as directory:
            csv_path = Path(directory) / "designs.csv"
            write_designs(csv_path)
            designs = read_designs(csv_path)
        selected = select_designs(designs, ["second"])
        self.assertEqual([item.design_id for item in selected], ["second"])
        with self.assertRaisesRegex(ValueError, "unknown design_id"):
            select_designs(designs, ["missing"])

    def test_duplicate_csv_id_is_rejected_before_build(self) -> None:
        with TemporaryDirectory() as directory:
            csv_path = Path(directory) / "designs.csv"
            write_designs(csv_path, ("same", "same"))
            with self.assertRaisesRegex(ValueError, "duplicate design_id"):
                read_designs(csv_path)

    def test_obsolete_angle_and_clearance_csv_gets_a_migration_error(self) -> None:
        with TemporaryDirectory() as directory:
            csv_path = Path(directory) / "old.csv"
            old_header = [
                *HEADER[:-2],
                "secondary_left_bottom_angle",
                "secondary_bottom_width",
                "secondary_bottom_clearance",
            ]
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(old_header)
                writer.writerow(
                    ["old", 1.46, 1.5, 12, 18, 20, 40, 70, 55, 30]
                )
            with self.assertRaisesRegex(ValueError, "obsolete"):
                read_designs(csv_path)

    def test_partial_build_and_overwrite_protection(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "designs.csv"
            write_designs(csv_path)
            output = root / "runs"
            first = build_designs(
                TEMPLATE,
                csv_path,
                output,
                design_ids=["second"],
            )
            self.assertEqual([item.design_id for item in first], ["second"])
            self.assertTrue(first[0].succeeded)
            self.assertFalse((output / "first").exists())
            second = build_designs(
                TEMPLATE,
                csv_path,
                output,
                design_ids=["second"],
            )
            self.assertFalse(second[0].succeeded)
            self.assertIn("--force", second[0].error_summary)


class ProjectSelectionTests(unittest.TestCase):
    def test_directory_glob_is_non_recursive_by_default(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "direct.gsz").write_bytes(b"x")
            nested = root / "nested"
            nested.mkdir()
            (nested / "nested.gsz").write_bytes(b"x")
            direct = resolve_projects(project_dir=root)
            recursive = resolve_projects(project_dir=root, glob_pattern="**/*.gsz")
        self.assertEqual([item.name for item in direct], ["direct.gsz"])
        self.assertEqual({item.name for item in recursive}, {"direct.gsz", "nested.gsz"})

    def test_output_project_requires_one_input_and_output_dir_checks_collisions(self) -> None:
        left = Path("a") / "same.gsz"
        right = Path("b") / "same.gsz"
        with self.assertRaisesRegex(ValueError, "exactly one"):
            plan_project_destinations(
                [left, right], output_project=Path("out.gsz")
            )
        with self.assertRaisesRegex(ValueError, "same destination"):
            plan_project_destinations([left, right], output_dir=Path("out"))

    def test_existing_copied_output_is_rejected_during_preflight(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.gsz"
            destination = root / "destination.gsz"
            source.write_bytes(b"source")
            destination.write_bytes(b"existing")
            with self.assertRaisesRegex(ValueError, "--overwrite-output"):
                plan_project_destinations(
                    [source], output_project=destination
                )
            planned = plan_project_destinations(
                [source],
                output_project=destination,
                overwrite_output=True,
            )
        self.assertEqual(planned, [destination.resolve()])


class IndependentStageTests(unittest.TestCase):
    def test_remesh_operates_without_build_report_and_can_copy(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "manual.gsz"
            source.write_bytes(b"manual project")
            destination = root / "copied.gsz"
            with patch(
                "geostudio_automation.workflow.force_native_remesh_and_validate",
                return_value=matching_mesh(),
            ) as remesh:
                result = remesh_project(
                    source,
                    output_project=destination,
                    runtime=fake_runtime(),
                )
            self.assertTrue(result.succeeded)
            self.assertEqual(destination.read_bytes(), b"manual project")
            self.assertFalse((root / "build_report.json").exists())
            remesh.assert_called_once()

    def test_solve_validates_mesh_and_records_all_ten_stages(self) -> None:
        with TemporaryDirectory() as directory:
            project = Path(directory) / "manual.gsz"
            project.write_bytes(b"manual project")
            names = [f"stage-{index}" for index in range(1, 11)]

            def fake_solve(path, analysis_names, *, on_result=None):
                results = [AnalysisSolve(name, True, "") for name in analysis_names]
                for item in results:
                    if on_result:
                        on_result(item)
                return results

            with (
                patch(
                    "geostudio_automation.workflow.validate_gsz_mesh",
                    return_value=matching_mesh(),
                ),
                patch(
                    "geostudio_automation.workflow.analysis_names",
                    return_value=names,
                ),
                patch(
                    "geostudio_automation.workflow.solve_analyses",
                    side_effect=fake_solve,
                ),
            ):
                result = solve_project(project, runtime=fake_runtime())
            self.assertTrue(result.succeeded, result.error_summary)
            self.assertEqual(len(result.analyses), 10)
            report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "solved")
            self.assertEqual(len(report["analyses"]), 10)

    def test_solve_refuses_a_mismatched_mesh_before_loading_gsi(self) -> None:
        with TemporaryDirectory() as directory:
            project = Path(directory) / "manual.gsz"
            project.write_bytes(b"manual project")
            mismatch = MeshValidation(
                2,
                "mesh_2.ply",
                100,
                97,
                98,
                1.0e-5,
                False,
                (),
            )
            with (
                patch(
                    "geostudio_automation.workflow.validate_gsz_mesh",
                    return_value=mismatch,
                ),
                patch("geostudio_automation.workflow.solve_analyses") as solve,
            ):
                result = solve_project(project, runtime=fake_runtime())
            self.assertFalse(result.succeeded)
            self.assertIn("mesh validation failed", result.error_summary)
            solve.assert_not_called()

    def test_batch_continues_by_default_and_fail_fast_stops(self) -> None:
        sources = [Path("one.gsz"), Path("two.gsz")]
        destinations = list(sources)
        failed = type("R", (), {"succeeded": False})()
        passed = type("R", (), {"succeeded": True})()
        with patch(
            "geostudio_automation.workflow.remesh_project",
            side_effect=[failed, passed],
        ) as remesh:
            results = remesh_projects(
                sources,
                destinations,
                runtime=fake_runtime(),
            )
        self.assertEqual(len(results), 2)
        self.assertEqual(remesh.call_count, 2)
        with patch(
            "geostudio_automation.workflow.remesh_project",
            return_value=failed,
        ) as remesh:
            results = remesh_projects(
                sources,
                destinations,
                runtime=fake_runtime(),
                fail_fast=True,
            )
        self.assertEqual(len(results), 1)
        self.assertEqual(remesh.call_count, 1)


if __name__ == "__main__":
    unittest.main()
