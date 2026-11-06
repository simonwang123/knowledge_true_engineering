from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from geostudio_automation.cli import main

ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_optimize_rejects_stress_constraints_before_starting(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            config_path = Path(directory) / "invalid.json"
            config = json.loads((ROOT / "examples/optimization.json").read_text(encoding="utf-8"))
            config["constraints"]["dam_sigma1_effective_stress_cvar99p5_kpa"] = {
                "limit": 1000, "direction": "<="
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with patch("geostudio_automation.optimization.GeoStudioEvaluator") as evaluator:
                code = main(["optimize", "--config", str(config_path), "--output", str(output)])
            self.assertEqual(code, 2)
            evaluator.assert_not_called()
            self.assertFalse(output.exists())

    def test_synthetic_configuration_requires_explicit_simulate_flag(self) -> None:
        with TemporaryDirectory() as directory:
            with patch("geostudio_automation.optimization.GeoStudioEvaluator") as evaluator:
                code = main(["optimize", "--config", str(ROOT / "examples/optimization.simulation.json"), "--output", directory])
            self.assertEqual(code, 2)
            evaluator.assert_not_called()

    def test_optimize_cli_passes_mode_resume_and_simulate(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory)
            result = SimpleNamespace(report_path=output/"report.json", dataset_path=output/"dataset.csv",
                                     attempts_spent=23, successful_samples=23, best_design={"design_id": "x"}, stop_reason="budget_exhausted")
            with patch("geostudio_automation.optimization.run_optimization", return_value=result) as run:
                code = main(["optimize", "--config", str(ROOT / "examples/optimization.simulation.json"),
                             "--output", directory, "--simulate", "--resume", "--mode", "plain"])
            self.assertEqual(code, 0)
            self.assertEqual(run.call_args.kwargs["mode"], "plain")
            self.assertTrue(run.call_args.kwargs["simulate"])
            self.assertTrue(run.call_args.kwargs["resume"])

    def test_build_selected_design(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "runs"
            code = main(
                [
                    "build",
                    "--template",
                    str(ROOT / "templates" / "geo_studio_model_sample"),
                    "--designs",
                    str(ROOT / "examples" / "designs.csv"),
                    "--output",
                    str(output),
                    "--design-id",
                    "baseline_feasible",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(
                (output / "baseline_feasible" / "baseline_feasible.gsz").is_file()
            )

    def test_unknown_design_is_preflight_error(self) -> None:
        with TemporaryDirectory() as directory:
            code = main(
                [
                    "build",
                    "--template",
                    str(ROOT / "templates" / "geo_studio_model_sample"),
                    "--designs",
                    str(ROOT / "examples" / "designs.csv"),
                    "--output",
                    str(Path(directory) / "runs"),
                    "--design-id",
                    "missing",
                ]
            )
            self.assertEqual(code, 2)
            self.assertFalse((Path(directory) / "runs").exists())

    def test_extract_command_uses_independent_batch_stage(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "design.gsz"
            project.write_bytes(b"project")
            output = root / "extracted"
            outputs = SimpleNamespace(
                dataset_path=output / "dataset.csv",
                audit_path=output / "extraction_audit.csv",
                report_path=output / "extraction_report.json",
                failed_projects=0,
            )
            with patch(
                "geostudio_automation.cli.extract_projects",
                return_value=outputs,
            ) as extract:
                code = main(
                    [
                        "extract",
                        "--project",
                        str(project),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(code, 0)
            extract.assert_called_once()
            self.assertEqual(extract.call_args.kwargs["step"], 3)

    def test_extract_command_accepts_zero_step(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "design.gsz"
            project.write_bytes(b"project")
            output = root / "extracted"
            outputs = SimpleNamespace(
                dataset_path=output / "dataset.csv",
                audit_path=output / "extraction_audit.csv",
                report_path=output / "extraction_report.json",
                failed_projects=0,
            )
            with patch(
                "geostudio_automation.cli.extract_projects",
                return_value=outputs,
            ) as extract:
                code = main(
                    [
                        "extract",
                        "--project",
                        str(project),
                        "--output",
                        str(output),
                        "--step",
                        "0",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(extract.call_args.kwargs["step"], 0)

    def test_calibrate_kg_command_uses_independent_dataset_stage(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.csv"
            dataset.write_text("placeholder", encoding="utf-8")
            output = root / "calibration"
            outputs = SimpleNamespace(
                weights_path=output / "kg_weights.json",
                edge_calibration_path=output / "kg_edge_calibration.csv",
                report_path=output / "calibration_report.json",
                active_edges=4,
                edges=tuple(range(40)),
            )
            with patch(
                "geostudio_automation.cli.calibrate_kg_edges",
                return_value=outputs,
            ) as calibrate:
                code = main(
                    [
                        "calibrate-kg",
                        "--dataset",
                        str(dataset),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(code, 0)
            calibrate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
