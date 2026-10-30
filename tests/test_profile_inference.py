from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import profile_inference


class FakeEvent:
    def __init__(
        self,
        key: str,
        count: int,
        self_cpu_time_total: float,
        cpu_time_total: float,
        input_shapes: list[list[int]],
    ) -> None:
        self.key = key
        self.count = count
        self.self_cpu_time_total = self_cpu_time_total
        self.cpu_time_total = cpu_time_total
        self.input_shapes = input_shapes


def controller_args(directory: str) -> argparse.Namespace:
    root = Path(directory)
    environment = root / "environment.json"
    environment.write_text("{}\n", encoding="utf-8")
    return argparse.Namespace(
        batch_size=32,
        num_threads=4,
        seed=20261029,
        rtol=1e-4,
        atol=1e-5,
        warmup_iterations=5,
        profile_iterations=3,
        environment=environment,
        output_dir=root / "profiler",
        worker=False,
        execution_mode=None,
        run_id=None,
    )


class ProfileInferenceTests(unittest.TestCase):
    def test_operator_rows_rank_and_preserve_all_events(self) -> None:
        events = [
            FakeEvent("small", 2, 20.0, 30.0, [[1, 2]]),
            FakeEvent("large", 1, 80.0, 90.0, [[2, 2]]),
        ]
        rows = profile_inference.operator_rows(events, "run", "eager", 3)
        self.assertEqual([row["operator"] for row in rows], ["large", "small"])
        self.assertEqual([row["rank_by_self_cpu"] for row in rows], [1, 2])
        self.assertAlmostEqual(sum(row["self_cpu_percent"] for row in rows), 100.0)
        self.assertEqual(rows[1]["self_cpu_time_avg_us"], 10.0)
        self.assertEqual(rows[0]["input_shapes"], "[[2,2]]")

    def test_operator_rows_reject_invalid_event(self) -> None:
        event = FakeEvent("invalid", 0, 1.0, 1.0, [])
        with self.assertRaisesRegex(RuntimeError, "invalid count"):
            profile_inference.operator_rows([event], "run", "eager", 1)

    def test_controller_validation_rejects_nonpositive_counts(self) -> None:
        args = argparse.Namespace(
            batch_size=32,
            num_threads=4,
            warmup_iterations=0,
            profile_iterations=3,
        )
        with self.assertRaisesRegex(ValueError, "warmup-iterations"):
            profile_inference.validate_controller_args(args)

    def test_controller_writes_both_modes_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = controller_args(directory)
            environment = {
                "cpu": {"logical_cores": 4},
            }

            def fake_profile_worker(
                worker_args: argparse.Namespace,
                execution_mode: str,
            ) -> dict[str, object]:
                return {
                    "execution_mode": execution_mode,
                    "profile_started_at": "2026-10-30T00:00:00+00:00",
                    "profiled_wall_time_ms": 1.0,
                    "operator_count": 1,
                    "operators": [
                        {
                            "run_id": worker_args.run_id,
                            "execution_mode": execution_mode,
                            "rank_by_self_cpu": 1,
                            "operator": f"{execution_mode}_operator",
                            "input_shapes": "[]",
                            "call_count": 1,
                            "self_cpu_time_total_us": 10.0,
                            "self_cpu_time_avg_us": 10.0,
                            "cpu_time_total_us": 10.0,
                            "cpu_time_avg_us": 10.0,
                            "self_cpu_percent": 100.0,
                            "profile_iterations": worker_args.profile_iterations,
                        }
                    ],
                }

            correctness = {
                "allclose": True,
                "max_abs_error": 1e-6,
                "rtol": args.rtol,
                "atol": args.atol,
            }
            required_environment = {
                "CONDA_DEFAULT_ENV": "ai-infra-compile",
                "CC": "/compiler/cc",
                "CXX": "/compiler/cxx",
                "LD_LIBRARY_PATH": "/compiler/lib",
            }
            with (
                mock.patch.dict(os.environ, required_environment, clear=True),
                mock.patch.object(
                    profile_inference.benchmark,
                    "load_environment",
                    return_value=environment,
                ),
                mock.patch.object(
                    profile_inference.benchmark,
                    "validate_environment_snapshot",
                ),
                mock.patch.object(
                    profile_inference.benchmark,
                    "run_fresh_worker",
                    return_value=correctness,
                ),
                mock.patch.object(
                    profile_inference,
                    "run_profile_worker",
                    side_effect=fake_profile_worker,
                ),
            ):
                profile_inference.controller_main(args)

            operator_path = args.output_dir / "operator_summary.csv"
            metadata_path = args.output_dir / "metadata.json"
            with operator_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                {row["execution_mode"] for row in rows},
                {"eager", "compiled"},
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["run_id"], rows[0]["run_id"])
            self.assertFalse(metadata["isolation"]["compilation_profiled"])
            self.assertIn("not be used", metadata["interpretation_limit"])

    def test_existing_profiler_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "operator_summary.csv"
            output.write_text("protected\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                profile_inference.write_operator_csv(output, [])
            self.assertEqual(output.read_text(encoding="utf-8"), "protected\n")


if __name__ == "__main__":
    unittest.main()
