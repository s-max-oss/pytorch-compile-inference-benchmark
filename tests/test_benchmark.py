from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import benchmark


def valid_args() -> argparse.Namespace:
    return argparse.Namespace(
        matrix=False,
        resume=False,
        overwrite=False,
        batch_size=1,
        num_threads=4,
        repetitions=1,
        warmup_iterations=5,
        min_run_time=1.0,
        rtol=1e-4,
        atol=1e-5,
    )


def matrix_args(directory: str, resume: bool = False) -> argparse.Namespace:
    root = Path(directory)
    environment_path = root / "environment.json"
    environment_path.write_text("{}\n", encoding="utf-8")
    return argparse.Namespace(
        matrix=True,
        resume=resume,
        overwrite=False,
        batch_size=1,
        num_threads=4,
        repetitions=5,
        warmup_iterations=5,
        min_run_time=1.0,
        rtol=1e-4,
        atol=1e-5,
        seed=benchmark.DEFAULT_SEED,
        environment=environment_path,
        output=root / "raw_results.csv",
        metadata_output=None,
    )


def fake_environment() -> dict[str, object]:
    return {
        "os": {"platform": "Linux-test"},
        "cpu": {
            "model": "Test CPU",
            "physical_cores": 4,
            "logical_cores": 4,
        },
        "gpus": [],
        "python": {"version": "3.12.0 test"},
    }


def fake_worker(
    args: argparse.Namespace,
    worker: str,
    execution_mode: str | None = None,
) -> dict[str, object]:
    if worker == "correctness":
        return {
            "allclose": True,
            "max_abs_error": 1e-6,
            "rtol": args.rtol,
            "atol": args.atol,
        }
    return {
        "first_timestamp": "2026-10-30T00:00:00+00:00",
        "first_latency_ms": 10.0,
        "steady_timestamp": "2026-10-30T00:00:01+00:00",
        "steady_latency_ms": 5.0,
        "steady_iqr_ms": 0.1,
        "timer_block_count": 4,
        "number_per_run": 2,
    }


class BenchmarkValidationTests(unittest.TestCase):
    def test_throughput(self) -> None:
        self.assertEqual(benchmark.throughput_samples_s(8, 2.0), 4000.0)

    def test_non_finite_timing_and_tolerances_are_rejected(self) -> None:
        environment = {
            "CONDA_DEFAULT_ENV": "ai-infra-compile",
            "CC": "/tmp/cc",
            "CXX": "/tmp/cxx",
            "LD_LIBRARY_PATH": "/tmp/lib",
        }
        for field in ("min_run_time", "rtol", "atol"):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(field=field, value=value):
                    args = valid_args()
                    setattr(args, field, value)
                    with mock.patch.dict(os.environ, environment, clear=True):
                        with self.assertRaises(ValueError):
                            benchmark.validate_controller_environment(args)

    def test_matrix_requires_five_repetitions(self) -> None:
        args = valid_args()
        args.matrix = True
        args.repetitions = 4
        with self.assertRaisesRegex(ValueError, "at least 5"):
            benchmark.validate_controller_environment(args)

    def test_matrix_disables_overwrite(self) -> None:
        args = valid_args()
        args.matrix = True
        args.repetitions = 5
        args.overwrite = True
        with self.assertRaisesRegex(ValueError, "disabled for --matrix"):
            benchmark.validate_controller_environment(args)

    def test_environment_snapshot_rejects_changed_core_allocation(self) -> None:
        environment = {
            "conda": {"prefix": os.environ.get("CONDA_PREFIX")},
            "python": {
                "executable": benchmark.sys.executable,
                "version": benchmark.platform.python_version(),
            },
            "os": {"platform": benchmark.platform.platform()},
            "cpu": {
                "model": "Test CPU",
                "physical_cores": 4,
                "logical_cores": 4,
            },
            "pytorch": {
                "torch_version": benchmark.torch.__version__,
                "torchvision_version": benchmark.torchvision.__version__,
            },
            "environment_variables": benchmark.performance_environment(
                benchmark.INITIAL_PROCESS_ENVIRONMENT
            ),
        }
        with (
            mock.patch.object(benchmark, "current_cpu_model", return_value="Test CPU"),
            mock.patch.object(benchmark, "current_cpu_counts", return_value=(2, 4)),
        ):
            with self.assertRaisesRegex(RuntimeError, "physical CPU cores"):
                benchmark.validate_environment_snapshot(environment)

    def test_matrix_thread_counts_deduplicate_physical_cores(self) -> None:
        self.assertEqual(
            benchmark.matrix_thread_counts({"cpu": {"physical_cores": 4}}),
            (1, 4),
        )
        self.assertEqual(
            benchmark.matrix_thread_counts({"cpu": {"physical_cores": 2}}),
            (1, 4, 2),
        )

    def test_input_and_output_paths_must_be_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shared = Path(directory) / "same-file"
            with self.assertRaises(ValueError):
                benchmark.validate_distinct_paths(
                    {
                        "environment": shared,
                        "output": shared,
                        "metadata_output": Path(directory) / "metadata.json",
                    }
                )

    def test_performance_environment_filters_relevant_values(self) -> None:
        selected = benchmark.performance_environment(
            {
                "CC": "/compiler/cc",
                "OMP_NUM_THREADS": "4",
                "GOMP_CPU_AFFINITY": "0-3",
                "DNNL_MAX_CPU_ISA": "AVX2",
                "ATEN_CPU_CAPABILITY": "avx2",
                "TORCH_LOGS": "graph_breaks",
                "UNRELATED": "ignored",
            }
        )
        self.assertEqual(
            selected,
            {
                "CC": "/compiler/cc",
                "ATEN_CPU_CAPABILITY": "avx2",
                "DNNL_MAX_CPU_ISA": "AVX2",
                "GOMP_CPU_AFFINITY": "0-3",
                "OMP_NUM_THREADS": "4",
                "TORCH_LOGS": "graph_breaks",
            },
        )

    def test_csv_temp_file_cannot_overwrite_similarly_named_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.csv"
            protected_input = Path(directory) / "result.csv.tmp"
            protected_input.write_text("environment snapshot", encoding="utf-8")
            row = {column: "" for column in benchmark.CSV_COLUMNS}
            benchmark.write_csv(output, [row], overwrite=False)
            self.assertEqual(
                protected_input.read_text(encoding="utf-8"),
                "environment snapshot",
            )
            self.assertTrue(output.is_file())

    def test_matrix_writes_complete_120_row_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = matrix_args(directory)
            metadata_path = benchmark.metadata_output_path(args)
            with mock.patch.object(
                benchmark, "run_fresh_worker", side_effect=fake_worker
            ):
                benchmark.matrix_controller(args, fake_environment(), metadata_path)

            with args.output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 120)
            self.assertEqual(
                {
                    (int(row["batch_size"]), int(row["num_threads"]))
                    for row in rows
                },
                {(1, 1), (1, 4), (8, 1), (8, 4), (32, 1), (32, 4)},
            )
            metadata = benchmark.load_json(metadata_path)
            self.assertEqual(metadata["checkpoint"]["completed_measurements"], 60)
            self.assertEqual(len(metadata["correctness"]), 6)
            partial_csv, state_json = benchmark.checkpoint_paths(args.output)
            self.assertFalse(partial_csv.exists())
            self.assertFalse(state_json.exists())

    def test_matrix_resume_skips_completed_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = matrix_args(directory, resume=True)
            environment = fake_environment()
            configurations = tuple(
                (batch_size, num_threads)
                for batch_size in benchmark.MATRIX_BATCH_SIZES
                for num_threads in benchmark.matrix_thread_counts(environment)
            )
            correctness = {
                benchmark.matrix_config_key(batch_size, num_threads): {
                    "allclose": True,
                    "max_abs_error": 1e-6,
                    "rtol": args.rtol,
                    "atol": args.atol,
                }
                for batch_size, num_threads in configurations
            }
            fingerprint = benchmark.matrix_fingerprint(
                args,
                environment,
                benchmark.MATRIX_BATCH_SIZES,
                benchmark.matrix_thread_counts(environment),
            )
            state = {
                "schema_version": 1,
                "run_id": "test-run-id",
                "status": "in_progress",
                "started_at_utc": "2026-10-30T00:00:00+00:00",
                "last_updated_at_utc": "2026-10-30T00:00:00+00:00",
                "resume_count": 0,
                "command": "python benchmark.py --matrix",
                "fingerprint": fingerprint,
                "correctness": correctness,
            }
            partial_csv, state_json = benchmark.checkpoint_paths(args.output)
            benchmark.write_json(state_json, state, overwrite=False)
            completed_args = benchmark.configured_args(args, 1, 1)
            completed_args.run_id = state["run_id"]
            initial_rows = benchmark.result_rows(
                environment,
                completed_args,
                correctness[benchmark.matrix_config_key(1, 1)],
                "eager",
                1,
                fake_worker(completed_args, "benchmark", "eager"),
            )
            benchmark.write_csv(partial_csv, initial_rows, overwrite=False)

            calls: list[tuple[str, str | None]] = []

            def recording_worker(
                worker_args: argparse.Namespace,
                worker: str,
                execution_mode: str | None = None,
            ) -> dict[str, object]:
                calls.append((worker, execution_mode))
                return fake_worker(worker_args, worker, execution_mode)

            metadata_path = benchmark.metadata_output_path(args)
            with mock.patch.object(
                benchmark, "run_fresh_worker", side_effect=recording_worker
            ):
                benchmark.matrix_controller(args, environment, metadata_path)

            self.assertEqual(calls.count(("benchmark", "eager")), 29)
            self.assertEqual(calls.count(("benchmark", "compiled")), 30)
            self.assertNotIn(("correctness", None), calls)
            with args.output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 120)
            metadata = benchmark.load_json(metadata_path)
            self.assertEqual(metadata["checkpoint"]["resume_count"], 1)

    def test_checkpoint_rows_must_match_state_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = matrix_args(directory)
            environment = fake_environment()
            fingerprint = benchmark.matrix_fingerprint(
                args,
                environment,
                benchmark.MATRIX_BATCH_SIZES,
                benchmark.matrix_thread_counts(environment),
            )
            row_args = benchmark.configured_args(args, 1, 1)
            row_args.run_id = "different-run"
            correctness = {
                "allclose": True,
                "max_abs_error": 1e-6,
                "rtol": args.rtol,
                "atol": args.atol,
            }
            rows = benchmark.result_rows(
                environment,
                row_args,
                correctness,
                "eager",
                1,
                fake_worker(row_args, "benchmark", "eager"),
            )
            with self.assertRaisesRegex(RuntimeError, "run_id"):
                benchmark.completed_checkpoint_measurements(
                    rows,
                    ((1, 1),),
                    args,
                    {benchmark.matrix_config_key(1, 1): correctness},
                    "expected-run",
                    fingerprint,
                )

    def test_resume_rejects_unrelated_final_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "raw_results.csv"
            metadata = root / "raw_results.metadata.json"
            partial = root / "raw_results.partial.csv"
            output.write_text("unrelated\n", encoding="utf-8")
            partial.write_text("checkpoint\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "not identical"):
                benchmark.validate_resume_final_artifacts(
                    output, metadata, partial, "expected-run"
                )

            output.write_text("checkpoint\n", encoding="utf-8")
            metadata.write_text(
                '{"run_id": "another-run"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(FileExistsError, "another run"):
                benchmark.validate_resume_final_artifacts(
                    output, metadata, partial, "expected-run"
                )


if __name__ == "__main__":
    unittest.main()
