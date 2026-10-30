from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import analyze


def make_normalized_rows(
    eager_steady: list[float],
    compiled_steady: list[float],
    eager_first: list[float] | None = None,
    compiled_first: list[float] | None = None,
) -> list[dict[str, object]]:
    eager_first = eager_first or [20.0] * len(eager_steady)
    compiled_first = compiled_first or [1020.0] * len(eager_steady)
    rows: list[dict[str, object]] = []
    for repetition in range(1, len(eager_steady) + 1):
        for mode, first_values, steady_values in (
            ("eager", eager_first, eager_steady),
            ("compiled", compiled_first, compiled_steady),
        ):
            for phase, values in (
                ("first_call", first_values),
                ("steady", steady_values),
            ):
                latency = values[repetition - 1]
                rows.append(
                    {
                        "run_id": "run-1",
                        "model": "resnet18",
                        "device": "cpu",
                        "seed": 20261029,
                        "batch_size": 1,
                        "num_threads": 1,
                        "repetition": repetition,
                        "execution_mode": mode,
                        "phase": phase,
                        "latency_ms": latency,
                        "throughput_samples_s": 1000.0 / latency,
                        "max_abs_error": 1e-6,
                    }
                )
    return rows


def make_raw_fixture() -> tuple[list[dict[str, str]], dict[str, object]]:
    configuration: dict[str, object] = {
        "model": "resnet18",
        "device": "cpu",
        "batch_sizes": [1, 8, 32],
        "num_threads": [1, 4],
        "num_interop_threads": 1,
        "repetitions": 5,
        "seed": 20261029,
        "rtol": 1e-4,
        "atol": 1e-5,
        "warmup_iterations": 5,
        "min_run_time_s": 1.0,
        "environment_platform": "Linux-test",
        "cpu_model": "Test CPU",
        "physical_cores": 4,
        "logical_cores": 4,
        "python_version": "3.12.16",
        "torch_version": "2.12.1+cpu",
        "torchvision_version": "0.27.1+cpu",
    }
    metadata: dict[str, object] = {
        "run_id": "run-1",
        "configuration": configuration,
        "correctness": {
            f"batch={batch},threads={threads}": {"allclose": True}
            for batch in (1, 8, 32)
            for threads in (1, 4)
        },
    }
    rows: list[dict[str, str]] = []
    for batch in (1, 8, 32):
        for threads in (1, 4):
            for repetition in range(1, 6):
                for mode in ("eager", "compiled"):
                    for phase in ("first_call", "steady"):
                        base = float(batch * 10 + threads + repetition)
                        if phase == "first_call":
                            latency = base + (1000.0 if mode == "compiled" else 1.0)
                        else:
                            latency = base * (0.8 if mode == "compiled" else 1.0)
                        rows.append(
                            {
                                "run_id": "run-1",
                                "os": "Linux-test",
                                "cpu_model": "Test CPU",
                                "physical_cores": "4",
                                "logical_cores": "4",
                                "gpu_model": "",
                                "python_version": "3.12.16",
                                "torch_version": "2.12.1+cpu",
                                "torchvision_version": "0.27.1+cpu",
                                "seed": "20261029",
                                "model": "resnet18",
                                "device": "cpu",
                                "execution_mode": mode,
                                "batch_size": str(batch),
                                "num_threads": str(threads),
                                "repetition": str(repetition),
                                "phase": phase,
                                "latency_ms": str(latency),
                                "throughput_samples_s": str(batch * 1000.0 / latency),
                                "allclose": "True",
                                "max_abs_error": "0.000001",
                                "rtol": "0.0001",
                                "atol": "0.00001",
                                "num_interop_threads": "1",
                                "warmup_iterations": "5",
                                "min_run_time_s": "1.0",
                            }
                        )
    return rows, metadata


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AnalysisTests(unittest.TestCase):
    def test_break_even_formula(self) -> None:
        self.assertEqual(
            analyze.calculate_break_even_requests(1000.0, 20.0, 10.0),
            100.0,
        )

    def test_break_even_is_none_without_steady_gain(self) -> None:
        self.assertIsNone(
            analyze.calculate_break_even_requests(1000.0, 10.0, 10.0)
        )
        self.assertIsNone(
            analyze.calculate_break_even_requests(1000.0, 10.0, 12.0)
        )

    def test_break_even_never_returns_negative_requests(self) -> None:
        self.assertEqual(
            analyze.calculate_break_even_requests(-10.0, 20.0, 10.0),
            0.0,
        )

    def test_summary_uses_medians_and_preserves_ranges(self) -> None:
        rows = make_normalized_rows(
            eager_steady=[18.0, 19.0, 20.0, 21.0, 40.0],
            compiled_steady=[8.0, 9.0, 10.0, 11.0, 30.0],
            eager_first=[18.0, 19.0, 20.0, 21.0, 22.0],
            compiled_first=[1018.0, 1019.0, 1020.0, 1021.0, 1022.0],
        )
        summary = analyze.summarize_rows(rows)
        self.assertEqual(len(summary), 1)
        result = summary[0]
        self.assertEqual(result["eager_steady_median_ms"], 20.0)
        self.assertEqual(result["compiled_steady_median_ms"], 10.0)
        self.assertEqual(result["eager_steady_min_ms"], 18.0)
        self.assertEqual(result["eager_steady_max_ms"], 40.0)
        self.assertEqual(result["steady_speedup"], 2.0)
        self.assertEqual(result["compile_overhead_ms"], 1000.0)
        self.assertEqual(result["break_even_requests"], 100.0)

    def test_summary_marks_slower_compiled_configuration_not_recoverable(self) -> None:
        rows = make_normalized_rows(
            eager_steady=[10.0] * 5,
            compiled_steady=[12.0] * 5,
        )
        result = analyze.summarize_rows(rows)[0]
        self.assertLess(result["steady_speedup"], 1.0)
        self.assertIsNone(result["break_even_requests"])
        self.assertEqual(analyze._csv_value(None), "null")

    def test_full_matrix_validation_accepts_complete_fixture(self) -> None:
        rows, metadata = make_raw_fixture()
        normalized = analyze.validate_and_normalize_rows(rows, metadata)
        self.assertEqual(len(normalized), 120)
        self.assertEqual(len(analyze.summarize_rows(normalized)), 6)

    def test_validation_rejects_corrupt_or_incomplete_evidence(self) -> None:
        def shorten_matrix(
            rows: list[dict[str, str]], metadata: dict[str, object]
        ) -> None:
            configuration = metadata["configuration"]
            assert isinstance(configuration, dict)
            configuration["batch_sizes"] = [1, 8]

        def reduce_repetitions(
            rows: list[dict[str, str]], metadata: dict[str, object]
        ) -> None:
            configuration = metadata["configuration"]
            assert isinstance(configuration, dict)
            configuration["repetitions"] = 4

        mutations = {
            "missing row": lambda rows, metadata: rows.pop(),
            "duplicate row": lambda rows, metadata: rows.append(dict(rows[0])),
            "failed correctness": lambda rows, metadata: rows[0].__setitem__(
                "allclose", "False"
            ),
            "bad throughput": lambda rows, metadata: rows[0].__setitem__(
                "throughput_samples_s", "1"
            ),
            "wrong model": lambda rows, metadata: rows[0].__setitem__(
                "model", "mobilenet_v3_small"
            ),
            "short matrix": shorten_matrix,
            "too few repetitions": reduce_repetitions,
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                rows, metadata = make_raw_fixture()
                mutate(rows, metadata)
                with self.assertRaises(RuntimeError):
                    analyze.validate_and_normalize_rows(rows, metadata)

    def test_output_path_protection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "summary.csv"
            existing.write_text("protected\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                analyze.ensure_outputs_available([existing], overwrite=False)
            with self.assertRaises(ValueError):
                analyze.validate_distinct_paths([existing], [existing])

    def test_controller_end_to_end_preserves_raw_input(self) -> None:
        rows, metadata = make_raw_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.csv"
            raw_metadata = root / "raw.metadata.json"
            with raw_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            raw_metadata.write_text(json.dumps(metadata), encoding="utf-8")
            raw_before_hash = sha256(raw_path)
            metadata_before_hash = sha256(raw_metadata)
            summary_path = root / "derived" / "summary.csv"
            figures_dir = root / "figures"
            analysis_metadata_path = root / "derived" / "analysis.json"
            command = [
                sys.executable,
                str(Path(analyze.__file__).resolve()),
                "--input",
                str(raw_path),
                "--input-metadata",
                str(raw_metadata),
                "--summary-output",
                str(summary_path),
                "--figures-dir",
                str(figures_dir),
                "--metadata-output",
                str(analysis_metadata_path),
            ]
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Validated 120 raw rows", completed.stdout)
            self.assertEqual(sha256(raw_path), raw_before_hash)
            self.assertEqual(sha256(raw_metadata), metadata_before_hash)
            with summary_path.open(newline="", encoding="utf-8") as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(len(summary), 6)
            output_files = {
                "summary_csv": summary_path,
                "latency_figure": figures_dir / "latency.png",
                "throughput_figure": figures_dir / "throughput.png",
                "break_even_figure": figures_dir / "break_even.png",
            }
            for path in output_files.values():
                self.assertTrue(path.is_file())
            provenance = json.loads(
                analysis_metadata_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                provenance["source"]["raw_results_sha256"], raw_before_hash
            )
            self.assertEqual(
                provenance["source"]["benchmark_metadata_sha256"],
                metadata_before_hash,
            )
            self.assertEqual(
                provenance["analysis_environment"]["analysis_script_sha256"],
                sha256(Path(analyze.__file__).resolve()),
            )
            for label, path in output_files.items():
                self.assertEqual(
                    provenance["outputs"]["sha256"][label],
                    sha256(path),
                )
            summary_before_hash = sha256(summary_path)
            refused = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("derived output already exists", refused.stderr)
            self.assertEqual(sha256(summary_path), summary_before_hash)


if __name__ == "__main__":
    unittest.main()
