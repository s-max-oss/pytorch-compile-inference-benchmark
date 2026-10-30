"""Validate and summarize the full benchmark matrix, then generate figures.

All reported statistics are derived from ``results/raw_results.csv``.  The
script never edits the raw input and refuses to replace derived artifacts
unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXECUTION_MODES = ("eager", "compiled")
PHASES = ("first_call", "steady")
MIN_REPETITIONS = 5
REQUIRED_BATCH_SIZES = (1, 8, 32)
REQUIRED_THREAD_COUNTS = (1, 4)
REQUIRED_MODEL = "resnet18"
REQUIRED_DEVICE = "cpu"

REQUIRED_RAW_COLUMNS = {
    "run_id",
    "os",
    "cpu_model",
    "physical_cores",
    "logical_cores",
    "gpu_model",
    "python_version",
    "torch_version",
    "torchvision_version",
    "seed",
    "model",
    "device",
    "execution_mode",
    "batch_size",
    "num_threads",
    "repetition",
    "phase",
    "latency_ms",
    "throughput_samples_s",
    "allclose",
    "max_abs_error",
    "rtol",
    "atol",
    "num_interop_threads",
    "warmup_iterations",
    "min_run_time_s",
}

SUMMARY_COLUMNS = [
    "run_id",
    "model",
    "device",
    "seed",
    "batch_size",
    "num_threads",
    "repetitions",
    "allclose",
    "max_abs_error",
    "eager_first_median_ms",
    "eager_first_min_ms",
    "eager_first_max_ms",
    "compiled_first_median_ms",
    "compiled_first_min_ms",
    "compiled_first_max_ms",
    "eager_steady_median_ms",
    "eager_steady_min_ms",
    "eager_steady_max_ms",
    "compiled_steady_median_ms",
    "compiled_steady_min_ms",
    "compiled_steady_max_ms",
    "eager_steady_throughput_median_samples_s",
    "eager_steady_throughput_min_samples_s",
    "eager_steady_throughput_max_samples_s",
    "compiled_steady_throughput_median_samples_s",
    "compiled_steady_throughput_min_samples_s",
    "compiled_steady_throughput_max_samples_s",
    "steady_speedup",
    "compile_overhead_ms",
    "steady_latency_saving_ms",
    "break_even_requests",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_provenance(project_root: Path) -> dict[str, Any]:
    """Return Git identity when available without requiring a Git repository."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(project_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"repository_detected": False, "commit": None, "dirty": None}
    return {"repository_detected": True, "commit": commit, "dirty": dirty}


def calculate_break_even_requests(
    compile_overhead_ms: float,
    eager_steady_ms: float,
    compiled_steady_ms: float,
) -> float | None:
    """Return requests needed to recover compile overhead, or None.

    A compiled configuration that is not faster at steady state cannot repay
    its compile cost.  A non-positive measured overhead is treated as already
    repaid, so this function never emits a negative request count.
    """
    values = (compile_overhead_ms, eager_steady_ms, compiled_steady_ms)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("break-even inputs must be finite")
    if eager_steady_ms <= 0 or compiled_steady_ms <= 0:
        raise ValueError("steady-state latencies must be positive")
    steady_saving_ms = eager_steady_ms - compiled_steady_ms
    if steady_saving_ms <= 0:
        return None
    return max(0.0, compile_overhead_ms) / steady_saving_ms


def _parse_positive_float(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"raw results contain invalid {field}") from exc
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"raw results contain non-positive {field}")
    return value


def _parse_nonnegative_float(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"raw results contain invalid {field}") from exc
    if not math.isfinite(value) or value < 0:
        raise RuntimeError(f"raw results contain negative {field}")
    return value


def read_raw_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"raw results not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError("raw results have no CSV header")
        missing = REQUIRED_RAW_COLUMNS - set(reader.fieldnames)
        if missing:
            raise RuntimeError(
                "raw results are missing required columns: "
                + ", ".join(sorted(missing))
            )
        return list(reader)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"metadata not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return data


def validate_and_normalize_rows(
    raw_rows: list[dict[str, str]],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate the matrix against metadata and convert numeric fields."""
    configuration = metadata.get("configuration")
    if not isinstance(configuration, dict):
        raise RuntimeError("benchmark metadata has no configuration object")
    try:
        batch_sizes = tuple(int(value) for value in configuration["batch_sizes"])
        thread_counts = tuple(int(value) for value in configuration["num_threads"])
        repetitions = int(configuration["repetitions"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("benchmark metadata matrix is invalid") from exc
    if repetitions < MIN_REPETITIONS:
        raise RuntimeError(
            f"analysis requires at least {MIN_REPETITIONS} independent repetitions"
        )
    if not batch_sizes or not thread_counts:
        raise RuntimeError("benchmark metadata matrix is empty")
    if batch_sizes != REQUIRED_BATCH_SIZES:
        raise RuntimeError(
            "benchmark metadata does not contain the required batch matrix "
            f"{REQUIRED_BATCH_SIZES}"
        )
    try:
        physical_cores = int(configuration["physical_cores"])
        logical_cores = int(configuration["logical_cores"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("benchmark metadata core counts are invalid") from exc
    if physical_cores <= 0 or logical_cores <= 0:
        raise RuntimeError("benchmark metadata core counts must be positive")
    expected_thread_counts = tuple(
        sorted(set((*REQUIRED_THREAD_COUNTS, physical_cores)))
    )
    if tuple(sorted(set(thread_counts))) != expected_thread_counts:
        raise RuntimeError(
            "benchmark metadata does not contain the required thread matrix "
            f"{expected_thread_counts}"
        )
    if len(thread_counts) != len(set(thread_counts)):
        raise RuntimeError("benchmark metadata thread matrix contains duplicates")
    for field, expected in {
        "model": REQUIRED_MODEL,
        "device": REQUIRED_DEVICE,
    }.items():
        if configuration.get(field) != expected:
            raise RuntimeError(f"benchmark metadata {field} must be {expected!r}")
    try:
        expected_seed = int(configuration["seed"])
        expected_rtol = float(configuration["rtol"])
        expected_atol = float(configuration["atol"])
        expected_interop_threads = int(configuration["num_interop_threads"])
        expected_warmup = int(configuration["warmup_iterations"])
        expected_min_run_time = float(configuration["min_run_time_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("benchmark metadata experiment settings are invalid") from exc
    expected_text_fields = {
        "os": str(configuration.get("environment_platform", "")),
        "cpu_model": str(configuration.get("cpu_model", "")),
        "python_version": str(configuration.get("python_version", "")),
        "torch_version": str(configuration.get("torch_version", "")),
        "torchvision_version": str(configuration.get("torchvision_version", "")),
        "model": REQUIRED_MODEL,
        "device": REQUIRED_DEVICE,
    }
    if any(not value for value in expected_text_fields.values()):
        raise RuntimeError("benchmark metadata experiment identity is incomplete")

    run_id = metadata.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("benchmark metadata has no valid run_id")

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, str, str]] = set()
    allowed_configs = {
        (batch_size, num_threads)
        for batch_size in batch_sizes
        for num_threads in thread_counts
    }
    for row in raw_rows:
        try:
            batch_size = int(row["batch_size"])
            num_threads = int(row["num_threads"])
            repetition = int(row["repetition"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("raw results contain invalid integer fields") from exc
        mode = row.get("execution_mode", "")
        phase = row.get("phase", "")
        if row.get("run_id") != run_id:
            raise RuntimeError("raw result run_id differs from benchmark metadata")
        mismatched_text = [
            field
            for field, expected in expected_text_fields.items()
            if row.get(field) != expected
        ]
        if mismatched_text:
            raise RuntimeError(
                "raw result experiment identity differs from benchmark metadata: "
                + ", ".join(mismatched_text)
            )
        try:
            row_seed = int(row["seed"])
            row_physical_cores = int(row["physical_cores"])
            row_logical_cores = int(row["logical_cores"])
            row_interop_threads = int(row["num_interop_threads"])
            row_warmup = int(row["warmup_iterations"])
            row_rtol = float(row["rtol"])
            row_atol = float(row["atol"])
            row_min_run_time = float(row["min_run_time_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("raw results contain invalid experiment settings") from exc
        if (
            row_seed != expected_seed
            or row_physical_cores != physical_cores
            or row_logical_cores != logical_cores
            or row_interop_threads != expected_interop_threads
            or row_warmup != expected_warmup
            or not math.isclose(row_rtol, expected_rtol, rel_tol=0, abs_tol=1e-15)
            or not math.isclose(row_atol, expected_atol, rel_tol=0, abs_tol=1e-15)
            or not math.isclose(
                row_min_run_time,
                expected_min_run_time,
                rel_tol=0,
                abs_tol=1e-15,
            )
        ):
            raise RuntimeError(
                "raw result experiment settings differ from benchmark metadata"
            )
        if (batch_size, num_threads) not in allowed_configs:
            raise RuntimeError("raw results contain an unexpected configuration")
        if not 1 <= repetition <= repetitions:
            raise RuntimeError("raw results contain an unexpected repetition")
        if mode not in EXECUTION_MODES or phase not in PHASES:
            raise RuntimeError("raw results contain an unexpected mode or phase")
        if row.get("allclose", "").lower() != "true":
            raise RuntimeError(
                "raw results contain a failed correctness check; analysis aborted"
            )

        latency_ms = _parse_positive_float(row, "latency_ms")
        throughput = _parse_positive_float(row, "throughput_samples_s")
        max_abs_error = _parse_nonnegative_float(row, "max_abs_error")
        expected_throughput = batch_size * 1000.0 / latency_ms
        if not math.isclose(throughput, expected_throughput, rel_tol=1e-9):
            raise RuntimeError("throughput is inconsistent with batch size and latency")

        key = (batch_size, num_threads, repetition, mode, phase)
        if key in seen:
            raise RuntimeError(f"raw results contain a duplicate row: {key}")
        seen.add(key)
        normalized.append(
            {
                "run_id": run_id,
                "model": row["model"],
                "device": row["device"],
                "seed": row_seed,
                "batch_size": batch_size,
                "num_threads": num_threads,
                "repetition": repetition,
                "execution_mode": mode,
                "phase": phase,
                "latency_ms": latency_ms,
                "throughput_samples_s": throughput,
                "max_abs_error": max_abs_error,
            }
        )

    expected_keys = {
        (batch_size, num_threads, repetition, mode, phase)
        for batch_size, num_threads in allowed_configs
        for repetition in range(1, repetitions + 1)
        for mode in EXECUTION_MODES
        for phase in PHASES
    }
    missing = expected_keys - seen
    if missing:
        first_missing = sorted(missing)[0]
        raise RuntimeError(
            f"raw results are incomplete: missing {len(missing)} rows; "
            f"first missing key is {first_missing}"
        )
    if seen - expected_keys:
        raise RuntimeError("raw results contain rows outside the expected matrix")

    correctness = metadata.get("correctness")
    if not isinstance(correctness, dict):
        raise RuntimeError("benchmark metadata has no correctness evidence")
    for batch_size, num_threads in sorted(allowed_configs):
        item = correctness.get(f"batch={batch_size},threads={num_threads}")
        if not isinstance(item, dict) or item.get("allclose") is not True:
            raise RuntimeError(
                "benchmark metadata lacks passing correctness evidence for "
                f"batch={batch_size}, threads={num_threads}"
            )
    return normalized


def _latencies(
    rows: Iterable[dict[str, Any]], mode: str, phase: str
) -> list[float]:
    return [
        float(row["latency_ms"])
        for row in rows
        if row["execution_mode"] == mode and row["phase"] == phase
    ]


def _throughputs(rows: Iterable[dict[str, Any]], mode: str) -> list[float]:
    return [
        float(row["throughput_samples_s"])
        for row in rows
        if row["execution_mode"] == mode and row["phase"] == "steady"
    ]


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["batch_size"], row["num_threads"])].append(row)

    summary: list[dict[str, Any]] = []
    for (batch_size, num_threads), group in sorted(grouped.items()):
        stats: dict[tuple[str, str], list[float]] = {}
        for mode in EXECUTION_MODES:
            for phase in PHASES:
                values = _latencies(group, mode, phase)
                if not values:
                    raise RuntimeError(
                        f"no values for batch={batch_size}, threads={num_threads}, "
                        f"mode={mode}, phase={phase}"
                    )
                stats[(mode, phase)] = values

        eager_first = stats[("eager", "first_call")]
        compiled_first = stats[("compiled", "first_call")]
        eager_steady = stats[("eager", "steady")]
        compiled_steady = stats[("compiled", "steady")]
        repetitions = len(eager_steady)
        if any(len(values) != repetitions for values in stats.values()):
            raise RuntimeError("summary groups have inconsistent repetition counts")

        eager_first_median = median(eager_first)
        compiled_first_median = median(compiled_first)
        eager_steady_median = median(eager_steady)
        compiled_steady_median = median(compiled_steady)
        compile_overhead = compiled_first_median - eager_first_median
        steady_saving = eager_steady_median - compiled_steady_median
        break_even = calculate_break_even_requests(
            compile_overhead,
            eager_steady_median,
            compiled_steady_median,
        )
        eager_throughputs = _throughputs(group, "eager")
        compiled_throughputs = _throughputs(group, "compiled")
        summary.append(
            {
                "run_id": group[0]["run_id"],
                "model": group[0]["model"],
                "device": group[0]["device"],
                "seed": group[0]["seed"],
                "batch_size": batch_size,
                "num_threads": num_threads,
                "repetitions": repetitions,
                "allclose": True,
                "max_abs_error": max(row["max_abs_error"] for row in group),
                "eager_first_median_ms": eager_first_median,
                "eager_first_min_ms": min(eager_first),
                "eager_first_max_ms": max(eager_first),
                "compiled_first_median_ms": compiled_first_median,
                "compiled_first_min_ms": min(compiled_first),
                "compiled_first_max_ms": max(compiled_first),
                "eager_steady_median_ms": eager_steady_median,
                "eager_steady_min_ms": min(eager_steady),
                "eager_steady_max_ms": max(eager_steady),
                "compiled_steady_median_ms": compiled_steady_median,
                "compiled_steady_min_ms": min(compiled_steady),
                "compiled_steady_max_ms": max(compiled_steady),
                "eager_steady_throughput_median_samples_s": median(
                    eager_throughputs
                ),
                "eager_steady_throughput_min_samples_s": min(eager_throughputs),
                "eager_steady_throughput_max_samples_s": max(eager_throughputs),
                "compiled_steady_throughput_median_samples_s": median(
                    compiled_throughputs
                ),
                "compiled_steady_throughput_min_samples_s": min(
                    compiled_throughputs
                ),
                "compiled_steady_throughput_max_samples_s": max(
                    compiled_throughputs
                ),
                "steady_speedup": eager_steady_median / compiled_steady_median,
                "compile_overhead_ms": compile_overhead,
                "steady_latency_saving_ms": steady_saving,
                "break_even_requests": break_even,
            }
        )
    return summary


def _csv_value(value: Any) -> Any:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return format(value, ".12g")
    return value


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row[key]) for key in SUMMARY_COLUMNS})


def _config_labels(summary: list[dict[str, Any]]) -> list[str]:
    return [f"B{row['batch_size']}\nT{row['num_threads']}" for row in summary]


def _add_error_bars(
    axis: Any,
    positions: list[float],
    medians: list[float],
    minima: list[float],
    maxima: list[float],
    color: str,
) -> None:
    lower = [middle - low for middle, low in zip(medians, minima)]
    upper = [high - middle for middle, high in zip(medians, maxima)]
    axis.errorbar(
        positions,
        medians,
        yerr=[lower, upper],
        fmt="none",
        ecolor=color,
        elinewidth=1.1,
        capsize=3,
        zorder=3,
    )


def plot_latency(summary: list[dict[str, Any]], path: Path) -> None:
    labels = _config_labels(summary)
    x = list(range(len(summary)))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    panels = (
        (axes[0], "first", "First-call latency", True),
        (axes[1], "steady", "Steady-state latency", False),
    )
    colors = {"eager": "#4C78A8", "compiled": "#F58518"}
    for axis, phase, title, log_scale in panels:
        for index, mode in enumerate(EXECUTION_MODES):
            prefix = f"{mode}_{phase}"
            values = [row[f"{prefix}_median_ms"] for row in summary]
            minima = [row[f"{prefix}_min_ms"] for row in summary]
            maxima = [row[f"{prefix}_max_ms"] for row in summary]
            offset = (-width / 2) if index == 0 else (width / 2)
            positions = [value + offset for value in x]
            axis.bar(
                positions,
                values,
                width,
                label=mode.capitalize(),
                color=colors[mode],
                zorder=2,
            )
            _add_error_bars(
                axis, positions, values, minima, maxima, colors[mode]
            )
        if log_scale:
            axis.set_yscale("log")
        axis.set_title(title)
        axis.set_ylabel("Latency (ms, median; whiskers=min-max)")
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25, zorder=1)
        axis.legend()
    fig.suptitle(
        f"{summary[0]['model']} {summary[0]['device'].upper()} inference latency"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_throughput(summary: list[dict[str, Any]], path: Path) -> None:
    labels = _config_labels(summary)
    x = list(range(len(summary)))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    colors = {"eager": "#4C78A8", "compiled": "#F58518"}
    for index, mode in enumerate(EXECUTION_MODES):
        key = f"{mode}_steady_throughput_median_samples_s"
        values = [row[key] for row in summary]
        minima = [
            row[f"{mode}_steady_throughput_min_samples_s"] for row in summary
        ]
        maxima = [
            row[f"{mode}_steady_throughput_max_samples_s"] for row in summary
        ]
        offset = (-width / 2) if index == 0 else (width / 2)
        positions = [value + offset for value in x]
        axis.bar(
            positions,
            values,
            width,
            label=mode.capitalize(),
            color=colors[mode],
        )
        _add_error_bars(
            axis, positions, values, minima, maxima, colors[mode]
        )
    axis.set_title(
        f"{summary[0]['model']} {summary[0]['device'].upper()} steady-state throughput"
    )
    axis.set_ylabel("Throughput (samples/s, median; whiskers=min-max)")
    axis.set_xticks(x, labels)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_break_even(summary: list[dict[str, Any]], path: Path) -> None:
    labels = _config_labels(summary)
    x = list(range(len(summary)))
    values = [row["break_even_requests"] for row in summary]
    drawable = [0.0 if value is None else value for value in values]
    colors = ["#B9B9B9" if value is None else "#54A24B" for value in values]
    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    bars = axis.bar(x, drawable, color=colors)
    positive_values = [value for value in values if value is not None]
    y_padding = max(positive_values, default=1.0) * 0.035
    for bar, value in zip(bars, values):
        if value is None:
            label = "N/A\n(no steady gain)"
            y = y_padding
        else:
            label = f"{value:.1f}"
            y = value + y_padding
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_title("Requests needed to recover compile overhead")
    axis.set_ylabel("Break-even requests (median-based estimate)")
    axis.set_xticks(x, labels)
    axis.grid(axis="y", alpha=0.25)
    axis.set_ylim(0, max(positive_values, default=1.0) * 1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def ensure_outputs_available(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "derived output already exists; use --overwrite to regenerate: "
            + ", ".join(existing)
        )


def validate_distinct_paths(inputs: Iterable[Path], outputs: Iterable[Path]) -> None:
    input_paths = {path.resolve() for path in inputs}
    output_paths = [path.resolve() for path in outputs]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("analysis output paths must be distinct")
    overlap = input_paths.intersection(output_paths)
    if overlap:
        raise ValueError(f"analysis output overlaps an input: {sorted(overlap)[0]}")


def controller_main(args: argparse.Namespace) -> None:
    output_paths = (
        args.summary_output,
        args.figures_dir / "latency.png",
        args.figures_dir / "throughput.png",
        args.figures_dir / "break_even.png",
        args.metadata_output,
    )
    validate_distinct_paths(
        (args.input, args.input_metadata),
        output_paths,
    )
    ensure_outputs_available(output_paths, args.overwrite)

    metadata = read_json(args.input_metadata)
    raw_rows = read_raw_rows(args.input)
    normalized = validate_and_normalize_rows(raw_rows, metadata)
    summary = summarize_rows(normalized)

    common_parent = Path(
        os.path.commonpath([str(path.resolve().parent) for path in output_paths])
    )
    common_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="analysis-", dir=common_parent
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        temporary_summary = temporary / "summary.csv"
        temporary_latency = temporary / "latency.png"
        temporary_throughput = temporary / "throughput.png"
        temporary_break_even = temporary / "break_even.png"
        temporary_metadata = temporary / "analysis_metadata.json"

        write_summary_csv(temporary_summary, summary)
        plot_latency(summary, temporary_latency)
        plot_throughput(summary, temporary_throughput)
        plot_break_even(summary, temporary_break_even)

        analysis_metadata = {
            "schema_version": 1,
            "generated_at_utc": utc_now(),
            "command": " ".join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
            "source": {
                "raw_results": str(args.input),
                "raw_results_sha256": file_sha256(args.input),
                "benchmark_metadata": str(args.input_metadata),
                "benchmark_metadata_sha256": file_sha256(args.input_metadata),
                "run_id": metadata["run_id"],
            },
            "analysis_environment": {
                "analysis_script": str(Path(__file__).resolve()),
                "analysis_script_sha256": file_sha256(Path(__file__).resolve()),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "python_executable": sys.executable,
                "matplotlib_version": matplotlib.__version__,
                "matplotlib_backend": matplotlib.get_backend(),
                "git": git_provenance(Path(__file__).resolve().parent),
                "reproducibility_note": (
                    "summary values are expected to reproduce from identical inputs "
                    "and analysis code; PNG bytes may differ across rendering stacks"
                ),
            },
            "method": {
                "central_statistic": "median across independent repetitions",
                "range": "minimum and maximum across independent repetitions",
                "compile_overhead_ms": (
                    "compiled first-call median minus eager first-call median"
                ),
                "steady_speedup": (
                    "eager steady median divided by compiled steady median"
                ),
                "break_even_requests": (
                    "max(0, compile overhead) divided by steady latency saving; "
                    "null when compiled steady latency is not lower"
                ),
            },
            "validation": {
                "raw_rows": len(normalized),
                "summary_rows": len(summary),
                "minimum_repetitions_required": MIN_REPETITIONS,
                "all_correctness_checks_passed": True,
                "required_batch_sizes": list(REQUIRED_BATCH_SIZES),
                "required_thread_counts": sorted(
                    set(
                        (
                            *REQUIRED_THREAD_COUNTS,
                            metadata["configuration"]["physical_cores"],
                        )
                    )
                ),
                "required_matrix_complete": True,
            },
            "outputs": {
                "summary_csv": str(args.summary_output),
                "latency_figure": str(args.figures_dir / "latency.png"),
                "throughput_figure": str(args.figures_dir / "throughput.png"),
                "break_even_figure": str(args.figures_dir / "break_even.png"),
                "sha256": {
                    "summary_csv": file_sha256(temporary_summary),
                    "latency_figure": file_sha256(temporary_latency),
                    "throughput_figure": file_sha256(temporary_throughput),
                    "break_even_figure": file_sha256(temporary_break_even),
                },
            },
        }
        temporary_metadata.write_text(
            json.dumps(analysis_metadata, indent=2) + "\n",
            encoding="utf-8",
        )

        moves = (
            (temporary_summary, args.summary_output),
            (temporary_latency, args.figures_dir / "latency.png"),
            (temporary_throughput, args.figures_dir / "throughput.png"),
            (temporary_break_even, args.figures_dir / "break_even.png"),
            (temporary_metadata, args.metadata_output),
        )
        for source, destination in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)

    print(f"Validated {len(normalized)} raw rows across {len(summary)} configurations")
    print(f"Wrote summary to {args.summary_output}")
    print(f"Wrote figures to {args.figures_dir}")
    print(f"Wrote analysis metadata to {args.metadata_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/raw_results.csv"),
        help="full-matrix raw benchmark CSV",
    )
    parser.add_argument(
        "--input-metadata",
        type=Path,
        default=Path("results/raw_results.metadata.json"),
        help="metadata sidecar for the raw benchmark CSV",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/summary.csv"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("results/analysis_metadata.json"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing derived analysis outputs; raw inputs remain untouched",
    )
    return parser.parse_args()


def main() -> None:
    controller_main(parse_args())


if __name__ == "__main__":
    main()
