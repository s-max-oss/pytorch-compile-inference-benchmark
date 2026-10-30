"""Profile representative Eager and torch.compile ResNet18 CPU inference.

Profiler measurements explain operator structure only.  They are intentionally
kept separate from the formal latency results in results/raw_results.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import benchmark
import torch
from torch.profiler import ProfilerActivity, profile


DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_THREADS = 4
DEFAULT_WARMUP_ITERATIONS = 5
DEFAULT_PROFILE_ITERATIONS = 3

PROFILE_COLUMNS = [
    "run_id",
    "execution_mode",
    "rank_by_self_cpu",
    "operator",
    "input_shapes",
    "call_count",
    "self_cpu_time_total_us",
    "self_cpu_time_avg_us",
    "cpu_time_total_us",
    "cpu_time_avg_us",
    "self_cpu_percent",
    "profile_iterations",
]


def operator_rows(
    events: Iterable[Any],
    run_id: str,
    execution_mode: str,
    profile_iterations: int,
) -> list[dict[str, Any]]:
    """Convert all grouped profiler events into deterministic structured rows."""
    prepared: list[dict[str, Any]] = []
    for event in events:
        count = int(event.count)
        self_cpu_total = float(event.self_cpu_time_total)
        cpu_total = float(event.cpu_time_total)
        if count < 1:
            raise RuntimeError(f"profiler event has invalid count: {event.key!r}")
        if not all(math.isfinite(value) and value >= 0 for value in (self_cpu_total, cpu_total)):
            raise RuntimeError(f"profiler event has invalid CPU time: {event.key!r}")
        prepared.append(
            {
                "run_id": run_id,
                "execution_mode": execution_mode,
                "operator": str(event.key),
                "input_shapes": json.dumps(
                    event.input_shapes,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "call_count": count,
                "self_cpu_time_total_us": self_cpu_total,
                "self_cpu_time_avg_us": self_cpu_total / count,
                "cpu_time_total_us": cpu_total,
                "cpu_time_avg_us": cpu_total / count,
                "profile_iterations": profile_iterations,
            }
        )

    prepared.sort(
        key=lambda row: (
            -float(row["self_cpu_time_total_us"]),
            str(row["operator"]),
            str(row["input_shapes"]),
        )
    )
    total_self_cpu = sum(float(row["self_cpu_time_total_us"]) for row in prepared)
    if total_self_cpu <= 0:
        raise RuntimeError("profiler captured no positive self CPU time")
    for rank, row in enumerate(prepared, start=1):
        row["rank_by_self_cpu"] = rank
        row["self_cpu_percent"] = (
            float(row["self_cpu_time_total_us"]) * 100.0 / total_self_cpu
        )
    return prepared


def profile_worker(args: argparse.Namespace) -> dict[str, Any]:
    benchmark.configure_threads(args.num_threads)
    model, inputs = benchmark.make_model_and_input(args.seed, args.batch_size)
    target = (
        model
        if args.execution_mode == "eager"
        else torch.compile(model)
    )

    @torch.inference_mode()
    def run_once() -> torch.Tensor:
        return target(inputs)

    # The priming call includes graph capture/compilation for compiled mode.
    # All priming and warmup calls finish before the profiler starts.
    priming_output = run_once()
    _ = priming_output.shape
    for _ in range(args.warmup_iterations):
        warmup_output = run_once()
        _ = warmup_output.shape

    profile_started_at = benchmark.utc_now()
    wall_start_ns = time.perf_counter_ns()
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
    ) as profiler:
        for _ in range(args.profile_iterations):
            output = run_once()
            _ = output.shape
    profiled_wall_time_ms = (time.perf_counter_ns() - wall_start_ns) / 1_000_000

    rows = operator_rows(
        profiler.key_averages(group_by_input_shape=True),
        run_id=args.run_id,
        execution_mode=args.execution_mode,
        profile_iterations=args.profile_iterations,
    )
    return {
        "execution_mode": args.execution_mode,
        "profile_started_at": profile_started_at,
        "profiled_wall_time_ms": profiled_wall_time_ms,
        "operator_count": len(rows),
        "operators": rows,
    }


def worker_main(args: argparse.Namespace) -> None:
    print(json.dumps(profile_worker(args), separators=(",", ":")))


def run_profile_worker(
    args: argparse.Namespace,
    execution_mode: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--execution-mode",
        execution_mode,
        "--run-id",
        args.run_id,
        "--batch-size",
        str(args.batch_size),
        "--num-threads",
        str(args.num_threads),
        "--seed",
        str(args.seed),
        "--warmup-iterations",
        str(args.warmup_iterations),
        "--profile-iterations",
        str(args.profile_iterations),
    ]
    worker_environment = os.environ.copy()
    worker_environment["OMP_NUM_THREADS"] = str(args.num_threads)
    worker_environment["MKL_NUM_THREADS"] = str(args.num_threads)

    with tempfile.TemporaryDirectory(
        prefix=f"torchinductor-profiler-{execution_mode}-"
    ) as cache_dir:
        worker_environment["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=worker_environment,
        )

    if completed.returncode != 0:
        raise RuntimeError(
            "profiler worker failed\n"
            f"command: {shlex.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("profiler worker produced no JSON output")
    try:
        payload = json.loads(output_lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"profiler worker ended with invalid JSON: {output_lines[-1]!r}"
        ) from exc
    if payload.get("execution_mode") != execution_mode:
        raise RuntimeError("profiler worker returned the wrong execution mode")
    return payload


def write_operator_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite profiler data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(
                handle,
                fieldnames=PROFILE_COLUMNS,
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def validate_controller_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "num_threads", "warmup_iterations", "profile_iterations"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if os.environ.get("CONDA_DEFAULT_ENV") != "ai-infra-compile":
        raise RuntimeError("activate ai-infra-compile before profiling")
    for variable in ("CC", "CXX", "LD_LIBRARY_PATH"):
        if not os.environ.get(variable):
            raise RuntimeError(
                f"{variable} is unset; run through scripts/run_in_env.sh"
            )


def controller_main(args: argparse.Namespace) -> None:
    validate_controller_args(args)
    environment = benchmark.load_environment(args.environment)
    benchmark.validate_environment_snapshot(environment)
    if args.num_threads > int(environment["cpu"]["logical_cores"]):
        raise ValueError("--num-threads exceeds the VM logical core count")

    operator_path = args.output_dir / "operator_summary.csv"
    metadata_path = args.output_dir / "metadata.json"
    benchmark.validate_distinct_paths(
        {
            "environment": args.environment,
            "operator_summary": operator_path,
            "metadata": metadata_path,
        }
    )
    benchmark.ensure_outputs_available((operator_path, metadata_path), overwrite=False)

    args.run_id = str(uuid.uuid4())
    correctness_args = argparse.Namespace(
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        seed=args.seed,
        rtol=args.rtol,
        atol=args.atol,
        warmup_iterations=args.warmup_iterations,
        min_run_time=1.0,
    )
    print("Running isolated correctness check...", flush=True)
    correctness = benchmark.run_fresh_worker(correctness_args, "correctness")
    print(
        "Correctness: "
        f"allclose={correctness['allclose']}, "
        f"max_abs_error={correctness['max_abs_error']:.6g}",
        flush=True,
    )
    if not correctness["allclose"]:
        raise RuntimeError("Eager and compiled outputs differ; profiling aborted")

    payloads: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for execution_mode in ("eager", "compiled"):
        print(f"Profiling {execution_mode} after warmup...", flush=True)
        payload = run_profile_worker(args, execution_mode)
        payloads.append(payload)
        rows.extend(payload["operators"])

    if not rows:
        raise RuntimeError("profiler produced no operator rows")
    if {row["execution_mode"] for row in rows} != {"eager", "compiled"}:
        raise RuntimeError("profiler output is missing an execution mode")

    metadata = {
        "schema_version": 1,
        "run_id": args.run_id,
        "completed_at_utc": benchmark.utc_now(),
        "command": shlex.join(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
        ),
        "working_directory": str(Path.cwd()),
        "environment_snapshot": str(args.environment),
        "environment_snapshot_sha256": benchmark.file_sha256(args.environment),
        "configuration": {
            "model": benchmark.MODEL_NAME,
            "device": benchmark.DEVICE,
            "batch_size": args.batch_size,
            "num_threads": args.num_threads,
            "num_interop_threads": 1,
            "seed": args.seed,
            "priming_iterations": 1,
            "warmup_iterations": args.warmup_iterations,
            "profile_iterations": args.profile_iterations,
            "activities": ["CPU"],
            "record_shapes": True,
            "profile_memory": False,
            "with_stack": False,
        },
        "controller_performance_environment": benchmark.performance_environment(
            benchmark.INITIAL_PROCESS_ENVIRONMENT
        ),
        "worker_performance_environment": {
            **benchmark.performance_environment(dict(os.environ)),
            "OMP_NUM_THREADS": str(args.num_threads),
            "MKL_NUM_THREADS": str(args.num_threads),
            "TORCHINDUCTOR_CACHE_DIR": "<fresh temporary directory per worker>",
        },
        "isolation": {
            "process_policy": "fresh subprocess for correctness and each execution mode",
            "inductor_cache_policy": (
                "fresh TemporaryDirectory per worker; deleted after worker exits"
            ),
            "compilation_profiled": False,
        },
        "correctness": correctness,
        "profiles": [
            {
                "execution_mode": payload["execution_mode"],
                "profile_started_at": payload["profile_started_at"],
                "profiled_wall_time_ms": payload["profiled_wall_time_ms"],
                "operator_count": payload["operator_count"],
            }
            for payload in payloads
        ],
        "interpretation_limit": (
            "Profiler data is for operator attribution only and must not be used "
            "as formal benchmark latency."
        ),
    }
    write_operator_csv(operator_path, rows)
    benchmark.write_json(metadata_path, metadata, overwrite=False)
    print(f"Wrote {len(rows)} operator rows to {operator_path}")
    print(f"Wrote profiler metadata to {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument("--seed", type=int, default=benchmark.DEFAULT_SEED)
    parser.add_argument("--rtol", type=float, default=benchmark.DEFAULT_RTOL)
    parser.add_argument("--atol", type=float, default=benchmark.DEFAULT_ATOL)
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=DEFAULT_WARMUP_ITERATIONS,
    )
    parser.add_argument(
        "--profile-iterations",
        type=int,
        default=DEFAULT_PROFILE_ITERATIONS,
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=Path("results/environment.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/profiler"),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--execution-mode",
        choices=("eager", "compiled"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        if args.execution_mode is None or not args.run_id:
            raise ValueError("profiler worker requires execution mode and run ID")
        worker_main(args)
    else:
        controller_main(args)


if __name__ == "__main__":
    main()
