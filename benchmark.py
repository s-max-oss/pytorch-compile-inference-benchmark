"""Benchmark PyTorch Eager and torch.compile inference for ResNet18.

The default command preserves the stage-2 minimal closure.  ``--matrix`` runs
the complete CPU matrix with resumable checkpoints.  Every first-call
measurement uses a fresh worker process and a fresh Inductor cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Capture user/system-provided values before importing PyTorch.  PyTorch may
# populate TORCHINDUCTOR_CACHE_DIR during import; that generated default is not
# an external experimental setting and every worker replaces it explicitly.
INITIAL_PROCESS_ENVIRONMENT = dict(os.environ)

import torch
import torch.utils.benchmark as torch_benchmark
import torchvision
from torchvision.models import resnet18


MODEL_NAME = "resnet18"
DEVICE = "cpu"
DEFAULT_SEED = 20261029
DEFAULT_RTOL = 1e-4
DEFAULT_ATOL = 1e-5
MATRIX_BATCH_SIZES = (1, 8, 32)
MATRIX_REQUIRED_THREADS = (1, 4)
MATRIX_MIN_REPETITIONS = 5

PERFORMANCE_ENV_PREFIXES = (
    "ATEN_",
    "BLIS_",
    "CUDA",
    "CUDNN",
    "DNNL_",
    "GOMP_",
    "KMP_",
    "MKL_",
    "NUMEXPR_",
    "ONEDNN_",
    "OMP_",
    "OPENBLAS_",
    "PYTORCH",
    "TBB_",
    "TORCH",
    "VECLIB_",
)
PERFORMANCE_ENV_EXACT_KEYS = (
    "CC",
    "CXX",
    "GLIBC_TUNABLES",
    "LD_LIBRARY_PATH",
    "MALLOC_CONF",
)

CSV_COLUMNS = [
    "timestamp",
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
    "steady_iqr_ms",
    "timer_block_count",
    "number_per_run",
    "total_invocations",
    "warmup_iterations",
    "min_run_time_s",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_model_and_input(seed: int, batch_size: int) -> tuple[torch.nn.Module, torch.Tensor]:
    """Create deterministic weights and input without downloading a dataset."""
    torch.manual_seed(seed)
    model = resnet18(weights=None).eval()
    input_generator = torch.Generator(device=DEVICE).manual_seed(seed)
    inputs = torch.randn(
        batch_size,
        3,
        224,
        224,
        generator=input_generator,
        device=DEVICE,
    )
    return model, inputs


def configure_threads(num_threads: int) -> None:
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)


def correctness_worker(args: argparse.Namespace) -> dict[str, Any]:
    configure_threads(args.num_threads)
    model, inputs = make_model_and_input(args.seed, args.batch_size)
    with torch.inference_mode():
        eager_output = model(inputs)
        compiled_output = torch.compile(model)(inputs)
    max_abs_error = float((eager_output - compiled_output).abs().max().item())
    return {
        "allclose": bool(
            torch.allclose(
                eager_output,
                compiled_output,
                rtol=args.rtol,
                atol=args.atol,
            )
        ),
        "max_abs_error": max_abs_error,
        "rtol": args.rtol,
        "atol": args.atol,
    }


def benchmark_worker(args: argparse.Namespace) -> dict[str, Any]:
    configure_threads(args.num_threads)
    model, inputs = make_model_and_input(args.seed, args.batch_size)
    target = model if args.execution_mode == "eager" else torch.compile(model)

    @torch.inference_mode()
    def run_once() -> torch.Tensor:
        return target(inputs)

    first_timestamp = utc_now()
    start_ns = time.perf_counter_ns()
    first_output = run_once()
    first_latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
    # Force the result to remain live through the end of the timed call.
    _ = first_output.shape

    for _ in range(args.warmup_iterations):
        run_once()

    timer = torch_benchmark.Timer(
        stmt="run_once()",
        globals={"run_once": run_once},
        num_threads=args.num_threads,
        label=MODEL_NAME,
        sub_label=args.execution_mode,
        description=f"batch_size={args.batch_size}",
    )
    steady_timestamp = utc_now()
    measurement = timer.blocked_autorange(min_run_time=args.min_run_time)

    return {
        "first_timestamp": first_timestamp,
        "first_latency_ms": first_latency_ms,
        "steady_timestamp": steady_timestamp,
        "steady_latency_ms": measurement.median * 1_000,
        "steady_iqr_ms": measurement.iqr * 1_000,
        "timer_block_count": len(measurement.raw_times),
        "number_per_run": measurement.number_per_run,
    }


def worker_main(args: argparse.Namespace) -> None:
    if args.worker == "correctness":
        result = correctness_worker(args)
    elif args.worker == "benchmark":
        result = benchmark_worker(args)
    else:
        raise ValueError(f"unknown worker type: {args.worker}")
    print(json.dumps(result, separators=(",", ":")))


def run_fresh_worker(
    args: argparse.Namespace,
    worker: str,
    execution_mode: str | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        worker,
        "--batch-size",
        str(args.batch_size),
        "--num-threads",
        str(args.num_threads),
        "--seed",
        str(args.seed),
        "--rtol",
        str(args.rtol),
        "--atol",
        str(args.atol),
        "--warmup-iterations",
        str(args.warmup_iterations),
        "--min-run-time",
        str(args.min_run_time),
    ]
    if execution_mode is not None:
        command.extend(["--execution-mode", execution_mode])

    worker_environment = os.environ.copy()
    worker_environment["OMP_NUM_THREADS"] = str(args.num_threads)
    worker_environment["MKL_NUM_THREADS"] = str(args.num_threads)

    with tempfile.TemporaryDirectory(prefix=f"torchinductor-{worker}-") as cache_dir:
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
            "worker process failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("worker process produced no JSON output")
    try:
        return json.loads(output_lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"worker process ended with invalid JSON: {output_lines[-1]!r}"
        ) from exc


def throughput_samples_s(batch_size: int, latency_ms: float) -> float:
    if latency_ms <= 0:
        raise ValueError("latency_ms must be positive")
    return batch_size * 1_000 / latency_ms


def load_environment(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"environment file not found: {path}; run collect_environment.py first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def validate_controller_environment(args: argparse.Namespace) -> None:
    if args.matrix:
        if args.repetitions < MATRIX_MIN_REPETITIONS:
            raise ValueError(
                f"--matrix requires at least {MATRIX_MIN_REPETITIONS} repetitions"
            )
        if args.overwrite:
            raise ValueError(
                "--overwrite is disabled for --matrix; preserve existing raw data "
                "and choose a new --output path"
            )
        if args.batch_size != 1 or args.num_threads != 4:
            raise ValueError(
                "--batch-size and --num-threads are fixed by --matrix; "
                "do not override them"
            )
    else:
        if args.batch_size != 1:
            raise ValueError("stage 2 permits only --batch-size 1")
        if args.num_threads != 4:
            raise ValueError("stage 2 minimal closure uses exactly --num-threads 4")
        if args.resume:
            raise ValueError("--resume is available only with --matrix")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1")
    if args.warmup_iterations < 1:
        raise ValueError("--warmup-iterations must be at least 1")
    if not math.isfinite(args.min_run_time) or args.min_run_time <= 0:
        raise ValueError("--min-run-time must be positive")
    if not math.isfinite(args.rtol) or not math.isfinite(args.atol):
        raise ValueError("--rtol and --atol must be finite")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("--rtol and --atol must be non-negative")
    if os.environ.get("CONDA_DEFAULT_ENV") != "ai-infra-compile":
        raise RuntimeError(
            "activate ai-infra-compile before benchmarking: "
            "source ~/miniconda3/etc/profile.d/conda.sh && "
            "conda activate ai-infra-compile"
        )
    for variable in ("CC", "CXX", "LD_LIBRARY_PATH"):
        if not os.environ.get(variable):
            raise RuntimeError(
                f"{variable} is unset; run through: scripts/run_in_env.sh python benchmark.py"
            )


def current_cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return platform.processor() or None
    for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "model name":
            return value.strip()
    return platform.processor() or None


def current_cpu_counts() -> tuple[int | None, int | None]:
    """Return physical and logical CPU counts using snapshot-compatible logic."""
    logical_cores = os.cpu_count()
    physical_cores: int | None = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        processors: list[dict[str, str]] = []
        current: dict[str, str] = {}
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                if current:
                    processors.append(current)
                    current = {}
                continue
            key, separator, value = line.partition(":")
            if separator:
                current[key.strip()] = value.strip()
        if current:
            processors.append(current)
        core_ids = {
            (item.get("physical id", "0"), item["core id"])
            for item in processors
            if "core id" in item
        }
        if core_ids:
            physical_cores = len(core_ids)

    if (
        physical_cores is None
        and sys.platform.startswith("linux")
        and shutil.which("lscpu") is not None
    ):
        completed = subprocess.run(
            ["lscpu", "--parse=CORE,SOCKET"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode == 0:
            pairs = {
                tuple(line.split(",", maxsplit=1))
                for line in completed.stdout.splitlines()
                if line and not line.startswith("#") and "," in line
            }
            physical_cores = len(pairs) or None
    return physical_cores, logical_cores


def performance_environment(environment: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(environment.items())
        if key in PERFORMANCE_ENV_EXACT_KEYS
        or key.startswith(PERFORMANCE_ENV_PREFIXES)
    }


def validate_environment_snapshot(environment: dict[str, Any]) -> None:
    """Reject stale or cross-host environment metadata before measuring."""
    expected_prefix = environment["conda"]["prefix"]
    expected_python = environment["python"]["executable"]
    current_physical_cores, current_logical_cores = current_cpu_counts()
    checks = {
        "CONDA_PREFIX": (expected_prefix, os.environ.get("CONDA_PREFIX")),
        "python executable": (expected_python, sys.executable),
        "OS platform": (environment["os"]["platform"], platform.platform()),
        "CPU model": (environment["cpu"]["model"], current_cpu_model()),
        "physical CPU cores": (
            environment["cpu"]["physical_cores"],
            current_physical_cores,
        ),
        "logical CPU cores": (
            environment["cpu"]["logical_cores"],
            current_logical_cores,
        ),
        "Python version": (
            environment["python"]["version"].split()[0],
            platform.python_version(),
        ),
        "torch version": (environment["pytorch"]["torch_version"], torch.__version__),
        "torchvision version": (
            environment["pytorch"]["torchvision_version"],
            torchvision.__version__,
        ),
    }
    mismatches = [
        f"{name}: snapshot={expected!r}, runtime={actual!r}"
        for name, (expected, actual) in checks.items()
        if expected != actual
    ]
    if mismatches:
        raise RuntimeError(
            "environment snapshot does not match the benchmark runtime:\n- "
            + "\n- ".join(mismatches)
            + "\nRegenerate results/environment.json in this environment."
        )

    snapshot_variables = performance_environment(
        environment.get("environment_variables", {})
    )
    runtime_variables = performance_environment(INITIAL_PROCESS_ENVIRONMENT)
    if snapshot_variables != runtime_variables:
        all_keys = sorted(set(snapshot_variables) | set(runtime_variables))
        differences = [
            (
                f"{key}: snapshot={snapshot_variables.get(key)!r}, "
                f"runtime={runtime_variables.get(key)!r}"
            )
            for key in all_keys
            if snapshot_variables.get(key) != runtime_variables.get(key)
        ]
        raise RuntimeError(
            "performance-related environment variables differ from the snapshot:\n- "
            + "\n- ".join(differences)
            + "\nRegenerate results/environment.json or restore the recorded variables."
        )


def base_row(
    environment: dict[str, Any],
    args: argparse.Namespace,
    correctness: dict[str, Any],
    execution_mode: str,
    repetition: int,
) -> dict[str, Any]:
    gpu_names = [gpu.get("name", "") for gpu in environment.get("gpus", [])]
    return {
        "run_id": getattr(args, "run_id", ""),
        "os": environment["os"]["platform"],
        "cpu_model": environment["cpu"]["model"],
        "physical_cores": environment["cpu"]["physical_cores"],
        "logical_cores": environment["cpu"]["logical_cores"],
        "gpu_model": "; ".join(name for name in gpu_names if name),
        "python_version": environment["python"]["version"].split()[0],
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "seed": args.seed,
        "model": MODEL_NAME,
        "device": DEVICE,
        "execution_mode": execution_mode,
        "batch_size": args.batch_size,
        "num_threads": args.num_threads,
        "repetition": repetition,
        "allclose": correctness["allclose"],
        "max_abs_error": correctness["max_abs_error"],
        "rtol": correctness["rtol"],
        "atol": correctness["atol"],
        "num_interop_threads": 1,
        "warmup_iterations": args.warmup_iterations,
        "min_run_time_s": args.min_run_time,
    }


def result_rows(
    environment: dict[str, Any],
    args: argparse.Namespace,
    correctness: dict[str, Any],
    execution_mode: str,
    repetition: int,
    measurement: dict[str, Any],
) -> list[dict[str, Any]]:
    common = base_row(environment, args, correctness, execution_mode, repetition)
    first_latency = float(measurement["first_latency_ms"])
    steady_latency = float(measurement["steady_latency_ms"])
    first_row = {
        **common,
        "timestamp": measurement["first_timestamp"],
        "phase": "first_call",
        "latency_ms": first_latency,
        "throughput_samples_s": throughput_samples_s(args.batch_size, first_latency),
        "steady_iqr_ms": "",
        "timer_block_count": 1,
        "number_per_run": 1,
        "total_invocations": 1,
    }
    steady_row = {
        **common,
        "timestamp": measurement["steady_timestamp"],
        "phase": "steady",
        "latency_ms": steady_latency,
        "throughput_samples_s": throughput_samples_s(args.batch_size, steady_latency),
        "steady_iqr_ms": measurement["steady_iqr_ms"],
        "timer_block_count": measurement["timer_block_count"],
        "number_per_run": measurement["number_per_run"],
        "total_invocations": (
            measurement["timer_block_count"] * measurement["number_per_run"]
        ),
    }
    return [first_row, steady_row]


def write_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing data: {path}; choose a new --output path"
        )
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
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing metadata: {path}; choose a new path"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def metadata_output_path(args: argparse.Namespace) -> Path:
    if args.metadata_output is not None:
        return args.metadata_output
    return args.output.with_suffix(".metadata.json")


def ensure_outputs_available(paths: tuple[Path, ...], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing result files: " + ", ".join(existing)
        )


def validate_distinct_paths(paths: dict[str, Path]) -> None:
    resolved: dict[Path, list[str]] = {}
    for label, path in paths.items():
        canonical = path.expanduser().resolve(strict=False)
        resolved.setdefault(canonical, []).append(label)
    conflicts = [
        f"{', '.join(labels)} -> {path}"
        for path, labels in resolved.items()
        if len(labels) > 1
    ]
    if conflicts:
        raise ValueError("result/input paths must be distinct: " + "; ".join(conflicts))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_thread_counts(environment: dict[str, Any]) -> tuple[int, ...]:
    physical_cores = int(environment["cpu"]["physical_cores"])
    if physical_cores < 1:
        raise ValueError("environment snapshot reports no physical CPU cores")
    return tuple(dict.fromkeys((*MATRIX_REQUIRED_THREADS, physical_cores)))


def configured_args(
    args: argparse.Namespace,
    batch_size: int,
    num_threads: int,
) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(batch_size=batch_size, num_threads=num_threads)
    return argparse.Namespace(**values)


def matrix_config_key(batch_size: int, num_threads: int) -> str:
    return f"batch={batch_size},threads={num_threads}"


def measurement_key(
    batch_size: int,
    num_threads: int,
    repetition: int,
    execution_mode: str,
) -> tuple[int, int, int, str]:
    return batch_size, num_threads, repetition, execution_mode


def checkpoint_paths(output: Path) -> tuple[Path, Path]:
    partial_csv = output.with_name(f"{output.stem}.partial{output.suffix}")
    state_json = output.with_name(f"{output.stem}.state.json")
    return partial_csv, state_json


def matrix_fingerprint(
    args: argparse.Namespace,
    environment: dict[str, Any],
    batch_sizes: tuple[int, ...],
    thread_counts: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "device": DEVICE,
        "batch_sizes": list(batch_sizes),
        "num_threads": list(thread_counts),
        "num_interop_threads": 1,
        "repetitions": args.repetitions,
        "seed": args.seed,
        "rtol": args.rtol,
        "atol": args.atol,
        "warmup_iterations": args.warmup_iterations,
        "min_run_time_s": args.min_run_time,
        "environment_snapshot_sha256": file_sha256(args.environment),
        "benchmark_script_sha256": file_sha256(Path(__file__).resolve()),
        "environment_platform": environment["os"]["platform"],
        "cpu_model": environment["cpu"]["model"],
        "physical_cores": environment["cpu"]["physical_cores"],
        "logical_cores": environment["cpu"]["logical_cores"],
        "python_version": environment["python"]["version"].split()[0],
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid checkpoint JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint JSON must contain an object: {path}")
    return payload


def load_checkpoint_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_COLUMNS:
            raise RuntimeError(
                f"checkpoint CSV columns do not match the current schema: {path}"
            )
        return list(reader)


def completed_checkpoint_measurements(
    rows: list[dict[str, Any]],
    configurations: tuple[tuple[int, int], ...],
    args: argparse.Namespace,
    correctness_by_config: dict[str, Any],
    run_id: str,
    fingerprint: dict[str, Any],
) -> set[tuple[int, int, int, str]]:
    allowed_configurations = set(configurations)
    phases_by_measurement: dict[tuple[int, int, int, str], set[str]] = {}
    for row in rows:
        try:
            batch_size = int(row["batch_size"])
            num_threads = int(row["num_threads"])
            repetition = int(row["repetition"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("checkpoint CSV contains invalid integer fields") from exc
        execution_mode = row.get("execution_mode", "")
        phase = row.get("phase", "")
        if (batch_size, num_threads) not in allowed_configurations:
            raise RuntimeError("checkpoint CSV contains an unexpected configuration")
        if not 1 <= repetition <= args.repetitions:
            raise RuntimeError("checkpoint CSV contains an unexpected repetition")
        if execution_mode not in ("eager", "compiled"):
            raise RuntimeError("checkpoint CSV contains an unexpected execution mode")
        if phase not in ("first_call", "steady"):
            raise RuntimeError("checkpoint CSV contains an unexpected phase")
        if str(row.get("allclose", "")).lower() != "true":
            raise RuntimeError("checkpoint CSV contains a failed correctness result")
        expected_fields = {
            "run_id": run_id,
            "os": str(fingerprint["environment_platform"]),
            "cpu_model": str(fingerprint["cpu_model"]),
            "physical_cores": str(fingerprint["physical_cores"]),
            "logical_cores": str(fingerprint["logical_cores"]),
            "python_version": str(fingerprint["python_version"]),
            "torch_version": str(fingerprint["torch_version"]),
            "torchvision_version": str(fingerprint["torchvision_version"]),
            "seed": str(args.seed),
            "model": MODEL_NAME,
            "device": DEVICE,
            "rtol": str(args.rtol),
            "atol": str(args.atol),
            "num_interop_threads": "1",
            "warmup_iterations": str(args.warmup_iterations),
            "min_run_time_s": str(args.min_run_time),
        }
        mismatched_fields = [
            field
            for field, expected in expected_fields.items()
            if row.get(field) != expected
        ]
        if mismatched_fields:
            raise RuntimeError(
                "checkpoint CSV settings differ from the checkpoint state: "
                + ", ".join(mismatched_fields)
            )
        try:
            latency_ms = float(row["latency_ms"])
            throughput = float(row["throughput_samples_s"])
            max_abs_error = float(row["max_abs_error"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("checkpoint CSV contains invalid numeric results") from exc
        if (
            not math.isfinite(latency_ms)
            or latency_ms <= 0
            or not math.isfinite(throughput)
            or throughput <= 0
            or not math.isfinite(max_abs_error)
            or max_abs_error < 0
        ):
            raise RuntimeError("checkpoint CSV contains invalid numeric results")
        config_key = matrix_config_key(batch_size, num_threads)
        correctness = correctness_by_config.get(config_key)
        if not correctness or not correctness.get("allclose"):
            raise RuntimeError(
                f"checkpoint rows lack passing correctness evidence for {config_key}"
            )
        key = measurement_key(
            batch_size, num_threads, repetition, execution_mode
        )
        phases = phases_by_measurement.setdefault(key, set())
        if phase in phases:
            raise RuntimeError(f"checkpoint CSV contains a duplicate row: {key}/{phase}")
        phases.add(phase)

    incomplete = [
        key
        for key, phases in phases_by_measurement.items()
        if phases != {"first_call", "steady"}
    ]
    if incomplete:
        raise RuntimeError(
            "checkpoint CSV contains incomplete measurements: "
            + ", ".join(map(str, incomplete))
        )
    return set(phases_by_measurement)


def validate_resume_final_artifacts(
    output: Path,
    metadata_path: Path,
    partial_csv: Path,
    run_id: str,
) -> None:
    """Reject unrelated final files before a resumed run may replace them."""
    if output.exists():
        if not partial_csv.is_file() or file_sha256(output) != file_sha256(partial_csv):
            raise FileExistsError(
                "resume found an existing final CSV that is not identical to "
                f"this run's checkpoint: {output}"
            )
    if metadata_path.exists():
        metadata = load_json(metadata_path)
        if metadata.get("run_id") != run_id:
            raise FileExistsError(
                "resume found existing metadata from another run: "
                f"{metadata_path}"
            )


def minimal_controller(
    args: argparse.Namespace,
    environment: dict[str, Any],
    metadata_path: Path,
) -> None:
    controller_started_at = utc_now()

    print("Running isolated correctness check...", flush=True)
    correctness = run_fresh_worker(args, "correctness")
    print(
        "Correctness: "
        f"allclose={correctness['allclose']}, "
        f"max_abs_error={correctness['max_abs_error']:.6g}",
        flush=True,
    )
    if not correctness["allclose"]:
        raise RuntimeError("Eager and compiled outputs differ; performance run aborted")

    rows: list[dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        for execution_mode in ("eager", "compiled"):
            print(
                f"Running {execution_mode}, repetition {repetition}/{args.repetitions}...",
                flush=True,
            )
            measurement = run_fresh_worker(
                args,
                worker="benchmark",
                execution_mode=execution_mode,
            )
            rows.extend(
                result_rows(
                    environment,
                    args,
                    correctness,
                    execution_mode,
                    repetition,
                    measurement,
                )
            )

    metadata = {
        "schema_version": 1,
        "started_at_utc": controller_started_at,
        "completed_at_utc": utc_now(),
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "working_directory": str(Path.cwd()),
        "python_executable": sys.executable,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "environment_snapshot": str(args.environment),
        "result_csv": str(args.output),
        "configuration": {
            "model": MODEL_NAME,
            "device": DEVICE,
            "batch_size": args.batch_size,
            "num_threads": args.num_threads,
            "num_interop_threads": 1,
            "repetitions": args.repetitions,
            "seed": args.seed,
            "rtol": args.rtol,
            "atol": args.atol,
            "warmup_iterations": args.warmup_iterations,
            "min_run_time_s": args.min_run_time,
        },
        "controller_performance_environment": performance_environment(
            INITIAL_PROCESS_ENVIRONMENT
        ),
        "worker_performance_environment": {
            **performance_environment(dict(os.environ)),
            "OMP_NUM_THREADS": str(args.num_threads),
            "MKL_NUM_THREADS": str(args.num_threads),
            "TORCHINDUCTOR_CACHE_DIR": "<fresh temporary directory per worker>",
        },
        "isolation": {
            "process_policy": "fresh subprocess per correctness check and mode/repetition",
            "inductor_cache_policy": (
                "fresh TemporaryDirectory per worker; deleted after worker exits"
            ),
        },
        "correctness": correctness,
    }
    write_csv(args.output, rows, overwrite=args.overwrite)
    write_json(metadata_path, metadata, overwrite=args.overwrite)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Wrote run metadata to {metadata_path}")


def matrix_controller(
    args: argparse.Namespace,
    environment: dict[str, Any],
    metadata_path: Path,
) -> None:
    batch_sizes = MATRIX_BATCH_SIZES
    thread_counts = matrix_thread_counts(environment)
    configurations = tuple(
        (batch_size, num_threads)
        for batch_size in batch_sizes
        for num_threads in thread_counts
    )
    partial_csv, state_json = checkpoint_paths(args.output)
    fingerprint = matrix_fingerprint(
        args, environment, batch_sizes, thread_counts
    )

    validate_distinct_paths(
        {
            "environment": args.environment,
            "output": args.output,
            "metadata_output": metadata_path,
            "partial_csv": partial_csv,
            "state_json": state_json,
        }
    )

    if args.resume:
        if not state_json.is_file():
            raise FileNotFoundError(
                f"cannot resume without checkpoint state: {state_json}"
            )
        state = load_json(state_json)
        if state.get("fingerprint") != fingerprint:
            raise RuntimeError(
                "checkpoint settings or environment differ from this command; "
                "resume with the original settings or choose a new --output"
            )
        run_id = state.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError("checkpoint state has no valid run_id")
        validate_resume_final_artifacts(
            args.output, metadata_path, partial_csv, run_id
        )
        state["resume_count"] = int(state.get("resume_count", 0)) + 1
        rows = load_checkpoint_rows(partial_csv)
        state["last_updated_at_utc"] = utc_now()
        write_json(state_json, state, overwrite=True)
        print(
            f"Resuming from {partial_csv} with {len(rows)} checkpoint rows...",
            flush=True,
        )
    else:
        ensure_outputs_available((args.output, metadata_path), args.overwrite)
        checkpoint_existing = [
            str(path) for path in (partial_csv, state_json) if path.exists()
        ]
        if checkpoint_existing:
            raise FileExistsError(
                "checkpoint data already exists; use --resume to continue it: "
                + ", ".join(checkpoint_existing)
            )
        state = {
            "schema_version": 1,
            "run_id": str(uuid.uuid4()),
            "status": "in_progress",
            "started_at_utc": utc_now(),
            "last_updated_at_utc": utc_now(),
            "resume_count": 0,
            "command": shlex.join(
                [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
            ),
            "fingerprint": fingerprint,
            "correctness": {},
        }
        rows = []
        write_json(state_json, state, overwrite=False)
        write_csv(partial_csv, rows, overwrite=False)

    correctness_by_config = state.get("correctness")
    if not isinstance(correctness_by_config, dict):
        raise RuntimeError("checkpoint correctness data is invalid")

    completed = completed_checkpoint_measurements(
        rows,
        configurations,
        args,
        correctness_by_config,
        state["run_id"],
        fingerprint,
    )

    for batch_size, num_threads in configurations:
        config_key = matrix_config_key(batch_size, num_threads)
        correctness = correctness_by_config.get(config_key)
        if correctness is not None:
            if not correctness.get("allclose"):
                raise RuntimeError(
                    f"saved correctness check failed for {config_key}; aborting"
                )
            continue
        config_args = configured_args(args, batch_size, num_threads)
        config_args.run_id = state["run_id"]
        print(
            "Running isolated correctness check for "
            f"batch={batch_size}, threads={num_threads}...",
            flush=True,
        )
        correctness = run_fresh_worker(config_args, "correctness")
        print(
            f"Correctness {config_key}: "
            f"allclose={correctness['allclose']}, "
            f"max_abs_error={correctness['max_abs_error']:.6g}",
            flush=True,
        )
        correctness_by_config[config_key] = correctness
        state["last_updated_at_utc"] = utc_now()
        write_json(state_json, state, overwrite=True)
        if not correctness["allclose"]:
            raise RuntimeError(
                f"Eager and compiled outputs differ for {config_key}; "
                "performance run aborted"
            )

    total_measurements = len(configurations) * args.repetitions * 2
    for batch_size, num_threads in configurations:
        config_args = configured_args(args, batch_size, num_threads)
        config_args.run_id = state["run_id"]
        correctness = correctness_by_config[
            matrix_config_key(batch_size, num_threads)
        ]
        for repetition in range(1, args.repetitions + 1):
            modes = (
                ("eager", "compiled")
                if repetition % 2 == 1
                else ("compiled", "eager")
            )
            for execution_mode in modes:
                key = measurement_key(
                    batch_size, num_threads, repetition, execution_mode
                )
                if key in completed:
                    print(
                        "Skipping completed measurement "
                        f"batch={batch_size}, threads={num_threads}, "
                        f"repetition={repetition}, mode={execution_mode}",
                        flush=True,
                    )
                    continue
                print(
                    f"Running {len(completed) + 1}/{total_measurements}: "
                    f"batch={batch_size}, threads={num_threads}, "
                    f"repetition={repetition}/{args.repetitions}, "
                    f"mode={execution_mode}...",
                    flush=True,
                )
                measurement = run_fresh_worker(
                    config_args,
                    worker="benchmark",
                    execution_mode=execution_mode,
                )
                rows.extend(
                    result_rows(
                        environment,
                        config_args,
                        correctness,
                        execution_mode,
                        repetition,
                        measurement,
                    )
                )
                write_csv(partial_csv, rows, overwrite=True)
                completed.add(key)
                state["last_updated_at_utc"] = utc_now()
                write_json(state_json, state, overwrite=True)

    expected_rows = total_measurements * 2
    if len(completed) != total_measurements or len(rows) != expected_rows:
        raise RuntimeError(
            "matrix ended with missing data: "
            f"{len(completed)}/{total_measurements} measurements, "
            f"{len(rows)}/{expected_rows} rows"
        )

    metadata = {
        "schema_version": 1,
        "run_id": state["run_id"],
        "started_at_utc": state["started_at_utc"],
        "completed_at_utc": utc_now(),
        "command": state["command"],
        "working_directory": str(Path.cwd()),
        "python_executable": sys.executable,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "environment_snapshot": str(args.environment),
        "environment_snapshot_sha256": fingerprint[
            "environment_snapshot_sha256"
        ],
        "result_csv": str(args.output),
        "configuration": {
            **fingerprint,
            "execution_mode_order": (
                "odd repetitions: eager then compiled; "
                "even repetitions: compiled then eager"
            ),
        },
        "controller_performance_environment": performance_environment(
            INITIAL_PROCESS_ENVIRONMENT
        ),
        "worker_performance_environment": {
            **performance_environment(dict(os.environ)),
            "OMP_NUM_THREADS": "<num_threads for current configuration>",
            "MKL_NUM_THREADS": "<num_threads for current configuration>",
            "TORCHINDUCTOR_CACHE_DIR": "<fresh temporary directory per worker>",
        },
        "isolation": {
            "process_policy": (
                "fresh subprocess per correctness check and mode/repetition"
            ),
            "inductor_cache_policy": (
                "fresh TemporaryDirectory per worker; deleted after worker exits"
            ),
        },
        "checkpoint": {
            "resume_count": state["resume_count"],
            "completed_measurements": len(completed),
            "checkpoint_files_removed_after_success": True,
        },
        "correctness": correctness_by_config,
    }

    # Checkpoints remain available if either final write fails.  A resumed run
    # may safely replace a half-written finalization with the same validated data.
    write_csv(args.output, rows, overwrite=args.overwrite or args.resume)
    write_json(metadata_path, metadata, overwrite=args.overwrite or args.resume)
    state_json.unlink()
    partial_csv.unlink()
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Wrote run metadata to {metadata_path}")


def controller_main(args: argparse.Namespace) -> None:
    validate_controller_environment(args)
    metadata_path = metadata_output_path(args)
    validate_distinct_paths(
        {
            "environment": args.environment,
            "output": args.output,
            "metadata_output": metadata_path,
        }
    )
    environment = load_environment(args.environment)
    validate_environment_snapshot(environment)
    if args.matrix:
        matrix_controller(args, environment, metadata_path)
    else:
        ensure_outputs_available((args.output, metadata_path), args.overwrite)
        minimal_controller(args, environment, metadata_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        action="store_true",
        help=(
            "run batch sizes 1/8/32 and thread counts 1/4/physical cores; "
            "defaults to five independent repetitions"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument(
        "--repetitions",
        type=int,
        help="independent repetitions; defaults to 1, or 5 with --matrix",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--min-run-time", type=float, default=1.0)
    parser.add_argument(
        "--environment",
        type=Path,
        default=Path("results/environment.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "result CSV; defaults to results/minimal_raw_results.csv, "
            "or results/raw_results.csv with --matrix"
        ),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="JSON sidecar path; defaults to OUTPUT with .metadata.json suffix",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace an existing output file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a matching interrupted --matrix run from its checkpoint",
    )
    parser.add_argument(
        "--worker",
        choices=("correctness", "benchmark"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--execution-mode",
        choices=("eager", "compiled"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.repetitions is None:
        args.repetitions = MATRIX_MIN_REPETITIONS if args.matrix else 1
    if args.output is None:
        args.output = Path(
            "results/raw_results.csv"
            if args.matrix
            else "results/minimal_raw_results.csv"
        )
    return args


def main() -> None:
    args = parse_args()
    if args.worker:
        if args.worker == "benchmark" and args.execution_mode is None:
            raise ValueError("benchmark worker requires --execution-mode")
        worker_main(args)
    else:
        controller_main(args)


if __name__ == "__main__":
    main()
