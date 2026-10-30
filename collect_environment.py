"""Collect reproducible host and PyTorch environment metadata.

Run this script inside the exact Conda environment used for the experiment:

    python collect_environment.py --output results/environment.json

The script is intentionally benchmark-free.  It records facts about the
environment without downloading data or changing the Python installation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RELEVANT_ENV_PREFIXES = (
    "ATEN_",
    "BLIS_",
    "CC",
    "CONDA",
    "CXX",
    "CUDA",
    "CUDNN",
    "DNNL_",
    "GLIBC_TUNABLES",
    "GOMP_",
    "KMP",
    "LD_LIBRARY_PATH",
    "MKL",
    "MALLOC_CONF",
    "NUMEXPR_",
    "ONEDNN_",
    "OMP",
    "OPENBLAS",
    "PYTORCH",
    "TBB_",
    "TORCH",
    "VECLIB_",
)


def run_command(command: list[str], timeout: int = 15) -> dict[str, Any]:
    """Run a read-only diagnostic command and retain success or failure."""
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "command": command,
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"{command[0]} was not found on PATH",
        }

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return {
            "command": command,
            "available": True,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "available": True,
            "returncode": None,
            "stdout": (exc.stdout or "").strip(),
            "stderr": f"timed out after {timeout} seconds",
        }


def first_output_line(result: dict[str, Any]) -> str | None:
    if result["returncode"] != 0 or not result["stdout"]:
        return None
    return result["stdout"].splitlines()[0]


def linux_cpu_info() -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": platform.processor() or None,
        "physical_cores": None,
        "logical_cores": os.cpu_count(),
    }
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
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

        if processors:
            result["model"] = processors[0].get("model name", result["model"])
            core_ids = {
                (item.get("physical id", "0"), item["core id"])
                for item in processors
                if "core id" in item
            }
            if core_ids:
                result["physical_cores"] = len(core_ids)

    if result["physical_cores"] is None:
        lscpu = run_command(["lscpu", "--parse=CORE,SOCKET"])
        if lscpu["returncode"] == 0:
            pairs = {
                tuple(line.split(",", maxsplit=1))
                for line in lscpu["stdout"].splitlines()
                if line and not line.startswith("#") and "," in line
            }
            result["physical_cores"] = len(pairs) or None
    return result


def windows_hardware_info() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cpu_script = (
        "Get-CimInstance Win32_Processor | "
        "Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | "
        "ConvertTo-Json -Compress"
    )
    gpu_script = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion | "
        "ConvertTo-Json -Compress"
    )
    cpu_command = run_command(["powershell", "-NoProfile", "-Command", cpu_script])
    gpu_command = run_command(["powershell", "-NoProfile", "-Command", gpu_script])

    cpu: dict[str, Any] = {
        "model": platform.processor() or None,
        "physical_cores": None,
        "logical_cores": os.cpu_count(),
    }
    gpus: list[dict[str, Any]] = []
    if cpu_command["returncode"] == 0:
        payload = json.loads(cpu_command["stdout"])
        entries = payload if isinstance(payload, list) else [payload]
        cpu = {
            "model": "; ".join(item["Name"].strip() for item in entries),
            "physical_cores": sum(int(item["NumberOfCores"]) for item in entries),
            "logical_cores": sum(
                int(item["NumberOfLogicalProcessors"]) for item in entries
            ),
        }
    if gpu_command["returncode"] == 0:
        payload = json.loads(gpu_command["stdout"])
        entries = payload if isinstance(payload, list) else [payload]
        gpus = [
            {
                "name": item.get("Name"),
                "memory_bytes": item.get("AdapterRAM"),
                "driver_version": item.get("DriverVersion"),
            }
            for item in entries
        ]
    return cpu, gpus


def nvidia_gpu_info() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[dict[str, Any]] = []
    if query["returncode"] == 0:
        for line in query["stdout"].splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 3:
                gpus.append(
                    {
                        "name": fields[0],
                        "memory_mib": int(fields[1]),
                        "driver_version": fields[2],
                    }
                )
    return gpus, query


def torch_info() -> dict[str, Any]:
    try:
        import torch
        import torchvision
    except Exception as exc:  # Preserve the exact import failure as evidence.
        return {"import_ok": False, "error": f"{type(exc).__name__}: {exc}"}

    info: dict[str, Any] = {
        "import_ok": True,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "torch_compile_available": hasattr(torch, "compile"),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "parallel_info": torch.__config__.parallel_info(),
    }
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    return info


def collect() -> dict[str, Any]:
    # Capture external values before importing PyTorch in torch_info().
    initial_environment = dict(os.environ)
    system = platform.system()
    if system == "Windows":
        cpu, platform_gpus = windows_hardware_info()
    else:
        cpu = linux_cpu_info()
        platform_gpus = []

    nvidia_gpus, nvidia_smi = nvidia_gpu_info()
    compiler_commands = {
        name: run_command([name, "--version"])
        for name in ("gcc", "g++", "clang", "clang++")
    }
    configured_compiler_commands = {
        name: run_command([path, "--version"])
        for name, path in (("CC", os.environ.get("CC")), ("CXX", os.environ.get("CXX")))
        if path
    }
    conda = run_command(["conda", "--version"])
    lscpu = run_command(["lscpu"])
    memory = run_command(["free", "-b"])
    libc = run_command(["ldd", "--version"])

    return {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection_command": shlex.join([sys.executable, *sys.argv]),
        "working_directory": str(Path.cwd()),
        "os": {
            "system": system,
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "cpu": cpu,
        "gpus": nvidia_gpus or platform_gpus,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "conda": {
            "version": first_output_line(conda),
            "environment_name": os.environ.get("CONDA_DEFAULT_ENV"),
            "prefix": os.environ.get("CONDA_PREFIX"),
        },
        "pytorch": torch_info(),
        "compilers": {
            "path_compilers": {
                name: first_output_line(result)
                for name, result in compiler_commands.items()
            },
            "configured_compilers": {
                name: {
                    "path": os.environ[name],
                    "version": first_output_line(result),
                }
                for name, result in configured_compiler_commands.items()
            },
        },
        "system_libraries": {
            "libc": first_output_line(libc),
        },
        "nvidia_smi": nvidia_smi,
        "environment_variables": {
            key: value
            for key, value in sorted(initial_environment.items())
            if key.startswith(RELEVANT_ENV_PREFIXES)
        },
        "diagnostic_commands": {
            "conda": conda,
            "compilers": compiler_commands,
            "configured_compilers": configured_compiler_commands,
            "lscpu": lscpu,
            "memory": memory,
            "libc": libc,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/environment.json"),
        help="Path for the generated JSON file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(collect(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote environment metadata to {args.output}")


if __name__ == "__main__":
    main()
