"""Verify that torch.compile can compile and run a tiny CPU function.

This is a toolchain smoke test, not a performance benchmark.
"""

from __future__ import annotations

import json

import torch


def function(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x) + 1


def main() -> None:
    torch.manual_seed(0)
    inputs = torch.randn(8)
    eager_output = function(inputs)
    compiled_output = torch.compile(function)(inputs)
    result = {
        "allclose": bool(torch.allclose(eager_output, compiled_output)),
        "max_abs_error": float((eager_output - compiled_output).abs().max()),
        "torch_version": torch.__version__,
        "device": "cpu",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
