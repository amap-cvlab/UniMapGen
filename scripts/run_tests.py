#!/usr/bin/env python3
"""Run the dependency-free function tests when pytest is unavailable."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tests = []
    for path in sorted((root / "tests").glob("test_*.py")):
        namespace = runpy.run_path(str(path))
        tests.extend(
            value
            for name, value in namespace.items()
            if name.startswith("test_") and callable(value)
        )
    for test in tests:
        test()
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
