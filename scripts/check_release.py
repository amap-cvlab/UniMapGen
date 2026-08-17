#!/usr/bin/env python3
"""Fail if a release contains secrets, private infrastructure, or model blobs."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "outputs", "results", "visualizations"}
BINARY_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
WEIGHT_SUFFIXES = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}
MAX_GITHUB_FILE_SIZE = 95 * 1024 * 1024
FORBIDDEN = {
    "private mount path": re.compile(r"/mnt/(?:nas-data|workspace)"),
    "private IP address": re.compile(r"\b(?:10|33)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "Alibaba internal host": re.compile(r"(?:alibaba-inc\.com|code\.alibaba-inc\.com)"),
    "hard-coded password": re.compile(r"password\s*[=:]\s*['\"]?\w+", re.IGNORECASE),
}


def main() -> None:
    failures = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        relative = path.relative_to(ROOT)
        if path.stat().st_size > MAX_GITHUB_FILE_SIZE:
            failures.append(f"file exceeds 95 MiB: {relative}")
            continue
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            failures.append(f"model weight tracked: {relative}")
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in FORBIDDEN.items():
            if pattern.search(text):
                failures.append(f"{label}: {relative}")
    if failures:
        print("Release check failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("Release check passed.")


if __name__ == "__main__":
    main()
