"""Enumerate project files without descending into build output or dependencies."""

import os
from fnmatch import fnmatch
from pathlib import Path

IGNORED = (
    ".git",
    "__pycache__",
    ".venv*",
    "venv",
    "env",
    "node_modules",
    "target",
    "build*",
    "cmake-build-*",
    "CMakeFiles",
    "_deps",
    "external",
    "third_party",
    "third-party",
    "vendor",
    "results",
    "reports",
    "artifacts",
    "generated",
    ".*cache*",
)


def ignore_files(directory, names):
    return {name for name in names if any(fnmatch(name, pattern) for pattern in IGNORED)}


def project_files(root: Path):
    for directory, dirs, names in os.walk(root):
        excluded = ignore_files(directory, dirs)
        dirs[:] = sorted(set(dirs) - excluded)
        for name in sorted(names):
            if name.endswith((".pyc", ".nsys-rep", ".ncu-rep")):
                continue
            yield Path(directory) / name
