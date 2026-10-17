#!/usr/bin/env python3
"""Apply/check the root formatter rules, including files not yet tracked by Git."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from fnmatch import fnmatch
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
ROOT = LABS.parent
CONFIG = ROOT
EXCLUDED_DIRS = (
    ".git",
    "build",
    "build-*",
    "cmake-build-*",
    "CMakeFiles",
    "out",
    "dist",
    "target",
    "_deps",
    "external",
    "third_party",
    "third-party",
    "vendor",
    "node_modules",
    ".venv*",
    "venv",
    "env",
    ".tox",
    ".nox",
    "__pycache__",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "results",
    "artifacts",
    "generated",
)
EXCLUDED_FILES = ("*.generated.*", "*.min.*", "package-lock.json", "Cargo.lock")


def files() -> list[Path]:
    selected = [p for p in ROOT.iterdir() if p.is_file()]
    for base in ("course", "labs", "solutions", ".vscode", ".obsidian"):
        for directory, dirs, names in os.walk(ROOT / base):
            dirs[:] = [d for d in dirs if not any(fnmatch(d, p) for p in EXCLUDED_DIRS)]
            if Path(directory) == ROOT / ".obsidian":
                dirs[:] = [d for d in dirs if d not in ("cache", "plugins")]
            for name in sorted(names):
                path = Path(directory) / name
                if any(fnmatch(name, p) for p in EXCLUDED_FILES):
                    continue
                if path.parent == ROOT / ".obsidian" and fnmatch(name, "workspace*.json"):
                    continue
                selected.append(path)
    return sorted(selected)


def resolve(name: str) -> str | None:
    local = LABS / ".venv-format"
    directories = [
        os.environ.get("MLSYS_FORMAT_BIN", ""),
        str(local / ("Scripts" if os.name == "nt" else "bin")),
        str(local / "node_modules/.bin"),
        os.environ.get("PATH", ""),
    ]
    return shutil.which(name, path=os.pathsep.join(filter(None, directories)))


def pinned_versions() -> dict[str, str]:
    return dict(
        line.strip().split("=", 1)
        for line in (LABS / "tools" / "formatter-versions.env").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="format files in place")
    parser.add_argument("--report", type=Path, help="save command output as JSON")
    args = parser.parse_args()
    pins = pinned_versions()
    mode = args.write
    groups = [
        (
            "C/C++/CUDA",
            "clang-format",
            "CLANG_FORMAT",
            (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".cu", ".cuh"),
            [
                f"--style=file:{CONFIG / '.clang-format'}",
                *(["-i"] if mode else ["--dry-run", "--Werror"]),
            ],
        ),
        (
            "Python",
            "ruff",
            "RUFF",
            (".py",),
            [
                "format",
                "--config",
                str(CONFIG / "pyproject.toml"),
                "--no-cache",
                *([] if mode else ["--check"]),
            ],
        ),
        (
            "CMake",
            "gersemi",
            "GERSEMI",
            ("CMakeLists.txt", ".cmake", ".cmake.in"),
            [
                "--config",
                str(CONFIG / ".gersemirc"),
                "--no-cache",
                "--in-place" if mode else "--check",
            ],
        ),
        (
            "Markdown/JSON/YAML/HTML/JS",
            "prettier",
            "PRETTIER",
            (".md", ".json", ".jsonc", ".yaml", ".yml", ".html", ".js", ".css"),
            [
                "--config",
                str(CONFIG / ".prettierrc.json"),
                "--no-editorconfig",
                "--write" if mode else "--check",
            ],
        ),
        (
            "Shell",
            "shfmt",
            "SHFMT",
            (".sh", ".bash", ".sbatch", ".envrc"),
            ["-i", "4", "-bn", "-ci", "-sr", "-w" if mode else "-d"],
        ),
        (
            "Rust",
            "rustfmt",
            "RUST",
            (".rs",),
            ["--edition", "2021", *([] if mode else ["--check"])],
        ),
    ]
    paths = files()
    report = {"operation": "write" if mode else "check", "groups": []}
    failed = False
    for name, tool, pin, suffixes, options in groups:
        targets = [str(p.relative_to(ROOT)) for p in paths if str(p).endswith(suffixes)]
        item = {"group": name, "files": len(targets), "commands": []}
        report["groups"].append(item)
        if not targets:
            item["status"] = "skipped_no_files"
            print(f"[SKIP] {name}: no files")
            continue
        executable = resolve(tool)
        version_exe = resolve("rustc") if tool == "rustfmt" else executable
        if executable is None or version_exe is None:
            item["status"] = "missing_tool"
            print(f"[FAIL] {name}: run bash tools/setup-format.sh")
            failed = True
            continue
        version = subprocess.run([version_exe, "--version"], capture_output=True, text=True)
        item["version"] = version.stdout.strip()
        detected = re.search(r"\d+\.\d+\.\d+", version.stdout)
        if version.returncode or detected is None or detected.group() != pins[pin]:
            item["status"] = "wrong_version"
            print(f"[FAIL] {tool}: expected {pins[pin]}, found {item['version']}")
            failed = True
            continue
        item["status"] = "passed"
        for start in range(0, len(targets), 80):
            command = [executable, *options, *targets[start : start + 80]]
            run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            item["commands"].append(
                {
                    "argv": command,
                    "returncode": run.returncode,
                    "stdout": run.stdout,
                    "stderr": run.stderr,
                }
            )
            if run.returncode:
                item["status"] = "failed"
                failed = True
                print((run.stdout + run.stderr)[:8000])
        print(f"[{item['status'].upper()}] {name}: {len(targets)} files")
    report["status"] = "failed" if failed else "passed"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
