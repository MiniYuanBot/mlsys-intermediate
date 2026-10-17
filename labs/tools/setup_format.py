#!/usr/bin/env python3
"""Install the pinned formatters locally on Windows, Linux, macOS or WSL."""

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import urllib.request

from platform_tools import node_package_command

LABS = Path(__file__).resolve().parents[1]


def shfmt_asset(system, machine, version):
    systems = {"Windows": "windows", "Linux": "linux", "Darwin": "darwin"}
    architectures = {"x86_64": "amd64", "amd64": "amd64", "arm64": "arm64", "aarch64": "arm64"}
    if system not in systems or machine.lower() not in architectures:
        raise RuntimeError(
            f"No automatic shfmt installation for {system}/{machine}; install it manually."
        )
    suffix = ".exe" if system == "Windows" else ""
    return f"shfmt_v{version}_{systems[system]}_{architectures[machine.lower()]}{suffix}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="show the plan without installing")
    args = parser.parse_args()
    pins = dict(
        line.split("=", 1)
        for line in (LABS / "tools/formatter-versions.env").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )
    env = LABS / ".venv-format"
    binaries = env / ("Scripts" if os.name == "nt" else "bin")
    python = binaries / ("python.exe" if os.name == "nt" else "python")
    npm = node_package_command("npm")
    asset = shfmt_asset(platform.system(), platform.machine(), pins["SHFMT"])
    url = f"https://github.com/mvdan/sh/releases/download/v{pins['SHFMT']}/{asset}"

    def run(command):
        print("+", subprocess.list2cmdline(list(map(str, command))), flush=True)
        if not args.dry_run:
            subprocess.run(list(map(str, command)), cwd=LABS, check=True)

    if not python.exists():
        run([sys.executable, "-m", "venv", env])
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            *[
                f"{p}=={pins[v]}"
                for p, v in [
                    ("clang-format", "CLANG_FORMAT"),
                    ("gersemi", "GERSEMI"),
                    ("ruff", "RUFF"),
                ]
            ],
        ]
    )
    run(
        [
            *npm,
            "install",
            "--prefix",
            env,
            "--no-audit",
            "--no-fund",
            f"prettier@{pins['PRETTIER']}",
        ]
    )
    print(f"download {url} -> {binaries}")
    if not args.dry_run:
        target = binaries / ("shfmt.exe" if os.name == "nt" else "shfmt")
        temporary = target.with_suffix(".download")
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
            temporary.chmod(0o755)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        if not shutil.which("rustc") or not shutil.which("rustup"):
            raise RuntimeError("Install Rust with rustup, then rerun this script.")
        version = subprocess.check_output(["rustc", "--version"], text=True).split()[1]
        if version != pins["RUST"]:
            raise RuntimeError(
                f"Use Rust {pins['RUST']} for the shared formatter rules; found {version}."
            )
    run(["rustup", "component", "add", "rustfmt"])
    run([sys.executable, LABS / "tools/format_check.py"])


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
