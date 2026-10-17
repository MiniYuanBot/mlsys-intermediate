"""Portability regressions, including native compilation in Unicode/space paths."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import platform_tools as tools
from setup_format import shfmt_asset

ROOT = Path(__file__).resolve().parents[2]


class PortabilityTests(unittest.TestCase):
    def test_native_compiler_with_unicode_and_spaces(self):
        with tempfile.TemporaryDirectory(prefix="mlsys platform ") as directory:
            directory = Path(directory)
            source = directory / "算例 with spaces.cpp"
            source.write_text(
                "#include <iostream>\n#include <optional>\nint main() { std::optional<int> x = 6; std::cout << *x; }\n",
                encoding="utf-8",
            )
            output = tools.executable_path(directory / "程序 with spaces")
            subprocess.run(
                tools.cxx_command(source, output), cwd=directory, check=True, capture_output=True
            )
            result = subprocess.run([str(output)], check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout, "6")

    def test_msvc_executable_and_flags(self):
        windows = SimpleNamespace(name="nt", environ={"CXX": "C:/Visual Studio/cl.exe"})
        with (
            patch.object(tools, "os", windows),
            patch.object(tools.shutil, "which", return_value="C:/Visual Studio/cl.exe"),
        ):
            command = tools.cxx_command("source with spaces.cpp", "program")
            self.assertIn("/std:c++17", command)
            self.assertIn("/EHsc", command)
            self.assertIn("/utf-8", command)
            self.assertFalse(any(arg.startswith("-std=") for arg in command))
            self.assertTrue(
                any(arg.startswith("/Fe") and arg.endswith("program.exe") for arg in command)
            )
            self.assertTrue(
                any(arg.startswith("/Fo") and arg.endswith("program.obj") for arg in command)
            )

    def test_macos_rejects_cuda(self):
        with patch.object(tools, "host_platform", return_value="macos"):
            with self.assertRaisesRegex(RuntimeError, "use CPU on macOS"):
                tools.cuda_command("kernel.cu", "kernel")

    def test_cuda_architecture_is_configurable(self):
        with (
            patch.object(tools, "host_platform", return_value="linux"),
            patch.object(tools, "find_tool", return_value="nvcc"),
        ):
            with patch.dict(os.environ, {"MLSYS_CUDA_ARCH": "sm_80"}):
                self.assertIn("-arch=sm_80", tools.cuda_command("kernel.cu", "kernel"))
            with patch.dict(os.environ, {"MLSYS_CUDA_ARCH": "invalid;target"}):
                with self.assertRaisesRegex(RuntimeError, "MLSYS_CUDA_ARCH"):
                    tools.cuda_command("kernel.cu", "kernel")

    def test_node_cli_avoids_windows_batch_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory) / "Node JS"
            cli = directory / "node_modules/npm/bin/npm-cli.js"
            cli.parent.mkdir(parents=True)
            cli.write_text("", encoding="utf-8")
            with patch.object(tools.shutil, "which", return_value=str(directory / "node.exe")):
                command = tools.node_package_command("npm", str(directory / "npm.cmd"))
            self.assertEqual(command, [str(directory / "node.exe"), str(cli)])

    def test_shfmt_assets(self):
        for system, expected in [("Windows", "windows"), ("Linux", "linux"), ("Darwin", "darwin")]:
            for machine, arch in [("AMD64", "amd64"), ("arm64", "arm64")]:
                suffix = ".exe" if system == "Windows" else ""
                self.assertEqual(
                    shfmt_asset(system, machine, "3.12.0"),
                    f"shfmt_v3.12.0_{expected}_{arch}{suffix}",
                )
        with self.assertRaises(RuntimeError):
            shfmt_asset("Linux", "unknown", "3.12.0")

    def test_materialized_lab_runs_outside_repository(self):
        with tempfile.TemporaryDirectory(prefix="mlsys standalone ") as directory:
            destination = Path(directory) / "独立 lab"
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "labs/tools/materialize.py"),
                    "--lab",
                    "1",
                    "--impl",
                    "reference",
                    "--to",
                    str(destination),
                ],
                env=env,
                check=True,
                capture_output=True,
            )
            self.assertTrue((destination / "_tools/platform_tools.py").is_file())
            result = subprocess.run(
                [sys.executable, str(destination / "exercise.py"), "--mode", "test", "--part", "A"],
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS A", result.stdout)


if __name__ == "__main__":
    unittest.main()
