"""Portability regressions, including native compilation in Unicode/space paths."""

import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import lab_meta
import platform_tools as tools
from setup_format import shfmt_asset

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "labs/tools"
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def compiler_family():
    try:
        return tools.compiler_family(tools.find_cxx())
    except RuntimeError:
        return "missing"


# MinGW ld cannot write some non-ASCII output paths (a known toolchain bug, reported by
# doctor/check_env); MSVC, GCC and Clang are tested for real.
MINGW = compiler_family().startswith("mingw")
MINGW_REASON = "MinGW ld fails on non-ASCII paths (known); use MSVC or WSL for this check"


def run_python(*args, env=None, cwd=None):
    return subprocess.run(
        [sys.executable, *map(str, args)],
        env=env or ENV,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


class PortabilityTests(unittest.TestCase):
    @unittest.skipIf(MINGW, MINGW_REASON)
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

    @unittest.skipIf(MINGW, MINGW_REASON)
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


class EnvironmentSystemTests(unittest.TestCase):
    TABLE = re.compile(r"^\s*(\d+)\s*[)=]\s*'?(comps=\"[^\"]*\" models=\"[^\"]*\")'?", re.M)

    def test_script_lab_tables_match_lab_meta(self):
        expected = {n: lab_meta.table_line(n) for n in lab_meta.LAB_NUMBERS}
        for name in ["setup_env.sh", "setup_env.ps1", "check_env.sh", "check_env.ps1"]:
            text = (TOOLS / name).read_text(encoding="utf-8")
            block = text.split("BEGIN LAB TABLE", 1)[1].split("END LAB TABLE", 1)[0]
            found = {int(n): line for n, line in self.TABLE.findall(block)}
            self.assertEqual(found, expected, f"{name}: regenerate with lab_meta.py --table")
            if name.endswith(".sh"):
                self.assertIn(f'ALL_LABS="{" ".join(map(str, lab_meta.LAB_NUMBERS))}"', block)

    def test_lab_meta_components(self):
        self.assertEqual(
            lab_meta.install_components(5),
            (("gpu:R", "torch:R", "hf:R", "cargo"), ("Qwen3-0.6B:R",)),
        )
        self.assertEqual(
            lab_meta.install_components(4),
            (("gpu:R", "torch:R", "hf:R", "cargo"), ("Qwen3-0.6B:R", "Qwen3-1.7B:R")),
        )
        self.assertEqual(lab_meta.install_components(8), (("cargo",), ()))  # R is a guide
        self.assertEqual(lab_meta.feasibility(2, "B", "windows"), "wsl")
        self.assertEqual(lab_meta.feasibility(2, "C", "macos"), "cpu-only")
        self.assertEqual(lab_meta.feasibility(5, "C", "windows"), "full")
        for n, lab in lab_meta.LABS.items():
            r_required = "R" in lab.stages and lab.r_implemented and not lab.stages["R"].guide_only
            self.assertEqual(
                lab.required_stages(), ("A", "B", "C", "R") if r_required else ("A", "B", "C"), n
            )
            if lab.r_implemented and lab.exists():
                # A lab may only claim R once its sources carry R TODO regions.
                self.assertTrue(
                    any(
                        "BEGIN TODO R" in f.read_text(encoding="utf-8", errors="ignore")
                        for f in lab.directory.rglob("*")
                        if f.is_file()
                    ),
                    n,
                )

    def test_pins_hash_matches_shell_definition(self):
        text = (TOOLS / "env-versions.env").read_text(encoding="utf-8")
        lines = [x for x in text.splitlines() if re.match(r"^[A-Z_][A-Z0-9_]*=", x)]
        digest = hashlib.sha256("".join(x + "\n" for x in lines).encode()).hexdigest()
        self.assertEqual(lab_meta.pins_hash(), digest)
        self.assertTrue(any(x.startswith("TORCH=") for x in lines))

    @unittest.skipUnless(
        (ROOT.parent / "mlsys-foundations/labs/tools/env-versions.env").is_file(),
        "no sibling mlsys-foundations checkout",
    )
    def test_pins_identical_to_foundations(self):
        sibling = ROOT.parent / "mlsys-foundations/labs/tools/env-versions.env"
        self.assertEqual(
            lab_meta.pin_lines(sibling), lab_meta.pin_lines(TOOLS / "env-versions.env")
        )

    def test_windows_cuda_gate_prints_wsl_command(self):
        labs = "D:" + "\\course dir\\labs"
        with (
            patch.object(tools, "host_platform", return_value="windows"),
            patch.object(tools, "labs_directory", return_value=labs),
            patch.dict(os.environ, {"MLSYS_RERUN": "--lab 2 --device cuda"}),
        ):
            with self.assertRaises(tools.MissingDependency) as caught:
                tools.require_cuda(2, "B", "GEMV kernels")
        message = str(caught.exception)
        self.assertTrue(message.startswith("MISSING CUDA"), message)
        self.assertIn("\n  fix: wsl -d Ubuntu-24.04 -- bash -lc", message)
        self.assertIn("cd '/mnt/d/course dir/labs'", message)
        self.assertIn("python tools/run.py --lab 2 --device cuda", message)

    def test_wsl_path(self):
        self.assertEqual(tools.wsl_path("D:" + "\\a\\b c"), "/mnt/d/a/b c")
        unc = "\\\\" + "wsl.localhost\\Ubuntu-24.04\\home\\u"
        self.assertEqual(tools.wsl_path(unc), "/home/u")
        self.assertEqual(tools.wsl_path("/home/u/labs"), "/home/u/labs")

    def test_missing_tool_exits_3_without_traceback(self):
        env = {**ENV, "CARGO": "mlsys-no-such-cargo"}
        result = run_python(ROOT / "labs/lab5/exercise.py", "--mode", "test", env=env)
        self.assertEqual(result.returncode, tools.MISSING_EXIT, result.stdout + result.stderr)
        self.assertRegex(result.stderr, r"^MISSING cargo for lab 5 stage A/B/C/R: .+\n  fix: ")
        self.assertNotIn("Traceback", result.stderr)

    def test_unimplemented_r_stage(self):
        pending = [
            n
            for n, lab in lab_meta.LABS.items()
            if lab.exists() and "R" in lab.stages and not lab.r_implemented
        ]
        if not pending:
            self.skipTest("every lab with an R stage has implemented it")
        result = run_python(TOOLS / "run.py", "--lab", pending[0], "--part", "R", "--mode", "test")
        self.assertEqual(result.returncode, 2)
        self.assertIn("NOT RUN R", result.stderr)

    def test_registered_but_missing_lab(self):
        missing = [n for n, lab in lab_meta.LABS.items() if not lab.exists()]
        if not missing:
            self.skipTest("every registered lab exists")
        result = run_python(TOOLS / "run.py", "--lab", missing[0], "--mode", "test")
        self.assertEqual(result.returncode, 2)
        self.assertIn("does not exist yet", result.stderr)


if __name__ == "__main__":
    unittest.main()
