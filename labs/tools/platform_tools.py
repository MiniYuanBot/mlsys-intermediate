"""Portable compiler commands and standalone lab support (standard library only)."""

import os
from pathlib import Path
import platform
import re
import shutil


def host_platform():
    if os.name == "nt":
        return "windows"
    if platform.system() == "Darwin":
        return "macos"
    if "microsoft" in platform.release().lower() or os.environ.get("WSL_DISTRO_NAME"):
        return "wsl"
    return platform.system().lower()


def find_tool(name, variable=None):
    configured = os.environ.get(variable, "") if variable else ""
    executable = configured.strip('"') or name
    found = shutil.which(executable)
    if found:
        return found
    if configured:
        raise RuntimeError(f"{variable} executable not found: {executable}")
    return None


def find_cxx():
    if os.environ.get("CXX"):
        return find_tool("c++", "CXX")
    candidates = ["cl", "clang++", "g++", "c++"] if os.name == "nt" else ["c++", "g++", "clang++"]
    for name in candidates:
        if executable := find_tool(name):
            return executable
    raise RuntimeError(
        "C++17 compiler not found. Windows: use an x64 Visual Studio Developer terminal; "
        "macOS: install Xcode Command Line Tools; Linux/WSL: install g++ or clang++. "
        "CXX may name an executable or its full path, without extra flags."
    )


def executable_path(path):
    path = Path(path).resolve()
    if os.name == "nt" and path.suffix.lower() != ".exe":
        return path.with_name(path.name + ".exe")
    return path


def cxx_command(source, output, includes=()):
    compiler = find_cxx()
    output = executable_path(output)
    source = str(Path(source).resolve())
    if Path(compiler).name.lower() in {"cl", "cl.exe", "clang-cl", "clang-cl.exe"}:
        return [
            compiler,
            "/nologo",
            "/std:c++17",
            "/O2",
            "/EHsc",
            "/utf-8",
            "/DNOMINMAX",
            "/W4",
            *[f"/I{Path(p).resolve()}" for p in includes],
            source,
            f"/Fo{output.with_suffix('.obj')}",
            f"/Fe{output}",
        ]
    return [
        compiler,
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        *[f"-I{Path(p).resolve()}" for p in includes],
        source,
        "-o",
        str(output),
    ]


def cuda_command(source, output, includes=()):
    if host_platform() == "macos":
        raise RuntimeError(
            "CUDA labs require an NVIDIA GPU on Linux, WSL2 or Windows; use CPU on macOS."
        )
    compiler = find_tool("nvcc", "NVCC")
    if compiler is None:
        raise RuntimeError(
            "nvcc not found; install a compatible CUDA Toolkit and add its bin directory to PATH."
        )
    if os.name == "nt" and not find_tool("cl"):
        raise RuntimeError(
            "nvcc on Windows needs cl.exe; open an x64 Visual Studio Developer terminal."
        )
    arch = os.environ.get("MLSYS_CUDA_ARCH", "native")
    if not re.fullmatch(r"native|(?:sm|compute)_\d+[af]?", arch):
        raise RuntimeError("MLSYS_CUDA_ARCH must be native or an nvcc target such as sm_89.")
    return [
        compiler,
        "-std=c++17",
        "-O2",
        "-lineinfo",
        f"-arch={arch}",
        *(["-DNOMINMAX", "-Xcompiler=/utf-8,/EHsc"] if os.name == "nt" else []),
        *[f"-I{Path(p).resolve()}" for p in includes],
        str(Path(source).resolve()),
        "-o",
        str(executable_path(output)),
    ]


def stage_helpers(destination):
    """Include the helper in temporary/materialized labs without touching their parent."""
    target = Path(destination) / "_tools"
    target.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), target / "platform_tools.py")


def node_package_command(package, launcher=None):
    """Run npm/Prettier via node, avoiding Windows .cmd shell quoting."""
    node = shutil.which("node")
    if not node:
        raise RuntimeError("node not found; install Node.js with npm and add it to PATH.")
    entry = "bin/npm-cli.js" if package == "npm" else "bin/prettier.cjs"
    launcher = launcher or shutil.which(package)
    candidates = []
    if launcher:
        resolved = Path(launcher).resolve()
        if resolved.suffix in {".js", ".cjs", ".mjs"}:
            candidates.append(resolved)
        candidates.append(Path(launcher).parent / "node_modules" / package / entry)
    candidates.append(Path(node).parent / "node_modules" / package / entry)
    for path in candidates:
        if path.is_file():
            return [node, str(path)]
    raise RuntimeError(f"{package} JavaScript entry point not found; reinstall {package} with npm.")
