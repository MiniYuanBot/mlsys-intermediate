"""Portable compiler commands, dependency gates and standalone lab support (stdlib only).

Every labN/exercise.py checks its tools through require_tool/require_module/require_cuda.
A missing dependency raises MissingDependency; exit_on_missing() turns it into one line
plus a `fix:` hint and exit code 3, which tools/run.py records as status env-missing.
"""

import importlib.util
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

MISSING_EXIT = 3
WSL_DISTRO = "Ubuntu-24.04"


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
        "C++17 compiler not found. Windows: use an x64 Visual Studio Developer terminal or "
        "MSYS2 UCRT64 g++ (tools\\setup_env.ps1 installs it); macOS: install Xcode Command "
        "Line Tools; Linux/WSL: install g++ or clang++. "
        "CXX may name an executable or its full path, without extra flags."
    )


def compiler_family(compiler):
    """msvc | clang-cl | mingw-gcc | mingw-clang | gcc | clang | apple-clang | unknown."""
    name = Path(compiler).name.lower()
    if name in {"cl", "cl.exe"}:
        return "msvc"
    if name.startswith("clang-cl"):
        return "clang-cl"
    try:
        text = subprocess.run(
            [compiler, "--version"], capture_output=True, text=True, errors="replace", timeout=30
        ).stdout.lower()
    except (OSError, subprocess.SubprocessError):
        text = ""
    mingw = os.name == "nt" or "mingw" in text or "msys" in text
    if "apple" in text:
        return "apple-clang"
    if "clang" in text:
        return "mingw-clang" if mingw else "clang"
    if "gcc" in text or "g++" in text or "free software foundation" in text:
        return "mingw-gcc" if mingw else "gcc"
    return "unknown"


def mingw_path_warning(family, *paths):
    """MinGW ld fails on some non-ASCII paths; return a warning string or None."""
    if not family.startswith("mingw"):
        return None
    bad = [str(p) for p in paths if p and not str(p).isascii()]
    if not bad:
        return None
    return (
        f"MinGW ({family}) is known to fail linking under non-ASCII paths: {', '.join(bad)}. "
        "Move the repository/TEMP to an ASCII path, use MSVC (Developer terminal), or use WSL."
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


def find_nvcc():
    """nvcc from NVCC/PATH, else the default toolkit location (non-login WSL shells)."""
    if os.environ.get("NVCC"):
        return shutil.which(os.environ["NVCC"].strip('"'))
    if found := find_tool("nvcc"):
        return found
    if os.name == "nt":
        return None
    for base in filter(None, [os.environ.get("CUDA_HOME"), "/usr/local/cuda"]):
        candidate = Path(base) / "bin" / "nvcc"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def cargo_command():
    """cargo from CARGO/PATH or ~/.cargo/bin; that directory is added to PATH for rustc."""
    if os.environ.get("CARGO"):
        return shutil.which(os.environ["CARGO"].strip('"'))
    if found := find_tool("cargo"):
        return found
    home = Path(os.environ.get("CARGO_HOME") or Path.home() / ".cargo") / "bin"
    candidate = home / ("cargo.exe" if os.name == "nt" else "cargo")
    if candidate.is_file():
        os.environ["PATH"] = str(home) + os.pathsep + os.environ.get("PATH", "")
        return str(candidate)
    return None


def cuda_command(source, output, includes=()):
    if host_platform() == "macos":
        raise RuntimeError(
            "CUDA labs require an NVIDIA GPU on Linux, WSL2 or Windows; use CPU on macOS."
        )
    compiler = find_nvcc()
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


class MissingDependency(RuntimeError):
    """A tool, module or device a lab stage needs is not available on this machine."""

    def __init__(self, what, lab, stage, purpose, fix):
        super().__init__(f"MISSING {what} for lab {lab} stage {stage}: {purpose}\n  fix: {fix}")
        self.what, self.lab, self.stage, self.fix = what, lab, stage, fix


def setup_hint(lab):
    if host_platform() == "windows":
        return f"powershell -ExecutionPolicy Bypass -File tools\\setup_env.ps1 -Lab {lab}"
    return f"bash tools/setup_env.sh --lab {lab}"


def labs_directory():
    """The repository's labs/ (run.py exports it; temp copies cannot find it themselves)."""
    if os.environ.get("MLSYS_LABS_DIR"):
        return Path(os.environ["MLSYS_LABS_DIR"])
    here = Path(__file__).resolve().parents[1]
    return here if (here / "tools" / "run.py").is_file() else Path.cwd()


def wsl_path(path):
    r"""D:\x\y -> /mnt/d/x/y; \\wsl$\Distro\home -> /home; POSIX paths stay unchanged."""
    text = str(path)
    unc = re.match(r"^\\\\wsl(?:\$|\.localhost)\\[^\\]+(.*)$", text, re.IGNORECASE)
    if unc:
        return unc[1].replace("\\", "/") or "/"
    drive = re.match(r"^([A-Za-z]):[\\/]?(.*)$", text)
    if drive:
        return f"/mnt/{drive[1].lower()}/" + drive[2].replace("\\", "/")
    return text.replace("\\", "/")


def wsl_command(lab, stage):
    """A command to paste into PowerShell/cmd that reruns the same request inside WSL."""
    rerun = os.environ.get("MLSYS_RERUN") or f"--lab {lab} --part {stage} --device cuda"
    target = wsl_path(labs_directory())
    if re.search(r"[^\w/.\-]", target):
        target = "'" + target.replace("'", "'\\''") + "'"
    distro = os.environ.get("MLSYS_WSL_DISTRO", WSL_DISTRO)
    return (
        f'wsl -d {distro} -- bash -lc "cd {target} && . ~/venvs/mlsys/bin/activate '
        f'&& python tools/run.py {rerun}"'
    )


def require_tool(name, lab, stage, purpose):
    """Path of a command-line tool; c++/cxx, nvcc and cargo use the dedicated finders."""
    if name in {"c++", "cxx"}:
        try:
            return find_cxx()
        except RuntimeError as error:
            raise MissingDependency(
                "C++17 compiler", lab, stage, purpose, setup_hint(lab)
            ) from error
    finder = {"nvcc": find_nvcc, "cargo": cargo_command}.get(name, lambda: find_tool(name))
    found = finder()
    if not found:
        raise MissingDependency(name, lab, stage, purpose, setup_hint(lab))
    return found


def require_module(module, lab, stage, purpose=None):
    if importlib.util.find_spec(module) is None:
        raise MissingDependency(
            f"Python module {module} (interpreter {sys.executable})",
            lab,
            stage,
            purpose or f"imported by lab{lab} stage {stage}",
            setup_hint(lab) + "; then use the course venv (new terminal, or run `mlsys`)",
        )


def gpu_visible():
    smi = shutil.which("nvidia-smi")
    if not smi and Path("/usr/lib/wsl/lib/nvidia-smi").exists():
        smi = "/usr/lib/wsl/lib/nvidia-smi"
    if not smi:
        return False
    try:
        return subprocess.run([smi, "-L"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def require_cuda(lab, stage, purpose="CUDA kernels", nvcc=True):
    """Platform gate (CUDA runs only in WSL2/Linux), then nvcc, then a visible GPU.

    Returns the nvcc path, or None with nvcc=False (GPU stages that only need torch).
    """
    where = host_platform()
    if where == "windows":
        raise MissingDependency(
            "CUDA (native Windows is CPU-only in this course)",
            lab,
            stage,
            f"{purpose}; CUDA stages run in WSL2",
            wsl_command(lab, stage),
        )
    if where == "macos":
        raise MissingDependency(
            "CUDA (macOS has no NVIDIA GPU)",
            lab,
            stage,
            purpose,
            "run the CPU parts here; CUDA stages need WSL2 or Linux with an NVIDIA GPU",
        )
    compiler = None
    if nvcc:
        compiler = find_nvcc()
        if compiler is None:
            raise MissingDependency("nvcc (CUDA toolkit)", lab, stage, purpose, setup_hint(lab))
    if not gpu_visible():
        raise MissingDependency(
            "NVIDIA GPU (nvidia-smi -L failed)",
            lab,
            stage,
            purpose,
            "install or update the NVIDIA driver (in WSL: the Windows driver); see setup.md",
        )
    return compiler


def exit_on_missing(main):
    """Run main(); MissingDependency -> exit 3, a failed child command -> exit 1.

    Child exit codes are folded to 1 so that 3 always means "environment missing".
    """
    try:
        result = main()
    except MissingDependency as error:
        print(error, file=sys.stderr, flush=True)
        raise SystemExit(MISSING_EXIT) from None
    except subprocess.CalledProcessError as error:
        command = " ".join(map(str, error.cmd)) if isinstance(error.cmd, list) else error.cmd
        print(f"FAILED (exit {error.returncode}): {command}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
    if isinstance(result, int) and result:
        raise SystemExit(1 if result == MISSING_EXIT else result)


def stage_label(part):
    return "A/B/C" if part == "all" else part


def r_stage(lab, part, has_r, implemented=False):
    """--part R / all for labs whose R stage is absent or not in code yet."""
    if part not in {"R", "all"}:
        return
    if not has_r:
        message = f"NOT RUN R (lab{lab} has no R stage)"
    elif not implemented:
        message = "NOT RUN R (not implemented yet in this lab)"
    else:
        return
    if part == "R":
        print(message, file=sys.stderr, flush=True)
        raise SystemExit(2)
    if has_r:
        print(message, flush=True)


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
