#!/usr/bin/env python3
"""Inject known bugs into a temporary reference copy and check that the tests catch them.

Each fault in solutions/labN/faults.json (schema: labs/templates/faults.example.json,
rules: labs/docs/authoring.md) is applied to a fresh copy built like tools/run.py does
(labs/labN + solutions/labN overlay + staged helpers). The lab's exercise.py then runs
with --mode test --part P --device D and the outcome is classified:

  CAUGHT   nonzero exit other than 3: the tests detected the fault
  MISSED   exit 0: the tests passed with the bug in place
  ENV      exit 3 (missing dependency) in the fault run or the baseline: not verified
  STALE    the `find` text does not occur exactly `count` times in the reference
  TIMEOUT  the run exceeded --timeout (reported; treat it as a weak test)

The unmodified reference is run once per needed (part, device) first; if it fails the
whole run aborts because the baseline is broken. Exit status: 0 all caught, 1 any
MISSED/STALE/invalid fault, 2 baseline broken, 3 some faults could not be verified (ENV).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from platform_tools import stage_helpers  # noqa: E402
from source_files import ignore_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LABS = ROOT / "labs"
SOLUTIONS = ROOT / "solutions"
PARTS = ("A", "B", "C", "R", "all")
DEVICES = ("cpu", "cuda")
REQUIRED = ("id", "todo", "file", "find", "replace", "expect_fail_part", "why")
TODO_MARK = re.compile(r"BEGIN TODO ([ABCR]\d+)")
EVIDENCE = re.compile(r"FAIL|Error|error:|panicked|assert|mismatch|Traceback|expected", re.I)
GPU_LOCK = "/tmp/mlsys-gpu.lock"


def known_labs():
    return sorted(int(p.name[3:]) for p in LABS.glob("lab*") if p.name[3:].isdigit() and p.is_dir())


def make_reference_copy(lab, dest):
    """Same layout as run.py --impl reference: copy lab, overlay solution, stage helpers."""
    shutil.copytree(LABS / f"lab{lab}", dest, ignore=ignore_files)
    shutil.copytree(SOLUTIONS / f"lab{lab}", dest, dirs_exist_ok=True, ignore=ignore_files)
    for extra in ("faults.json",):
        (dest / extra).unlink(missing_ok=True)
    stage_helpers(dest)
    return dest


def load_faults(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    faults = data["faults"] if isinstance(data, dict) else data
    problems = []
    seen = set()
    for k, f in enumerate(faults):
        where = f.get("id", f"#{k}")
        missing = [key for key in REQUIRED if key not in f]
        if missing:
            problems.append(f"{where}: missing keys {missing}")
        if f.get("expect_fail_part") not in PARTS:
            problems.append(f"{where}: expect_fail_part must be one of {PARTS}")
        if f.get("device", "cpu") not in DEVICES:
            problems.append(f"{where}: device must be cpu or cuda")
        if not re.fullmatch(r"[ABCR]\d+", str(f.get("todo", ""))):
            problems.append(f"{where}: todo must look like A1/B2/C3/R1")
        if f.get("find") == f.get("replace"):
            problems.append(f"{where}: find and replace are identical")
        if not isinstance(f.get("count", 1), int) or f.get("count", 1) < 1:
            problems.append(f"{where}: count must be a positive integer")
        if where in seen:
            problems.append(f"{where}: duplicate id")
        seen.add(where)
    return faults, problems


def reference_todos(lab):
    todos = set()
    for path in (SOLUTIONS / f"lab{lab}").rglob("*"):
        if path.is_file() and path.suffix not in (".md", ".json"):
            try:
                todos.update(TODO_MARK.findall(path.read_text(encoding="utf-8", errors="ignore")))
            except OSError:
                pass
    return todos


def run_exercise(work, part, device, timeout, env):
    cmd = [sys.executable, "exercise.py", "--mode", "test", "--device", device, "--part", part]
    if device == "cuda" and shutil.which("flock"):
        cmd = ["flock", GPU_LOCK, *cmd]
    started = time.perf_counter()
    try:
        out = subprocess.run(
            cmd,
            cwd=work,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout,
        )
        rc, text = out.returncode, out.stdout + out.stderr
    except subprocess.TimeoutExpired as error:
        rc = None
        text = (
            (error.stdout or b"").decode("utf-8", "replace")
            if isinstance(error.stdout, bytes)
            else (error.stdout or "")
        )
    return rc, text, time.perf_counter() - started, cmd


def evidence(text):
    """Pick the most informative failure line: explicit FAIL, exception what(), then any error."""
    for pattern in (re.compile(r"\bFAIL\b"), re.compile(r"what\(\)|panicked at|Error:"), EVIDENCE):
        for line in text.splitlines():
            if (
                pattern.search(line)
                and "Traceback" not in line
                and not line.startswith("terminate called")
            ):
                return line.strip()[:120]
    lines = [l for l in text.splitlines() if l.strip()]
    return lines[-1].strip()[:120] if lines else ""


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--lab", type=int, required=True)
    p.add_argument(
        "--device",
        choices=DEVICES,
        help="only faults for this device (default: each fault's own, cpu if unset)",
    )
    p.add_argument("--only", nargs="+", metavar="ID", help="run only these fault ids")
    p.add_argument("--list", action="store_true", help="list faults and TODO coverage, run nothing")
    p.add_argument("--faults", type=Path, help="fault file (default: solutions/labN/faults.json)")
    p.add_argument("--json", type=Path, help="write a machine-readable result here")
    p.add_argument("--timeout", type=float, default=900, help="seconds per run (default 900)")
    p.add_argument("--keep", action="store_true", help="keep the temporary copies for debugging")
    p.add_argument(
        "--require-coverage", action="store_true", help="fail if a reference TODO has no fault"
    )
    p.add_argument("-v", "--verbose", action="store_true", help="print full output of every run")
    a = p.parse_args(argv)

    if a.lab not in known_labs():
        p.error(f"lab{a.lab} not found; known labs: {known_labs()}")
    path = a.faults or SOLUTIONS / f"lab{a.lab}" / "faults.json"
    if not path.is_file():
        print(f"no fault file: {path}  (template: labs/templates/faults.example.json)")
        return 1
    faults, problems = load_faults(path)
    for problem in problems:
        print(f"INVALID {problem}")
    if problems:
        return 1
    for f in faults:
        f.setdefault("count", 1)
        f["device"] = f.get("device") or a.device or "cpu"
    if a.only:
        unknown = set(a.only) - {f["id"] for f in faults}
        if unknown:
            p.error(f"unknown fault ids: {sorted(unknown)}")
        faults = [f for f in faults if f["id"] in a.only]
    if a.device:
        skipped = [f["id"] for f in faults if f["device"] != a.device]
        if skipped:
            print(f"skipping {len(skipped)} faults for other devices: {', '.join(skipped)}")
        faults = [f for f in faults if f["device"] == a.device]

    todos = reference_todos(a.lab)
    covered = {}
    for f in faults:
        covered[f["todo"]] = covered.get(f["todo"], 0) + 1
    uncovered = sorted(todos - set(covered))
    print(f"lab{a.lab}: {len(faults)} faults from {path}")
    print(
        "TODO coverage: "
        + " ".join(f"{t}:{covered.get(t, 0)}" for t in sorted(todos | set(covered)))
    )
    if uncovered:
        print(f"WARN TODOs without a fault: {' '.join(uncovered)}")
    if a.list:
        for f in faults:
            print(
                f"  {f['id']:<24} {f['todo']:<4} part={f['expect_fail_part']:<3} {f['device']:<4} {f['file']}  # {f['why']}"
            )
        return 0

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "MLSYS_LABS_DIR": str(LABS),  # read by staged platform_tools for redirect hints
    }
    temp_root = Path(tempfile.mkdtemp(prefix=f"mlsys-fault-lab{a.lab}-"))
    if (LABS / f"lab{a.lab}" / "Cargo.toml").is_file() and "CARGO_TARGET_DIR" not in env:
        env["CARGO_TARGET_DIR"] = str(temp_root / "cargo-target")  # reuse dependency builds
    results = []
    baseline = {}
    code = 0
    try:
        for part, device in sorted({(f["expect_fail_part"], f["device"]) for f in faults}):
            work = make_reference_copy(
                a.lab, temp_root / f"baseline-{part}-{device}" / f"lab{a.lab}"
            )
            rc, text, seconds, cmd = run_exercise(work, part, device, a.timeout, env)
            status = (
                "OK" if rc == 0 else "ENV" if rc == 3 else "TIMEOUT" if rc is None else "BROKEN"
            )
            baseline[(part, device)] = status
            print(f"BASELINE part={part} device={device}: {status} ({seconds:.1f}s)")
            if a.verbose or status in ("BROKEN", "TIMEOUT"):
                print(text)
            if status in ("BROKEN", "TIMEOUT"):
                print(
                    f"ABORT: unmodified reference fails `{' '.join(cmd)}` (exit {rc}); fix the baseline first"
                )
                code = 2
                return code
            if status == "ENV":
                print(f"  {evidence(text)}")

        for n, f in enumerate(faults):
            row = {k: f[k] for k in ("id", "todo", "file", "expect_fail_part", "device", "why")}
            key = (f["expect_fail_part"], f["device"])
            work = make_reference_copy(a.lab, temp_root / f"fault-{n:03d}" / f"lab{a.lab}")
            target = work / f["file"]
            if not target.is_file():
                row.update(status="STALE", detail=f"file not found: {f['file']}")
                results.append(row)
                print(f"STALE   {f['id']}: {row['detail']}")
                continue
            source = target.read_text(encoding="utf-8")
            hits = source.count(f["find"])
            if hits != f["count"]:
                row.update(
                    status="STALE", detail=f"find matched {hits} times, expected {f['count']}"
                )
                results.append(row)
                print(f"STALE   {f['id']}: {row['detail']}")
                continue
            if baseline.get(key) == "ENV":
                row.update(status="ENV", detail="baseline needs a missing dependency")
                results.append(row)
                print(f"ENV     {f['id']}: {row['detail']}")
                continue
            target.write_text(source.replace(f["find"], f["replace"]), encoding="utf-8", newline="")
            rc, text, seconds, _ = run_exercise(
                work, f["expect_fail_part"], f["device"], a.timeout, env
            )
            status = (
                "MISSED" if rc == 0 else "ENV" if rc == 3 else "TIMEOUT" if rc is None else "CAUGHT"
            )
            row.update(
                status=status, returncode=rc, seconds=round(seconds, 2), detail=evidence(text)
            )
            results.append(row)
            print(f"{status:<7} {f['id']} ({seconds:.1f}s) {row['detail']}")
            if a.verbose:
                print(text)
    finally:
        if a.keep:
            print(f"kept temporary copies under {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)
        if a.json and results:
            a.json.parent.mkdir(parents=True, exist_ok=True)
            a.json.write_text(
                json.dumps(
                    {
                        "lab": a.lab,
                        "faults_file": str(path),
                        "baseline": [
                            {"part": k[0], "device": k[1], "status": v} for k, v in baseline.items()
                        ],
                        "todo_coverage": covered,
                        "todos_without_fault": uncovered,
                        "results": results,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )

    width = max([len(r["id"]) for r in results] + [2])
    print(f"\n{'ID':<{width}}  TODO  PART  DEV   STATUS   EVIDENCE")
    for r in results:
        print(
            f"{r['id']:<{width}}  {r['todo']:<4}  {r['expect_fail_part']:<4}  {r['device']:<4}  {r['status']:<7}  {r.get('detail', '')}"
        )
    counts = {
        s: sum(r["status"] == s for r in results)
        for s in ("CAUGHT", "MISSED", "STALE", "ENV", "TIMEOUT")
    }
    checked = counts["CAUGHT"] + counts["MISSED"]
    rate = f"{100 * counts['CAUGHT'] / checked:.0f}%" if checked else "n/a"
    print("\n" + "  ".join(f"{k}={v}" for k, v in counts.items()) + f"  caught-rate={rate}")
    if counts["MISSED"] or counts["STALE"]:
        return 1
    if a.require_coverage and uncovered:
        print("FAIL: --require-coverage and some TODOs have no fault")
        return 1
    if counts["ENV"]:
        print("NOT VERIFIED: some faults need a missing dependency (exit 3)")
        return 3
    return code


if __name__ == "__main__":
    raise SystemExit(main())
