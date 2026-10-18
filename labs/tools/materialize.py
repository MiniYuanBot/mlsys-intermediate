#!/usr/bin/env python3
"""Create a new standalone lab copy; refuse to overwrite an existing destination."""

import argparse
import shutil
from pathlib import Path

import lab_meta
from platform_tools import stage_helpers
from source_files import ignore_files

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--lab", type=int, choices=lab_meta.LAB_NUMBERS, required=True)
p.add_argument("--to", type=Path, required=True)
p.add_argument("--impl", choices=["student", "reference"], default="reference")
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
if not lab_meta.get(a.lab).exists():
    p.error(lab_meta.missing_lab_message(a.lab))
if a.to.exists():
    p.error("destination exists; choose a new path")
shutil.copytree(
    root / "labs" / f"lab{a.lab}",
    a.to,
    ignore=ignore_files,
)
if a.impl == "reference":
    shutil.copytree(
        root / "solutions" / f"lab{a.lab}", a.to, dirs_exist_ok=True, ignore=ignore_files
    )
stage_helpers(a.to)
print(a.to.resolve())
