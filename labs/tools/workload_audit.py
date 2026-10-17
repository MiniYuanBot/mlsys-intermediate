#!/usr/bin/env python3
"""Audit paired student/reference TODO regions; count implementation work, not files.

ELOC is a reproducible proxy: nonblank physical lines within solution TODO regions,
excluding comment-only lines, preprocessor directives, and brace-only lines. It is
not a required student line count and does not measure difficulty or originality.
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BEGIN = re.compile(r"(?:#|//)\s*BEGIN TODO\s+([ABC]\d+)\b")
END = re.compile(r"(?:#|//)\s*END TODO\s+([ABC]\d+)\b")
EXTENSIONS = {".py", ".hpp", ".h", ".cpp", ".cu", ".cuh", ".rs", ".c"}


def effective(lines):
    return sum(
        bool(line.strip())
        and not line.lstrip().startswith(("#", "//", "/*", "*"))
        and bool(line.strip().strip("{}[]();,"))
        for line in lines
    )


def regions(path):
    found, active, body = {}, None, []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        begin, end = BEGIN.search(line), END.search(line)
        if begin:
            if active or begin[1] in found:
                raise ValueError(f"nested/duplicate marker {path}:{number}")
            active, body = begin[1], []
        elif end:
            if active != end[1]:
                raise ValueError(f"mismatched end {path}:{number}")
            found[active] = {"lines": body, "end_line": number}
            active = None
        elif active:
            body.append(line)
    if active:
        raise ValueError(f"unclosed {path}: {active}")
    return found


def audit():
    output, errors = [], []
    for lab in range(1, 9):
        rows, identifiers = [], set()
        student_root = ROOT / "labs" / f"lab{lab}"
        for solution in sorted((ROOT / "solutions" / f"lab{lab}").rglob("*")):
            if not solution.is_file() or solution.suffix not in EXTENSIONS:
                continue
            if any(p in {"target", "__pycache__", "build"} for p in solution.parts):
                continue
            relative = solution.relative_to(ROOT / "solutions" / f"lab{lab}")
            student = student_root / relative
            reference_regions = regions(solution)
            if not reference_regions:
                continue
            if not student.exists():
                errors.append(f"missing student file: {student}")
                continue
            student_regions = regions(student)
            if reference_regions.keys() != student_regions.keys():
                errors.append(f"region parity: lab{lab}/{relative}")
            for identifier, region in reference_regions.items():
                if identifier in identifiers:
                    errors.append(f"duplicate lab ID: lab{lab}/{identifier}")
                identifiers.add(identifier)
                body = "\n".join(region["lines"])
                if re.search(
                    r"NotImplementedError|todo!\(|unimplemented!\(|Complete [ABC]\d+", body
                ):
                    errors.append(f"reference stub: lab{lab}/{identifier}")
                rows.append(
                    {
                        "id": identifier,
                        "file": str(relative),
                        "reference_eloc": effective(region["lines"]),
                        "student_eloc": effective(
                            student_regions.get(identifier, {}).get("lines", [])
                        ),
                    }
                )
        stages = {
            stage: {
                "todos": sum(row["id"].startswith(stage) for row in rows),
                "reference_eloc": sum(
                    row["reference_eloc"] for row in rows if row["id"].startswith(stage)
                ),
            }
            for stage in "ABC"
        }
        if any(value["todos"] == 0 for value in stages.values()):
            errors.append(f"empty stage: lab{lab}")
        output.append(
            {
                "lab": lab,
                "stages": stages,
                "total_todos": len(rows),
                "total_reference_eloc": sum(row["reference_eloc"] for row in rows),
                "regions": rows,
            }
        )
    return {"method": __doc__, "labs": output, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = audit()
    for row in data["labs"]:
        print(
            f"lab{row['lab']}: {row['total_todos']} TODO, {row['total_reference_eloc']} ELOC; {row['stages']}"
        )
    for error in data["errors"]:
        print("ERROR:", error)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return bool(data["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
