#!/usr/bin/env python3
"""Extract a CSV/JSONL table from run.py records; list table numbers by default."""

import argparse
import csv
import io
import json
import re
from pathlib import Path


def tables(record):
    found = []
    for command in record["commands"]:
        lines = command.get("stdout", "").splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("{"):
                rows = []
                while i < len(lines) and lines[i].startswith("{"):
                    rows.append(json.loads(lines[i]))
                    i += 1
                found.append(rows)
                continue
            fields = next(csv.reader([line]))
            if len(fields) > 1 and all(re.fullmatch(r"[a-z][a-z0-9_]*", x) for x in fields):
                rows = []
                i += 1
                while i < len(lines):
                    values = next(csv.reader([lines[i]]))
                    if len(values) != len(fields) or all(
                        re.fullmatch(r"[a-z][a-z0-9_]*", x) for x in values
                    ):
                        break
                    rows.append(dict(zip(fields, values, strict=True)))
                    i += 1
                if rows:
                    found.append(rows)
                continue
            i += 1
    return found


def columns(rows):
    """JSONL policies may emit different optional metrics; preserve their union."""
    return list(dict.fromkeys(key for row in rows for key in row))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("record", type=Path)
    p.add_argument("--table", type=int)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    found = tables(json.loads(a.record.read_text(encoding="utf-8")))
    if a.table is None:
        for i, rows in enumerate(found):
            print(i, len(rows), "rows", columns(rows))
        return
    if not 0 <= a.table < len(found):
        p.error("invalid table index")
    rows = found[a.table]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns(rows))
    writer.writeheader()
    writer.writerows(rows)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(buf.getvalue(), encoding="utf-8", newline="\n")
    else:
        print(buf.getvalue(), end="")


if __name__ == "__main__":
    main()
