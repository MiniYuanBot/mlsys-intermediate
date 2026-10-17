#!/usr/bin/env python3
"""Plot exported CSV. Error bars are sample SD across rows, not confidence intervals."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("csv", type=Path)
p.add_argument("--x", required=True)
p.add_argument("--y", required=True)
p.add_argument("--group", required=True)
p.add_argument("--filter", action="append", default=[])
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
filters = dict(pair.split("=", 1) for pair in a.filter)
data = defaultdict(lambda: defaultdict(list))
for row in csv.DictReader(a.csv.open()):
    if all(row[k] == v for k, v in filters.items()):
        data[row[a.group]][float(row[a.x])].append(float(row[a.y]))
if not data:
    p.error("no data after filtering")
fig, ax = plt.subplots(figsize=(7, 4))
for label, points in sorted(data.items()):
    xs = sorted(points)
    ys = [statistics.mean(points[x]) for x in xs]
    sd = [statistics.stdev(points[x]) if len(points[x]) > 1 else 0 for x in xs]
    ax.errorbar(xs, ys, yerr=sd, marker="o", capsize=3, label=label)
ax.set(
    xlabel=a.x,
    ylabel=a.y,
    title="Mean ± sample SD (not CI)" + (" | " + ", ".join(a.filter) if a.filter else ""),
)
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
a.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out, dpi=160)
print(a.out)
