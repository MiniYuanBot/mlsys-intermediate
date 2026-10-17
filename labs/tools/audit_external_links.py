#!/usr/bin/env python3
"""Audit HTTP reachability of Markdown links; this is not a scientific review."""

import concurrent.futures
import datetime
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
urls = {}
for path in ROOT.rglob("*.md"):
    if any(
        x.startswith(".venv") or x in ("build", "target", ".git", "node_modules")
        for x in path.parts
    ):
        continue
    for url in re.findall(r"\]\((https?://[^\s)]+)\)", path.read_text()):
        urls.setdefault(url, []).append(str(path.relative_to(ROOT)))


def check(item):
    url, files = item
    result = {"url": url, "referenced_by": sorted(set(files))}
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (MLSys curriculum reference audit)"}
        )
        with urllib.request.urlopen(req, timeout=18) as response:
            response.read(1024)
            result.update(http_status=response.status, final_url=response.url, status="reachable")
    except urllib.error.HTTPError as e:
        result.update(
            http_status=e.code,
            status="restricted_or_rate_limited" if e.code in [401, 403, 429] else "http_error",
        )
    except Exception as e:
        result.update(status="network_error", error=str(e))
    return result


with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    results = list(pool.map(check, sorted(urls.items())))
output = ROOT / "course/readings/link-audit.json"
output.write_text(
    json.dumps(
        {
            "checked_client_date": datetime.datetime.now(ZoneInfo("Asia/Shanghai"))
            .date()
            .isoformat(),
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scope": "HTTP reachability only; some pages were also independently read via web search. HTTP errors do not prove missing content.",
            "links": results,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)
print("URLs", len(results), "reachable", sum(r["status"] == "reachable" for r in results))
for r in results:
    if r["status"] != "reachable":
        print(r["status"], r.get("http_status"), r["url"])
