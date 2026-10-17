#!/usr/bin/env python3
"""Audit HTTP reachability of Markdown links; this is not a scientific review."""

import argparse
import concurrent.futures
import datetime
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]


def collect_urls():
    from source_files import project_files

    urls = {}
    for path in project_files(ROOT):
        if path.suffix != ".md":
            continue
        for url in re.findall(r"\]\((https?://[^\s)]+)\)", path.read_text()):
            urls.setdefault(url, []).append(str(path.relative_to(ROOT)))
    return urls


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "labs/results/external-links.json")
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(check, sorted(collect_urls().items())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "checked_client_date": datetime.datetime.now(ZoneInfo("Asia/Shanghai"))
                .date()
                .isoformat(),
                "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "scope": "HTTP reachability only; HTTP errors do not prove missing content.",
                "links": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print("URLs", len(results), "reachable", sum(r["status"] == "reachable" for r in results))
    for result in results:
        if result["status"] != "reachable":
            print(result["status"], result.get("http_status"), result["url"])
    print(f"RECORD: {args.output}")


if __name__ == "__main__":
    main()
