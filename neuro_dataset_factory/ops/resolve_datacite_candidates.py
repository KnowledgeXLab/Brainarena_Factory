#!/usr/bin/env python3
"""Resolve generic dataset DOIs to repository locators through DataCite."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from neuro_dataset_factory.remote_data import normalize_locator
from neuro_dataset_factory.storage import read_jsonl, write_json, write_jsonl


def resolve(doi: str, timeout: int) -> dict[str, Any]:
    url = "https://api.datacite.org/dois/" + urllib.parse.quote(doi, safe="")
    request = urllib.request.Request(
        url, headers={"User-Agent": "BrainArenaDatasetVerifier/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    attributes = ((payload.get("data") or {}).get("attributes") or {})
    types = attributes.get("types") or {}
    return {
        "doi": str(attributes.get("doi") or doi),
        "url": str(attributes.get("url") or ""),
        "publisher": str(attributes.get("publisher") or ""),
        "resource_type": str(types.get("resourceTypeGeneral") or ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    rows = read_jsonl(args.candidates)
    dois: set[str] = set()
    for row in rows:
        for locator in row.get("data_locators") or []:
            if locator.get("repository") == "generic" and locator.get("locator_kind") == "doi":
                doi = str(locator.get("record_id") or "").removeprefix("doi:").strip().casefold()
                if doi.startswith("10."):
                    dois.add(doi)

    resolved: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(resolve, doi, args.timeout): doi for doi in sorted(dois)}
        for future in as_completed(futures):
            doi = futures[future]
            try:
                resolved[doi] = future.result()
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                failures[doi] = f"{type(exc).__name__}: {exc}"

    output: list[dict[str, Any]] = []
    resolved_locators = 0
    dataset_dois = 0
    for row in rows:
        locators: dict[tuple[str, str, str], dict[str, Any]] = {}
        for original in row.get("data_locators") or []:
            if original.get("repository") != "generic" or original.get("locator_kind") != "doi":
                continue
            doi = str(original.get("record_id") or "").removeprefix("doi:").strip().casefold()
            metadata = resolved.get(doi) or {}
            if metadata.get("resource_type") != "Dataset":
                continue
            dataset_dois += 1
            target = str(metadata.get("url") or "")
            locator = normalize_locator(target) if target else {}
            if locator.get("repository") in {None, "", "generic"}:
                continue
            locator.update({
                "datacite_doi": str(metadata.get("doi") or doi),
                "datacite_publisher": str(metadata.get("publisher") or ""),
            })
            key = (
                str(locator.get("repository") or ""),
                str(locator.get("record_id") or ""),
                str(locator.get("version") or ""),
            )
            locators[key] = locator
        if locators:
            resolved_locators += len(locators)
            output.append({
                **row,
                "data_locators": list(locators.values()),
                "status": "needs_remote_verification",
            })

    write_jsonl(args.out, output)
    report = {
        "schema_version": 2,
        "ok": True,
        "stats": {
            "candidate_papers": len(rows),
            "unique_generic_dois": len(dois),
            "datacite_resolved": len(resolved),
            "datacite_failures": len(failures),
            "dataset_doi_occurrences": dataset_dois,
            "papers_with_supported_repository": len(output),
            "resolved_repository_locators": resolved_locators,
        },
        "failures": failures,
        "issues": [],
    }
    write_json(args.out.with_suffix(".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
