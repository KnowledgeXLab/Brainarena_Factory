#!/usr/bin/env python3
"""Small deterministic JSONL helpers for resumable remote-task batches."""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from neuro_dataset_factory.remote_data import normalize_locator


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def select_missing(args: argparse.Namespace) -> None:
    completed = {
        str(row[args.key])
        for path in args.completed
        for row in read_jsonl(path)
        if args.key in row
    }
    rows = (row for row in read_jsonl(args.source) if str(row.get(args.key, "")) not in completed)
    if args.max_rows is not None:
        rows = islice(rows, args.max_rows)
    count = write_jsonl(args.out, rows)
    print(json.dumps({"completed_keys": len(completed), "selected_rows": count, "out": str(args.out)}))


def select_mapped(args: argparse.Namespace) -> None:
    with args.mapping.open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict):
        raise ValueError(f"{args.mapping}: expected a JSON object")
    if args.mapping_field:
        mapping = mapping.get(args.mapping_field) or {}
        if not isinstance(mapping, dict):
            raise ValueError(
                f"{args.mapping}: field {args.mapping_field!r} must be a JSON object"
            )
    selected_keys = {str(value) for value in mapping.keys()}
    count = write_jsonl(
        args.out,
        (row for row in read_jsonl(args.source) if str(row.get(args.key, "")) in selected_keys),
    )
    print(json.dumps({"mapped_keys": len(selected_keys), "selected_rows": count, "out": str(args.out)}))


def merge_unique(args: argparse.Namespace) -> None:
    seen: set[str] = set()

    def rows() -> Iterable[dict[str, Any]]:
        for path in args.inputs:
            for row in read_jsonl(path):
                value = str(row.get(args.key, ""))
                if not value:
                    raise ValueError(f"{path}: missing required key {args.key!r}")
                if value in seen:
                    raise ValueError(f"duplicate {args.key} {value!r} across inputs")
                seen.add(value)
                yield row

    count = write_jsonl(args.out, rows())
    print(json.dumps({"merged_rows": count, "unique_keys": len(seen), "out": str(args.out)}))


def merge_paper_records(args: argparse.Namespace) -> None:
    """Merge repeated paper rows while retaining the best result per locator."""
    priority = {
        "rejected": 1,
        "verification_deferred_network": 2,
        "needs_manual_review": 3,
        "verified_downloadable": 4,
    }
    papers: dict[str, dict[str, Any]] = {}
    locators: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    input_rows = 0
    for path in args.inputs:
        for row in read_jsonl(path):
            input_rows += 1
            paper_id = str(row.get("paper_id") or "")
            if not paper_id:
                raise ValueError(f"{path}: missing required key 'paper_id'")
            current = papers.get(paper_id)
            if current is None:
                current = {**row, "data_locators": []}
                papers[paper_id] = current
                locators[paper_id] = {}
            elif priority.get(str(row.get("status") or ""), 0) > priority.get(
                str(current.get("status") or ""), 0
            ):
                current["status"] = row.get("status")
            for locator in row.get("data_locators") or []:
                verification = locator.get("verification") or {}
                key = (
                    str(locator.get("repository") or ""),
                    str(verification.get("record_id") or locator.get("record_id") or ""),
                    str(locator.get("original_url") or ""),
                )
                prior = locators[paper_id].get(key)
                prior_status = str(((prior or {}).get("verification") or {}).get("status") or "")
                status = str(verification.get("status") or "")
                if prior is None or priority.get(status, 0) > priority.get(prior_status, 0):
                    locators[paper_id][key] = locator
    for paper_id, row in papers.items():
        row["data_locators"] = list(locators[paper_id].values())
        row["verified_locator_count"] = sum(
            ((locator.get("verification") or {}).get("status") == "verified_downloadable")
            for locator in row["data_locators"]
        )
        if row["verified_locator_count"]:
            row["status"] = "verified_downloadable"
    count = write_jsonl(args.out, (papers[key] for key in sorted(papers)))
    print(json.dumps({"input_rows": input_rows, "merged_papers": count, "out": str(args.out)}))


def renormalize_locators(args: argparse.Namespace) -> None:
    def rows() -> Iterable[dict[str, Any]]:
        for row in read_jsonl(args.source):
            normalized = []
            for locator in row.get("data_locators") or []:
                url = str(locator.get("original_url") or "")
                normalized.append({**locator, **normalize_locator(url)})
            yield {**row, "data_locators": normalized}

    count = write_jsonl(args.out, rows())
    print(json.dumps({"renormalized_papers": count, "out": str(args.out)}))


def select_verified_size(args: argparse.Namespace) -> None:
    excluded: set[str] = set()
    if args.exclude_registry:
        with args.exclude_registry.open(encoding="utf-8") as handle:
            registry = json.load(handle)
        excluded = {
            str(row.get("paper_id") or "")
            for row in registry.get("tasks") or []
            if row.get("paper_id")
        }

    selected_papers = 0
    selected_locators = 0

    def rows() -> Iterable[dict[str, Any]]:
        nonlocal selected_papers, selected_locators
        for row in read_jsonl(args.source):
            paper_id = str(row.get("paper_id") or "")
            if paper_id in excluded:
                continue
            verified = [
                locator for locator in row.get("data_locators") or []
                if ((locator.get("verification") or {}).get("status") == "verified_downloadable")
            ]
            sizes = [
                int((locator.get("verification") or {}).get("total_bytes"))
                for locator in verified
                if (locator.get("verification") or {}).get("total_bytes") not in (None, "")
            ]
            if not sizes or max(sizes) <= args.min_bytes:
                continue
            selected_papers += 1
            selected_locators += len(verified)
            yield {**row, "status": "verified_downloadable", "data_locators": verified}

    count = write_jsonl(args.out, rows())
    print(json.dumps({
        "selected_papers": count,
        "selected_verified_locators": selected_locators,
        "excluded_papers": len(excluded),
        "min_bytes": args.min_bytes,
        "out": str(args.out),
    }))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    missing = subparsers.add_parser("select-missing")
    missing.add_argument("--source", type=Path, required=True)
    missing.add_argument("--completed", type=Path, nargs="+", required=True)
    missing.add_argument("--key", default="paper_id")
    missing.add_argument("--max-rows", type=int)
    missing.add_argument("--out", type=Path, required=True)
    missing.set_defaults(func=select_missing)

    mapped = subparsers.add_parser("select-mapped")
    mapped.add_argument("--source", type=Path, required=True)
    mapped.add_argument("--mapping", type=Path, required=True)
    mapped.add_argument("--mapping-field", default="")
    mapped.add_argument("--key", default="paper_id")
    mapped.add_argument("--out", type=Path, required=True)
    mapped.set_defaults(func=select_mapped)

    merge = subparsers.add_parser("merge-unique")
    merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge.add_argument("--key", default="task_id")
    merge.add_argument("--out", type=Path, required=True)
    merge.set_defaults(func=merge_unique)

    paper_merge = subparsers.add_parser("merge-paper-records")
    paper_merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    paper_merge.add_argument("--out", type=Path, required=True)
    paper_merge.set_defaults(func=merge_paper_records)

    renormalize = subparsers.add_parser("renormalize-locators")
    renormalize.add_argument("--source", type=Path, required=True)
    renormalize.add_argument("--out", type=Path, required=True)
    renormalize.set_defaults(func=renormalize_locators)

    sized = subparsers.add_parser("select-verified-size")
    sized.add_argument("--source", type=Path, required=True)
    sized.add_argument("--exclude-registry", type=Path)
    sized.add_argument("--min-bytes", type=int, required=True)
    sized.add_argument("--out", type=Path, required=True)
    sized.set_defaults(func=select_verified_size)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
