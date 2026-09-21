#!/usr/bin/env python3
"""Anonymize paper identities in an already materialized remote task package.

Hidden provenance keeps the source title for evaluator audit.  Solver-visible
directory names, task IDs, registry fields, and data locators use stable opaque
paper IDs.  No model call or task regeneration is required.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


PAPER_DIR_PARENTS = (
    "benchmark/querys",
    "benchmark/rubrics",
    "benchmark/gt/rubric",
    "benchmark/gt/query_gt_data",
    "benchmark/gt/provenance",
    "benchmark/gt/gt_figure",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def opaque_paper_id(old_id: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9]", "", old_id)[-12:] or "unknown"
    return f"paper--{suffix}"


def replace_strings(value: Any, pattern: re.Pattern[str], replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            pattern.sub(lambda match: replacements[match.group(0)], key) if isinstance(key, str) else key:
            replace_strings(item, pattern, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [replace_strings(item, pattern, replacements) for item in value]
    if isinstance(value, str):
        return pattern.sub(lambda match: replacements[match.group(0)], value)
    return value


def rewrite_json_file(path: Path, pattern: re.Pattern[str], replacements: dict[str, str]) -> bool:
    raw_text = path.read_text(encoding="utf-8")
    if pattern.search(raw_text) is None:
        return False
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
        text = "".join(json.dumps(replace_strings(row, pattern, replacements), ensure_ascii=False) + "\n" for row in rows)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    else:
        write_json(path, replace_strings(json.loads(raw_text), pattern, replacements))
    return True


def migrate(root: Path, *, preserve_old_data_paths: bool, jobs: int = 16) -> dict[str, Any]:
    registry_path = root / "benchmark/task_registry.json"
    registry = read_json(registry_path)
    tasks = registry.get("tasks") or []
    mapping_path = root / "benchmark/paper_data_mapping.json"
    existing_mapping = read_json(mapping_path) if mapping_path.is_file() else {}
    data_root = root / "data"
    legacy_data_ids = {
        path.name for path in data_root.iterdir()
        if path.is_dir() and not path.name.startswith("paper--")
    } if data_root.is_dir() else set()
    candidate_ids = {
        str(task.get("paper_id") or "") for task in tasks if task.get("paper_id")
    } | {str(key) for key in existing_mapping} | legacy_data_ids
    paper_map = {
        old: opaque_paper_id(old)
        for old in sorted(candidate_ids)
        if old != opaque_paper_id(old)
    }
    old_ids = sorted(paper_map)
    target_ids = set(paper_map.values()) | {
        str(task.get("paper_id") or "") for task in tasks
        if str(task.get("paper_id") or "").startswith("paper--")
    }
    if len(set(paper_map.values())) != len(paper_map):
        raise ValueError("opaque paper ID collision")

    replacements = paper_map
    pattern = (
        re.compile("|".join(re.escape(old) for old in sorted(replacements, key=len, reverse=True)))
        if replacements else None
    )

    # Rename solver-visible metadata trees first.  This is idempotent.
    rename_operations: list[tuple[Path, Path]] = []
    for relative_parent in PAPER_DIR_PARENTS:
        parent = root / relative_parent
        if not parent.is_dir():
            continue
        for old, new in paper_map.items():
            source, target = parent / old, parent / new
            if source.exists() and not target.exists():
                rename_operations.append((source, target))

    def rename_directory(operation: tuple[Path, Path]) -> int:
        source, target = operation
        if not source.exists() or target.exists():
            return 0
        source.rename(target)
        return 1

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        renamed_dirs = sum(pool.map(rename_directory, rename_operations))

    def migrate_data_directory(pair: tuple[str, str]) -> int:
        old, new = pair
        source, target = data_root / old, data_root / new
        if target.exists() or target.is_symlink() or not source.exists():
            if not (preserve_old_data_paths and target.is_dir() and source.is_dir()):
                return 0
        if preserve_old_data_paths:
            target.mkdir(parents=True, exist_ok=True)
            for child in source.iterdir():
                destination = target / child.name
                if child.name == "REMOTE_DATA_LOCATOR.json":
                    shutil.copy2(child, destination)
                elif child.is_symlink() and not destination.exists() and not destination.is_symlink():
                    destination.symlink_to(os.readlink(child), target_is_directory=True)
        else:
            source.rename(target)
        return 1

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        renamed_dirs += sum(pool.map(migrate_data_directory, paper_map.items()))

    # Rewrite machine-readable metadata after directory migration.  Markdown
    # query prose is deliberately unchanged; the tasks themselves are preserved.
    rewritten = 0
    json_paths = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl"}
    ]
    if pattern is not None:
        def rewrite(path: Path) -> int:
            return int(rewrite_json_file(path, pattern, replacements))

        # Large acquisition manifests can be hundreds of MiB each.  Keep this
        # pool smaller than the metadata-operation pool to bound peak memory.
        with ThreadPoolExecutor(max_workers=max(1, min(jobs, 4))) as pool:
            rewritten = sum(pool.map(rewrite, json_paths))

    registry = read_json(registry_path)
    for task in registry.get("tasks") or []:
        paper_id = str(task.get("paper_id") or "")
        task["paper_display_name"] = paper_id
        task["data_id"] = paper_id
        task["data_path"] = str(root / "data" / paper_id)
    write_json(registry_path, registry)

    mapping = read_json(mapping_path)
    mapping = {
        paper_id: str(root / "data" / paper_id)
        for paper_id in mapping
    }
    write_json(mapping_path, mapping)

    locator_count = 0
    for paper_id in target_ids:
        locator_path = root / "data" / paper_id / "REMOTE_DATA_LOCATOR.json"
        if not locator_path.is_file():
            continue
        locator = read_json(locator_path)
        locator["paper_id"] = paper_id
        locator["paper_display_name"] = paper_id
        locator["data_path"] = str(root / "data" / paper_id)
        write_json(locator_path, locator)
        locator_count += 1

    # Hidden provenance remains the authoritative title-bearing audit surface.
    titles: dict[str, str] = {}
    for path in (root / "benchmark/gt/provenance").glob("*/*.json"):
        item = read_json(path)
        paper = item.get("paper") or {}
        paper_id = path.parent.name
        if paper.get("title"):
            titles[paper_id] = str(paper["title"])
    exact_query_title_hits = 0
    for task in registry.get("tasks") or []:
        title = titles.get(str(task.get("paper_id") or ""), "")
        query_path = root / str(task.get("query_path") or "")
        if title and query_path.is_file() and title.casefold() in query_path.read_text(encoding="utf-8").casefold():
            exact_query_title_hits += 1

    report = {
        "schema_version": 1,
        "status": "complete" if exact_query_title_hits == 0 else "review_required",
        "tasks": len(tasks),
        "papers_anonymized": len(paper_map),
        "renamed_or_aliased_directories": renamed_dirs,
        "rewritten_json_jsonl_files": rewritten,
        "locators_anonymized": locator_count,
        "exact_paper_title_hits_in_query_markdown": exact_query_title_hits,
        "hidden_provenance_titles_retained": len(titles),
        "preserved_old_data_paths_for_active_downloader": preserve_old_data_paths,
    }
    write_json(root / "TITLE_REDACTION_REPORT.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--preserve-old-data-paths",
        action="store_true",
        help="create anonymous data aliases while an existing downloader still targets old paths",
    )
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    report = migrate(
        args.root.resolve(),
        preserve_old_data_paths=args.preserve_old_data_paths,
        jobs=max(1, args.jobs),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
