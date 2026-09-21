#!/usr/bin/env python3
"""Merge one full BrainArena remote release and any number of disjoint deltas."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from neuro_dataset_factory.storage import read_json, read_jsonl, write_json, write_jsonl


TREE_PARENTS = (
    "benchmark/querys",
    "benchmark/rubrics",
    "benchmark/gt/rubric",
    "benchmark/gt/query_gt_data",
    "benchmark/gt/provenance",
    "benchmark/gt/gt_figure",
    "data",
)


def _queue_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("repository") or ""),
        str(row.get("record_id") or ""),
        str(row.get("version") or ""),
    )


def merge_roots(
    roots: list[Path],
    output_root: Path,
    *,
    hardlink: bool = False,
    replace_overlap: bool = False,
) -> dict[str, Any]:
    if not roots:
        raise ValueError("at least one source root is required")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {output_root}")

    # Later recovery batches may intentionally replace an older representation
    # of the same paper (for example, to add a formerly size-capped dataset).
    # Resolve ownership before copying so replacement is whole-paper atomic.
    paper_owner: dict[str, Path] = {}
    overlaps: set[str] = set()
    for root in roots:
        registry = read_json(root / "benchmark/task_registry.json")
        for task in registry.get("tasks") or []:
            paper_id = str(task.get("paper_id") or "")
            if paper_id in paper_owner and paper_owner[paper_id] != root:
                overlaps.add(paper_id)
            paper_owner[paper_id] = root
    if overlaps and not replace_overlap:
        raise ValueError(f"papers occur in multiple roots: {sorted(overlaps)[:5]}")

    output_root.mkdir(parents=True)
    tasks: dict[str, dict[str, Any]] = {}
    paper_sources: dict[str, Path] = {}
    queue: dict[tuple[str, str, str], dict[str, Any]] = {}
    figures: dict[tuple[str, str], dict[str, Any]] = {}
    schema_version = 2

    for root in roots:
        registry = read_json(root / "benchmark/task_registry.json")
        schema_version = int(registry.get("schema_version") or schema_version)
        source_papers: set[str] = set()
        for raw_task in registry.get("tasks") or []:
            task = dict(raw_task)
            task_id = str(task.get("task_id") or "")
            paper_id = str(task.get("paper_id") or "")
            if not task_id or not paper_id:
                raise ValueError(f"invalid task identity in {root}")
            if paper_owner.get(paper_id) != root:
                continue
            if task_id in tasks:
                raise ValueError(f"duplicate task_id across roots: {task_id}")
            prior = paper_sources.get(paper_id)
            if prior is not None and prior != root:
                raise ValueError(f"paper occurs in multiple roots: {paper_id}")
            paper_sources[paper_id] = root
            source_papers.add(paper_id)
            task["data_path"] = str((output_root.resolve() / "data" / paper_id))
            tasks[task_id] = task

        for paper_id in sorted(source_papers):
            for parent in TREE_PARENTS:
                source = root / parent / paper_id
                if not source.exists():
                    continue
                target = output_root / parent / paper_id
                if target.exists():
                    raise ValueError(f"merge target collision: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, target, copy_function=os.link if hardlink else shutil.copy2)

        figure_path = root / "benchmark/gt/gt_figure_manifest.jsonl"
        if figure_path.is_file():
            for row in read_jsonl(figure_path):
                if paper_owner.get(str(row.get("paper_id") or "")) != root:
                    continue
                key = (str(row.get("paper_id") or ""), str(row.get("task_tag") or ""))
                if key in figures:
                    raise ValueError(f"duplicate figure mapping across roots: {key}")
                figures[key] = row

    # Rebuild the queue only from the retained paper trees.  This prevents
    # superseded locator records from leaking in from an older source root.
    for paper_id in sorted(paper_sources):
        locator = read_json(output_root / "data" / paper_id / "REMOTE_DATA_LOCATOR.json")
        for row in locator.get("datasets") or []:
            key = _queue_key(row)
            current = queue.get(key)
            if current is None:
                queue[key] = dict(row)
                continue
            current_papers = {str(value) for value in current.get("source_papers") or []}
            current_papers.update(str(value) for value in row.get("source_papers") or [])
            current["source_papers"] = sorted(current_papers)

    output_resolved = output_root.resolve()
    task_rows = [tasks[key] for key in sorted(tasks)]
    registry = {
        "schema_version": schema_version,
        "project_root": str(output_resolved),
        "description": (
            "BrainArena Nature Communications remote-data tasks with verified object-level "
            "manifests; payloads remain awaiting_data_download."
        ),
        "fields": list(task_rows[0].keys()) if task_rows else [],
        "tasks": task_rows,
    }
    write_json(output_root / "benchmark/task_registry.json", registry)
    write_json(output_root / "benchmark/paper_data_mapping.json", {
        paper_id: str(output_resolved / "data" / paper_id)
        for paper_id in sorted(paper_sources)
    })
    write_jsonl(output_root / "DOWNLOAD_QUEUE.jsonl", [queue[key] for key in sorted(queue)])
    figure_rows = [figures[key] for key in sorted(figures)]
    write_jsonl(output_root / "benchmark/gt/gt_figure_manifest.jsonl", figure_rows)

    gt_stats = {
        "tasks": len(figure_rows),
        "matched": sum(row.get("match_status") == "matched" for row in figure_rows),
        "model_candidate_matches": sum(row.get("model_match_status") == "matched" for row in figure_rows),
        "review_required": sum(row.get("match_status") == "review_required" for row in figure_rows),
        "files_written": sum(bool(row.get("gt_figure_path")) for row in figure_rows),
        "no_exact_match": sum(row.get("match_status") == "no_exact_match" for row in figure_rows),
    }
    write_json(output_root / "GT_FIGURE_REPORT.json", {
        "schema_version": schema_version, "ok": True, "stats": gt_stats, "issues": [],
    })
    report = {
        "schema_version": schema_version,
        "ok": True,
        "sources": [str(root.resolve()) for root in roots],
        "stats": {
            "source_roots": len(roots),
            "tasks_materialized": len(task_rows),
            "papers": len(paper_sources),
            "unique_remote_datasets": len(queue),
            "gt_figure_files": gt_stats["files_written"],
            "enabled": sum(bool(row.get("enabled")) for row in task_rows),
            "papers_replaced_by_later_roots": len(overlaps),
        },
        "issues": [],
    }
    write_json(output_root / "MATERIALIZE_REPORT.json", report)
    (output_root / "README.md").write_text(
        "# BrainArena Nature Communications remote all-processable release\n\n"
        f"- Tasks: {len(task_rows)}\n"
        f"- Papers: {len(paper_sources)}\n"
        f"- Verified remote dataset records: {len(queue)}\n"
        f"- High-confidence ground-truth figures: {gt_stats['files_written']}\n"
        "- State: awaiting_data_download; enabled=false\n"
        "- Raw dataset payloads are not included.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="hard-link files when all roots and the output are on one filesystem",
    )
    parser.add_argument(
        "--replace-overlap",
        action="store_true",
        help="when a paper occurs in multiple roots, retain the later root's whole-paper version",
    )
    args = parser.parse_args()
    print(json.dumps(
        merge_roots(
            args.root, args.out_root,
            hardlink=args.hardlink,
            replace_overlap=args.replace_overlap,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
