#!/usr/bin/env python3
"""Fill missing paper-level figure mappings conservatively as no exact match."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from neuro_dataset_factory.storage import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--mappings", type=Path, required=True)
    args = parser.parse_args()

    tasks: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(args.candidates):
        tasks[str(row.get("paper_id") or "")].append(row)
    figures = {str(row.get("paper_id") or ""): row for row in read_jsonl(args.figures)}
    existing_rows = read_jsonl(args.mappings) if args.mappings.is_file() else []
    by_paper = {str(row.get("paper_id") or ""): row for row in existing_rows}
    fallback_papers = 0
    fallback_tasks = 0
    for paper_id in sorted(tasks):
        if paper_id in by_paper:
            continue
        paper = figures.get(paper_id) or {}
        mappings = []
        for task in sorted(tasks[paper_id], key=lambda row: str(row.get("task_tag") or "")):
            mappings.append({
                "task_tag": str(task.get("task_tag") or ""),
                "match_status": "no_exact_match",
                "image_object": "",
                "figure_label": "",
                "caption": "",
                "confidence": "low",
                "reason": "figure mapping request timed out; no exact source figure was asserted",
            })
        by_paper[paper_id] = {
            "schema_version": 2,
            "paper_id": paper_id,
            "doi": paper.get("doi", ""),
            "title": paper.get("title", ""),
            "source_prefix": paper.get("source_prefix", ""),
            "mappings": mappings,
            "generation": {
                "model": "conservative-timeout-fallback",
                "prompt_version": "no-exact-match-v1",
                "cache_key": "",
                "cache_hit": False,
            },
        }
        fallback_papers += 1
        fallback_tasks += len(mappings)
    output = [by_paper[key] for key in sorted(by_paper)]
    write_jsonl(args.mappings, output)
    report = {
        "schema_version": 2,
        "ok": True,
        "stats": {
            "papers": len(output),
            "tasks": sum(len(row.get("mappings") or []) for row in output),
            "existing_papers": len(existing_rows),
            "fallback_papers": fallback_papers,
            "fallback_tasks": fallback_tasks,
        },
        "issues": [],
    }
    write_json(args.mappings.with_suffix(".completion_report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
