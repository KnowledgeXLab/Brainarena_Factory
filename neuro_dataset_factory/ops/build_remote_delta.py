#!/usr/bin/env python3
"""Build a self-contained BrainArena remote-task delta against an earlier delivery."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from neuro_dataset_factory.storage import read_json, read_jsonl, write_json, write_jsonl


def _queue_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("repository") or ""),
        str(row.get("record_id") or ""),
        str(row.get("version") or ""),
    )


def _normalized(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _source_journal(candidate: dict[str, Any]) -> str:
    doi = str(candidate.get("doi") or "").casefold()
    article = str(candidate.get("article_url") or "").casefold()
    if "s41467" in doi or "s41467" in article:
        return "nature communications"
    return _normalized(candidate.get("journal") or "")


def _eval_matches(
    eval_manifest: Path,
    candidates_path: Path,
    source_root: Path,
) -> tuple[set[str], list[dict[str, Any]]]:
    payload = read_json(eval_manifest)
    eval_papers = payload.get("papers") or {}
    candidates = {str(row.get("paper_id") or ""): row for row in read_jsonl(candidates_path)}
    exported: dict[str, tuple[str, dict[str, Any]]] = {}
    for paper_dir in (source_root / "benchmark/gt/provenance").iterdir():
        if not paper_dir.is_dir():
            continue
        paths = sorted(paper_dir.glob("*.json"))
        if not paths:
            continue
        provenance = read_json(paths[0])
        internal = str((provenance.get("paper") or {}).get("paper_id") or "")
        exported[paper_dir.name] = (internal, candidates.get(internal, {}))

    matched_exported: set[str] = set()
    screening: list[dict[str, Any]] = []
    for eval_key, raw in eval_papers.items():
        item = raw if isinstance(raw, dict) else {}
        key_text = str(eval_key)
        display = str(item.get("display_name") or key_text)
        year_match = re.search(r"\b(20\d{2})\b", f"{key_text} {display}")
        year = year_match.group(1) if year_match else ""
        key_parts = key_text.split("_")
        year_index = next((i for i, part in enumerate(key_parts) if re.fullmatch(r"20\d{2}", part)), -1)
        surnames = {
            _normalized(part)
            for part in (key_parts[:year_index] if year_index > 0 else key_parts[:1])
            if _normalized(part) not in {"ibl", "international brain laboratory"}
        }
        journal = _normalized(" ".join(key_parts[year_index + 1 :])) if year_index >= 0 else ""
        exact_doi = _normalized(item.get("doi"))
        exact_title = _normalized(item.get("title"))
        hits: list[dict[str, str]] = []
        for exported_id, (internal_id, candidate) in exported.items():
            candidate_doi = _normalized(candidate.get("doi"))
            candidate_title = _normalized(candidate.get("title"))
            reason = ""
            if exact_doi and exact_doi == candidate_doi:
                reason = "exact_doi"
            elif exact_title and exact_title == candidate_title:
                reason = "exact_title"
            else:
                authors = _normalized(candidate.get("authors"))
                publication = str(candidate.get("publication_date") or candidate.get("doi") or "")
                journal_match = bool(journal and journal == _source_journal(candidate))
                author_match = bool(surnames and all(surname in authors for surname in surnames))
                if year and year in publication and journal_match and author_match:
                    reason = "author_year_journal"
            if reason:
                matched_exported.add(exported_id)
                hits.append({
                    "paper_id": exported_id,
                    "internal_paper_id": internal_id,
                    "doi": str(candidate.get("doi") or ""),
                    "title": str(candidate.get("title") or ""),
                    "reason": reason,
                })
        screening.append({
            "eval_key": key_text,
            "display_name": display,
            "matched_source_papers": hits,
        })
    return matched_exported, screening


def _copy_paper_tree(
    source_root: Path,
    output_root: Path,
    relative_parent: str,
    paper_id: str,
    *,
    hardlink: bool,
) -> None:
    source = source_root / relative_parent / paper_id
    if not source.exists():
        return
    target = output_root / relative_parent / paper_id
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, copy_function=os.link if hardlink else shutil.copy2)


def build_delta(
    source_root: Path,
    exclude_root: Path,
    output_root: Path,
    eval_manifest: Path,
    candidates_path: Path,
    *,
    hardlink: bool = False,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {output_root}")
    source_registry = read_json(source_root / "benchmark/task_registry.json")
    exclude_registry = read_json(exclude_root / "benchmark/task_registry.json")
    source_tasks = {str(row["task_id"]): row for row in source_registry.get("tasks") or []}
    exclude_tasks = {str(row["task_id"]): row for row in exclude_registry.get("tasks") or []}
    missing = sorted(set(exclude_tasks) - set(source_tasks))
    if missing:
        raise ValueError(f"excluded delivery is not a subset; missing tasks: {missing[:5]}")

    eval_papers, eval_screening = _eval_matches(eval_manifest, candidates_path, source_root)
    excluded_papers = {str(row.get("paper_id") or "") for row in exclude_tasks.values()} | eval_papers
    retained = [row for row in source_registry["tasks"] if str(row.get("paper_id") or "") not in excluded_papers]
    retained_papers = {str(row.get("paper_id") or "") for row in retained}
    source_papers = {str(row.get("paper_id") or "") for row in source_registry["tasks"]}
    partial = {
        paper_id for paper_id in source_papers
        if any(str(row.get("paper_id") or "") == paper_id for row in retained)
        and any(str(row.get("paper_id") or "") == paper_id for row in exclude_tasks.values())
    }
    if partial:
        raise ValueError(f"partial-paper exclusion is not allowed: {sorted(partial)[:5]}")

    output_root.mkdir(parents=True)
    tree_parents = [
        "benchmark/querys",
        "benchmark/rubrics",
        "benchmark/gt/rubric",
        "benchmark/gt/query_gt_data",
        "benchmark/gt/provenance",
        "benchmark/gt/gt_figure",
        "data",
    ]
    for paper_id in sorted(retained_papers):
        for parent in tree_parents:
            _copy_paper_tree(
                source_root, output_root, parent, paper_id, hardlink=hardlink,
            )

    output_root_resolved = output_root.resolve()
    for task in retained:
        paper_id = str(task.get("paper_id") or "")
        task["data_path"] = str(output_root_resolved / "data" / paper_id)
    registry = {
        **source_registry,
        "project_root": str(output_root_resolved),
        "description": (
            "BrainArena remote neuroscience task delta excluding the previously delivered v2 "
            "and any eval-manifest article matches."
        ),
        "tasks": retained,
    }
    write_json(output_root / "benchmark/task_registry.json", registry)

    source_mapping = read_json(source_root / "benchmark/paper_data_mapping.json")
    mapping = {
        paper_id: str(output_root_resolved / "data" / paper_id)
        for paper_id in sorted(retained_papers)
        if paper_id in source_mapping
    }
    if len(mapping) != len(retained_papers):
        raise ValueError("one or more retained papers are missing from paper_data_mapping")
    write_json(output_root / "benchmark/paper_data_mapping.json", mapping)

    queue: dict[tuple[str, str, str], dict[str, Any]] = {}
    for paper_id in retained_papers:
        locator = read_json(output_root / "data" / paper_id / "REMOTE_DATA_LOCATOR.json")
        for record in locator.get("datasets") or []:
            queue.setdefault(_queue_key(record), record)
    queue_rows = [queue[key] for key in sorted(queue)]
    write_jsonl(output_root / "DOWNLOAD_QUEUE.jsonl", queue_rows)
    excluded_queue = {_queue_key(row) for row in read_jsonl(exclude_root / "DOWNLOAD_QUEUE.jsonl")}
    net_new_queue = [row for row in queue_rows if _queue_key(row) not in excluded_queue]
    write_jsonl(output_root / "DOWNLOAD_QUEUE_NET_NEW_VS_V2.jsonl", net_new_queue)

    figure_rows = [
        row for row in read_jsonl(source_root / "benchmark/gt/gt_figure_manifest.jsonl")
        if str(row.get("paper_id") or "") in retained_papers
    ]
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
        "schema_version": 2, "ok": True, "stats": gt_stats, "issues": [],
    })

    report = {
        "schema_version": 2,
        "ok": True,
        "source_root": str(source_root.resolve()),
        "excluded_delivery": str(exclude_root.resolve()),
        "eval_manifest": str(eval_manifest.resolve()),
        "stats": {
            "source_tasks": len(source_tasks),
            "excluded_v2_tasks": len(exclude_tasks),
            "eval_entries_screened": len(eval_screening),
            "eval_papers_matched": len(eval_papers),
            "retained_tasks": len(retained),
            "retained_papers": len(retained_papers),
            "required_remote_datasets": len(queue_rows),
            "net_new_remote_datasets_vs_v2": len(net_new_queue),
            "gt_figure_files": gt_stats["files_written"],
        },
        "eval_screening": eval_screening,
        "issues": [],
    }
    write_json(output_root / "DELTA_REPORT.json", report)
    write_json(output_root / "MATERIALIZE_REPORT.json", {
        "schema_version": 2,
        "ok": True,
        "stats": {
            "packages_seen": len(retained),
            "tasks_materialized": len(retained),
            "papers": len(retained_papers),
            "unique_remote_datasets": len(queue_rows),
        },
        "issues": [],
    })
    (output_root / "README.md").write_text(
        "# BrainArena Nature Communications remaining-data delta\n\n"
        f"- Retained tasks: {len(retained)}\n"
        f"- Retained papers: {len(retained_papers)}\n"
        f"- Excluded previously delivered v2 tasks: {len(exclude_tasks)}\n"
        f"- Eval entries screened: {len(eval_screening)}; matched source papers: {len(eval_papers)}\n"
        f"- Required remote dataset records: {len(queue_rows)}\n"
        f"- Net-new remote dataset records versus v2: {len(net_new_queue)}\n"
        "- State: awaiting_data_download; enabled=false\n\n"
        "`DOWNLOAD_QUEUE.jsonl` is the complete self-contained queue for retained tasks. "
        "`DOWNLOAD_QUEUE_NET_NEW_VS_V2.jsonl` removes records already present in the delivered v2 queue.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--exclude-root", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="hard-link retained files when source and output share a filesystem",
    )
    args = parser.parse_args()
    report = build_delta(
        args.source_root, args.exclude_root, args.out_root,
        args.eval_manifest, args.candidates, hardlink=args.hardlink,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
