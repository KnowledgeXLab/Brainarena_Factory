#!/usr/bin/env python3
"""Build a delivery containing only papers whose remote payloads are complete."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TREE_PARENTS = (
    "benchmark/querys",
    "benchmark/rubrics",
    "benchmark/gt/rubric",
    "benchmark/gt/query_gt_data",
    "benchmark/gt/provenance",
    "benchmark/gt/gt_figure",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def completed_records(
    queue: list[dict[str, Any]],
    state_root: Path,
    payload_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    complete: dict[str, dict[str, Any]] = {}
    reports: dict[str, dict[str, Any]] = {}
    for record in queue:
        dataset_id = str(record.get("dataset_id") or "")
        report_path = state_root / "datasets" / f"{dataset_id}.json"
        payload_path = payload_root / dataset_id
        if not report_path.is_file() or not payload_path.is_dir():
            continue
        try:
            report = read_json(report_path)
        except (OSError, json.JSONDecodeError):
            continue
        expected_files = len(record.get("files") or [])
        expected_bytes = int(record.get("total_bytes") or 0)
        if not (
            report.get("complete") is True
            and int(report.get("expected_files") or -1) == expected_files
            and int(report.get("verified_files") or -1) == expected_files
            and int(report.get("expected_bytes") or -1) == expected_bytes
            and int(report.get("verified_bytes") or -1) == expected_bytes
        ):
            continue
        complete[dataset_id] = record
        reports[dataset_id] = report
    return complete, reports


def copy_paper_tree(source_root: Path, output_root: Path, parent: str, paper_id: str) -> None:
    source = source_root / parent / paper_id
    if not source.exists():
        return
    target = output_root / parent / paper_id
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, copy_function=shutil.copy2)


def build(
    source_root: Path,
    output_root: Path,
    state_root: Path,
    payload_root: Path,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {output_root}")
    source_registry = read_json(source_root / "benchmark/task_registry.json")
    source_tasks = source_registry.get("tasks") or []
    queue = read_jsonl(source_root / "DOWNLOAD_QUEUE.jsonl")
    queue_by_id = {str(row["dataset_id"]): row for row in queue}
    complete, reports = completed_records(queue, state_root, payload_root)

    paper_records: dict[str, dict[str, dict[str, Any]]] = {}
    for data_dir in (source_root / "data").iterdir():
        if not data_dir.is_dir() or not data_dir.name.startswith("paper--"):
            continue
        locator_path = data_dir / "REMOTE_DATA_LOCATOR.json"
        if not locator_path.is_file():
            continue
        locator = read_json(locator_path)
        records = {
            str(row.get("dataset_id") or ""): row
            for row in locator.get("datasets") or []
            if row.get("dataset_id")
        }
        paper_records[data_dir.name] = records

    ready_papers = {
        paper_id for paper_id, records in paper_records.items()
        if records and set(records) <= set(complete)
    }
    retained = [copy.deepcopy(row) for row in source_tasks if str(row.get("paper_id") or "") in ready_papers]
    output_root.mkdir(parents=True)
    output_resolved = output_root.resolve()

    for paper_id in sorted(ready_papers):
        for parent in TREE_PARENTS:
            copy_paper_tree(source_root, output_root, parent, paper_id)
        source_locator = read_json(source_root / "data" / paper_id / "REMOTE_DATA_LOCATOR.json")
        records = []
        data_dir = output_root / "data" / paper_id
        data_dir.mkdir(parents=True, exist_ok=True)
        for raw in source_locator.get("datasets") or []:
            record = copy.deepcopy(raw)
            dataset_id = str(record.get("dataset_id") or "")
            if dataset_id not in complete:
                raise ValueError(f"ready paper unexpectedly references incomplete dataset: {paper_id}: {dataset_id}")
            record["acquisition_status"] = "downloaded_and_verified"
            record["payload_downloaded"] = True
            record["payload_hash_verified"] = True
            record["target_relative_path"] = f"data/{paper_id}/{dataset_id}"
            target = data_dir / dataset_id
            target.symlink_to(payload_root / dataset_id, target_is_directory=True)
            records.append(record)
        locator = {
            **source_locator,
            "status": "downloaded_and_verified",
            "note": "All listed payloads are present and verified; dataset directories are OSS-local symbolic links.",
            "paper_id": paper_id,
            "paper_display_name": paper_id,
            "data_path": str(data_dir.resolve()),
            "datasets": records,
        }
        write_json(data_dir / "REMOTE_DATA_LOCATOR.json", locator)

    for task in retained:
        paper_id = str(task.get("paper_id") or "")
        records = paper_records[paper_id]
        task["data_path"] = str(output_resolved / "data" / paper_id)
        task["n_data_files"] = sum(
            int(queue_by_id[dataset_id].get("file_count") or len(queue_by_id[dataset_id].get("files") or []))
            for dataset_id in records
        )
        task["needs_data_note"] = "payload_downloaded_and_verified; reference_pending"
        task["data_supports_task"] = "full"
        task["trajectory_verdict"] = "ready_for_trajectory_collection"
        task["source_verdict"] = "PAYLOAD_VERIFIED_REFERENCE_PENDING"
        # Final canonical enablement remains gated on reference execution.
        task["enabled"] = False

    registry = {
        **source_registry,
        "project_root": str(output_resolved),
        "description": "BrainArena neuroscience tasks with complete verified OSS payloads; reference execution pending.",
        "tasks": retained,
    }
    write_json(output_root / "benchmark/task_registry.json", registry)
    write_json(output_root / "benchmark/paper_data_mapping.json", {
        paper_id: str(output_resolved / "data" / paper_id)
        for paper_id in sorted(ready_papers)
    })

    required_ids = sorted({dataset_id for paper_id in ready_papers for dataset_id in paper_records[paper_id]})
    delivery_queue = []
    completion_rows = []
    for dataset_id in required_ids:
        record = copy.deepcopy(queue_by_id[dataset_id])
        record["acquisition_status"] = "downloaded_and_verified"
        record["payload_downloaded"] = True
        record["payload_hash_verified"] = True
        delivery_queue.append(record)
        completion_rows.append({
            "dataset_id": dataset_id,
            "repository": record.get("repository"),
            "record_id": record.get("record_id"),
            "version": record.get("version"),
            "expected_files": reports[dataset_id].get("expected_files"),
            "verified_files": reports[dataset_id].get("verified_files"),
            "expected_bytes": reports[dataset_id].get("expected_bytes"),
            "verified_bytes": reports[dataset_id].get("verified_bytes"),
            "payload_path": str(payload_root / dataset_id),
            "complete": True,
        })
    write_jsonl(output_root / "DOWNLOAD_QUEUE.jsonl", delivery_queue)
    write_jsonl(output_root / "DATASET_COMPLETION_MANIFEST.jsonl", completion_rows)

    figure_manifest = source_root / "benchmark/gt/gt_figure_manifest.jsonl"
    figure_rows = [
        row for row in read_jsonl(figure_manifest)
        if str(row.get("paper_id") or "") in ready_papers
    ] if figure_manifest.is_file() else []
    write_jsonl(output_root / "benchmark/gt/gt_figure_manifest.jsonl", figure_rows)

    task_counts = Counter(str(row.get("paper_id") or "") for row in retained)
    issues: list[str] = []
    if set(task_counts.values()) != {4}:
        issues.append(f"unexpected tasks-per-paper values: {sorted(set(task_counts.values()))}")
    for task in retained:
        for field in ("query_path", "rubric_path"):
            if not (output_root / str(task.get(field) or "")).is_file():
                issues.append(f"missing {field}: {task.get('task_id')}")
    for paper_id in ready_papers:
        for dataset_id in paper_records[paper_id]:
            link = output_root / "data" / paper_id / dataset_id
            if not link.is_symlink() or link.resolve() != (payload_root / dataset_id).resolve():
                issues.append(f"bad payload link: {paper_id}/{dataset_id}")

    verified_bytes = sum(int(reports[dataset_id].get("verified_bytes") or 0) for dataset_id in required_ids)
    snapshot = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": 1,
        "ok": not issues,
        "snapshot_at": snapshot,
        "source_root": str(source_root),
        "payload_root": str(payload_root),
        "stats": {
            "source_tasks": len(source_tasks),
            "source_papers": len(paper_records),
            "complete_datasets_in_full_queue": len(complete),
            "delivery_tasks": len(retained),
            "delivery_papers": len(ready_papers),
            "delivery_unique_datasets": len(required_ids),
            "delivery_verified_files": sum(int(reports[d].get("verified_files") or 0) for d in required_ids),
            "delivery_verified_bytes": verified_bytes,
            "delivery_verified_gib": round(verified_bytes / 2**30, 3),
            "tasks_per_paper": sorted(set(task_counts.values())),
        },
        "issues": issues,
    }
    write_json(output_root / "COMPLETED_DELIVERY_REPORT.json", report)
    write_json(output_root / "DELIVERY_AUDIT_REPORT.json", {
        "schema_version": 1,
        "ok": not issues,
        "stats": report["stats"],
        "issues": issues,
    })
    (output_root / "README.md").write_text(
        "# BrainArena completed-payload delivery\n\n"
        f"- Snapshot: `{snapshot}`\n"
        f"- Tasks: {len(retained)}\n"
        f"- Papers: {len(ready_papers)}\n"
        f"- Unique verified remote datasets: {len(required_ids)}\n"
        f"- Verified payload: {verified_bytes / 2**30:.3f} GiB\n"
        "- State: payload downloaded and verified; reference execution pending.\n"
        "- Payload directories are symbolic links into the shared OSS dataset store.\n\n"
        "Only papers whose complete dependency set passed file-count, byte-count, and checksum/size validation are included.\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    args = parser.parse_args()
    report = build(
        args.source_root.resolve(),
        args.output_root,
        args.state_root.resolve(),
        args.payload_root.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
