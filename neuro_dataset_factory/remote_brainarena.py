"""Export provisional remote tasks in the established BrainArena data layout."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from neuro_dataset_factory.contracts import SCHEMA_VERSION, ValidationReport
from neuro_dataset_factory.storage import read_json, write_json, write_jsonl


REGISTRY_FIELDS = [
    "task_id", "paper_id", "paper_display_name", "figure", "query_path",
    "rubric_path", "data_id", "data_path", "self_contained", "enabled",
    "needs_data_note", "n_data_files", "data_supports_task", "code_leak",
    "trajectory_verdict", "source_verdict",
]


def _paper_directory(title: str, internal_id: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9]", "", internal_id)[-12:] or "unknown"
    # Paper titles are evaluator-only provenance.  Solver-visible paths and task
    # IDs must remain opaque so a trajectory cannot search for the source article.
    return f"paper--{suffix}"


def _relative_required_path(path: str, dataset_ids: list[str]) -> str:
    normalized = str(path).replace("\\", "/")
    for dataset_id in dataset_ids:
        prefix = f"data/{dataset_id}/"
        if normalized.startswith(prefix):
            return f"{dataset_id}/{normalized[len(prefix):]}"
    return normalized.removeprefix("data/")


def _query_markdown(info: dict[str, Any], locator: dict[str, Any]) -> str:
    dataset_ids = [str(row.get("dataset_id") or "") for row in locator.get("datasets", [])]
    query = str(info.get("query") or "").strip()
    for dataset_id in dataset_ids:
        query = query.replace(f"data/{dataset_id}/", f"{dataset_id}/")
    required = [_relative_required_path(path, dataset_ids) for path in info.get("required_files") or []]
    lines = [query, "", "**Expected outputs.**", str(info.get("deliverable") or "Analysis code, figures, machine-readable summary, and report."), ""]
    lines.extend([
        "**Data acquisition state.** The dataset payload is not bundled yet. The data directory contains "
        "`REMOTE_DATA_LOCATOR.json` with verified record versions, per-file download URLs, sizes, and checksums. "
        "Materialize and verify those files before executing this task; synthetic replacements are not allowed.",
        "",
        "**Required files after materialization** (paths relative to this paper's data directory):",
    ])
    lines.extend(f"- `{path}`" for path in required)
    return "\n".join(lines).strip() + "\n"


def materialize_remote_brainarena(
    provisional_root: Path,
    output_root: Path,
) -> ValidationReport:
    report = ValidationReport()
    if output_root.exists():
        report.error("output_exists", f"refusing to overwrite: {output_root}")
        return report
    package_dirs = sorted(path for path in provisional_root.iterdir() if (path / "task/task_info.json").is_file())
    tasks: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    paper_ids: dict[str, str] = {}
    locator_by_paper: dict[str, dict[str, Any]] = {}
    queue: dict[tuple[str, str, str], dict[str, Any]] = {}

    for package in package_dirs:
        status = read_json(package / "package_status.json")
        if status.get("status") != "awaiting_data_download":
            report.warn("non_remote_package_skipped", package.name)
            continue
        task_root = package / "task"
        info = read_json(task_root / "task_info.json")
        checklist = read_json(task_root / "target_study/checklist.json")
        ground_truth = read_json(task_root / "target_study/ground_truth.json")
        provenance = read_json(task_root / "target_study/provenance.json")
        locator = read_json(task_root / "data/REMOTE_DATA_LOCATOR.json")
        internal_paper = str(info.get("paper_id") or "")
        title = str(info.get("paper_display_name") or internal_paper)
        paper_id = paper_ids.setdefault(internal_paper, _paper_directory(title, internal_paper))
        public_paper_name = paper_id
        tag = str(info.get("task_tag") or "Main")
        task_id = f"{paper_id}__{tag}"
        query_rel = Path("benchmark/querys") / paper_id / f"{tag}.md"
        rubric_rel = Path("benchmark/rubrics") / paper_id / f"{tag}.json"
        gt_rubric_rel = Path("benchmark/gt/rubric") / paper_id / f"{tag}.json"
        gt_data_rel = Path("benchmark/gt/query_gt_data") / paper_id / f"{tag}.json"
        provenance_rel = Path("benchmark/gt/provenance") / paper_id / f"{tag}.json"
        data_dir = (output_root / "data" / paper_id).resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = str(data_dir)
        previous = mapping.setdefault(paper_id, data_path)
        if previous != data_path:
            report.error("paper_data_path_conflict", paper_id)
            continue
        for record in locator.get("datasets", []):
            dataset_id = str(record.get("dataset_id") or "")
            record["target_relative_path"] = f"data/{paper_id}/{dataset_id}"
            for item in record.get("files", []):
                item["materialized_relative_path"] = f"{dataset_id}/{item.get('path', 'download')}"
            key = (str(record.get("repository") or ""), str(record.get("record_id") or ""), str(record.get("version") or ""))
            queue.setdefault(key, record)
        paper_locator = {
            "schema_version": SCHEMA_VERSION,
            "status": "awaiting_data_download",
            "paper_id": paper_id,
            "paper_display_name": public_paper_name,
            "data_path": data_path,
            "datasets": locator.get("datasets", []),
        }
        existing_locator = locator_by_paper.get(paper_id)
        if existing_locator is None:
            locator_by_paper[paper_id] = paper_locator
            write_json(data_dir / "REMOTE_DATA_LOCATOR.json", paper_locator)
        else:
            existing_records = {
                str(record.get("dataset_id") or ""): record
                for record in existing_locator.get("datasets", [])
            }
            for record in paper_locator.get("datasets", []):
                dataset_id = str(record.get("dataset_id") or "")
                previous_record = existing_records.get(dataset_id)
                if previous_record is None:
                    existing_locator["datasets"].append(record)
                    existing_records[dataset_id] = record
                elif previous_record != record:
                    report.error("dataset_locator_conflict", f"{paper_id}: {dataset_id}")
                    continue
            existing_locator["datasets"] = sorted(
                existing_locator["datasets"], key=lambda record: str(record.get("dataset_id") or ""),
            )
            paper_locator = existing_locator
            write_json(data_dir / "REMOTE_DATA_LOCATOR.json", existing_locator)
        (output_root / query_rel).parent.mkdir(parents=True, exist_ok=True)
        (output_root / query_rel).write_text(_query_markdown(info, paper_locator), encoding="utf-8")
        write_json(output_root / rubric_rel, checklist)
        write_json(output_root / gt_rubric_rel, checklist)
        write_json(output_root / gt_data_rel, ground_truth)
        write_json(output_root / provenance_rel, provenance)
        expected_count = sum(int(record.get("file_count") or len(record.get("files", []))) for record in paper_locator["datasets"])
        tasks.append({
            "task_id": task_id,
            "paper_id": paper_id,
            "paper_display_name": public_paper_name,
            "figure": tag,
            "query_path": query_rel.as_posix(),
            "rubric_path": rubric_rel.as_posix(),
            "data_id": paper_id,
            "data_path": data_path,
            "self_contained": False,
            "enabled": False,
            "needs_data_note": f"awaiting_data_download; see REMOTE_DATA_LOCATOR.json; expected_files={expected_count}",
            "n_data_files": 0,
            "data_supports_task": "partial",
            "code_leak": "unknown",
            "trajectory_verdict": "awaiting_data_download",
            "source_verdict": "AWAITING_DATA_DOWNLOAD",
        })

    expected_by_paper = {
        paper_id: sum(
            int(record.get("file_count") or len(record.get("files", [])))
            for record in locator.get("datasets", [])
        )
        for paper_id, locator in locator_by_paper.items()
    }
    for task in tasks:
        task["needs_data_note"] = (
            "awaiting_data_download; see REMOTE_DATA_LOCATOR.json; "
            f"expected_files={expected_by_paper.get(str(task.get('paper_id') or ''), 0)}"
        )

    registry = {
        "schema_version": SCHEMA_VERSION,
        "project_root": str(output_root.resolve()),
        "description": "BrainArena-format neuroscience end-to-end tasks with verified remote data locators awaiting materialization.",
        "fields": REGISTRY_FIELDS,
        "tasks": tasks,
    }
    write_json(output_root / "benchmark/task_registry.json", registry)
    write_json(output_root / "benchmark/paper_data_mapping.json", mapping)
    write_jsonl(output_root / "DOWNLOAD_QUEUE.jsonl", sorted(queue.values(), key=lambda row: (str(row.get("repository")), str(row.get("record_id")))))
    (output_root / "README.md").write_text(
        "# BrainArena-format remote neuroscience tasks\n\n"
        f"- Tasks: {len(tasks)}\n"
        f"- Papers: {len(mapping)}\n"
        f"- Unique remote dataset records: {len(queue)}\n"
        "- Current state: `awaiting_data_download`\n\n"
        "This directory follows the existing `data/brainarena_trainingdata_rcb_neuro` layout. "
        "Each `data/<paper_id>/REMOTE_DATA_LOCATOR.json` contains verified, versioned download records. "
        "Download payload files into the specified dataset subdirectory, verify sizes/checksums, then update "
        "`enabled`, `n_data_files`, `data_supports_task`, and `source_verdict` after reference validation.\n",
        encoding="utf-8",
    )
    report.stats.update({
        "packages_seen": len(package_dirs),
        "tasks_materialized": len(tasks),
        "papers": len(mapping),
        "unique_remote_datasets": len(queue),
    })
    write_json(output_root / "MATERIALIZE_REPORT.json", report.to_dict())
    return report


def audit_remote_brainarena(root: Path) -> ValidationReport:
    report = ValidationReport()
    try:
        registry = read_json(root / "benchmark/task_registry.json")
        mapping = read_json(root / "benchmark/paper_data_mapping.json")
    except (OSError, json.JSONDecodeError) as exc:
        report.error("invalid_registry", str(exc))
        return report
    if registry.get("fields") != REGISTRY_FIELDS:
        report.error("registry_fields_mismatch", str(registry.get("fields")))
    tasks = registry.get("tasks") or []
    seen: set[str] = set()
    locations: set[tuple[str, str]] = set()
    paper_counts: dict[str, int] = {}
    datasets: set[tuple[str, str, str]] = set()
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        paper_id = str(task.get("paper_id") or "")
        tag = str(task.get("figure") or "")
        if not task_id or task_id in seen:
            report.error("duplicate_task_id", task_id)
        seen.add(task_id)
        if not paper_id or not tag or (paper_id, tag) in locations:
            report.error("duplicate_task_location", f"{paper_id}::{tag}")
        locations.add((paper_id, tag))
        paper_counts[paper_id] = paper_counts.get(paper_id, 0) + 1
        for field in ("query_path", "rubric_path"):
            path = root / str(task.get(field) or "")
            if not path.is_file():
                report.error("missing_task_file", f"{task_id}: {field}")
        rubric_path = root / str(task.get("rubric_path") or "")
        if rubric_path.is_file():
            try:
                rubric = read_json(rubric_path)
                items = rubric["rubrics"]["score_100"]["rubric"]["Scoring items"]
                if sum(int(item.get("point", 0)) for item in items) != 100:
                    report.error("rubric_points_not_100", task_id)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                report.error("invalid_rubric", f"{task_id}: {exc}")
        data_path = Path(str(task.get("data_path") or "")).absolute()
        expected = root.resolve() / "data" / paper_id
        if data_path != expected or not data_path.is_dir():
            report.error("data_path_mismatch", task_id)
            continue
        locator_path = data_path / "REMOTE_DATA_LOCATOR.json"
        if not locator_path.is_file():
            report.error("missing_remote_locator", paper_id)
            continue
        payload_files = [path for path in data_path.rglob("*") if path.is_file() and path != locator_path]
        if payload_files:
            report.error("unexpected_payload", paper_id)
        locator = read_json(locator_path)
        if locator.get("status") != "awaiting_data_download":
            report.error("bad_locator_status", paper_id)
        for record in locator.get("datasets", []):
            if not record.get("record_id") or not record.get("version") or not record.get("files"):
                report.error("incomplete_remote_record", paper_id)
            for item in record.get("files", []):
                if not item.get("download_url") or not item.get("materialized_relative_path"):
                    report.error("incomplete_remote_file", paper_id)
            datasets.add((str(record.get("repository")), str(record.get("record_id")), str(record.get("version"))))
        if task.get("enabled") is not False or int(task.get("n_data_files", -1)) != 0:
            report.error("premature_task_enablement", task_id)
        if task.get("source_verdict") != "AWAITING_DATA_DOWNLOAD":
            report.error("source_status_mismatch", task_id)
        if mapping.get(paper_id) != str(expected):
            report.error("paper_mapping_mismatch", paper_id)
    report.stats.update({
        "tasks": len(tasks),
        "papers": len(paper_counts),
        "tasks_per_paper": sorted(set(paper_counts.values())),
        "unique_remote_datasets": len(datasets),
        "enabled": sum(bool(task.get("enabled")) for task in tasks),
    })
    write_json(root / "FORMAT_AUDIT_REPORT.json", report.to_dict())
    return report
