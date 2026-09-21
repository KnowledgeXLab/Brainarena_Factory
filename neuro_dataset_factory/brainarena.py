"""Materialize validated canonical packages into portable BrainArena layout."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from neuro_dataset_factory.contracts import END_TO_END_STAGES, SCHEMA_VERSION, ValidationReport
from neuro_dataset_factory.manifest import sha256_file
from neuro_dataset_factory.packages import validate_task_package
from neuro_dataset_factory.storage import read_json, write_json


def _merge_task_data(
    source: Path,
    destination: Path,
    report: ValidationReport,
    task_id: str,
) -> bool:
    """Merge one task's selected files into a shared immutable dataset asset."""
    destination.mkdir(parents=True, exist_ok=True)
    ok = True
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        if path.is_symlink():
            report.error("symlink", f"{task_id}: {rel}")
            ok = False
            continue
        if not path.is_file():
            continue
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not target.is_file() or sha256_file(target) != sha256_file(path):
                report.error("shared_data_conflict", f"{task_id}: {rel}")
                ok = False
            continue
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
    return ok


def materialize_brainarena(packages_root: Path, output_root: Path) -> ValidationReport:
    report = ValidationReport()
    if output_root.exists():
        report.error("output_exists", f"refusing to overwrite: {output_root}")
        return report
    tasks: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    seen_locations: set[tuple[str, str]] = set()
    dataset_paths: dict[str, str] = {}
    package_dirs = sorted(path for path in packages_root.iterdir() if (path / "task").is_dir())
    for package_dir in package_dirs:
        task_root = package_dir / "task"
        validation = validate_task_package(task_root)
        if not validation.ok:
            report.warn("invalid_package_skipped", package_dir.name)
            continue
        info = read_json(task_root / "task_info.json")
        checklist = read_json(task_root / "target_study" / "checklist.json")
        ground_truth = read_json(task_root / "target_study" / "ground_truth.json")
        provenance = read_json(task_root / "target_study" / "provenance.json")
        task_id = str(info["task_id"])
        paper_id = str(info.get("paper_id") or task_id)
        task_tag = str(info.get("task_tag") or "Main")
        location = (paper_id, task_tag)
        if location in seen_locations:
            report.error("duplicate_paper_task_tag", f"paper_id={paper_id} task_tag={task_tag}")
            continue
        seen_locations.add(location)
        figure = task_tag
        query_rel = Path("benchmark") / "querys" / paper_id / f"{figure}.md"
        rubric_rel = Path("benchmark") / "rubrics" / paper_id / f"{figure}.json"
        gt_rubric_rel = Path("benchmark") / "gt" / "rubric" / paper_id / f"{figure}.json"
        gt_data_rel = Path("benchmark") / "gt" / "query_gt_data" / paper_id / f"{figure}.json"
        provenance_rel = Path("benchmark") / "gt" / "provenance" / paper_id / f"{figure}.json"
        data_id = str(info.get("dataset_id") or task_id)
        data_rel = Path("data") / data_id
        dataset_info_rel = Path("benchmark") / "datasets" / data_id / "dataset_info.json"
        dataset_manifest_rel = Path("benchmark") / "datasets" / data_id / "manifest.jsonl"
        portable_data_path = data_rel.as_posix()
        previous_mapping = mapping.get(paper_id)
        if previous_mapping is not None and previous_mapping != portable_data_path:
            report.error(
                "paper_uses_multiple_datasets",
                f"{paper_id}: {previous_mapping} != {portable_data_path}",
            )
            continue
        for path in (query_rel, rubric_rel, gt_rubric_rel, gt_data_rel, provenance_rel):
            (output_root / path).parent.mkdir(parents=True, exist_ok=True)
        (output_root / query_rel).write_text(str(info["query"]).strip() + "\n", encoding="utf-8")
        write_json(output_root / rubric_rel, checklist)
        write_json(output_root / gt_rubric_rel, checklist)
        write_json(output_root / gt_data_rel, ground_truth)
        write_json(output_root / provenance_rel, provenance)
        shared_source = packages_root / "_datasets" / data_id / "data"
        if shared_source.is_dir():
            if data_id not in dataset_paths:
                if not _merge_task_data(shared_source, output_root / data_rel, report, task_id):
                    continue
                source_info = packages_root / "_datasets" / data_id / "dataset_info.json"
                source_manifest = packages_root / "_datasets" / data_id / "manifest.jsonl"
                if not source_info.is_file() or not source_manifest.is_file():
                    report.error("missing_shared_asset_metadata", data_id)
                    continue
                (output_root / dataset_info_rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_info, output_root / dataset_info_rel)
                shutil.copy2(source_manifest, output_root / dataset_manifest_rel)
        elif not _merge_task_data(task_root / "data", output_root / data_rel, report, task_id):
            # Backward-compatible fallback for schema-v1 package roots.
            continue
        mapping[paper_id] = portable_data_path
        dataset_paths[data_id] = portable_data_path
        paper_link = provenance.get("paper_link") or {}
        leakage_risk = str(paper_link.get("leakage_risk") or "unknown")
        tasks.append({
            "task_id": task_id,
            "paper_id": paper_id,
            "paper_display_name": info.get("paper_display_name", paper_id),
            "figure": figure,
            "task_tag": task_tag,
            "task_format": info.get("task_format", "end_to_end"),
            "workflow_stages": info.get("workflow_stages", []),
            "query_path": query_rel.as_posix(),
            "rubric_path": rubric_rel.as_posix(),
            "provenance_path": provenance_rel.as_posix(),
            "data_id": data_id,
            "data_path": portable_data_path,
            "dataset_info_path": dataset_info_rel.as_posix(),
            "dataset_manifest_path": dataset_manifest_rel.as_posix(),
            "self_contained": False,
            "enabled": True,
            "needs_data_note": "",
            "n_data_files": int(
                provenance.get("dataset", {}).get("file_count")
                or len(info.get("provided_data") or [])
            ),
            "data_supports_task": paper_link.get("data_support", "unknown"),
            "code_leak": leakage_risk,
            "trajectory_verdict": provenance.get("trajectory_verdict", "unreviewed"),
            "source_verdict": "VERIFIED",
            "dataset_id": info.get("dataset_id", ""),
        })
    registry = {
        "schema_version": SCHEMA_VERSION,
        "project_root": ".",
        "description": "Dataset-centric neuroscience end-to-end training tasks; portable relative data paths.",
        "fields": [
            "task_id", "paper_id", "paper_display_name", "figure", "task_tag",
            "task_format", "workflow_stages",
            "query_path", "rubric_path", "provenance_path", "data_id", "data_path",
            "dataset_info_path", "dataset_manifest_path", "self_contained",
            "enabled", "needs_data_note", "n_data_files", "data_supports_task",
            "code_leak", "trajectory_verdict", "source_verdict", "dataset_id",
        ],
        "tasks": tasks,
    }
    write_json(output_root / "benchmark" / "task_registry.json", registry)
    write_json(output_root / "benchmark" / "paper_data_mapping.json", mapping)
    report.stats.update({
        "packages_seen": len(package_dirs),
        "tasks_materialized": len(tasks),
        "shared_dataset_assets": len(dataset_paths),
    })
    write_json(output_root / "MATERIALIZE_REPORT.json", report.to_dict())
    return report


def audit_brainarena(root: Path) -> ValidationReport:
    report = ValidationReport()
    registry_path = root / "benchmark" / "task_registry.json"
    if not registry_path.is_file():
        report.error("missing_registry", str(registry_path))
        return report
    try:
        registry = read_json(registry_path)
    except (OSError, json.JSONDecodeError) as exc:
        report.error("invalid_registry", str(exc))
        return report
    tasks = registry.get("tasks") or []
    seen: set[str] = set()
    seen_locations: set[tuple[str, str]] = set()
    dataset_paths: dict[str, str] = {}
    audited_datasets: set[str] = set()
    split_assignments: dict[tuple[str, str], str] = {}
    for row in tasks:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in seen:
            report.error("duplicate_or_empty_task_id", task_id)
        seen.add(task_id)
        location = (str(row.get("paper_id") or ""), str(row.get("task_tag") or row.get("figure") or ""))
        if not all(location) or location in seen_locations:
            report.error("duplicate_or_empty_paper_task_tag", f"{location[0]}::{location[1]}")
        seen_locations.add(location)
        if row.get("task_format") != "end_to_end":
            report.error("not_end_to_end", task_id)
        stages = row.get("workflow_stages") or []
        missing_stages = [stage for stage in END_TO_END_STAGES if stage not in stages]
        if missing_stages:
            report.error("missing_workflow_stages", f"{task_id}: {', '.join(missing_stages)}")
        split = str(row.get("split") or "")
        if split:
            if split not in {"train", "validation", "test"}:
                report.error("invalid_split", f"{task_id}: {split}")
            for field in ("dataset_id", "paper_id"):
                identity = (field, str(row.get(field) or ""))
                previous_split = split_assignments.setdefault(identity, split)
                if previous_split != split:
                    report.error(
                        "split_leakage",
                        f"{field}={identity[1]} appears in {previous_split} and {split}",
                    )
        for field in ("query_path", "rubric_path", "provenance_path", "dataset_info_path", "dataset_manifest_path"):
            raw = str(row.get(field) or "")
            if Path(raw).is_absolute():
                report.error("absolute_path", f"{task_id}: {field}={raw}")
            elif not (root / raw).is_file():
                report.error("missing_path", f"{task_id}: {field}={raw}")
        data_path = str(row.get("data_path") or "")
        if data_path:
            if Path(data_path).is_absolute():
                report.error("absolute_data_path", f"{task_id}: {data_path}")
            elif not (root / data_path).is_dir():
                report.error("missing_data_path", f"{task_id}: {data_path}")
        dataset_id = str(row.get("dataset_id") or "")
        if dataset_id:
            previous = dataset_paths.setdefault(dataset_id, data_path)
            if previous != data_path:
                report.error("dataset_path_not_shared", f"{dataset_id}: {previous} != {data_path}")
            info_path = root / str(row.get("dataset_info_path") or "")
            if dataset_id not in audited_datasets and info_path.is_file() and data_path and (root / data_path).is_dir():
                audited_datasets.add(dataset_id)
                try:
                    dataset_info = read_json(info_path)
                    files = [path for path in (root / data_path).rglob("*") if path.is_file()]
                    actual_bytes = sum(path.stat().st_size for path in files)
                    if len(files) != int(dataset_info.get("file_count", -1)):
                        report.error("dataset_file_count_mismatch", dataset_id)
                    if actual_bytes != int(dataset_info.get("total_bytes", -1)):
                        report.error("dataset_bytes_mismatch", dataset_id)
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    report.error("invalid_dataset_info", f"{dataset_id}: {exc}")
        rubric_path = root / str(row.get("rubric_path") or "")
        if rubric_path.is_file():
            try:
                rubric = read_json(rubric_path)
                items = rubric["rubrics"]["score_100"]["rubric"]["Scoring items"]
                total = sum(int(item.get("point", 0)) for item in items)
                if total != 100:
                    report.error("points_not_100", f"{task_id}: {total}")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                report.error("invalid_rubric", f"{task_id}: {exc}")
    mapping_path = root / "benchmark" / "paper_data_mapping.json"
    if not mapping_path.is_file():
        report.error("missing_mapping", str(mapping_path))
    else:
        try:
            mapping = read_json(mapping_path)
            for paper_id, data_path in mapping.items():
                if Path(str(data_path)).is_absolute():
                    report.error("absolute_mapping_path", f"{paper_id}: {data_path}")
                elif data_path and not (root / str(data_path)).is_dir():
                    report.error("missing_mapping_path", f"{paper_id}: {data_path}")
            for row in tasks:
                if mapping.get(str(row.get("paper_id") or "")) != row.get("data_path"):
                    report.error("task_mapping_mismatch", str(row.get("task_id") or ""))
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            report.error("invalid_mapping", str(exc))
    report.stats.update({
        "tasks": len(tasks),
        "enabled": sum(bool(row.get("enabled", True)) for row in tasks),
        "shared_dataset_assets": len(dataset_paths),
    })
    return report
