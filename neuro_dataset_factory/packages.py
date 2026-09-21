"""Create and validate canonical dataset-grounded task packages."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from neuro_dataset_factory.contracts import (
    END_TO_END_STAGES,
    SCHEMA_VERSION,
    TASK_FORMATS,
    ValidationReport,
    require_text,
    string_list,
)
from neuro_dataset_factory.manifest import sha256_file
from neuro_dataset_factory.storage import read_json, read_jsonl, write_json, write_jsonl


def _safe_relative(raw: str, field_name: str) -> str:
    pure = PurePosixPath(str(raw).replace("\\", "/"))
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{field_name} contains an unsafe path: {raw!r}")
    return pure.as_posix()


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _ensure_shared_dataset_asset(
    output_root: Path,
    dataset: dict[str, Any],
    source_root: Path,
    manifest: list[dict[str, Any]],
    report: ValidationReport,
) -> Path | None:
    """Materialize the complete registered dataset once for all of its tasks."""
    dataset_id = str(dataset["dataset_id"])
    asset_root = output_root / "_datasets" / dataset_id
    data_root = asset_root / "data"
    info_path = asset_root / "dataset_info.json"
    if info_path.is_file() and data_root.is_dir():
        info = read_json(info_path)
        if info.get("tree_sha256") == dataset.get("tree_sha256"):
            return data_root
        report.error("shared_asset_version_conflict", dataset_id)
        return None
    if asset_root.exists():
        report.error("incomplete_shared_asset", str(asset_root))
        return None
    data_root.mkdir(parents=True, exist_ok=False)
    try:
        for item in manifest:
            rel = str(item["path"])
            source = source_root / rel
            if not source.is_file() or sha256_file(source) != item["sha256"]:
                raise ValueError(f"source changed: {rel}")
            _link_or_copy(source, data_root / rel)
        write_jsonl(asset_root / "manifest.jsonl", manifest)
        write_json(asset_root / "dataset_info.json", {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "name": dataset.get("name", dataset_id),
            "version": dataset.get("version", ""),
            "license": dataset.get("license", "unknown"),
            "tree_sha256": dataset.get("tree_sha256", ""),
            "file_count": dataset.get("file_count", len(manifest)),
            "total_bytes": dataset.get("total_bytes", sum(int(item["bytes"]) for item in manifest)),
            "source": {
                "url": dataset.get("source", {}).get("url", ""),
                "doi": dataset.get("source", {}).get("doi", ""),
            },
        })
    except (OSError, ValueError) as exc:
        shutil.rmtree(asset_root)
        report.error("shared_asset_build_failed", f"{dataset_id}: {exc}")
        return None
    return data_root


def _candidate(row: dict[str, Any]) -> dict[str, Any]:
    task_id = require_text(row.get("task_id"), "task_id")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", task_id):
        raise ValueError(f"task_id contains unsafe characters: {task_id!r}")
    task_tag = str(row.get("task_tag") or "Main").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", task_tag):
        raise ValueError(f"task_tag contains unsafe characters: {task_tag!r}")
    task_format = str(row.get("task_format") or "end_to_end").strip()
    if task_format not in TASK_FORMATS:
        raise ValueError(f"task_format must be one of {sorted(TASK_FORMATS)}")
    workflow_stages = string_list(
        row.get("workflow_stages") or list(END_TO_END_STAGES),
        "workflow_stages",
    )
    missing_stages = [stage for stage in END_TO_END_STAGES if stage not in workflow_stages]
    if missing_stages:
        raise ValueError(
            "end-to-end task is missing workflow stages: " + ", ".join(missing_stages)
        )
    rubric = row.get("rubric") or {}
    if not isinstance(rubric, dict):
        raise ValueError("rubric must be an object")
    items = rubric.get("scoring_items") or []
    if not isinstance(items, list):
        raise ValueError("rubric.scoring_items must be a list")
    normalized_items: list[dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise ValueError(f"rubric.scoring_items[{index}] must be an object")
        point = int(item.get("point", 0))
        if point <= 0:
            raise ValueError(f"rubric.scoring_items[{index}].point must be positive")
        normalized_items.append({
            "point": point,
            "criterion": require_text(item.get("criterion"), f"scoring_items[{index}].criterion"),
            "keywords": string_list(item.get("keywords"), f"scoring_items[{index}].keywords"),
            **({"deduction_rule": str(item["deduction_rule"]).strip()} if item.get("deduction_rule") else {}),
        })
    if sum(item["point"] for item in normalized_items) != 100:
        raise ValueError("rubric points must sum to exactly 100")
    if not 3 <= len(normalized_items) <= 8:
        raise ValueError("rubric must contain 3-8 scoring items")
    rubric_text = " ".join(item["criterion"] for item in normalized_items).casefold()
    if not any(token in rubric_text for token in ("real", "provided", "actual", "fabricat", "synthetic", "真实", "提供")):
        raise ValueError("rubric needs a scoring item that enforces real provided-data use")
    reference = row.get("reference") or {}
    if not isinstance(reference, dict):
        raise ValueError("reference must be an object")
    metrics = reference.get("metrics") or []
    if not isinstance(metrics, list):
        raise ValueError("reference.metrics must be a list")
    command = reference.get("command") or ""
    if isinstance(command, list):
        command = [str(part) for part in command]
    else:
        command = str(command).strip()
    metrics_file = _safe_relative(
        str(reference.get("metrics_file") or "summary.json"), "reference.metrics_file",
    )
    artifacts = [
        _safe_relative(path, "reference.artifacts")
        for path in string_list(reference.get("artifacts") or [metrics_file], "reference.artifacts")
    ]
    reference_cwd = str(reference.get("cwd") or "").strip()
    if reference_cwd:
        reference_cwd = _safe_relative(reference_cwd, "reference.cwd")
    generation = row.get("generation") or {}
    if not isinstance(generation, dict):
        raise ValueError("generation must be an object")
    required_files = [_safe_relative(path, "required_files") for path in string_list(row.get("required_files"), "required_files")]
    deliverable = require_text(row.get("deliverable"), "deliverable")
    ai_capability = string_list(
        row.get("ai_capability") or ["Planning", "Coding", "Interpretation", "Reasoning"],
        "ai_capability",
    )
    capability_text = " ".join(ai_capability).casefold()
    for capability in ("planning", "coding", "interpret"):
        if capability not in capability_text:
            raise ValueError(f"end-to-end task ai_capability must include {capability!r}")
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "task_tag": task_tag,
        "task_format": task_format,
        "workflow_stages": workflow_stages,
        "dataset_id": require_text(row.get("dataset_id"), "dataset_id"),
        "link_id": require_text(row.get("link_id"), "link_id"),
        "query": require_text(row.get("query"), "query"),
        "deliverable": deliverable,
        "deliverables": string_list(row.get("deliverables") or [deliverable], "deliverables"),
        "required_files": required_files,
        "ai_capability": ai_capability,
        "construction_mode": str(row.get("construction_mode") or "reproduce").strip(),
        "leakage_reviewed": bool(row.get("leakage_reviewed", False)),
        "trajectory_verdict": str(row.get("trajectory_verdict") or "unreviewed").strip(),
        "reference": {
            "status": str(reference.get("status") or "unverified").strip(),
            "command": command,
            "cwd": reference_cwd,
            "timeout_seconds": int(reference.get("timeout_seconds") or 0),
            "metrics_file": metrics_file,
            "artifacts": artifacts,
            "result_summary": str(reference.get("result_summary") or "").strip(),
            "metrics": metrics,
            **({"execution": reference["execution"]} if isinstance(reference.get("execution"), dict) else {}),
        },
        "generation": {
            key: generation.get(key)
            for key in (
                "provider", "model", "prompt_version", "cache_key", "cache_hit",
                "source_link_id", "generated_at_utc",
            )
            if generation.get(key) is not None
        },
        "rubric": {
            "reason": str(rubric.get("reason") or "").strip(),
            "core_conclusion": require_text(rubric.get("core_conclusion"), "rubric.core_conclusion"),
            "scoring_items": normalized_items,
            "acceptable_deviations": string_list(rubric.get("acceptable_deviations"), "acceptable_deviations"),
            "scoring_instructions": str(rubric.get("scoring_instructions") or "").strip(),
        },
    }


def _query_leaks_reference(candidate: dict[str, Any]) -> list[str]:
    query = candidate["query"].casefold()
    leaks: list[str] = []
    conclusion = candidate["rubric"]["core_conclusion"].strip().casefold()
    if len(conclusion) >= 30 and conclusion in query:
        leaks.append("the full GT core conclusion appears in the public query")
    for metric in candidate["reference"]["metrics"]:
        if not isinstance(metric, dict) or "target" not in metric:
            continue
        target = str(metric["target"]).strip()
        if len(target) >= 2 and re.search(rf"(?<![\w.]){re.escape(target)}(?![\w.])", candidate["query"]):
            leaks.append(f"reference target {target!r} appears in the public query")
    return leaks


def validate_task_package(task_root: Path, *, max_bytes: int | None = None) -> ValidationReport:
    report = ValidationReport()
    info_path = task_root / "task_info.json"
    checklist_path = task_root / "target_study" / "checklist.json"
    gt_path = task_root / "target_study" / "ground_truth.json"
    provenance_path = task_root / "target_study" / "provenance.json"
    data_root = task_root / "data"
    for path, code in (
        (info_path, "missing_task_info"),
        (checklist_path, "missing_checklist"),
        (gt_path, "missing_ground_truth"),
        (provenance_path, "missing_provenance"),
    ):
        if not path.is_file():
            report.error(code, str(path))
    if not data_root.is_dir():
        report.error("missing_data", str(data_root))
    if not report.ok:
        return report
    try:
        info = read_json(info_path)
        checklist = read_json(checklist_path)
        ground_truth = read_json(gt_path)
        provenance = read_json(provenance_path)
    except (OSError, json.JSONDecodeError) as exc:
        report.error("invalid_json", str(exc))
        return report

    if not str(info.get("query") or "").strip():
        report.error("empty_query", "task_info.query is empty")
    if any(Path(str(info.get(key) or "")).is_absolute() for key in ("data_path", "query_path", "rubric_path")):
        report.error("absolute_public_path", "task_info contains an absolute path")
    try:
        items = checklist["rubrics"]["score_100"]["rubric"]["Scoring items"]
        total = sum(int(item.get("point", 0)) for item in items)
        if not items:
            report.error("empty_rubric", "score_100 has no scoring items")
        if total != 100:
            report.error("points_not_100", f"rubric points sum to {total}")
    except (KeyError, TypeError, ValueError) as exc:
        report.error("invalid_rubric", str(exc))
    if ground_truth.get("task_id") != info.get("task_id"):
        report.error("identity_mismatch", "ground_truth.task_id differs from task_info.task_id")
    if info.get("task_format") != "end_to_end":
        report.error("not_end_to_end", "task_info.task_format must be end_to_end")
    stages = info.get("workflow_stages") or []
    missing_stages = [stage for stage in END_TO_END_STAGES if stage not in stages]
    if missing_stages:
        report.error("missing_workflow_stages", ", ".join(missing_stages))
    if not provenance.get("dataset", {}).get("tree_sha256"):
        report.error("missing_dataset_hash", "provenance lacks dataset tree_sha256")

    total_bytes = 0
    file_count = 0
    for path in data_root.rglob("*"):
        if path.is_symlink():
            report.error("symlink", f"symlink in task data: {path.relative_to(data_root)}")
        elif path.is_file():
            total_bytes += path.stat().st_size
            file_count += 1
    if not file_count:
        report.error("empty_data", "task/data contains no files")
    if max_bytes is not None and total_bytes > max_bytes:
        report.error("task_data_too_large", f"{total_bytes} bytes > {max_bytes}")
    report.stats.update({"data_files": file_count, "data_bytes": total_bytes})
    return report


def build_packages(
    registry_path: Path,
    links_path: Path,
    candidates_path: Path,
    output_root: Path,
    *,
    max_bytes: int | None = None,
    allow_unverified: bool = False,
) -> ValidationReport:
    datasets = {row["dataset_id"]: row for row in read_jsonl(registry_path)}
    links = {row["link_id"]: row for row in read_jsonl(links_path)}
    report = ValidationReport()
    built = 0
    skipped = 0
    seen: set[str] = set()
    seen_locations: set[tuple[str, str]] = set()
    for index, raw in enumerate(read_jsonl(candidates_path), 1):
        try:
            candidate = _candidate(raw)
        except (TypeError, ValueError) as exc:
            report.error("bad_candidate", f"row {index}: {exc}")
            continue
        task_id = candidate["task_id"]
        if task_id in seen or (output_root / task_id).exists():
            report.error("duplicate_task_id", task_id)
            continue
        seen.add(task_id)
        dataset = datasets.get(candidate["dataset_id"])
        link = links.get(candidate["link_id"])
        if not dataset or not link or link.get("dataset_id") != candidate["dataset_id"]:
            report.error("bad_candidate_join", f"{task_id}: dataset/link join failed")
            continue
        task_location = (str(link["paper_id"]), candidate["task_tag"])
        if task_location in seen_locations:
            report.error(
                "duplicate_paper_task_tag",
                f"{task_id}: paper_id={task_location[0]} task_tag={task_location[1]}",
            )
            continue
        seen_locations.add(task_location)
        gate_failures: list[str] = []
        if dataset.get("profile_status") != "valid":
            gate_failures.append("dataset profile is invalid")
        if str(dataset.get("license") or "unknown").casefold() in {"", "unknown", "unspecified", "none"}:
            gate_failures.append("dataset license is not verified")
        if link.get("data_support") != "yes":
            gate_failures.append(f"data_support={link.get('data_support')}")
        if link.get("dataset_version_match") is not True:
            gate_failures.append("dataset version is not verified")
        if link.get("reproducibility") != "verified" or candidate["reference"]["status"] != "verified":
            gate_failures.append("reference reproduction is not verified")
        risk = str(link.get("leakage_risk") or dataset.get("leakage_risk") or "unknown")
        if risk in {"severe", "needs_review", "unknown"} and not candidate["leakage_reviewed"]:
            gate_failures.append(f"leakage_risk={risk} has not been reviewed")
        gate_failures.extend(_query_leaks_reference(candidate))
        if gate_failures and not allow_unverified:
            skipped += 1
            report.warn("candidate_gated", f"{task_id}: " + "; ".join(gate_failures))
            continue

        source_root = Path(dataset["source"]["local_path"])
        manifest_path = registry_path.parent / dataset["manifest_path"]
        manifest_rows = read_jsonl(manifest_path)
        manifest = {row["path"]: row for row in manifest_rows}
        required = candidate["required_files"] or list(link.get("required_files") or [])
        if not required:
            report.error("no_required_files", f"{task_id}: candidate/link names no data files")
            continue
        missing = [path for path in required if path not in manifest]
        if missing:
            report.error("required_files_missing", f"{task_id}: {missing[:5]}")
            continue
        dataset_bytes = int(dataset.get("total_bytes") or 0)
        if max_bytes is not None and dataset_bytes > max_bytes:
            report.warn("dataset_over_size_limit", f"{task_id}: {dataset_bytes} > {max_bytes}")
            skipped += 1
            continue

        shared_data_root = _ensure_shared_dataset_asset(
            output_root, dataset, source_root, manifest_rows, report,
        )
        if shared_data_root is None:
            continue

        task_root = output_root / task_id / "task"
        data_root = task_root / "data"
        hidden_root = task_root / "target_study"
        data_root.mkdir(parents=True, exist_ok=False)
        hidden_root.mkdir(parents=True, exist_ok=True)
        selected_manifest: list[dict[str, Any]] = []
        copy_failed = False
        for rel in required:
            src = shared_data_root / rel
            dst = data_root / rel
            if not src.is_file() or sha256_file(src) != manifest[rel]["sha256"]:
                report.error("source_changed", f"{task_id}: {rel} differs from registry manifest")
                copy_failed = True
                break
            dst.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(src, dst)
            selected_manifest.append(manifest[rel])
        if copy_failed:
            shutil.rmtree(output_root / task_id)
            continue

        info = {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "task_tag": candidate["task_tag"],
            "task_format": candidate["task_format"],
            "workflow_stages": candidate["workflow_stages"],
            "paper_id": link["paper_id"],
            "paper_display_name": link["title"],
            "dataset_id": dataset["dataset_id"],
            "construction_mode": candidate["construction_mode"],
            "ai_capability": candidate["ai_capability"],
            "science_axes": dataset.get("science", {}),
            "query": candidate["query"],
            "provided_data": [
                {"path": f"data/{item['path']}", "bytes": item["bytes"], "sha256": item["sha256"]}
                for item in selected_manifest
            ],
            "deliverable": candidate["deliverable"],
            "deliverables": candidate["deliverables"],
            "real_data_grounding": "MUST use the real provided data under data/; fabricated or synthetic replacements fail the task.",
        }
        rubric = candidate["rubric"]
        checklist = {
            "tag": candidate["task_tag"],
            "rubrics": {"score_100": {
                "reason": rubric["reason"],
                "rubric": {
                    "GT core conclusion": rubric["core_conclusion"],
                    "Scoring items": rubric["scoring_items"],
                    "Acceptable deviations": rubric["acceptable_deviations"],
                    "Scoring instructions": rubric["scoring_instructions"],
                },
            }},
        }
        ground_truth = {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "task_tag": candidate["task_tag"],
            "core_conclusion": rubric["core_conclusion"],
            "reference": candidate["reference"],
        }
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "task_tag": candidate["task_tag"],
            "task_format": candidate["task_format"],
            "workflow_stages": candidate["workflow_stages"],
            "dataset": {
                "dataset_id": dataset["dataset_id"],
                "name": dataset["name"],
                "version": dataset["version"],
                "source_url": dataset["source"].get("url", ""),
                "doi": dataset["source"].get("doi", ""),
                "license": dataset.get("license", "unknown"),
                "tree_sha256": dataset["tree_sha256"],
                "file_count": dataset.get("file_count", len(manifest_rows)),
                "total_bytes": dataset_bytes,
                "selected_manifest": selected_manifest,
            },
            "paper_link": link,
            "generation": candidate["generation"],
            "trajectory_verdict": candidate["trajectory_verdict"],
            "quality_gate": {"allow_unverified": allow_unverified, "gate_failures": gate_failures},
        }
        write_json(task_root / "task_info.json", info)
        write_json(hidden_root / "checklist.json", checklist)
        write_json(hidden_root / "ground_truth.json", ground_truth)
        write_json(hidden_root / "provenance.json", provenance)
        validation = validate_task_package(task_root, max_bytes=max_bytes)
        write_json(output_root / task_id / "validation_report.json", validation.to_dict())
        if validation.ok:
            built += 1
        else:
            report.error("invalid_built_package", task_id)
    report.stats.update({"built": built, "skipped_by_gate": skipped, "candidates": len(read_jsonl(candidates_path))})
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "build_report.json", report.to_dict())
    return report
