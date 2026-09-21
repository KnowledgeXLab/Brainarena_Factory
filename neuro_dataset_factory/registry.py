"""Dataset registry and paper-link normalization."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from neuro_dataset_factory.contracts import (
    DATA_SUPPORT,
    GT_TYPES,
    LEAKAGE_RISKS,
    REPRO_STATUSES,
    SCHEMA_VERSION,
    DatasetSeed,
    ValidationReport,
    require_text,
    stable_id,
    string_list,
)
from neuro_dataset_factory.manifest import build_manifest
from neuro_dataset_factory.storage import read_jsonl, write_json, write_jsonl


def profile_datasets(seeds_path: Path, output_root: Path, *, max_bytes: int | None = None) -> tuple[list[dict[str, Any]], ValidationReport]:
    output_root.mkdir(parents=True, exist_ok=True)
    combined = ValidationReport()
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(read_jsonl(seeds_path), 1):
        try:
            seed = DatasetSeed.from_dict(row)
        except ValueError as exc:
            combined.error("bad_seed", f"row {index}: {exc}")
            continue
        if seed.dataset_id in seen_ids:
            combined.error("duplicate_dataset_id", seed.dataset_id)
            continue
        seen_ids.add(seed.dataset_id)
        source_root = Path(seed.local_path).expanduser().resolve()
        manifest, summary, report = build_manifest(source_root, max_bytes=max_bytes)
        manifest_rel = Path("datasets") / seed.dataset_id / "manifest.jsonl"
        write_jsonl(output_root / manifest_rel, manifest)
        write_json(output_root / "datasets" / seed.dataset_id / "profile_report.json", report.to_dict())
        leakage_risk = "needs_review" if (
            summary.get("role_counts", {}).get("code")
            or summary.get("role_counts", {}).get("result_artifact")
        ) else "unknown"
        record = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": seed.dataset_id,
            "name": seed.name,
            "version": seed.version,
            "description": seed.description,
            "source": {
                "url": seed.source_url,
                "doi": seed.doi,
                "local_path": str(source_root),
            },
            "license": seed.license,
            "science": {
                "modalities": list(seed.modalities),
                "species": list(seed.species),
                "task_domains": list(seed.task_domains),
                "high_value_questions": list(seed.high_value_questions),
            },
            "manifest_path": manifest_rel.as_posix(),
            "file_count": summary.get("file_count", 0),
            "total_bytes": summary.get("total_bytes", 0),
            "tree_sha256": summary.get("tree_sha256", ""),
            "role_counts": summary.get("role_counts", {}),
            "leakage_risk": leakage_risk,
            "profile_status": "valid" if report.ok else "invalid",
        }
        if seed.license.casefold() in {"", "unknown", "unspecified", "none"}:
            combined.warn("license_unknown", f"{seed.dataset_id}: dataset license is not verified")
        records.append(record)
        for issue in report.issues:
            message = f"{seed.dataset_id}: {issue.message}"
            combined.error(issue.code, message) if issue.level == "error" else combined.warn(issue.code, message)
    write_jsonl(output_root / "dataset_registry.jsonl", records)
    combined.stats.update({"seed_rows": len(read_jsonl(seeds_path)), "registry_rows": len(records)})
    write_json(output_root / "profile_report.json", combined.to_dict())
    return records, combined


def _normalize_paper_link(row: dict[str, Any]) -> dict[str, Any]:
    dataset_id = require_text(row.get("dataset_id"), "dataset_id")
    title = require_text(row.get("title"), "title")
    paper_id = str(row.get("paper_id") or "").strip() or stable_id("paper", title, row.get("doi"), row.get("url"))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", paper_id):
        raise ValueError(f"paper_id contains unsafe characters: {paper_id!r}")
    link_id = str(row.get("link_id") or "").strip() or stable_id("link", dataset_id, paper_id)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", link_id):
        raise ValueError(f"link_id contains unsafe characters: {link_id!r}")
    reproducibility = str(row.get("reproducibility") or "unverified").strip()
    if reproducibility not in REPRO_STATUSES:
        raise ValueError(f"invalid reproducibility: {reproducibility}")
    data_support = str(row.get("data_support") or "unknown").strip()
    if data_support not in DATA_SUPPORT:
        raise ValueError(f"invalid data_support: {data_support}")
    leakage_risk = str(row.get("leakage_risk") or "unknown").strip()
    if leakage_risk not in LEAKAGE_RISKS:
        raise ValueError(f"invalid leakage_risk: {leakage_risk}")
    gt_type = str(row.get("gt_type") or "mixed").strip()
    if gt_type not in GT_TYPES:
        raise ValueError(f"invalid gt_type: {gt_type}")
    evidence = row.get("evidence") or {}
    if not isinstance(evidence, dict):
        raise ValueError("evidence must be an object")
    return {
        "schema_version": SCHEMA_VERSION,
        "link_id": link_id,
        "dataset_id": dataset_id,
        "paper_id": paper_id,
        "title": title,
        "doi": str(row.get("doi") or "").strip(),
        "url": str(row.get("url") or "").strip(),
        "dataset_version_match": row.get("dataset_version_match", "unknown"),
        "scientific_question": require_text(row.get("scientific_question"), "scientific_question"),
        "target_result": require_text(row.get("target_result"), "target_result"),
        "required_files": string_list(row.get("required_files"), "required_files"),
        "reference_code": str(row.get("reference_code") or "").strip(),
        "gt_type": gt_type,
        "reproducibility": reproducibility,
        "data_support": data_support,
        "leakage_risk": leakage_risk,
        "evidence": {
            "dataset_mention": str(evidence.get("dataset_mention") or "").strip(),
            "data_availability": str(evidence.get("data_availability") or "").strip(),
            "result_location": str(evidence.get("result_location") or "").strip(),
        },
    }


def validate_paper_links(registry_path: Path, links_path: Path, output_path: Path) -> ValidationReport:
    datasets = {row["dataset_id"]: row for row in read_jsonl(registry_path)}
    report = ValidationReport()
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(read_jsonl(links_path), 1):
        try:
            row = _normalize_paper_link(raw)
        except ValueError as exc:
            report.error("bad_paper_link", f"row {index}: {exc}")
            continue
        if row["dataset_id"] not in datasets:
            report.error("unknown_dataset", f"row {index}: {row['dataset_id']}")
            continue
        if row["link_id"] in seen:
            report.error("duplicate_link_id", row["link_id"])
            continue
        seen.add(row["link_id"])
        if row["dataset_version_match"] is not True:
            report.warn("dataset_version_unverified", row["link_id"])
        if row["data_support"] != "yes":
            report.warn("data_support_not_yes", f"{row['link_id']}: {row['data_support']}")
        if row["reproducibility"] != "verified":
            report.warn("reference_unverified", f"{row['link_id']}: {row['reproducibility']}")
        if not row["evidence"]["dataset_mention"] or not row["evidence"]["result_location"]:
            report.warn("weak_linkage_evidence", row["link_id"])
        normalized.append(row)
    write_jsonl(output_path, normalized)
    report.stats.update({"input_links": len(read_jsonl(links_path)), "valid_links": len(normalized)})
    write_json(output_path.with_suffix(".report.json"), report.to_dict())
    return report
