"""Deterministic candidate-level duplicate detection for batch task construction."""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

from neuro_dataset_factory.contracts import ValidationReport
from neuro_dataset_factory.storage import read_jsonl, write_json, write_jsonl


def _tokens(value: Any) -> set[str]:
    return {token for token in re.findall(r"\w+", str(value or "").casefold()) if len(token) > 1}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _normalized_text(value: Any) -> str:
    return " ".join(re.findall(r"\w+", str(value or "").casefold()))


def _metric_names(row: dict[str, Any]) -> set[str]:
    metrics = (row.get("reference") or {}).get("metrics") or []
    return {
        _normalized_text(metric.get("name"))
        for metric in metrics
        if isinstance(metric, dict) and _normalized_text(metric.get("name"))
    }


def candidate_similarity(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    threshold: float | None = None,
) -> float:
    """Return a conservative lexical/structural similarity score in [0, 1]."""
    left_query = _normalized_text(left.get("query"))
    right_query = _normalized_text(right.get("query"))
    if left_query and left_query == right_query:
        return 1.0
    query_jaccard = _jaccard(_tokens(left_query), _tokens(right_query))
    file_score = _jaccard(
        {_normalized_text(path) for path in left.get("required_files") or []},
        {_normalized_text(path) for path in right.get("required_files") or []},
    )
    metric_score = _jaccard(_metric_names(left), _metric_names(right))
    query_matcher = difflib.SequenceMatcher(None, left_query, right_query, autojunk=False)
    if threshold is not None:
        # quick_ratio is an upper bound on ratio.  If even the maximum possible
        # conclusion contribution cannot reach the decision threshold, avoid
        # both quadratic SequenceMatcher ratios for this unrelated pair.
        query_upper = max(query_matcher.quick_ratio(), query_jaccard)
        upper = 0.65 * query_upper + 0.15 * file_score + 0.10 * metric_score + 0.10
        if upper < threshold:
            return upper
    query_score = max(query_matcher.ratio(), query_jaccard)
    partial = 0.65 * query_score + 0.15 * file_score + 0.10 * metric_score
    if threshold is not None and partial + 0.10 < threshold:
        return partial + 0.10
    left_conclusion = _normalized_text((left.get("rubric") or {}).get("core_conclusion"))
    right_conclusion = _normalized_text((right.get("rubric") or {}).get("core_conclusion"))
    conclusion_score = difflib.SequenceMatcher(
        None, left_conclusion, right_conclusion, autojunk=False,
    ).ratio()
    return partial + 0.10 * conclusion_score


def deduplicate_task_candidates(
    candidates_path: Path,
    accepted_path: Path,
    duplicates_path: Path,
    *,
    threshold: float = 0.90,
) -> ValidationReport:
    """Keep the first candidate from each near-duplicate group within a dataset."""
    report = ValidationReport()
    if not 0.0 <= threshold <= 1.0:
        report.error("invalid_threshold", str(threshold))
        return report
    rows = read_jsonl(candidates_path)
    accepted: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    task_ids: set[str] = set()
    for index, row in enumerate(rows, 1):
        task_id = str(row.get("task_id") or "").strip()
        dataset_id = str(row.get("dataset_id") or "").strip()
        query = str(row.get("query") or "").strip()
        if not task_id or not dataset_id or not query:
            report.error("invalid_candidate_identity", f"row {index}")
            continue
        if task_id in task_ids:
            duplicates.append({
                "task_id": task_id,
                "duplicate_of": task_id,
                "similarity": 1.0,
                "reason": "duplicate_task_id",
                "candidate": row,
            })
            continue
        task_ids.add(task_id)
        match: tuple[dict[str, Any], float] | None = None
        for previous in by_dataset.get(dataset_id, []):
            score = candidate_similarity(previous, row, threshold=threshold)
            if score >= threshold and (match is None or score > match[1]):
                match = (previous, score)
        if match is not None:
            duplicates.append({
                "task_id": task_id,
                "duplicate_of": match[0]["task_id"],
                "dataset_id": dataset_id,
                "similarity": round(match[1], 6),
                "threshold": threshold,
                "reason": "near_duplicate_within_dataset",
                "candidate": row,
            })
            continue
        accepted.append(row)
        by_dataset.setdefault(dataset_id, []).append(row)
    write_jsonl(accepted_path, accepted)
    write_jsonl(duplicates_path, duplicates)
    report.stats.update({
        "input_candidates": len(rows),
        "accepted": len(accepted),
        "duplicates": len(duplicates),
        "datasets": len(by_dataset),
        "threshold": threshold,
    })
    write_json(accepted_path.with_suffix(".dedup_report.json"), report.to_dict())
    return report
