"""Leakage-safe deterministic train/validation/test assignment."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from neuro_dataset_factory.contracts import SCHEMA_VERSION, ValidationReport
from neuro_dataset_factory.storage import read_json, write_json


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _unit_interval(seed: str, group_key: str) -> float:
    digest = hashlib.sha256(f"{seed}\0{group_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_grouped_splits(
    registry_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: str = "brainarena-v1",
    group_fields: tuple[str, ...] = ("dataset_id", "paper_id"),
) -> ValidationReport:
    """Assign connected dataset/paper components to one split to prevent leakage."""
    report = ValidationReport()
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        report.error("invalid_split_ratios", str(ratios))
        return report
    if not group_fields:
        report.error("missing_group_fields", "at least one group field is required")
        return report
    registry = read_json(registry_path)
    tasks = registry.get("tasks") or []
    if not isinstance(tasks, list):
        report.error("invalid_registry_tasks", "tasks must be a list")
        return report
    disjoint = _DisjointSet(len(tasks))
    first_seen: dict[tuple[str, str], int] = {}
    for index, task in enumerate(tasks):
        for field in group_fields:
            value = str(task.get(field) or "").strip()
            if not value:
                report.error("missing_split_group", f"task {task.get('task_id')}: {field}")
                continue
            identity = (field, value)
            if identity in first_seen:
                disjoint.union(index, first_seen[identity])
            else:
                first_seen[identity] = index
    if not report.ok:
        return report
    components: dict[int, list[int]] = {}
    for index in range(len(tasks)):
        components.setdefault(disjoint.find(index), []).append(index)
    split_counts = {"train": 0, "validation": 0, "test": 0}
    group_counts = {"train": 0, "validation": 0, "test": 0}
    group_manifest: list[dict[str, Any]] = []
    train_end = train_ratio
    validation_end = train_ratio + validation_ratio
    for indices in components.values():
        identities = sorted({
            f"{field}={tasks[index].get(field)}"
            for index in indices
            for field in group_fields
        })
        group_key = "|".join(identities)
        position = _unit_interval(seed, group_key)
        split = "train" if position < train_end else "validation" if position < validation_end else "test"
        for index in indices:
            tasks[index]["split"] = split
            tasks[index]["split_group"] = group_key
        split_counts[split] += len(indices)
        group_counts[split] += 1
        group_manifest.append({
            "split_group": group_key,
            "split": split,
            "task_ids": [tasks[index].get("task_id") for index in indices],
        })
    fields = registry.setdefault("fields", [])
    for field in ("split", "split_group"):
        if field not in fields:
            fields.append(field)
    write_json(output_path, registry)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "ratios": {"train": train_ratio, "validation": validation_ratio, "test": test_ratio},
        "group_fields": list(group_fields),
        "task_counts": split_counts,
        "group_counts": group_counts,
        "groups": sorted(group_manifest, key=lambda row: row["split_group"]),
    }
    write_json(manifest_path, manifest)
    report.stats.update({
        "tasks": len(tasks),
        "groups": len(components),
        "task_counts": split_counts,
        "group_counts": group_counts,
    })
    return report
