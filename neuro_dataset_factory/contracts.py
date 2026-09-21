"""Stable contracts shared by profiling, linking, packaging, and export."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 2
GT_TYPES = {"numeric", "table", "figure", "scientific_claim", "mixed"}
REPRO_STATUSES = {"verified", "partial", "unverified"}
DATA_SUPPORT = {"yes", "partial", "no", "unknown"}
LEAKAGE_RISKS = {"none", "partial", "severe", "needs_review", "unknown"}
TASK_FORMATS = {"end_to_end"}
END_TO_END_STAGES = (
    "planning",
    "data_inspection",
    "coding",
    "execution",
    "statistical_analysis",
    "visualization",
    "scientific_interpretation",
)


def require_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a string or list")
    out: list[str] = []
    for raw in value:
        item = str(raw or "").strip()
        if item and item not in out:
            out.append(item)
    return out


def slug(value: str, max_length: int = 64) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return (cleaned or "item")[:max_length]


def stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


@dataclass(frozen=True)
class DatasetSeed:
    name: str
    version: str
    local_path: str
    dataset_id: str = ""
    description: str = ""
    source_url: str = ""
    doi: str = ""
    license: str = "unknown"
    modalities: tuple[str, ...] = ()
    species: tuple[str, ...] = ()
    task_domains: tuple[str, ...] = ()
    high_value_questions: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "DatasetSeed":
        name = require_text(row.get("name"), "name")
        version = str(row.get("version") or "unspecified").strip()
        local_path = require_text(row.get("local_path"), "local_path")
        dataset_id = str(row.get("dataset_id") or "").strip()
        if not dataset_id:
            dataset_id = f"{slug(name, 42)}--{stable_id('ds', name, version, row.get('doi'), row.get('source_url'))[-12:]}"
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", dataset_id):
            raise ValueError(f"dataset_id contains unsafe characters: {dataset_id!r}")
        return cls(
            name=name,
            version=version,
            local_path=local_path,
            dataset_id=dataset_id,
            description=str(row.get("description") or "").strip(),
            source_url=str(row.get("source_url") or "").strip(),
            doi=str(row.get("doi") or "").strip(),
            license=str(row.get("license") or "unknown").strip(),
            modalities=tuple(string_list(row.get("modalities"), "modalities")),
            species=tuple(string_list(row.get("species"), "species")),
            task_domains=tuple(string_list(row.get("task_domains"), "task_domains")),
            high_value_questions=tuple(string_list(row.get("high_value_questions"), "high_value_questions")),
        )


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "code": self.code, "message": self.message}


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    def error(self, code: str, message: str) -> None:
        self.issues.append(ValidationIssue("error", code, message))

    def warn(self, code: str, message: str) -> None:
        self.issues.append(ValidationIssue("warning", code, message))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": self.ok,
            "stats": self.stats,
            "issues": [issue.to_dict() for issue in self.issues],
        }
