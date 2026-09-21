"""Build content-addressed manifests without discarding provenance files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from neuro_dataset_factory.contracts import ValidationReport

LICENSE_PREFIXES = ("license", "licence", "copying")
CODE_EXTENSIONS = {".py", ".r", ".m", ".jl", ".ipynb", ".sh"}
ENV_NAMES = {
    "requirements.txt", "environment.yml", "environment.yaml", "pyproject.toml",
    "setup.py", "setup.cfg", "dockerfile", "makefile", "renv.lock", "package.json",
}
RESULT_HINTS = ("result", "output", "prediction", "metric", "figure", "plot", "answer", "gold")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_role(path: Path) -> str:
    name = path.name.lower()
    ext = path.suffix.lower()
    if name.startswith(LICENSE_PREFIXES):
        return "license"
    if name in ENV_NAMES:
        return "environment"
    if ext in CODE_EXTENSIONS:
        return "code"
    if name.startswith("readme") or ext in {".md", ".rst"}:
        return "documentation"
    if any(hint in name for hint in RESULT_HINTS):
        return "result_artifact"
    return "data"


def build_manifest(root: Path, *, max_bytes: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any], ValidationReport]:
    report = ValidationReport()
    if not root.is_dir():
        report.error("missing_dataset", f"dataset path is not a directory: {root}")
        return [], {}, report

    files: list[Path] = []
    total = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            report.error("symlink", f"symbolic links are not allowed: {path.relative_to(root)}")
        elif path.is_file():
            files.append(path)
            total += path.stat().st_size
    if max_bytes is not None and total > max_bytes:
        summary = {"file_count": len(files), "total_bytes": total, "tree_sha256": "", "role_counts": {}}
        report.error("dataset_too_large", f"dataset is {total} bytes; limit is {max_bytes}")
        report.stats.update(summary)
        return [], summary, report

    entries: list[dict[str, Any]] = []
    role_counts: dict[str, int] = {}
    tree = hashlib.sha256()
    for path in files:
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        role = classify_role(path)
        role_counts[role] = role_counts.get(role, 0) + 1
        entry = {
            "path": rel,
            "bytes": size,
            "sha256": digest,
            "extension": path.suffix.lower(),
            "role": role,
        }
        entries.append(entry)
        tree.update(rel.encode("utf-8"))
        tree.update(b"\0")
        tree.update(str(size).encode("ascii"))
        tree.update(b"\0")
        tree.update(digest.encode("ascii"))
        tree.update(b"\n")

    if not entries:
        report.error("empty_dataset", f"dataset contains no regular files: {root}")
    if not role_counts.get("license"):
        report.warn("license_file_missing", "no LICENSE/LICENCE/COPYING file was found")
    if role_counts.get("code"):
        report.warn("code_leakage_review", f"bundle contains {role_counts['code']} code/notebook files")
    if role_counts.get("result_artifact"):
        report.warn("result_leakage_review", f"bundle contains {role_counts['result_artifact']} result-like files")

    summary = {
        "file_count": len(entries),
        "total_bytes": total,
        "tree_sha256": tree.hexdigest(),
        "role_counts": role_counts,
    }
    report.stats.update(summary)
    return entries, summary, report
