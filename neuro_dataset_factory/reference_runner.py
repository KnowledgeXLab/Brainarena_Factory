"""Batch execution and metric verification for trusted reference analyses."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neuro_dataset_factory.contracts import ValidationReport
from neuro_dataset_factory.packages import _safe_relative
from neuro_dataset_factory.storage import read_json, read_jsonl, write_json, write_jsonl

_SECRET_ENV_NAMES = {
    "API_KEY", "API_BASE", "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "SERPER_KEY",
    "BOYUE_KEY", "GITHUB_TOKEN", "HF_TOKEN",
}
_PROXY_ENV_NAMES = {
    "http_proxy", "https_proxy", "all_proxy",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
}


def _flatten_json(value: Any, prefix: str = "") -> dict[str, Any]:
    leaves: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(_flatten_json(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}.{index}" if prefix else str(index)
            leaves.update(_flatten_json(child, path))
    else:
        leaves[prefix] = value
    return leaves


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _resolve_metric(metric: dict[str, Any], leaves: dict[str, Any]) -> tuple[str, Any]:
    explicit = str(metric.get("json_path") or "").strip()
    if explicit:
        if explicit not in leaves:
            raise KeyError(f"json_path not found: {explicit}")
        return explicit, leaves[explicit]
    name = _norm(str(metric.get("name") or ""))
    matches: list[tuple[str, Any]] = []
    for path, value in leaves.items():
        normalized_path = _norm(path)
        leaf = _norm(path.rsplit(".", 1)[-1])
        if (
            normalized_path == name
            or normalized_path.endswith("_" + name)
            or leaf == name
            or leaf.startswith(name + "_")
            or name.startswith(leaf + "_")
        ):
            matches.append((path, value))
    if len(matches) != 1:
        raise KeyError(f"metric {name!r} resolved to {len(matches)} JSON leaves")
    return matches[0]


def _compare(observed: Any, target: Any, tolerance: Any) -> tuple[bool, str]:
    if isinstance(observed, bool) or isinstance(target, bool):
        ok = observed == target
        return ok, f"observed={observed!r}, target={target!r}"
    try:
        observed_number = float(observed)
        target_number = float(target)
        tolerance_number = float(tolerance or 0)
    except (TypeError, ValueError):
        ok = observed == target
        return ok, f"observed={observed!r}, target={target!r}"
    ok = math.isfinite(observed_number) and abs(observed_number - target_number) <= tolerance_number
    return ok, (
        f"observed={observed_number}, target={target_number}, tolerance={tolerance_number}"
    )


def _safe_working_directory(workspace_root: Path, raw: Any) -> Path:
    workspace = workspace_root.resolve()
    candidate = workspace if not raw else (workspace / _safe_relative(str(raw), "reference.cwd")).resolve()
    if candidate != workspace and workspace not in candidate.parents:
        raise ValueError(f"reference cwd escapes workspace: {candidate}")
    if not candidate.is_dir():
        raise ValueError(f"reference cwd does not exist: {candidate}")
    return candidate


def _command_args(command: Any, data_root: Path, output_root: Path) -> list[str]:
    if isinstance(command, list):
        args = [str(part) for part in command]
    else:
        args = shlex.split(str(command or ""))
    if not args:
        raise ValueError("reference.command is empty")
    replacements = {"<data>": str(data_root), "<output>": str(output_root)}
    return [
        part.replace("<data>", replacements["<data>"]).replace("<output>", replacements["<output>"])
        for part in args
    ]


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in list(env):
        upper = key.upper()
        if (
            key in _SECRET_ENV_NAMES
            or key in _PROXY_ENV_NAMES
            or any(marker in upper for marker in ("API_KEY", "AUTH_TOKEN", "_SECRET", "PASSWORD", "CREDENTIAL"))
        ):
            env.pop(key, None)
    return env


def run_reference_analyses(
    registry_path: Path,
    candidates_path: Path,
    updated_candidates_path: Path,
    results_path: Path,
    output_root: Path,
    *,
    workspace_root: Path,
    allow_execution: bool = False,
    accept_observed_targets: bool = False,
    jobs: int = 2,
    default_timeout: int = 1800,
) -> ValidationReport:
    """Execute trusted reference commands and verify their machine-readable metrics."""
    report = ValidationReport()
    if not allow_execution:
        report.error(
            "execution_not_authorized",
            "pass --allow-execution only after reviewing generated reference commands",
        )
        return report
    if jobs <= 0 or default_timeout <= 0:
        report.error("invalid_reference_limits", f"jobs={jobs}, timeout={default_timeout}")
        return report
    datasets = {row["dataset_id"]: row for row in read_jsonl(registry_path)}
    candidates = read_jsonl(candidates_path)
    output_root.mkdir(parents=True, exist_ok=True)

    def execute(candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        updated = deepcopy(candidate)
        task_id = str(candidate.get("task_id") or "")
        dataset = datasets.get(str(candidate.get("dataset_id") or ""))
        if not task_id or dataset is None:
            raise ValueError("candidate task_id/dataset_id is invalid")
        reference = dict(updated.get("reference") or {})
        task_output = output_root / task_id
        task_output.mkdir(parents=True, exist_ok=True)
        data_root = Path(dataset["source"]["local_path"]).resolve()
        cwd = _safe_working_directory(workspace_root, reference.get("cwd"))
        args = _command_args(reference.get("command"), data_root, task_output)
        timeout = int(reference.get("timeout_seconds") or default_timeout)
        started = datetime.now(timezone.utc).isoformat()
        start_time = time.monotonic()
        try:
            completed = subprocess.run(
                args,
                cwd=cwd,
                env=_subprocess_env(),
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
                check=False,
            )
            exit_code = completed.returncode
            stdout, stderr = completed.stdout, completed.stderr
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            exit_code = None
            stdout = str(exc.stdout or "")
            stderr = str(exc.stderr or "")
            timed_out = True
        duration = time.monotonic() - start_time
        (task_output / "stdout.log").write_text(stdout, encoding="utf-8")
        (task_output / "stderr.log").write_text(stderr, encoding="utf-8")
        execution = {
            "started_at_utc": started,
            "duration_seconds": round(duration, 3),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "cwd": str(cwd),
            "argv": args,
            "stdout_log": str((task_output / "stdout.log").relative_to(output_root)),
            "stderr_log": str((task_output / "stderr.log").relative_to(output_root)),
        }
        reference["execution"] = execution
        reference["status"] = "unverified"
        result = {
            "task_id": task_id,
            "status": "failed",
            "execution": execution,
            "metric_checks": [],
            "artifact_checks": [],
        }
        if timed_out or exit_code != 0:
            reference["result_summary"] = "Reference execution timed out or returned non-zero."
            updated["reference"] = reference
            return updated, result

        metrics_file_rel = _safe_relative(
            str(reference.get("metrics_file") or "summary.json"), "reference.metrics_file",
        )
        metrics_path = task_output / metrics_file_rel
        if not metrics_path.is_file():
            reference["result_summary"] = f"Missing metrics file: {metrics_file_rel}"
            updated["reference"] = reference
            return updated, result
        metrics_document = read_json(metrics_path)
        leaves = _flatten_json(metrics_document)
        metrics = reference.get("metrics") or []
        metric_checks: list[dict[str, Any]] = []
        targets_complete = bool(metrics)
        all_metrics_ok = bool(metrics)
        normalized_metrics: list[dict[str, Any]] = []
        for raw_metric in metrics:
            metric = dict(raw_metric)
            try:
                json_path, observed = _resolve_metric(metric, leaves)
                metric["json_path"] = json_path
                metric["observed"] = observed
                if "target" not in metric:
                    if accept_observed_targets:
                        metric["target"] = observed
                        ok, detail = True, "observed value accepted as reviewed target"
                    else:
                        targets_complete = False
                        ok, detail = False, "target missing; observed value requires review"
                else:
                    ok, detail = _compare(observed, metric["target"], metric.get("tolerance", 0))
                all_metrics_ok = all_metrics_ok and ok
                metric_checks.append({
                    "name": metric.get("name"), "json_path": json_path,
                    "ok": ok, "detail": detail,
                })
            except (KeyError, TypeError, ValueError) as exc:
                all_metrics_ok = False
                targets_complete = False
                metric_checks.append({"name": metric.get("name"), "ok": False, "detail": str(exc)})
            normalized_metrics.append(metric)
        reference["metrics"] = normalized_metrics

        artifacts = reference.get("artifacts") or [metrics_file_rel]
        artifact_checks: list[dict[str, Any]] = []
        all_artifacts_ok = True
        for raw_artifact in artifacts:
            rel = _safe_relative(str(raw_artifact), "reference.artifacts")
            exists = (task_output / rel).is_file()
            artifact_checks.append({"path": rel, "ok": exists})
            all_artifacts_ok = all_artifacts_ok and exists
        verified = all_metrics_ok and targets_complete and all_artifacts_ok
        reference["status"] = "verified" if verified else "partial"
        reference["result_summary"] = (
            "Reference command completed and all declared metrics/artifacts were verified."
            if verified
            else "Reference command completed, but metrics or artifacts require review."
        )
        result.update({
            "status": reference["status"],
            "metrics_file": metrics_file_rel,
            "metric_checks": metric_checks,
            "artifact_checks": artifact_checks,
        })
        updated["reference"] = reference
        return updated, result

    updated_by_id: dict[str, dict[str, Any]] = {}
    results_by_id: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(execute, row): str(row.get("task_id") or "") for row in candidates}
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                updated, result = future.result()
                updated_by_id[task_id] = updated
                results_by_id[task_id] = result
                if result["status"] == "failed":
                    report.error("reference_execution_failed", task_id)
                elif result["status"] != "verified":
                    report.warn("reference_needs_review", task_id)
            except Exception as exc:  # noqa: BLE001 - preserve other batch results
                report.error("reference_runner_error", f"{task_id}: {exc}")
                failed = deepcopy(next(row for row in candidates if row.get("task_id") == task_id))
                failed.setdefault("reference", {})["status"] = "unverified"
                updated_by_id[task_id] = failed
                results_by_id[task_id] = {"task_id": task_id, "status": "failed", "error": str(exc)}
    updated_rows = [updated_by_id.get(str(row.get("task_id") or ""), row) for row in candidates]
    result_rows = [results_by_id[str(row.get("task_id") or "")] for row in candidates]
    write_jsonl(updated_candidates_path, updated_rows)
    write_jsonl(results_path, result_rows)
    report.stats.update({
        "candidates": len(candidates),
        "verified": sum(row.get("status") == "verified" for row in result_rows),
        "partial": sum(row.get("status") == "partial" for row in result_rows),
        "failed": sum(row.get("status") == "failed" for row in result_rows),
        "accept_observed_targets": accept_observed_targets,
    })
    write_json(results_path.with_suffix(".report.json"), report.to_dict())
    return report
