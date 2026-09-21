"""Generate and package tasks whose verified data payload is awaiting download."""

from __future__ import annotations

import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neuro_dataset_factory.contracts import END_TO_END_STAGES, SCHEMA_VERSION, ValidationReport, stable_id
from neuro_dataset_factory.llm_client import OpenAICompatibleJSONClient
from neuro_dataset_factory.packages import _candidate, _safe_relative
from neuro_dataset_factory.storage import read_json, read_jsonl, write_json, write_jsonl


PROMPT_VERSION = "remote-data-end-to-end-v2-title-safe"
# Institutional repositories can expose thousands of deeply nested files.  A
# smaller, evenly sampled view keeps the request inside stricter compatible-API
# gateways while the full manifest remains available for post-generation path
# validation and acquisition.
MAX_PROMPT_MANIFEST_FILES = 48
SYSTEM_PROMPT = """You design rigorous end-to-end neuroscience data-analysis tasks.
The remote dataset record and file manifest have been verified, but payload files have
not yet been downloaded or inspected. Return JSON only. Design tasks that become
executable after the named files are materialized at their exact target paths. Require
planning, real-data inspection, coding and execution, statistics, visualization, and
scientific interpretation. Do not invent file schemas, variable names, sample sizes,
results, or download paths. If an archive's internal schema is unknown, explicitly make
schema inspection and adaptive loading part of the task. Never mark a reference as
verified and never expose evaluator-only conclusions in the public query. The source
article is evaluator-only provenance: never name or cite its title, DOI, authors,
journal, or URL, and never ask the solver to identify or retrieve the article."""


def _safe_tag(value: Any, index: int) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_.-")
    return tag[:48] or f"Analysis_{index:02d}"


def _contexts(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["paper_id"]): row for row in read_jsonl(path)}


def _manifest_paths(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
    manifest: list[dict[str, Any]] = []
    valid: set[str] = set()
    for record in records:
        dataset_id = str(record["dataset_id"])
        for item in record.get("files", []):
            try:
                remote_path = _safe_relative(str(item.get("path") or "download"), "remote file path")
                target = _safe_relative(f"data/{dataset_id}/{remote_path}", "target data path")
            except ValueError:
                continue
            manifest.append({
                "dataset_id": dataset_id,
                "path": target,
                "bytes": int(item.get("bytes") or 0),
                "checksum": str(item.get("checksum") or ""),
                "download_url": str(item.get("download_url") or ""),
            })
            valid.add(target)
    return manifest, valid


def _prompt_manifest(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep large repository listings usable without weakening path validation.

    The full file manifest remains in the acquisition handoff and is used by
    ``_normalize``.  The model only needs a bounded set of exact paths from
    which to choose task inputs; checksums and download URLs add substantial
    prompt volume without helping that choice.
    """
    compact = [{
        "dataset_id": item["dataset_id"],
        "path": item["path"],
        "bytes": item["bytes"],
    } for item in manifest]
    if len(compact) <= MAX_PROMPT_MANIFEST_FILES:
        return compact

    # Preserve coverage across the repository listing instead of taking only
    # one leading directory.  The stable evenly-spaced sample is deterministic
    # and includes both endpoints.
    last = len(compact) - 1
    indices = {
        round(index * last / (MAX_PROMPT_MANIFEST_FILES - 1))
        for index in range(MAX_PROMPT_MANIFEST_FILES)
    }
    return [compact[index] for index in sorted(indices)]


def _prompt(
    paper: dict[str, Any],
    datasets: list[dict[str, Any]],
    context: dict[str, Any],
    *,
    tasks_per_paper: int,
    max_context_chars: int,
) -> str:
    manifest, _ = _manifest_paths(datasets)
    prompt_manifest = _prompt_manifest(manifest)
    title = str(paper.get("title") or "").strip()
    context_text = str(context.get("text") or context.get("abstract") or "")[:max_context_chars]
    if title:
        context_text = re.sub(re.escape(title), "[ARTICLE TITLE WITHHELD]", context_text, flags=re.IGNORECASE)
    evidence = {
        "paper": {
            "paper_id": paper.get("paper_id"),
            "source_identity": "withheld_from_solver",
        },
        "paper_context": context_text,
        "verified_remote_datasets": [{
            "dataset_id": row.get("dataset_id"),
            "repository": row.get("repository"),
            "record_id": row.get("record_id"),
            "version": row.get("version"),
            "license": row.get("license"),
            "target_relative_path": row.get("target_relative_path"),
            "file_count": row.get("file_count"),
            "total_bytes": row.get("total_bytes"),
        } for row in datasets],
        "verified_file_manifest": prompt_manifest,
        "verified_file_manifest_scope": {
            "files_in_full_acquisition_manifest": len(manifest),
            "files_shown_to_task_designer": len(prompt_manifest),
            "note": "A deterministic prompt-sized subset of exact verified paths is shown when the full listing is large.",
        },
    }
    return f"""Generate exactly {tasks_per_paper} genuinely distinct end-to-end tasks.

Evidence:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Return:
{{"candidates":[{{
  "task_tag":"Analysis_01",
  "query":"solver-visible question that uses exact target paths but contains no answer",
  "deliverable":"analysis code, machine-readable summary, figures, and report",
  "required_files":["exact paths copied from verified_file_manifest"],
  "construction_mode":"reproduce or extend",
  "reference":{{
    "status":"awaiting_data_download",
    "command":"",
    "metrics_file":"summary.json",
    "artifacts":["summary.json","figure.png"],
    "metrics":[{{"name":"metric_name","tolerance":0.0}}]
  }},
  "rubric":{{
    "reason":"value of this real-data task",
    "core_conclusion":"evaluator-only claim supported by supplied paper evidence; no invented values",
    "scoring_items":[{{"point":20,"criterion":"criterion","keywords":["keyword"]}}],
    "acceptable_deviations":[],
    "scoring_instructions":"auditable scoring instructions"
  }}
}}]}}

Rules: required_files must contain one or more paths copied exactly from
verified_file_manifest; use only files
needed for the task. Include 3-8 positive-integer scoring items totaling exactly 100,
including one item that rejects fabricated/synthetic replacement data. Each task must
cover all seven workflow stages. Reference status must remain awaiting_data_download.
Do not include or closely restate the source paper title, DOI, authors, journal, or URL
in query, deliverable, task_tag, or any solver-visible text.
"""


def _normalize(
    raw: dict[str, Any],
    *,
    index: int,
    paper: dict[str, Any],
    records: list[dict[str, Any]],
    model: str,
    cache_meta: dict[str, Any],
) -> dict[str, Any]:
    manifest, valid_paths = _manifest_paths(records)
    requested = [str(path) for path in raw.get("required_files") or []]
    required = [path for path in requested if path in valid_paths]
    if not required:
        raise ValueError("candidate has no exact required_files from verified manifest")
    dataset_ids = []
    for item in manifest:
        if item["path"] in required and item["dataset_id"] not in dataset_ids:
            dataset_ids.append(item["dataset_id"])
    tag = _safe_tag(raw.get("task_tag"), index)
    task_id = "NEURO_REMOTE_" + stable_id("task", paper["paper_id"], tag).split("_", 1)[1].upper()
    reference = dict(raw.get("reference") or {})
    reference.update({"status": "awaiting_data_download", "command": ""})
    reference.setdefault("metrics_file", "summary.json")
    reference.setdefault("artifacts", [reference["metrics_file"]])
    reference.setdefault("metrics", [])
    link_id = stable_id("remote_link", paper["paper_id"], dataset_ids)
    query = str(raw.get("query") or "").strip()
    title = str(paper.get("title") or "").strip()
    # Exact paths are required for execution and can occasionally contain a paper
    # title chosen by the upstream repository.  Check prose after removing code
    # spans so those paths do not create a false positive.
    query_prose = re.sub(r"`[^`]*`", " ", query)
    normalized_query = re.sub(r"[^a-z0-9]+", " ", query_prose.casefold()).strip()
    normalized_title = re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()
    if normalized_title and normalized_title in normalized_query:
        raise ValueError("candidate exposes the source paper title in solver-visible query text")
    candidate = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "task_tag": tag,
        "task_format": "end_to_end",
        "workflow_stages": list(END_TO_END_STAGES),
        "dataset_id": dataset_ids[0],
        "dataset_ids": dataset_ids,
        "paper_id": paper["paper_id"],
        "link_id": link_id,
        "query": query,
        "deliverable": raw.get("deliverable"),
        "deliverables": raw.get("deliverables") or [raw.get("deliverable")],
        "required_files": required,
        "ai_capability": ["Planning", "Coding", "Statistical reasoning", "Neuroscience interpretation"],
        "construction_mode": raw.get("construction_mode") or "reproduce",
        "leakage_reviewed": False,
        "trajectory_verdict": "awaiting_data_download",
        "reference": reference,
        "rubric": raw.get("rubric") or {},
        "generation": {
            "provider": "openai_compatible",
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "cache_key": cache_meta["cache_key"],
            "cache_hit": cache_meta["cache_hit"],
            "source_link_id": link_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "package_status": "awaiting_data_download",
    }
    _candidate(candidate)
    return candidate


def generate_remote_task_candidates(
    handoff_root: Path,
    contexts_path: Path,
    output_path: Path,
    cache_dir: Path,
    *,
    model: str,
    tasks_per_paper: int = 4,
    jobs: int = 4,
    timeout: int = 180,
    max_retries: int = 3,
    max_tokens: int = 12000,
    temperature: float = 0.2,
    max_context_chars: int = 30000,
    max_papers: int | None = None,
    resume: bool = True,
    refresh_cache: bool = False,
    use_env_proxy: bool = False,
    client: OpenAICompatibleJSONClient | None = None,
) -> ValidationReport:
    report = ValidationReport()
    mapping = read_json(handoff_root / "paper_data_mapping.json").get("papers", {})
    queue = {row["dataset_id"]: row for row in read_jsonl(handoff_root / "acquisition_queue.jsonl")}
    contexts = _contexts(contexts_path)
    existing = read_jsonl(output_path) if resume and output_path.is_file() else []
    by_paper: dict[str, list[dict[str, Any]]] = {}
    for row in existing:
        by_paper.setdefault(str(row.get("paper_id") or ""), []).append(row)
    llm = client or OpenAICompatibleJSONClient(
        model=model, cache_dir=cache_dir, timeout=timeout, max_retries=max_retries,
        use_env_proxy=use_env_proxy,
    )

    def generate(paper: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        paper_id = str(paper["paper_id"])
        records = [queue[data_id] for data_id in paper["dataset_ids"] if data_id in queue]
        if not records:
            raise ValueError("paper has no verified remote dataset record")
        response, cache_meta = llm.chat_json(
            system=SYSTEM_PROMPT,
            user=_prompt(paper, records, contexts.get(paper_id, {}), tasks_per_paper=tasks_per_paper, max_context_chars=max_context_chars),
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_version=PROMPT_VERSION,
            refresh_cache=refresh_cache,
        )
        raw = response.get("candidates") or []
        if not isinstance(raw, list) or len(raw) != tasks_per_paper:
            raise ValueError(f"expected {tasks_per_paper} candidates, got {len(raw) if isinstance(raw, list) else 'non-list'}")
        normalized = [
            _normalize(item, index=index, paper=paper, records=records, model=model, cache_meta=cache_meta)
            for index, item in enumerate(raw, 1) if isinstance(item, dict)
        ]
        if len(normalized) != tasks_per_paper:
            raise ValueError("one or more candidates are not objects")
        return paper_id, normalized

    eligible = [
        paper for paper_id, paper in mapping.items()
        if paper_id in contexts and not (resume and len(by_paper.get(paper_id, [])) >= tasks_per_paper)
    ]
    if max_papers is not None:
        eligible = eligible[:max_papers]
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(generate, paper): paper["paper_id"] for paper in eligible}
        for future in as_completed(futures):
            paper_id = futures[future]
            try:
                key, candidates = future.result()
                by_paper[key] = candidates
                write_jsonl(output_path, [row for paper_key in sorted(by_paper) for row in by_paper[paper_key]])
            except Exception as exc:  # noqa: BLE001 - retain successful checkpoint rows
                report.error("remote_task_generation_failed", f"{paper_id}: {exc}")
    rows = [row for paper_key in sorted(by_paper) for row in by_paper[paper_key]]
    write_jsonl(output_path, rows)
    report.stats.update({
        "papers_seen": len(mapping),
        "papers_submitted": len(eligible),
        "candidates": len(rows),
        "api_calls": llm.call_count,
        "cache_hits": llm.cache_hits,
        "model": model,
        "prompt_version": PROMPT_VERSION,
    })
    write_json(output_path.with_suffix(".generation_report.json"), report.to_dict())
    return report


def build_provisional_remote_packages(
    candidates_path: Path,
    handoff_root: Path,
    output_root: Path,
) -> ValidationReport:
    report = ValidationReport()
    queue = {row["dataset_id"]: row for row in read_jsonl(handoff_root / "acquisition_queue.jsonl")}
    mapping = read_json(handoff_root / "paper_data_mapping.json").get("papers", {})
    built = 0
    for raw in read_jsonl(candidates_path):
        try:
            candidate = _candidate(raw)
        except (TypeError, ValueError) as exc:
            report.error("bad_remote_candidate", f"{raw.get('task_id')}: {exc}")
            continue
        task_id = candidate["task_id"]
        dataset_ids = [str(item) for item in raw.get("dataset_ids") or [candidate["dataset_id"]]]
        records = [queue[item] for item in dataset_ids if item in queue]
        if len(records) != len(dataset_ids):
            report.error("missing_remote_dataset", task_id)
            continue
        valid_paths = _manifest_paths(records)[1]
        if any(path not in valid_paths for path in candidate["required_files"]):
            report.error("remote_required_file_missing", task_id)
            continue
        package_root = output_root / task_id
        if package_root.exists():
            shutil.rmtree(package_root)
        task_root = package_root / "task"
        hidden = task_root / "target_study"
        data_dir = task_root / "data"
        hidden.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        paper = mapping.get(str(raw.get("paper_id") or ""), {})
        locator_payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "awaiting_data_download",
            "note": "This file is a locator only; dataset payload files are not present yet.",
            "datasets": records,
            "required_files_after_materialization": candidate["required_files"],
        }
        write_json(data_dir / "REMOTE_DATA_LOCATOR.json", locator_payload)
        info = {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "task_tag": candidate["task_tag"],
            "task_format": "end_to_end",
            "workflow_stages": candidate["workflow_stages"],
            "paper_id": raw.get("paper_id"),
            "paper_display_name": paper.get("title", ""),
            "dataset_ids": dataset_ids,
            "package_status": "awaiting_data_download",
            "query": candidate["query"],
            "provided_data": [{
                "dataset_id": record["dataset_id"],
                "target_relative_path": record["target_relative_path"],
                "locator": "data/REMOTE_DATA_LOCATOR.json",
                "payload_present": False,
            } for record in records],
            "required_files": candidate["required_files"],
            "deliverable": candidate["deliverable"],
            "deliverables": candidate["deliverables"],
            "real_data_grounding": "Task becomes executable only after the verified remote payload is materialized; synthetic replacement data is forbidden.",
        }
        rubric = candidate["rubric"]
        checklist = {"tag": candidate["task_tag"], "rubrics": {"score_100": {
            "reason": rubric["reason"],
            "rubric": {
                "GT core conclusion": rubric["core_conclusion"],
                "Scoring items": rubric["scoring_items"],
                "Acceptable deviations": rubric["acceptable_deviations"],
                "Scoring instructions": rubric["scoring_instructions"],
            },
        }}}
        ground_truth = {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "status": "awaiting_data_download",
            "core_conclusion": rubric["core_conclusion"],
            "reference": candidate["reference"],
        }
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "paper": paper,
            "remote_datasets": records,
            "generation": candidate["generation"],
            "quality_gate": {
                "remote_locator_verified": True,
                "payload_downloaded": False,
                "reference_executed": False,
                "canonical_release_ready": False,
            },
        }
        write_json(task_root / "task_info.json", info)
        write_json(hidden / "checklist.json", checklist)
        write_json(hidden / "ground_truth.json", ground_truth)
        write_json(hidden / "provenance.json", provenance)
        write_json(package_root / "package_status.json", {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "status": "awaiting_data_download",
            "canonical_package": False,
            "blocking_steps": ["download_payload", "profile_and_hash", "run_reference", "canonical_validation"],
        })
        write_json(package_root / "validation_report.json", {
            "schema_version": SCHEMA_VERSION,
            "ok": True,
            "package_class": "provisional_remote_data",
            "canonical_release_ready": False,
            "issues": [],
        })
        built += 1
    output_root.mkdir(parents=True, exist_ok=True)
    report.stats.update({"candidates": len(read_jsonl(candidates_path)), "built": built, "canonical_release_ready": 0})
    write_json(output_root / "build_report.json", report.to_dict())
    return report


def audit_provisional_remote_packages(output_root: Path) -> ValidationReport:
    report = ValidationReport()
    packages = sorted(path for path in output_root.iterdir() if path.is_dir()) if output_root.is_dir() else []
    papers: set[str] = set()
    datasets: set[str] = set()
    for package in packages:
        try:
            status = read_json(package / "package_status.json")
            info = read_json(package / "task/task_info.json")
            locator = read_json(package / "task/data/REMOTE_DATA_LOCATOR.json")
            checklist = read_json(package / "task/target_study/checklist.json")
            ground_truth = read_json(package / "task/target_study/ground_truth.json")
            provenance = read_json(package / "task/target_study/provenance.json")
        except (OSError, json.JSONDecodeError) as exc:
            report.error("invalid_provisional_package", f"{package.name}: {exc}")
            continue
        if status.get("status") != "awaiting_data_download" or status.get("canonical_package") is not False:
            report.error("bad_provisional_status", package.name)
        if info.get("package_status") != "awaiting_data_download" or ground_truth.get("status") != "awaiting_data_download":
            report.error("status_mismatch", package.name)
        if provenance.get("quality_gate", {}).get("canonical_release_ready") is not False:
            report.error("premature_release_ready", package.name)
        data_files = [path for path in (package / "task/data").rglob("*") if path.is_file()]
        if [path.name for path in data_files] != ["REMOTE_DATA_LOCATOR.json"]:
            report.error("unexpected_data_payload", package.name)
        valid: set[str] = set()
        for record in locator.get("datasets", []):
            datasets.add(str(record.get("dataset_id") or ""))
            valid.update(_manifest_paths([record])[1])
            if record.get("payload_downloaded") is not False or record.get("payload_hash_verified") is not False:
                report.error("payload_flag_mismatch", f"{package.name}: {record.get('dataset_id')}")
        required = set(info.get("required_files") or [])
        if not required or not required.issubset(valid):
            report.error("unresolved_required_files", package.name)
        try:
            items = checklist["rubrics"]["score_100"]["rubric"]["Scoring items"]
            if sum(int(item.get("point", 0)) for item in items) != 100:
                report.error("rubric_points_not_100", package.name)
        except (KeyError, TypeError, ValueError):
            report.error("invalid_provisional_rubric", package.name)
        conclusion = str(ground_truth.get("core_conclusion") or "").strip().casefold()
        query = str(info.get("query") or "").casefold()
        if len(conclusion) >= 30 and conclusion in query:
            report.error("ground_truth_leak", package.name)
        papers.add(str(info.get("paper_id") or ""))
    report.stats.update({
        "packages": len(packages),
        "papers": len(papers - {""}),
        "datasets_referenced": len(datasets - {""}),
        "canonical_release_ready": 0,
    })
    write_json(output_root / "audit_report.json", report.to_dict())
    return report
