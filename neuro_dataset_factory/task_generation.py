"""Batch LLM generation of grounded end-to-end task candidates."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neuro_dataset_factory.contracts import END_TO_END_STAGES, ValidationReport, stable_id
from neuro_dataset_factory.llm_client import OpenAICompatibleJSONClient
from neuro_dataset_factory.packages import _candidate
from neuro_dataset_factory.storage import read_jsonl, write_json, write_jsonl

PROMPT_VERSION = "end-to-end-candidate-v1"
SYSTEM_PROMPT = """You design rigorous end-to-end neuroscience data-analysis tasks.
Return JSON only. Every task must require planning, inspection of real provided data,
coding and execution, quantitative/statistical analysis, a scientifically legible
visualization, and calibrated neuroscience interpretation. Tasks sharing a dataset
must address genuinely different questions or methods, not paraphrases. Never expose
reference target values or the ground-truth conclusion in the public query. Do not
invent files, variables, sample sizes, results, licenses, or data capabilities.
Reference status must remain unverified until code has actually run."""


def _paper_contexts(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    contexts: dict[str, str] = {}
    for row in read_jsonl(path):
        paper_id = str(row.get("paper_id") or "").strip()
        text = str(row.get("text") or row.get("content") or row.get("abstract") or "").strip()
        if paper_id and text:
            contexts[paper_id] = text
    return contexts


def _safe_tag(value: Any, index: int) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_.-")
    return tag[:48] or f"Analysis_{index:02d}"


def _prompt(
    dataset: dict[str, Any],
    link: dict[str, Any],
    manifest: list[dict[str, Any]],
    paper_context: str,
    tasks_per_link: int,
    max_manifest_files: int,
    max_context_chars: int,
) -> str:
    visible_manifest = manifest[:max_manifest_files]
    payload = {
        "dataset": {
            key: dataset.get(key)
            for key in ("dataset_id", "name", "version", "description", "license", "science", "file_count", "total_bytes", "role_counts")
        },
        "paper_link": {
            key: link.get(key)
            for key in ("paper_id", "title", "doi", "scientific_question", "target_result", "gt_type", "required_files", "evidence")
        },
        "paper_context": paper_context[:max_context_chars],
        "manifest_sample": [
            {key: item.get(key) for key in ("path", "bytes", "extension", "role")}
            for item in visible_manifest
        ],
        "manifest_sample_is_complete": len(visible_manifest) == len(manifest),
    }
    return f"""Generate exactly {tasks_per_link} distinct task candidates from this evidence.

Input evidence:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return this shape:
{{
  "candidates": [
    {{
      "task_tag": "Analysis_01",
      "query": "solver-visible end-to-end question without answer values",
      "deliverable": "explicit code, tables, figures, machine-readable summary, and report",
      "required_files": ["paths copied exactly from manifest_sample or paper_link.required_files"],
      "construction_mode": "reproduce or extend",
      "reference": {{
        "status": "unverified",
        "command": "empty unless a real existing reference program is known",
        "metrics_file": "summary.json",
        "artifacts": ["summary.json"],
        "metrics": [{{"name": "machine_readable_metric_name", "json_path": "optional.path", "tolerance": 0.0}}]
      }},
      "rubric": {{
        "reason": "why this is a valuable real-data task",
        "core_conclusion": "evaluator-only conclusion grounded in supplied evidence",
        "scoring_items": [
          {{"point": 20, "criterion": "criterion", "keywords": ["keyword"]}}
        ],
        "acceptable_deviations": [],
        "scoring_instructions": "auditable scoring instructions"
      }}
    }}
  ]
}}

Requirements: 3-8 scoring items; positive integer points sum to exactly 100;
one item explicitly rejects fabricated/synthetic replacement data. Every required
file must occur in the supplied manifest sample or paper_link.required_files.
"""


def _normalize_generated(
    raw: dict[str, Any],
    *,
    index: int,
    dataset: dict[str, Any],
    link: dict[str, Any],
    manifest_paths: set[str],
    model: str,
    cache_meta: dict[str, Any],
) -> dict[str, Any]:
    task_tag = _safe_tag(raw.get("task_tag"), index)
    task_id = "NEURO_" + stable_id(
        "task", dataset["dataset_id"], link["paper_id"], task_tag,
    ).split("_", 1)[1].upper()
    requested = [str(path) for path in raw.get("required_files") or []]
    required = [path for path in requested if path in manifest_paths]
    if not required:
        required = [path for path in link.get("required_files") or [] if path in manifest_paths]
    if not required:
        raise ValueError("candidate has no valid required_files")
    reference = dict(raw.get("reference") or {})
    reference["status"] = "unverified"
    reference.setdefault("command", "")
    reference.setdefault("metrics_file", "summary.json")
    reference.setdefault("artifacts", [reference["metrics_file"]])
    reference.setdefault("metrics", [])
    candidate = {
        "task_id": task_id,
        "task_tag": task_tag,
        "task_format": "end_to_end",
        "workflow_stages": list(END_TO_END_STAGES),
        "dataset_id": dataset["dataset_id"],
        "link_id": link["link_id"],
        "query": raw.get("query"),
        "deliverable": raw.get("deliverable"),
        "required_files": required,
        "ai_capability": ["Planning", "Coding", "Statistical reasoning", "Neuroscience interpretation"],
        "construction_mode": raw.get("construction_mode") or "reproduce",
        "leakage_reviewed": False,
        "reference": reference,
        "rubric": raw.get("rubric") or {},
        "generation": {
            "provider": "openai_compatible",
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "cache_key": cache_meta["cache_key"],
            "cache_hit": cache_meta["cache_hit"],
            "source_link_id": link["link_id"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    _candidate(candidate)
    return candidate


def generate_task_candidates(
    registry_path: Path,
    links_path: Path,
    output_path: Path,
    cache_dir: Path,
    *,
    model: str,
    tasks_per_link: int = 4,
    jobs: int = 4,
    timeout: int = 180,
    max_retries: int = 3,
    max_tokens: int = 12000,
    temperature: float = 0.2,
    max_manifest_files: int = 500,
    max_context_chars: int = 20000,
    paper_contexts_path: Path | None = None,
    resume: bool = True,
    refresh_cache: bool = False,
    client: OpenAICompatibleJSONClient | None = None,
    use_env_proxy: bool = False,
) -> ValidationReport:
    report = ValidationReport()
    if tasks_per_link <= 0 or jobs <= 0:
        report.error("invalid_generation_limits", f"tasks_per_link={tasks_per_link}, jobs={jobs}")
        return report
    datasets = {row["dataset_id"]: row for row in read_jsonl(registry_path)}
    links = read_jsonl(links_path)
    contexts = _paper_contexts(paper_contexts_path)
    existing = read_jsonl(output_path) if resume and output_path.is_file() else []
    completed_links = {
        str((row.get("generation") or {}).get("source_link_id") or "")
        for row in existing
    }
    by_link: dict[str, list[dict[str, Any]]] = {}
    for row in existing:
        source_link = str((row.get("generation") or {}).get("source_link_id") or "")
        if source_link:
            by_link.setdefault(source_link, []).append(row)
    client = client or OpenAICompatibleJSONClient(
        model=model, cache_dir=cache_dir, timeout=timeout, max_retries=max_retries,
        use_env_proxy=use_env_proxy,
    )

    def generate(link: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        link_id = str(link["link_id"])
        dataset = datasets.get(link["dataset_id"])
        if dataset is None:
            raise ValueError(f"unknown dataset_id: {link['dataset_id']}")
        manifest_path = registry_path.parent / dataset["manifest_path"]
        manifest = read_jsonl(manifest_path)
        prompt = _prompt(
            dataset, link, manifest, contexts.get(str(link["paper_id"]), ""),
            tasks_per_link, max_manifest_files, max_context_chars,
        )
        response, cache_meta = client.chat_json(
            system=SYSTEM_PROMPT,
            user=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_version=PROMPT_VERSION,
            refresh_cache=refresh_cache,
        )
        raw_candidates = response.get("candidates") or []
        if not isinstance(raw_candidates, list) or len(raw_candidates) != tasks_per_link:
            raise ValueError(
                f"expected {tasks_per_link} candidates, got {len(raw_candidates) if isinstance(raw_candidates, list) else 'non-list'}"
            )
        manifest_paths = {str(item["path"]) for item in manifest}
        normalized = [
            _normalize_generated(
                raw, index=index, dataset=dataset, link=link,
                manifest_paths=manifest_paths, model=model, cache_meta=cache_meta,
            )
            for index, raw in enumerate(raw_candidates, 1)
            if isinstance(raw, dict)
        ]
        if len(normalized) != tasks_per_link:
            raise ValueError("one or more generated candidates are not objects")
        return link_id, normalized

    eligible: list[dict[str, Any]] = []
    for link in links:
        link_id = str(link.get("link_id") or "")
        if resume and link_id in completed_links and len(by_link.get(link_id, [])) >= tasks_per_link:
            continue
        if link.get("data_support") != "yes" or link.get("dataset_version_match") is not True:
            report.warn("link_not_generation_ready", link_id)
            continue
        eligible.append(link)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(generate, link): str(link["link_id"]) for link in eligible}
        for future in as_completed(futures):
            link_id = futures[future]
            try:
                completed_link, candidates = future.result()
                by_link[completed_link] = candidates
                checkpoint = [row for key in sorted(by_link) for row in by_link[key]]
                write_jsonl(output_path, checkpoint)
            except Exception as exc:  # noqa: BLE001 - preserve partial batch output
                report.error("llm_generation_failed", f"{link_id}: {exc}")
    rows = [row for key in sorted(by_link) for row in by_link[key]]
    write_jsonl(output_path, rows)
    report.stats.update({
        "links_seen": len(links),
        "links_submitted": len(eligible),
        "candidates": len(rows),
        "api_calls": client.call_count,
        "cache_hits": client.cache_hits,
        "model": model,
        "prompt_version": PROMPT_VERSION,
    })
    write_json(output_path.with_suffix(".generation_report.json"), report.to_dict())
    return report
