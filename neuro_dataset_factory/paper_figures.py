"""Extract paper figures, map them to tasks, and materialize gt_figure assets."""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from neuro_dataset_factory.contracts import SCHEMA_VERSION, ValidationReport
from neuro_dataset_factory.llm_client import OpenAICompatibleJSONClient
from neuro_dataset_factory.storage import read_json, read_jsonl, write_json, write_jsonl


PROMPT_VERSION = "paper-figure-task-mapping-v1"
SYSTEM_PROMPT = """Map neuroscience analysis tasks to figures from their source paper.
Use only the supplied figure objects and captions. Return JSON only. A match is valid
only when the paper figure directly supports the task's target result or scientific
claim. Do not choose a merely topically related image. For sensitivity analyses,
extensions, or tasks with no direct paper figure, return no_exact_match."""

_FIGURE_LABEL = re.compile(
    r"\b(?:(?:Extended\s+Data\s+)?Fig(?:ure)?\.?\s*(?:S?\d+[A-Za-z]?|[A-Za-z]\d+)|Supplementary\s+Fig(?:ure)?\.?\s*\w+)",
    re.IGNORECASE,
)
_CAPTION_HEADING = re.compile(
    r"(?im)^[ \t]*(?P<label>(?:(?:Extended\s+Data\s+)?Fig(?:ure)?\.?\s*"
    r"(?:S?\d+[A-Za-z]?|[A-Za-z]\d+)|Supplementary\s+Fig(?:ure)?\.?\s*\w+))"
    r"(?=\s*(?:\||:|\s))",
)
_MAIN_CAPTION_HEADING = re.compile(
    r"(?im)^[ \t]*(?P<label>Fig(?:ure)?\.?\s*(?P<number>\d+))"
    r"(?=\s*(?:\||:)|\s+[A-Z]|[A-Z])",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_figure_candidates_from_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    caption_candidates: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks):
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = str(block.get("content") or "").strip()
        matches = list(_MAIN_CAPTION_HEADING.finditer(text))
        for match_index, match in enumerate(matches):
            end = matches[match_index + 1].start() if match_index + 1 < len(matches) else len(text)
            caption_candidates.append({
                "block_index": block_index,
                "figure_number": int(match.group("number")),
                "figure_label": match.group("label").strip(),
                "caption": text[match.start():end].strip(),
            })

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(blocks):
        if not isinstance(item, dict) or item.get("type") != "image":
            continue
        image_object = str(item.get("content") or "").strip()
        if not image_object or image_object in seen:
            continue
        seen.add(image_object)
        ordinal = len(candidates) + 1
        embedded_caption = str(item.get("img_caption") or "").strip()
        previous_text = ""
        if index > 0 and isinstance(blocks[index - 1], dict) and blocks[index - 1].get("type") == "text":
            previous_text = str(blocks[index - 1].get("content") or "").strip()
        # SJT reading order can place a caption immediately before or after its image,
        # and occasionally splits surrounding body text across that image. Main-paper
        # image objects remain in figure order, so first prefer a caption with the same
        # figure number and then the closest text block (ties prefer the following block).
        matching_captions = [row for row in caption_candidates if row["figure_number"] == ordinal]
        pool = matching_captions or caption_candidates
        caption_match = min(
            pool,
            key=lambda row: (abs(int(row["block_index"]) - index), int(row["block_index"]) < index),
            default=None,
        )
        extracted_caption = str((caption_match or {}).get("caption") or "")
        caption = embedded_caption or extracted_caption
        nearby: list[str] = []
        for neighbor in blocks[max(0, index - 2): min(len(blocks), index + 3)]:
            if isinstance(neighbor, dict) and neighbor.get("type") == "text":
                text = str(neighbor.get("content") or "").strip()
                if text:
                    nearby.append(text[:2500])
        context = caption or "\n".join(nearby)
        label_match = _FIGURE_LABEL.search(embedded_caption) if embedded_caption else None
        trusted_caption_match = bool(embedded_caption or matching_captions)
        figure_label = str((caption_match or {}).get("figure_label") or "") if matching_captions else ""
        if label_match is None:
            matches = list(_FIGURE_LABEL.finditer(previous_text))
            label_match = matches[-1] if matches else _FIGURE_LABEL.search(context)
        if not figure_label and label_match:
            figure_label = label_match.group(0).strip()
        candidates.append({
            "image_object": image_object,
            "figure_label": figure_label,
            "caption": caption,
            "caption_context": context[:5000],
            "block_index": index,
            "figure_ordinal": ordinal,
            "caption_block_index": (caption_match or {}).get("block_index"),
            "caption_match_method": (
                "embedded" if embedded_caption else
                "same_figure_number_nearest_block" if matching_captions else
                "nearest_caption_fallback" if caption_match else
                "neighbor_context_only"
            ),
            "mapping_eligible": trusted_caption_match,
        })
    return candidates


def extract_s3_paper_figures(
    verified_path: Path,
    output_path: Path,
    *,
    s3_uri: str,
    endpoint_url: str,
    access_key: str | None = None,
    secret_key: str | None = None,
) -> ValidationReport:
    try:
        import boto3  # type: ignore
        from botocore.config import Config  # type: ignore
    except ImportError as exc:
        raise RuntimeError("boto3 is required for S3 figure extraction") from exc
    if not s3_uri.startswith("s3://") or "/" not in s3_uri[5:]:
        raise ValueError(f"invalid S3 URI: {s3_uri}")
    bucket, key = s3_uri[5:].split("/", 1)
    papers = read_jsonl(verified_path)
    targets = {
        str((row.get("corpus_source") or {}).get("track_id") or ""): row
        for row in papers if row.get("status") == "verified_downloadable"
    }
    client = boto3.client(
        "s3", endpoint_url=endpoint_url,
        aws_access_key_id=access_key, aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}, proxies={}),
    )
    report = ValidationReport()
    found: dict[str, dict[str, Any]] = {}
    rows_seen = 0
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    try:
        for raw in body.iter_lines(chunk_size=8 * 1024 * 1024):
            rows_seen += 1
            if rows_seen % 250 == 0:
                print(f"figure scan: rows={rows_seen}, matched={len(found)}/{len(targets)}", flush=True)
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            track_id = str(row.get("track_id") or "")
            paper = targets.get(track_id)
            if paper is None:
                continue
            figures = extract_figure_candidates_from_blocks(row.get("file_content") or [])
            found[track_id] = {
                "schema_version": SCHEMA_VERSION,
                "paper_id": paper["paper_id"],
                "doi": paper.get("doi", ""),
                "title": paper.get("title", ""),
                "track_id": track_id,
                "source_prefix": s3_uri.rsplit("/", 1)[0] + "/",
                "figures": figures,
            }
            if len(found) == len(targets):
                break
    finally:
        body.close()
    output = [found[track] for track in targets if track in found]
    write_jsonl(output_path, output)
    for track in sorted(set(targets) - set(found)):
        report.warn("paper_figures_missing", track)
    report.stats.update({
        "target_papers": len(targets),
        "papers_found": len(output),
        "figure_candidates": sum(len(row["figures"]) for row in output),
        "mapping_eligible_figures": sum(
            bool(figure.get("mapping_eligible"))
            for row in output for figure in row["figures"]
        ),
        "rows_scanned": rows_seen,
    })
    write_json(output_path.with_suffix(".report.json"), report.to_dict())
    return report


def _mapping_prompt(tasks: list[dict[str, Any]], paper: dict[str, Any]) -> str:
    figures = [{
        "image_object": row.get("image_object"),
        "figure_label": row.get("figure_label"),
        "caption": str(row.get("caption") or row.get("caption_context") or "")[:2200],
        "caption_match_method": row.get("caption_match_method"),
    } for row in paper.get("figures", []) if row.get("mapping_eligible")]
    payload = {
        "paper": {key: paper.get(key) for key in ("paper_id", "doi", "title")},
        "tasks": [{
            "task_tag": row.get("task_tag"),
            "query": row.get("query"),
            "target_conclusion": (row.get("rubric") or {}).get("core_conclusion", ""),
        } for row in tasks],
        "paper_figures": figures,
    }
    return f"""Map every task to at most one exact source-paper figure.

{json.dumps(payload, ensure_ascii=False, indent=2)}

Return exactly one entry per task:
{{"mappings":[{{
  "task_tag":"Analysis_01",
  "match_status":"matched or no_exact_match",
  "image_object":"exact supplied image_object, or empty",
  "figure_label":"supplied/inferred label, or empty",
  "confidence":"high, medium, or low",
  "reason":"brief evidence-based reason"
}}]}}

Use matched only when the caption/figure directly represents the task's target result.
Multiple tasks may share a figure only if it genuinely supports each target.
"""


def map_tasks_to_paper_figures(
    candidates_path: Path,
    figures_path: Path,
    output_path: Path,
    cache_dir: Path,
    *,
    model: str,
    jobs: int = 4,
    timeout: int = 180,
    max_retries: int = 3,
    max_tokens: int = 6000,
    max_papers: int | None = None,
    resume: bool = True,
    refresh_cache: bool = False,
    client: OpenAICompatibleJSONClient | None = None,
    use_env_proxy: bool = False,
) -> ValidationReport:
    report = ValidationReport()
    tasks_by_paper: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(candidates_path):
        tasks_by_paper.setdefault(str(row.get("paper_id") or ""), []).append(row)
    figures = {str(row.get("paper_id") or ""): row for row in read_jsonl(figures_path)}
    existing = read_jsonl(output_path) if resume and output_path.is_file() else []
    mapped_papers = {str(row.get("paper_id") or "") for row in existing}
    by_paper = {str(row.get("paper_id") or ""): row for row in existing}
    llm = client or OpenAICompatibleJSONClient(
        model=model, cache_dir=cache_dir, timeout=timeout,
        max_retries=max_retries, use_env_proxy=use_env_proxy,
    )

    def work(paper_id: str) -> tuple[str, dict[str, Any]]:
        tasks = sorted(tasks_by_paper[paper_id], key=lambda row: str(row.get("task_tag") or ""))
        paper = figures.get(paper_id)
        if paper is None:
            raise ValueError("paper has no extracted figure candidates")
        response, cache_meta = llm.chat_json(
            system=SYSTEM_PROMPT, user=_mapping_prompt(tasks, paper),
            temperature=0.0, max_tokens=max_tokens,
            prompt_version=PROMPT_VERSION,
            refresh_cache=refresh_cache,
        )
        raw = response.get("mappings") or []
        if not isinstance(raw, list) or len(raw) != len(tasks):
            raise ValueError(f"expected {len(tasks)} mappings")
        allowed = {
            str(item.get("image_object") or ""): item
            for item in paper.get("figures", []) if item.get("mapping_eligible")
        }
        expected_tags = {str(task.get("task_tag") or "") for task in tasks}
        normalized: list[dict[str, Any]] = []
        seen_tags: set[str] = set()
        for item in raw:
            tag = str(item.get("task_tag") or "")
            if tag not in expected_tags or tag in seen_tags:
                raise ValueError(f"invalid/duplicate task_tag: {tag}")
            seen_tags.add(tag)
            status = str(item.get("match_status") or "no_exact_match")
            image_object = str(item.get("image_object") or "")
            if status == "matched" and image_object not in allowed:
                raise ValueError(f"unknown image_object for {tag}: {image_object}")
            if status != "matched":
                status, image_object = "no_exact_match", ""
            source = allowed.get(image_object, {})
            normalized.append({
                "task_tag": tag,
                "match_status": status,
                "image_object": image_object,
                "figure_label": str(item.get("figure_label") or source.get("figure_label") or ""),
                "caption": str(source.get("caption") or source.get("caption_context") or ""),
                "confidence": str(item.get("confidence") or "low"),
                "reason": str(item.get("reason") or ""),
            })
        return paper_id, {
            "schema_version": SCHEMA_VERSION,
            "paper_id": paper_id,
            "doi": paper.get("doi", ""),
            "title": paper.get("title", ""),
            "source_prefix": paper.get("source_prefix", ""),
            "mappings": normalized,
            "generation": {
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "cache_key": cache_meta["cache_key"],
                "cache_hit": cache_meta["cache_hit"],
                "generated_at_utc": _utc_now(),
            },
        }

    eligible = [paper_id for paper_id in sorted(tasks_by_paper) if paper_id not in mapped_papers]
    if max_papers is not None:
        eligible = eligible[:max_papers]
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(work, paper_id): paper_id for paper_id in eligible}
        for future in as_completed(futures):
            paper_id = futures[future]
            try:
                key, row = future.result()
                by_paper[key] = row
                write_jsonl(output_path, [by_paper[key] for key in sorted(by_paper)])
            except Exception as exc:  # noqa: BLE001
                report.error("figure_mapping_failed", f"{paper_id}: {exc}")
    output = [by_paper[key] for key in sorted(by_paper)]
    write_jsonl(output_path, output)
    all_mappings = [item for row in output for item in row.get("mappings", [])]
    report.stats.update({
        "papers": len(output),
        "tasks": len(all_mappings),
        "matched": sum(item.get("match_status") == "matched" for item in all_mappings),
        "no_exact_match": sum(item.get("match_status") != "matched" for item in all_mappings),
        "api_calls": llm.call_count,
        "cache_hits": llm.cache_hits,
    })
    write_json(output_path.with_suffix(".report.json"), report.to_dict())
    return report


def _image_suffix(key: str, content_type: str) -> str:
    suffix = PurePosixPath(key).suffix.casefold()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}:
        return suffix
    return {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(content_type.casefold(), ".img")


def materialize_gt_figures(
    mappings_path: Path,
    brainarena_root: Path,
    *,
    endpoint_url: str,
    access_key: str | None = None,
    secret_key: str | None = None,
) -> ValidationReport:
    try:
        import boto3  # type: ignore
        from botocore.config import Config  # type: ignore
    except ImportError as exc:
        raise RuntimeError("boto3 is required for S3 figure materialization") from exc
    report = ValidationReport()
    internal_to_export: dict[tuple[str, str], str] = {}
    provenance_root = brainarena_root / "benchmark/gt/provenance"
    for path in provenance_root.glob("*/*.json"):
        row = read_json(path)
        internal = str((row.get("paper") or {}).get("paper_id") or "")
        internal_to_export[(internal, path.stem)] = path.parent.name
    clients: dict[tuple[str, str], Any] = {}
    cache: dict[tuple[str, str], tuple[bytes, str]] = {}
    manifest: list[dict[str, Any]] = []
    mappings = read_jsonl(mappings_path)
    for paper in mappings:
        source_prefix = str(paper.get("source_prefix") or "")
        if not source_prefix.startswith("s3://"):
            report.error("invalid_figure_source_prefix", str(paper.get("paper_id")))
            continue
        bucket, prefix = source_prefix[5:].split("/", 1)
        client_key = (bucket, endpoint_url)
        client = clients.get(client_key)
        if client is None:
            client = boto3.client(
                "s3", endpoint_url=endpoint_url,
                aws_access_key_id=access_key, aws_secret_access_key=secret_key,
                config=Config(signature_version="s3v4", s3={"addressing_style": "path"}, proxies={}),
            )
            clients[client_key] = client
        for mapping in paper.get("mappings", []):
            tag = str(mapping.get("task_tag") or "")
            internal_paper = str(paper.get("paper_id") or "")
            export_paper = internal_to_export.get((internal_paper, tag), "")
            model_match_status = str(mapping.get("match_status") or "no_exact_match")
            confidence = str(mapping.get("confidence") or "low").casefold()
            accepted_match = model_match_status == "matched" and confidence == "high"
            materialization_status = (
                "matched" if accepted_match else
                "review_required" if model_match_status == "matched" else
                "no_exact_match"
            )
            entry = {
                "schema_version": SCHEMA_VERSION,
                "paper_id": export_paper,
                "internal_paper_id": internal_paper,
                "task_tag": tag,
                "match_status": materialization_status,
                "model_match_status": model_match_status,
                "figure_label": mapping.get("figure_label", ""),
                "confidence": confidence,
                "reason": mapping.get("reason", ""),
                "caption": mapping.get("caption", ""),
                "source_object": "",
                "gt_figure_path": "",
                "sha256": "",
            }
            if not accepted_match:
                manifest.append(entry)
                continue
            if not export_paper:
                report.error("task_not_found_for_figure", f"{internal_paper}::{tag}")
                manifest.append(entry)
                continue
            image_object = str(mapping.get("image_object") or "")
            key = prefix + image_object
            object_id = (bucket, key)
            try:
                if object_id not in cache:
                    response = client.get_object(Bucket=bucket, Key=key)
                    payload = response["Body"].read()
                    content_type = str(response.get("ContentType") or "")
                    cache[object_id] = (payload, content_type)
                payload, content_type = cache[object_id]
                suffix = _image_suffix(key, content_type)
                rel = Path("benchmark/gt/gt_figure") / export_paper / f"{tag}{suffix}"
                target = brainarena_root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                entry.update({
                    "source_object": f"s3://{bucket}/{key}",
                    "gt_figure_path": rel.as_posix(),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                })
            except Exception as exc:  # noqa: BLE001
                report.error("figure_download_failed", f"{internal_paper}::{tag}: {exc}")
            manifest.append(entry)
    write_jsonl(brainarena_root / "benchmark/gt/gt_figure_manifest.jsonl", manifest)
    report.stats.update({
        "tasks": len(manifest),
        "matched": sum(row.get("match_status") == "matched" for row in manifest),
        "model_candidate_matches": sum(row.get("model_match_status") == "matched" for row in manifest),
        "review_required": sum(row.get("match_status") == "review_required" for row in manifest),
        "files_written": sum(bool(row.get("gt_figure_path")) for row in manifest),
        "unique_source_objects_read": len(cache),
        "no_exact_match": sum(row.get("match_status") == "no_exact_match" for row in manifest),
    })
    write_json(brainarena_root / "GT_FIGURE_REPORT.json", report.to_dict())
    return report


def audit_gt_figures(brainarena_root: Path) -> ValidationReport:
    report = ValidationReport()
    manifest_path = brainarena_root / "benchmark/gt/gt_figure_manifest.jsonl"
    if not manifest_path.is_file():
        report.error("missing_gt_figure_manifest", str(manifest_path))
        return report
    rows = read_jsonl(manifest_path)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        identity = (str(row.get("paper_id") or row.get("internal_paper_id") or ""), str(row.get("task_tag") or ""))
        if identity in seen:
            report.error("duplicate_figure_mapping", f"{identity[0]}::{identity[1]}")
        seen.add(identity)
        path = str(row.get("gt_figure_path") or "")
        if row.get("match_status") == "matched":
            target = brainarena_root / path
            if not path or not target.is_file():
                report.error("missing_gt_figure", f"{identity[0]}::{identity[1]}")
            elif hashlib.sha256(target.read_bytes()).hexdigest() != row.get("sha256"):
                report.error("gt_figure_hash_mismatch", path)
            if not str(row.get("source_object") or "").startswith("s3://"):
                report.error("missing_figure_provenance", f"{identity[0]}::{identity[1]}")
        elif path:
            report.error("unexpected_figure_for_unmatched_task", path)
    report.stats.update({
        "tasks": len(rows),
        "matched": sum(row.get("match_status") == "matched" for row in rows),
        "review_required": sum(row.get("match_status") == "review_required" for row in rows),
        "no_exact_match": sum(row.get("match_status") == "no_exact_match" for row in rows),
        "figure_files": len(list((brainarena_root / "benchmark/gt/gt_figure").glob("*/*"))),
    })
    write_json(brainarena_root / "GT_FIGURE_AUDIT_REPORT.json", report.to_dict())
    return report
