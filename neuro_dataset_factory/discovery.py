"""Discovery adapters produce candidates only; they never create verified assets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from neuro_dataset_factory.contracts import SCHEMA_VERSION, ValidationReport, require_text, string_list
from neuro_dataset_factory.storage import read_jsonl, write_json, write_jsonl

SERPER_URL = "https://google.serper.dev/search"
OPENALEX_URL = "https://api.openalex.org/works"
JINA_READER_URL = "https://r.jina.ai"

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+", re.I)
_DATA_URL_HINT = re.compile(
    r"(openneuro|dandiarchive|brain-map|crcns|gin\.g-node|figshare|zenodo|"
    r"datadryad|dryad|osf\.io|download|dataset|archive)",
    re.I,
)


def _id(prefix: str, *parts: str) -> str:
    value = "\0".join(parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(value).hexdigest()[:12]}"


def _post_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    return value


def _get_json(url: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "neuro-dataset-factory/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    return value


def _get_text(url: str, headers: dict[str, str], timeout: int) -> str:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def _string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        raise ValueError("expected a string or list of strings")
    return [str(item).strip() for item in value if str(item).strip()]


def _dataset_page_evidence(text: str) -> dict[str, Any]:
    dois: list[str] = []
    for raw in _DOI_RE.findall(text):
        doi = raw.rstrip(".,;:)]}")
        if doi not in dois:
            dois.append(doi)
    data_urls: list[str] = []
    for raw in _URL_RE.findall(text):
        url = raw.rstrip(".,;:)]}")
        if _DATA_URL_HINT.search(url) and url not in data_urls:
            data_urls.append(url)
    return {
        "associated_dois": dois[:50],
        "data_urls": data_urls[:100],
    }


def _inverted_abstract(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for token, indexes in value.items():
        if isinstance(indexes, list):
            positions.extend((int(index), str(token)) for index in indexes)
    return " ".join(token for _, token in sorted(positions))


def search_dataset_candidates(
    requests_path: Path,
    output_path: Path,
    *,
    api_key: str | None = None,
    api_key_env: str = "SERPER_KEY",
    results_per_query: int = 10,
    timeout: int = 60,
    enrich_with_jina: bool = False,
    jina_api_key: str | None = None,
    jina_key_env: str = "JINA_API_KEY",
    jina_reader_url: str = JINA_READER_URL,
    jina_max_chars: int = 20000,
    jina_timeout: int = 20,
    jina_jobs: int = 8,
    post_json: Callable[[str, dict[str, Any], dict[str, str], int], dict[str, Any]] = _post_json,
    get_text: Callable[[str, dict[str, str], int], str] = _get_text,
) -> ValidationReport:
    """Search dataset landing pages using question- or dataset-first requests.

    Dataset-first rows use ``dataset_focus`` and may constrain results with
    ``preferred_domains``. Legacy rows with ``scientific_question`` and
    ``capability_gap`` remain supported. Optional Jina enrichment reads the
    landing page and extracts DOI/download evidence. Output remains unverified.
    """
    key = (api_key or os.environ.get(api_key_env) or "").strip()
    jina_key = (jina_api_key or os.environ.get(jina_key_env) or "").strip()
    report = ValidationReport()
    if not key:
        report.error("missing_api_key", f"set {api_key_env} or pass api_key")
        return report
    candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    requests = read_jsonl(requests_path)
    for index, row in enumerate(requests, 1):
        try:
            focus = str(row.get("dataset_focus") or "").strip()
            question = str(row.get("scientific_question") or "").strip()
            capability = string_list(row.get("capability_gap"), "capability_gap")
            preferred_domains = _string_values(row.get("preferred_domains"))
            if not focus:
                question = require_text(question, "scientific_question")
                if not capability:
                    raise ValueError("capability_gap must be non-empty")
        except ValueError as exc:
            report.error("bad_discovery_request", f"row {index}: {exc}")
            continue
        modality = str(row.get("modality") or "").strip()
        species = str(row.get("species") or "").strip()
        if focus:
            domain_query = " OR ".join(f"site:{domain}" for domain in preferred_domains)
            query = " ".join(part for part in (
                focus,
                modality,
                species,
                domain_query,
                "widely used open neuroscience dataset official download license associated publication",
            ) if part)
        else:
            query = " ".join(part for part in (
                question,
                modality,
                species,
                "open neuroscience dataset download DOI license",
            ) if part)
        try:
            response = post_json(
                SERPER_URL,
                {"q": query, "num": max(1, min(results_per_query, 20))},
                {"X-API-KEY": key},
                timeout,
            )
        except Exception as exc:  # noqa: BLE001 - preserve other requests
            report.warn("dataset_search_failed", f"row {index}: {exc}")
            continue
        rows = [*(response.get("organic") or []), *(response.get("scholar") or [])]
        for result in rows:
            if not isinstance(result, dict):
                continue
            url = str(result.get("link") or result.get("url") or "").strip()
            title = str(result.get("title") or "").strip()
            if not url or not title or url in seen_urls:
                continue
            host = (urllib.parse.urlsplit(url).hostname or "").lower()
            if preferred_domains and not any(
                host == domain.lower() or host.endswith("." + domain.lower())
                for domain in preferred_domains
            ):
                continue
            seen_urls.add(url)
            candidate = {
                "schema_version": SCHEMA_VERSION,
                "discovery_id": _id("dataset_candidate", query, url),
                "status": "unverified",
                "provider": "serper",
                "request": {
                    "dataset_focus": focus,
                    "scientific_question": question,
                    "capability_gap": capability,
                    "modality": modality,
                    "species": species,
                    "preferred_domains": preferred_domains,
                    "query": query,
                },
                "title": title,
                "url": url,
                "snippet": str(result.get("snippet") or "").strip(),
            }
            candidates.append(candidate)

    # Persist Serper results before enrichment so an interrupted Jina batch can
    # resume from a useful raw discovery artifact.
    write_jsonl(output_path, candidates)
    if enrich_with_jina and candidates:
        headers = {
            "X-Return-Format": "markdown",
            "User-Agent": "neuro-dataset-factory/0.3",
        }
        if jina_key:
            headers["Authorization"] = f"Bearer {jina_key}"

        def enrich(candidate: dict[str, Any]) -> tuple[dict[str, Any], Exception | None]:
            reader_url = jina_reader_url.rstrip("/") + "/" + candidate["url"]
            try:
                page_text = get_text(reader_url, headers, jina_timeout)
                candidate.update({
                    "jina_status": "ok",
                    "page_text": page_text[:max(0, jina_max_chars)],
                    **_dataset_page_evidence(page_text),
                })
                return candidate, None
            except Exception as exc:  # noqa: BLE001 - preserve other candidates
                candidate.update({"jina_status": "failed", "jina_error": str(exc)[:500]})
                return candidate, exc

        completed = 0
        with ThreadPoolExecutor(max_workers=max(1, jina_jobs)) as executor:
            futures = {executor.submit(enrich, candidate): candidate for candidate in candidates}
            for future in as_completed(futures):
                candidate, error = future.result()
                if error is not None:
                    report.warn("jina_fetch_failed", f"{candidate['url']}: {error}")
                completed += 1
                if completed % 10 == 0:
                    write_jsonl(output_path, candidates)
    write_jsonl(output_path, candidates)
    report.stats.update({
        "requests": len(requests),
        "candidates": len(candidates),
        "jina_ok": sum(row.get("jina_status") == "ok" for row in candidates),
        "jina_failed": sum(row.get("jina_status") == "failed" for row in candidates),
    })
    write_json(output_path.with_suffix(".report.json"), report.to_dict())
    return report


def search_paper_candidates(
    registry_path: Path,
    output_path: Path,
    *,
    results_per_dataset: int = 20,
    mailto: str = "",
    timeout: int = 60,
    get_json: Callable[[str, int], dict[str, Any]] = _get_json,
) -> ValidationReport:
    """Find papers that may use a registered dataset via OpenAlex.

    These rows are not paper links. They intentionally set ``status=unverified``
    and must pass the explicit linkage contract before task construction.
    """
    report = ValidationReport()
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    datasets = read_jsonl(registry_path)
    for dataset in datasets:
        dataset_id = str(dataset.get("dataset_id") or "")
        name = str(dataset.get("name") or "")
        doi = str((dataset.get("source") or {}).get("doi") or "")
        query = f'"{name}"' + (f" {doi}" if doi else "")
        params = {
            "search": query,
            "per-page": str(max(1, min(results_per_dataset, 100))),
            "select": "id,doi,display_name,publication_year,primary_location,open_access,cited_by_count,abstract_inverted_index",
        }
        if mailto:
            params["mailto"] = mailto
        url = OPENALEX_URL + "?" + urllib.parse.urlencode(params)
        try:
            response = get_json(url, timeout)
        except Exception as exc:  # noqa: BLE001
            report.warn("paper_search_failed", f"{dataset_id}: {exc}")
            continue
        for result in response.get("results") or []:
            if not isinstance(result, dict):
                continue
            openalex_id = str(result.get("id") or "").rstrip("/").rsplit("/", 1)[-1]
            title = str(result.get("display_name") or "").strip()
            if not openalex_id or not title or (dataset_id, openalex_id) in seen:
                continue
            seen.add((dataset_id, openalex_id))
            location = result.get("primary_location") or {}
            source = location.get("source") or {} if isinstance(location, dict) else {}
            candidates.append({
                "schema_version": SCHEMA_VERSION,
                "paper_candidate_id": _id("paper_candidate", dataset_id, openalex_id),
                "status": "unverified",
                "provider": "openalex",
                "dataset_id": dataset_id,
                "dataset_name": name,
                "search_query": query,
                "paper_id": openalex_id,
                "title": title,
                "doi": str(result.get("doi") or "").replace("https://doi.org/", ""),
                "url": str(location.get("landing_page_url") or result.get("id") or "") if isinstance(location, dict) else str(result.get("id") or ""),
                "venue": str(source.get("display_name") or "") if isinstance(source, dict) else "",
                "year": result.get("publication_year"),
                "cited_by_count": int(result.get("cited_by_count") or 0),
                "is_open_access": bool((result.get("open_access") or {}).get("is_oa")),
                "abstract": _inverted_abstract(result.get("abstract_inverted_index"))[:4000],
            })
    write_jsonl(output_path, candidates)
    report.stats.update({"datasets": len(datasets), "candidates": len(candidates)})
    write_json(output_path.with_suffix(".report.json"), report.to_dict())
    return report
