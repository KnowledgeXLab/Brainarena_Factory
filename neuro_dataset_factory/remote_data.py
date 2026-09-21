"""Discover and verify remotely downloadable datasets referenced by papers.

This module deliberately stops before downloading dataset payloads.  It emits
object-level locators and repository manifests for the data-acquisition team.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator

from neuro_dataset_factory.contracts import SCHEMA_VERSION, ValidationReport, slug, stable_id
from neuro_dataset_factory.storage import read_jsonl, write_json, write_jsonl


VERIFIED = "verified_downloadable"
REVIEW = "needs_manual_review"
REJECTED = "rejected"
DEFERRED_NETWORK = "verification_deferred_network"
DEFAULT_REMOTE_MAX_BYTES = 50 * 1024 ** 3
MAX_OSF_LISTING_PAGES = 100
MAX_OSF_FILES = 10000
MAX_GEO_FILES = 500
MAX_OPENNEURO_FILES = 100000
MAX_NEUROVAULT_FILES = 10000
MAX_DANDI_FILES = 20000
MAX_EBI_FILES = 20000

_STRONG_NEURO_PATTERNS = {
    "brain": r"\bbrain(?:wide)?\b",
    "neuroscience": r"\bneuroscien\w*\b",
    "neuron": r"\bneuron\w*\b",
    "cortex": r"\b(?:cortex|cortical)\b",
    "hippocampus": r"\bhippocamp\w*\b",
    "synapse": r"\bsynap\w*\b",
    "glia": r"\b(?:glia|glial|astrocy\w*|microglia\w*)\b",
    "neurotransmitter": r"\b(?:dopamin\w*|seroton\w*|cholinerg\w*)\b",
    "neural_recording": r"\b(?:electrophysiolog\w*|electroencephal\w*|EEG|MEG)\b",
    "neuroimaging": r"\b(?:fMRI|BOLD|diffusion MRI|functional connectivity)\b",
    "neuroanatomy": r"\b(?:axon\w*|dendrit\w*|connectom\w*)\b",
    "sensory_system": r"\b(?:visual cortex|auditory cortex|olfactor\w*|retina\w*)\b",
    "spinal_cord": r"\bspinal cord\b",
}

_WEAK_NEURO_PATTERNS = {
    "neural": r"\bneural\b",
    "cognition": r"\bcogniti\w*\b",
    "behavior": r"\bbehavio(?:u)?r\w*\b",
    "memory": r"\bmemory\b",
}

_ARTIFICIAL_ONLY = re.compile(
    r"\b(?:artificial neural network|deep neural network|machine learning|shape[- ]memory)\b",
    re.IGNORECASE,
)

_DIRECT_FILE = re.compile(
    r"\.(?:zip|tar|tgz|gz|bz2|xz|7z|csv|tsv|json|jsonl|mat|nwb|h5|hdf5|nii|nii\.gz|edf|bdf|fif|pkl|rds)(?:\?|$)",
    re.IGNORECASE,
)

_DATA_EXTENSIONS = re.compile(
    r"\.(?:csv|tsv|json|jsonl|txt|mat|nwb|h5|hdf5|nii|nii\.gz|edf|bdf|fif|pkl|rds|zip|tar|tgz|gz|bz2|xz|7z)$",
    re.IGNORECASE,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_link_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        text = str(value).strip()
        if not text or text.casefold() in {"[]", "none", "null", "nan"}:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = [text]
        raw = parsed if isinstance(parsed, (list, tuple)) else [parsed]
    out: list[str] = []
    for item in raw:
        url = str(item or "").strip()
        if url.startswith(("http://", "https://")) and url not in out:
            out.append(url)
    return out


def neuroscience_evidence(row: dict[str, Any]) -> dict[str, Any]:
    title = str(row.get("title") or "")
    abstract = str(row.get("abstract") or "")
    keywords = str(row.get("keywords") or "")
    text = " ".join((title, abstract, keywords))
    strong = [name for name, pattern in _STRONG_NEURO_PATTERNS.items() if re.search(pattern, text, re.IGNORECASE)]
    weak = [name for name, pattern in _WEAK_NEURO_PATTERNS.items() if re.search(pattern, text, re.IGNORECASE)]
    artificial = bool(_ARTIFICIAL_ONLY.search(text))
    # A strong biological signal is sufficient. Weak terms alone require at
    # least two independent signals and are retained only for later review.
    accepted = bool(strong) or (len(weak) >= 2 and not artificial)
    return {
        "status": "heuristic_candidate" if accepted else "not_neuroscience",
        "strong_signals": strong,
        "weak_signals": weak,
        "artificial_system_signal": artificial,
    }


def normalize_locator(url: str) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(url)
    host = parsed.netloc.casefold().split(":", 1)[0]
    path = parsed.path
    repository = "generic"
    record_id = ""
    version = ""
    kind = "landing_page"

    match = re.search(r"/articles/(?:dataset/[^/]+/)?(\d+)(?:/(\d+))?", url, re.I)
    if match:
        repository, record_id, version, kind = "figshare", match.group(1), match.group(2) or "", "dataset_record"
    figshare_doi = re.search(r"10\.6084/m9\.figshare\.(\d+)(?:\.v(\d+))?", url, re.I)
    if figshare_doi:
        repository, record_id, version, kind = (
            "figshare", figshare_doi.group(1), figshare_doi.group(2) or "", "dataset_record",
        )
    match = match or re.search(r"osf\.io/([a-z0-9]{5})(?:/|$)", url, re.I)
    if repository == "generic" and match:
        repository, record_id, kind = "osf", match.group(1).lower(), "dataset_record"
    zenodo = re.search(r"(?:zenodo\.(?:org|\w+)/records?/|10\.5281/zenodo\.)(\d+)", url, re.I)
    if zenodo:
        repository, record_id, kind = "zenodo", zenodo.group(1), "dataset_record"
    github = re.search(r"github\.com/([^/]+)/([^/?#]+)(?:/(blob|tree)/([^/]+)/(.+))?", url, re.I)
    if github:
        repository = "github"
        record_id = f"{github.group(1)}/{github.group(2).removesuffix('.git')}"
        kind = "file" if github.group(3) == "blob" else "repository"
    if host == "gin.g-node.org":
        parts = [part for part in path.split("/") if part]
        if len(parts) >= 2:
            repository = "gin"
            record_id = f"{parts[0]}/{parts[1].removesuffix('.git')}"
            kind = "dataset_record"
    gin_doi = re.search(r"(10\.12751/g-node\.[A-Za-z0-9]+)", url, re.I)
    if gin_doi:
        repository, record_id, kind = "gin", gin_doi.group(1).lower(), "dataset_record"
    geo = re.search(r"(?:acc=|/)(GSE\d+)\b", url, re.I)
    if geo:
        repository, record_id, kind = "geo", geo.group(1).upper(), "dataset_record"
    openneuro = re.search(
        r"openneuro\.org/datasets/(ds\d+)(?:/versions/([^/?#]+))?", url, re.I,
    )
    openneuro_doi = re.search(r"10\.18112/openneuro\.(ds\d+)\.v([^/?#\s]+)", url, re.I)
    if openneuro_doi:
        repository, record_id, version, kind = (
            "openneuro", openneuro_doi.group(1).lower(),
            openneuro_doi.group(2).rstrip(".]),;"), "dataset_record",
        )
    if openneuro:
        repository, record_id, version, kind = (
            "openneuro", openneuro.group(1).lower(), openneuro.group(2) or "", "dataset_record",
        )
    dandi = re.search(
        r"dandiarchive\.org/dandiset/(\d+)(?:/([^/?#]+))?", url, re.I,
    )
    dandi_doi = re.search(r"10\.48324/dandi\.(\d+)(?:/([^?#\s]+))?", url, re.I)
    if dandi_doi:
        repository, record_id, version, kind = (
            "dandi", dandi_doi.group(1),
            (dandi_doi.group(2) or "").rstrip(".]),;"), "dataset_record",
        )
    if dandi:
        repository, record_id, version, kind = (
            "dandi", dandi.group(1), dandi.group(2) or "", "dataset_record",
        )
    dryad = re.search(r"10\.5061/(dryad\.[^?#\s]+)", url, re.I)
    if dryad:
        repository, record_id, kind = (
            "dryad", f"10.5061/{dryad.group(1).rstrip('.]),;')}", "dataset_record",
        )
    dataverse_doi = re.search(
        r"(10\.(?:7910/DVN|18738/T8|11588/DATA)/[A-Z0-9]+)", url, re.I,
    )
    if dataverse_doi:
        repository, record_id, kind = "dataverse", dataverse_doi.group(1).upper(), "dataset_record"
    neurovault = re.search(r"neurovault\.org/(collections|images)/([^/?#]+)/?", url, re.I)
    if neurovault:
        repository = "neurovault"
        record_id = f"{neurovault.group(1).casefold()}:{neurovault.group(2)}"
        kind = "dataset_record"
    encode = re.search(
        r"encodeproject\.org/(?:experiments|functional-characterization-experiments|"
        r"publication-data|references|files)/(ENC[A-Z0-9]+)",
        url,
        re.I,
    )
    if encode:
        repository, record_id, kind = "encode", encode.group(1).upper(), "dataset_record"
    sasbdb = re.search(r"sasbdb\.org/data/(SAS[A-Z0-9]{4})", url, re.I)
    if sasbdb:
        repository, record_id, kind = "sasbdb", sasbdb.group(1).upper(), "dataset_record"
    pride = re.search(r"(?:ebi\.ac\.uk/pride/archive/projects/|proteomexchange[^?#]*/)(PXD\d+)", url, re.I)
    if pride:
        repository, record_id, kind = "pride", pride.group(1).upper(), "dataset_record"
    biostudies = re.search(
        r"ebi\.ac\.uk/(?:biostudies/(?:arrayexpress/)?studies|arrayexpress/experiments)/"
        r"(E-[A-Z]+-\d+|S-[A-Z]+\d+)",
        url,
        re.I,
    )
    if biostudies:
        repository, record_id, kind = "biostudies", biostudies.group(1).upper(), "dataset_record"
    emdb = re.search(r"ebi\.ac\.uk/(?:emdb/entry|pdbe/entry/emdb)/(EMD-\d+)", url, re.I)
    if emdb:
        repository, record_id, kind = "emdb", emdb.group(1).upper(), "dataset_record"
    rcsb = re.search(r"rcsb\.org/structure/([0-9A-Za-z]{4})", url, re.I)
    rcsb_doi = re.search(r"10\.2210/pdb([0-9A-Za-z]{4})/pdb", url, re.I)
    if rcsb_doi:
        repository, record_id, kind = "rcsb", rcsb_doi.group(1).upper(), "dataset_record"
    if rcsb:
        repository, record_id, kind = "rcsb", rcsb.group(1).upper(), "dataset_record"
    ena = re.search(r"\b(PRJ(?:NA|EB|DB)\d+|[SED]RP\d+)\b", url, re.I)
    if ena:
        repository, record_id, kind = "ena", ena.group(1).upper(), "dataset_record"
    if host.endswith("doi.org") and repository == "generic":
        record_id, kind = path.lstrip("/"), "doi"
    if _DIRECT_FILE.search(url):
        kind = "file"

    preliminary = "candidate"
    if kind in {"landing_page", "repository", "doi"} and repository == "generic":
        preliminary = REVIEW
    return {
        "original_url": url,
        "repository": repository,
        "record_id": record_id,
        "version": version,
        "locator_kind": kind,
        "preliminary_status": preliminary,
    }


def scan_corpus_rows(
    rows: Iterable[dict[str, Any]],
    output_path: Path,
    *,
    source_uri: str,
    max_papers: int | None = None,
) -> ValidationReport:
    report = ValidationReport()
    output: list[dict[str, Any]] = []
    rows_seen = 0
    with_data_links = 0
    neuro_candidates = 0
    for row in rows:
        rows_seen += 1
        urls = parse_link_list(row.get("data_link"))
        if not urls:
            continue
        with_data_links += 1
        neuro = neuroscience_evidence(row)
        if neuro["status"] != "heuristic_candidate":
            continue
        neuro_candidates += 1
        doi = str(row.get("doi") or "").strip()
        title = str(row.get("title") or "").strip()
        paper_id = stable_id("paper", doi, title)
        output.append({
            "schema_version": SCHEMA_VERSION,
            "paper_id": paper_id,
            "doi": doi,
            "title": title,
            "article_url": str(row.get("article_url") or "").strip(),
            "publication_date": str(row.get("publication_date") or "").strip(),
            "authors": str(row.get("author") or "").strip(),
            "abstract": str(row.get("abstract") or "").strip(),
            "neuroscience": neuro,
            "data_locators": [normalize_locator(url) for url in urls],
            "code_links": parse_link_list(row.get("code_link")),
            "corpus_source": {
                "uri": source_uri,
                "track_id": str(row.get("track_id") or "").strip(),
                "paper_object": str(row.get("file_path") or "").strip(),
                "supplement_objects": parse_link_list(row.get("SI_path")),
            },
            "status": "needs_remote_verification",
        })
        if max_papers is not None and len(output) >= max_papers:
            break
    write_jsonl(output_path, output)
    report.stats.update({
        "rows_seen": rows_seen,
        "rows_with_data_links": with_data_links,
        "neuroscience_candidates": neuro_candidates,
        "written": len(output),
    })
    write_json(output_path.with_suffix(".report.json"), report.to_dict())
    return report


def select_remote_candidates(
    candidates_path: Path,
    output_path: Path,
    *,
    repositories: set[str],
    include_direct_files: bool = False,
    exclude_papers_path: Path | None = None,
    statuses: set[str] | None = None,
    max_papers: int | None = None,
) -> ValidationReport:
    """Select only locators supported by the next remote-verification batch.

    Pruning unrelated locators is intentional: a paper with a concrete Zenodo
    record should not also spend requests probing generic landing pages, and
    those weaker links must not leak into its acquisition handoff.
    """
    normalized_repositories = {
        str(repository).strip().casefold()
        for repository in repositories
        if str(repository).strip()
    }
    if not normalized_repositories and not include_direct_files:
        raise ValueError("at least one repository or --include-direct-files is required")

    excluded: set[str] = set()
    if exclude_papers_path is not None:
        excluded = {
            str(row.get("paper_id") or "")
            for row in read_jsonl(exclude_papers_path)
            if row.get("paper_id")
        }

    selected: list[dict[str, Any]] = []
    selected_locators = 0
    candidate_rows = read_jsonl(candidates_path)
    normalized_statuses = {
        str(status).strip() for status in (statuses or set()) if str(status).strip()
    }
    for row in candidate_rows:
        paper_id = str(row.get("paper_id") or "")
        if paper_id in excluded:
            continue
        if normalized_statuses and str(row.get("status") or "") not in normalized_statuses:
            continue
        locators = [
            locator for locator in row.get("data_locators", [])
            if str(locator.get("repository") or "").casefold() in normalized_repositories
            or (
                include_direct_files
                and locator.get("locator_kind") == "file"
                and str(locator.get("repository") or "generic").casefold() == "generic"
            )
        ]
        if not locators:
            continue
        selected.append({
            **row,
            "data_locators": locators,
            "status": "needs_remote_verification",
        })
        selected_locators += len(locators)
        if max_papers is not None and len(selected) >= max_papers:
            break

    write_jsonl(output_path, selected)
    report = ValidationReport()
    report.stats.update({
        "candidates_seen": len(candidate_rows),
        "papers_excluded": len(excluded),
        "papers_selected": len(selected),
        "locators_selected": selected_locators,
        "repositories": sorted(normalized_repositories),
        "include_direct_files": include_direct_files,
        "statuses": sorted(normalized_statuses),
    })
    write_json(output_path.with_suffix(".report.json"), report.to_dict())
    return report


@contextmanager
def open_s3_csv(
    s3_uri: str,
    *,
    endpoint_url: str,
    access_key: str | None = None,
    secret_key: str | None = None,
) -> Iterator[Iterable[dict[str, str]]]:
    try:
        import boto3  # type: ignore
        from botocore.config import Config  # type: ignore
    except ImportError as exc:
        raise RuntimeError("boto3 is required for --s3-uri input") from exc
    if not s3_uri.startswith("s3://") or "/" not in s3_uri[5:]:
        raise ValueError(f"invalid S3 URI: {s3_uri}")
    bucket, key = s3_uri[5:].split("/", 1)
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}, proxies={}),
    )
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    wrapper = io.TextIOWrapper(body, encoding="utf-8-sig", errors="replace", newline="")
    try:
        yield csv.DictReader(wrapper)
    finally:
        wrapper.close()


def open_local_csv(path: Path) -> tuple[io.TextIOBase, Iterable[dict[str, str]]]:
    handle = path.open("r", encoding="utf-8-sig", errors="replace", newline="")
    return handle, csv.DictReader(handle)


def select_paper_context(text: str, *, max_chars: int = 60000) -> str:
    """Prioritize data/method/result evidence from parsed paper text."""
    normalized = str(text or "").replace("\x00", "").strip()
    if len(normalized) <= max_chars:
        return normalized
    patterns = (
        ("DATA AVAILABILITY", r"\bdata availability\b"),
        ("CODE AVAILABILITY", r"\bcode availability\b"),
        ("METHODS", r"(?im)^\s*(?:#{1,4}\s*)?methods?\s*$"),
        ("STATISTICS", r"\b(?:statistical analysis|quantification and statistical analysis)\b"),
        ("RESULTS", r"(?im)^\s*(?:#{1,4}\s*)?results?\s*$"),
    )
    pieces: list[str] = []
    used: list[tuple[int, int]] = []
    budget = max_chars
    for label, pattern in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        start = max(0, match.start() - 1000)
        end = min(len(normalized), match.start() + 10000)
        if any(start < old_end and end > old_start for old_start, old_end in used):
            continue
        chunk = normalized[start:end]
        if len(chunk) > budget:
            chunk = chunk[:budget]
        pieces.append(f"\n## EXTRACTED {label}\n{chunk}")
        used.append((start, end))
        budget -= len(chunk)
        if budget <= 4000:
            break
    if budget > 0:
        pieces.append("\n## PAPER OPENING\n" + normalized[:budget])
    return "".join(pieces)[:max_chars]


def extract_s3_paper_contexts(
    verified_path: Path,
    output_path: Path,
    *,
    s3_uri: str,
    endpoint_url: str,
    access_key: str | None = None,
    secret_key: str | None = None,
    max_chars: int = 60000,
) -> ValidationReport:
    try:
        import boto3  # type: ignore
        from botocore.config import Config  # type: ignore
    except ImportError as exc:
        raise RuntimeError("boto3 is required for S3 context extraction") from exc
    if not s3_uri.startswith("s3://") or "/" not in s3_uri[5:]:
        raise ValueError(f"invalid S3 URI: {s3_uri}")
    bucket, key = s3_uri[5:].split("/", 1)
    papers = read_jsonl(verified_path)
    target_by_track = {
        str((row.get("corpus_source") or {}).get("track_id") or ""): row
        for row in papers
        if row.get("status") == VERIFIED and (row.get("corpus_source") or {}).get("track_id")
    }
    report = ValidationReport()
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}, proxies={}),
    )
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    found: dict[str, dict[str, Any]] = {}
    rows_seen = 0
    try:
        for raw in body.iter_lines(chunk_size=8 * 1024 * 1024):
            rows_seen += 1
            if rows_seen % 250 == 0:
                print(f"context scan: rows={rows_seen}, matched={len(found)}/{len(target_by_track)}", flush=True)
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                report.warn("bad_corpus_json", f"row {rows_seen}: {exc}")
                continue
            track_id = str(row.get("track_id") or "")
            paper = target_by_track.get(track_id)
            if paper is None:
                continue
            blocks = row.get("file_content") or []
            full_text = "\n\n".join(
                str(item.get("content") or "")
                for item in blocks
                if isinstance(item, dict) and item.get("type") == "text"
            )
            context = select_paper_context(full_text, max_chars=max_chars)
            found[track_id] = {
                "schema_version": SCHEMA_VERSION,
                "paper_id": paper["paper_id"],
                "doi": paper.get("doi", ""),
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "text": context,
                "full_text_chars": len(full_text),
                "context_chars": len(context),
                "corpus_source": {"uri": s3_uri, "track_id": track_id},
            }
            if len(found) == len(target_by_track):
                break
    finally:
        body.close()
    ordered = [found[track] for track in target_by_track if track in found]
    write_jsonl(output_path, ordered)
    missing = sorted(set(target_by_track) - set(found))
    for track_id in missing:
        report.warn("paper_context_missing", track_id)
    report.stats.update({
        "target_papers": len(target_by_track),
        "contexts_found": len(ordered),
        "rows_scanned": rows_seen,
        "missing": len(missing),
    })
    write_json(output_path.with_suffix(".report.json"), report.to_dict())
    return report


def _http_json(url: str, timeout: int) -> tuple[int, str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "BrainArenaDatasetVerifier/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.geturl(), json.load(response)


def _http_text(url: str, timeout: int) -> tuple[int, str, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "BrainArenaDatasetVerifier/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.geturl(), response.read().decode("utf-8", errors="replace")


def _http_probe(url: str, timeout: int) -> tuple[int, str, dict[str, str]]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BrainArenaDatasetVerifier/1.0", "Range": "bytes=0-0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.geturl(), dict(response.headers.items())


def _safe_remote_file_path(name: Any) -> str:
    """Convert repository paths to safe paths relative to a dataset root."""
    raw = str(name or "").replace("\\", "/").lstrip("/")
    pure = PurePosixPath(raw)
    if not raw or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe remote file path: {name!r}")
    return pure.as_posix()


def _file_entry(name: str, url: str, size: Any = None, checksum: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {"path": _safe_remote_file_path(name), "download_url": url}
    if size not in (None, ""):
        row["bytes"] = int(size)
    if checksum:
        row["checksum"] = checksum
    return row


def _response_total_bytes(headers: dict[str, str]) -> int | None:
    """Return the full object size from a ranged or ordinary HTTP response."""
    normalized = {str(key).casefold(): str(value) for key, value in headers.items()}
    content_range = normalized.get("content-range", "")
    match = re.search(r"/\s*(\d+)\s*$", content_range)
    if match:
        return int(match.group(1))
    content_length = normalized.get("content-length", "").strip()
    return int(content_length) if content_length.isdigit() else None


def _geo_series_prefix(accession: str) -> str:
    return re.sub(r"\d{3}$", "nnn", accession)


def _geo_file_links(index_url: str, html: str, accession: str) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    for href in re.findall(r'href=["\']([^"\']+)["\']', html, re.IGNORECASE):
        decoded = urllib.parse.unquote(href)
        name = Path(urllib.parse.urlsplit(decoded).path).name
        if not name or name in seen or name.casefold() == "filelist.txt":
            continue
        if not name.startswith(accession + "_"):
            continue
        url = urllib.parse.urljoin(index_url, href)
        seen.add(name)
        links.append((name, url))
    return links


def _verify_geo(
    locator: dict[str, Any],
    timeout: int,
    get_text: Callable[[str, int], tuple[int, str, str]],
    probe: Callable[[str, int], tuple[int, str, dict[str, str]]],
) -> dict[str, Any]:
    accession = str(locator.get("record_id") or "").upper()
    if not re.fullmatch(r"GSE\d+", accession):
        return {"status": REJECTED, "reason": "invalid GEO series accession", "files": [], "file_count": 0}

    base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{_geo_series_prefix(accession)}/{accession}/"
    listing_url = base + "suppl/"
    try:
        status, resolved, html = get_text(listing_url, timeout)
        candidates = _geo_file_links(resolved, html, accession) if status == 200 else []
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        candidates = []

    source_kind = "supplementary"
    if not candidates:
        source_kind = "series_matrix"
        listing_url = base + "matrix/"
        status, resolved, html = get_text(listing_url, timeout)
        candidates = _geo_file_links(resolved, html, accession) if status == 200 else []
    if not candidates:
        return {
            "status": REVIEW,
            "record_id": accession,
            "api_url": listing_url,
            "reason": "GEO record exists but no downloadable supplementary or series-matrix files were listed",
            "files": [],
            "file_count": 0,
        }
    if len(candidates) > MAX_GEO_FILES:
        return {
            "status": REVIEW,
            "record_id": accession,
            "api_url": listing_url,
            "reason": f"GEO listing exceeds safety cap of {MAX_GEO_FILES} files",
            "files": [],
            "file_count": len(candidates),
        }

    files: list[dict[str, Any]] = []
    unresolved = 0
    for name, url in candidates:
        file_status, download_url, headers = probe(url, timeout)
        total_bytes = _response_total_bytes(headers)
        if file_status not in {200, 206} or total_bytes is None:
            unresolved += 1
            continue
        files.append(_file_entry(name, download_url, total_bytes))
    return {
        "status": VERIFIED if files and unresolved == 0 else REVIEW,
        "record_id": accession,
        "api_url": listing_url,
        "source_kind": source_kind,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "unresolved_files": unresolved,
        **({"reason": "one or more GEO files could not be size-verified"} if unresolved else {}),
    }


def _verify_figshare(locator: dict[str, Any], timeout: int, get_json: Callable) -> dict[str, Any]:
    record_id = locator["record_id"]
    status, resolved, payload = get_json(f"https://api.figshare.com/v2/articles/{record_id}", timeout)
    files = [
        _file_entry(str(item.get("name") or item.get("id")), str(item.get("download_url") or ""), item.get("size"), str(item.get("computed_md5") or ""))
        for item in payload.get("files", []) if item.get("download_url")
    ]
    return {
        "status": VERIFIED if status == 200 and files else REVIEW,
        "api_url": resolved,
        "record_id": str(payload.get("id") or record_id),
        "version": str(payload.get("version") or locator.get("version") or ""),
        "license": (payload.get("license") or {}).get("name", "") if isinstance(payload.get("license"), dict) else str(payload.get("license") or ""),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item.get("bytes", 0)) for item in files),
    }


def _verify_zenodo(locator: dict[str, Any], timeout: int, get_json: Callable) -> dict[str, Any]:
    record_id = locator["record_id"]
    status, resolved, payload = get_json(f"https://zenodo.org/api/records/{record_id}", timeout)
    files = []
    for item in payload.get("files", []):
        links = item.get("links") or {}
        url = links.get("content") or links.get("self") or ""
        if url:
            files.append(_file_entry(str(item.get("key") or item.get("id")), str(url), item.get("size"), str(item.get("checksum") or "")))
    metadata = payload.get("metadata") or {}
    return {
        "status": VERIFIED if status == 200 and files else REVIEW,
        "api_url": resolved,
        "record_id": str(payload.get("id") or record_id),
        "version": str(metadata.get("version") or ""),
        "license": str(metadata.get("license") or ""),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item.get("bytes", 0)) for item in files),
    }


def _verify_github(locator: dict[str, Any], timeout: int, get_json: Callable) -> dict[str, Any]:
    repository = locator["record_id"]
    status, _, repo = get_json(f"https://api.github.com/repos/{repository}", timeout)
    if status != 200:
        return {"status": REJECTED, "reason": f"GitHub repository returned HTTP {status}"}
    branch = str(repo.get("default_branch") or "HEAD")
    _, tree_url, tree = get_json(f"https://api.github.com/repos/{repository}/git/trees/{branch}?recursive=1", timeout)
    sha = str(tree.get("sha") or branch)
    files = []
    for item in tree.get("tree", []):
        path = str(item.get("path") or "")
        if item.get("type") == "blob" and _DATA_EXTENSIONS.search(path):
            raw = f"https://raw.githubusercontent.com/{repository}/{sha}/{urllib.parse.quote(path)}"
            files.append(_file_entry(path, raw, item.get("size"), str(item.get("sha") or "")))
    return {
        "status": VERIFIED if files else REJECTED,
        "reason": "" if files else "repository contains no recognizable data files",
        "api_url": tree_url,
        "record_id": repository,
        "version": sha,
        "archive_url": f"https://api.github.com/repos/{repository}/zipball/{sha}",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item.get("bytes", 0)) for item in files),
    }


def _verify_gin(
    locator: dict[str, Any],
    timeout: int,
    get_json: Callable,
    get_text: Callable,
    probe: Callable,
) -> dict[str, Any]:
    """Verify a public GIN repository as a commit-pinned archive."""
    record_id = str(locator.get("record_id") or "")
    if record_id.startswith("10.12751/"):
        status, resolved, _ = get_text(str(locator.get("original_url") or ""), timeout)
        if status != 200:
            return {
                "status": REVIEW,
                "reason": f"GIN DOI resolver returned HTTP {status}",
                "record_id": record_id,
                "files": [],
                "file_count": 0,
            }
        match = re.search(r"gin\.g-node\.org/([^/]+)/([^/?#]+)", resolved, re.I)
        if not match:
            return {
                "status": REVIEW,
                "reason": "GIN DOI did not resolve to a repository",
                "record_id": record_id,
                "files": [],
                "file_count": 0,
            }
        record_id = f"{match.group(1)}/{match.group(2).removesuffix('.git')}"
    if not re.fullmatch(r"[^/]+/[^/]+", record_id):
        return {
            "status": REVIEW,
            "reason": "GIN link does not identify a repository",
            "record_id": record_id,
            "files": [],
            "file_count": 0,
        }
    owner, name = record_id.split("/", 1)
    status, api_url, metadata = get_json(
        f"https://gin.g-node.org/api/v1/repos/{owner}/{name}", timeout,
    )
    if status != 200 or bool(metadata.get("private")):
        return {
            "status": REVIEW if status in {200, 401, 403} else REJECTED,
            "reason": "GIN repository is unavailable or not public",
            "api_url": api_url,
            "record_id": record_id,
            "files": [],
            "file_count": 0,
        }
    branch = str(metadata.get("default_branch") or "master")
    refs_url = f"https://gin.g-node.org/{owner}/{name}.git/info/refs?service=git-upload-pack"
    refs_status, _, refs = get_text(refs_url, timeout)
    sha_match = re.search(
        rf"([0-9a-f]{{40}}) refs/heads/{re.escape(branch)}(?:\n|\x00)", refs, re.I,
    )
    if refs_status != 200 or not sha_match:
        return {
            "status": REVIEW,
            "reason": "GIN repository did not disclose the default-branch commit",
            "api_url": api_url,
            "record_id": record_id,
            "files": [],
            "file_count": 0,
        }
    sha = sha_match.group(1).lower()
    archive_url = f"https://gin.g-node.org/{owner}/{name}/archive/{sha}.zip"
    archive_status, resolved, headers = probe(archive_url, timeout)
    total_bytes = _response_total_bytes(headers)
    if archive_status not in {200, 206} or total_bytes is None:
        return {
            "status": REVIEW,
            "reason": "GIN archive did not disclose the complete object size",
            "api_url": api_url,
            "record_id": record_id,
            "files": [],
            "file_count": 0,
        }
    filename = f"{name}-{sha}.zip"
    return {
        "status": VERIFIED,
        "api_url": api_url,
        "record_id": record_id,
        "version": sha,
        "files": [_file_entry(filename, resolved, total_bytes)],
        "file_count": 1,
        "total_bytes": total_bytes,
    }


def _verify_osf(locator: dict[str, Any], timeout: int, get_json: Callable) -> dict[str, Any]:
    node = locator["record_id"]
    status, resolved, payload = get_json(f"https://api.osf.io/v2/nodes/{node}/files/", timeout)
    files: list[dict[str, Any]] = []
    seen_files: set[str] = set()
    listing_urls: list[str] = []
    for provider in payload.get("data", []):
        attributes = provider.get("attributes") or {}
        provider_name = str(attributes.get("name") or provider.get("id") or "").casefold()
        # A linked source-code provider is not evidence that the OSF node hosts
        # the paper's experimental data.  Keep native OSF storage and other
        # non-code storage providers, but fail closed on Git hosting mounts.
        if "github" in provider_name or "gitlab" in provider_name:
            continue
        related = ((provider.get("relationships") or {}).get("files") or {}).get("links", {}).get("related", {}).get("href")
        if not related:
            related = (provider.get("links") or {}).get("related")
        if related:
            listing_urls.append(str(related))

    visited: set[str] = set()
    listing_pages = 0
    while listing_urls:
        url = listing_urls.pop()
        if not url or url in visited:
            continue
        visited.add(url)
        listing_pages += 1
        if listing_pages > MAX_OSF_LISTING_PAGES:
            return {
                "status": REVIEW,
                "reason": f"OSF listing exceeded {MAX_OSF_LISTING_PAGES} pages/folders",
                "api_url": resolved,
                "record_id": node,
                "files": [],
                "file_count": 0,
            }
        _, _, listing = get_json(url, timeout)
        for item in listing.get("data", []):
            attrs = item.get("attributes") or {}
            links = item.get("links") or {}
            if attrs.get("kind") == "file" and links.get("download"):
                download = str(links["download"])
                file_id = str(item.get("id") or download)
                if file_id in seen_files:
                    continue
                seen_files.add(file_id)
                hashes = (attrs.get("extra") or {}).get("hashes") or {}
                checksum = ""
                if hashes.get("sha256"):
                    checksum = f"sha256:{hashes['sha256']}"
                elif hashes.get("md5"):
                    checksum = f"md5:{hashes['md5']}"
                files.append(_file_entry(
                    str(attrs.get("materialized_path") or attrs.get("name") or item.get("id")),
                    download,
                    attrs.get("size"),
                    checksum,
                ))
                if len(files) > MAX_OSF_FILES:
                    return {
                        "status": REVIEW,
                        "reason": f"OSF listing exceeded {MAX_OSF_FILES} files",
                        "api_url": resolved,
                        "record_id": node,
                        "files": [],
                        "file_count": 0,
                    }
            elif attrs.get("kind") == "folder":
                child = (((item.get("relationships") or {}).get("files") or {}).get("links") or {}).get("related") or {}
                child_url = child.get("href") if isinstance(child, dict) else child
                if child_url:
                    listing_urls.append(str(child_url))
        next_url = (listing.get("links") or {}).get("next")
        if next_url:
            listing_urls.append(str(next_url))
    return {
        "status": VERIFIED if status == 200 and files else REVIEW,
        "api_url": resolved,
        "record_id": node,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item.get("bytes", 0)) for item in files),
        "listing_pages": listing_pages,
    }


def _verify_openneuro(locator: dict[str, Any], timeout: int, get_text: Callable) -> dict[str, Any]:
    dataset = str(locator["record_id"])
    prefix = f"{dataset}/"
    base = "https://s3.amazonaws.com/openneuro.org"
    continuation = ""
    files: list[dict[str, Any]] = []
    pages = 0
    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if continuation:
            query["continuation-token"] = continuation
        url = base + "?" + urllib.parse.urlencode(query)
        status, resolved, payload = get_text(url, timeout)
        if status != 200:
            return {"status": DEFERRED_NETWORK, "reason": f"OpenNeuro S3 returned HTTP {status}"}
        root = ET.fromstring(payload)
        namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for content in root.findall("s3:Contents", namespace):
            key = str(content.findtext("s3:Key", default="", namespaces=namespace))
            if not key or key.endswith("/") or not key.startswith(prefix):
                continue
            path = key[len(prefix):]
            if not path:
                continue
            size = int(content.findtext("s3:Size", default="0", namespaces=namespace) or 0)
            etag = str(content.findtext("s3:ETag", default="", namespaces=namespace)).strip('"')
            checksum = f"md5:{etag}" if re.fullmatch(r"[0-9a-fA-F]{32}", etag) else ""
            files.append(_file_entry(
                path,
                f"{base}/{urllib.parse.quote(key, safe='/')}",
                size,
                checksum,
            ))
            if len(files) > MAX_OPENNEURO_FILES:
                return {
                    "status": REVIEW,
                    "reason": f"OpenNeuro listing exceeded {MAX_OPENNEURO_FILES} files",
                    "record_id": dataset,
                    "files": [],
                    "file_count": 0,
                }
        pages += 1
        truncated = str(root.findtext("s3:IsTruncated", default="false", namespaces=namespace)).casefold() == "true"
        continuation = str(root.findtext("s3:NextContinuationToken", default="", namespaces=namespace))
        if not truncated or not continuation:
            break
    return {
        "status": VERIFIED if files else REVIEW,
        "reason": "" if files else "OpenNeuro dataset contains no listed objects",
        "api_url": base + "?" + urllib.parse.urlencode({"list-type": "2", "prefix": prefix}),
        "record_id": dataset,
        "version": "s3-current",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item.get("bytes", 0)) for item in files),
        "listing_pages": pages,
    }


def _verify_neurovault(locator: dict[str, Any], timeout: int, get_json: Callable) -> dict[str, Any]:
    kind, _, identifier = str(locator["record_id"]).partition(":")
    if kind == "images":
        first_url = f"https://neurovault.org/api/images/{identifier}/?format=json"
    else:
        first_url = f"https://neurovault.org/api/collections/{identifier}/images/?format=json&limit=100"
    next_url = first_url
    files: list[dict[str, Any]] = []
    pages = 0
    while next_url:
        status, resolved, payload = get_json(next_url, timeout)
        if status != 200:
            return {"status": DEFERRED_NETWORK, "reason": f"NeuroVault returned HTTP {status}"}
        rows = payload.get("results") if isinstance(payload, dict) else None
        rows = rows if isinstance(rows, list) else [payload]
        for image in rows:
            file_url = str((image or {}).get("file") or "")
            if not file_url:
                continue
            name = Path(urllib.parse.urlsplit(file_url).path).name or f"image-{image.get('id')}.nii.gz"
            files.append(_file_entry(name, file_url, image.get("file_size")))
            if len(files) > MAX_NEUROVAULT_FILES:
                return {
                    "status": REVIEW,
                    "reason": f"NeuroVault listing exceeded {MAX_NEUROVAULT_FILES} files",
                    "record_id": identifier,
                    "files": [],
                    "file_count": 0,
                }
        pages += 1
        next_url = str(payload.get("next") or "") if kind != "images" else ""
    return {
        "status": VERIFIED if files else REVIEW,
        "reason": "" if files else "NeuroVault record contains no downloadable images",
        "api_url": first_url,
        "record_id": identifier,
        "version": "api-current",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item.get("bytes", 0)) for item in files),
        "listing_pages": pages,
    }


def _verify_encode(locator: dict[str, Any], timeout: int, get_json: Callable) -> dict[str, Any]:
    accession = str(locator["record_id"]).upper()
    api_url = f"https://www.encodeproject.org/{accession}/?format=json"
    status, resolved, payload = get_json(api_url, timeout)
    if status != 200:
        return {"status": DEFERRED_NETWORK, "reason": f"ENCODE returned HTTP {status}"}

    if accession.startswith("ENCFF"):
        metadata_rows = [payload]
    else:
        file_refs = payload.get("files") or []
        if len(file_refs) > MAX_EBI_FILES:
            return {
                "status": REVIEW,
                "reason": f"ENCODE record exceeded {MAX_EBI_FILES} files",
                "record_id": accession,
                "files": [],
                "file_count": 0,
            }
        metadata_rows = []
        for ref in file_refs:
            if isinstance(ref, dict):
                metadata_rows.append(ref)
                continue
            file_url = urllib.parse.urljoin("https://www.encodeproject.org", str(ref))
            separator = "&" if "?" in file_url else "?"
            file_status, _, file_payload = get_json(file_url + separator + "format=json", timeout)
            if file_status == 200 and isinstance(file_payload, dict):
                metadata_rows.append(file_payload)

    files: list[dict[str, Any]] = []
    for item in metadata_rows:
        if str(item.get("status") or "released").casefold() not in {"released", "archived"}:
            continue
        href = str(item.get("href") or "")
        size = item.get("file_size")
        if not href or not str(size or "").isdigit():
            continue
        download_url = urllib.parse.urljoin("https://www.encodeproject.org", href)
        name = Path(urllib.parse.urlsplit(download_url).path).name or str(item.get("accession") or accession)
        files.append(_file_entry(name, download_url, int(size), _checksum_with_algorithm(item.get("md5sum"))))
    return {
        "status": VERIFIED if files else REVIEW,
        "reason": "" if files else "ENCODE record contains no size-verified public files",
        "api_url": resolved,
        "record_id": accession,
        "version": str(payload.get("date_created") or payload.get("schema_version") or "api-current"),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item.get("bytes", 0)) for item in files),
    }


def _verify_dandi(locator: dict[str, Any], timeout: int, get_json: Callable) -> dict[str, Any]:
    dandiset = str(locator["record_id"])
    version = str(locator.get("version") or "")
    if not version:
        metadata_url = f"https://api.dandiarchive.org/api/dandisets/{dandiset}/"
        status, _, metadata = get_json(metadata_url, timeout)
        if status != 200:
            return {"status": DEFERRED_NETWORK, "reason": f"DANDI metadata returned HTTP {status}"}
        recent = metadata.get("most_recent_published_version") or {}
        version = str(recent.get("version") or "draft")
    first_url = (
        f"https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/"
        f"{urllib.parse.quote(version, safe='')}/assets/?page_size=100"
    )
    next_url = first_url
    files: list[dict[str, Any]] = []
    pages = 0
    while next_url:
        status, resolved, payload = get_json(next_url, timeout)
        if status != 200:
            return {"status": DEFERRED_NETWORK, "reason": f"DANDI returned HTTP {status}"}
        for asset in payload.get("results", []):
            asset_id = str(asset.get("asset_id") or asset.get("assetId") or asset.get("identifier") or "")
            path = str(asset.get("path") or asset_id)
            if not asset_id or not path:
                continue
            files.append(_file_entry(
                path,
                f"https://api.dandiarchive.org/api/assets/{asset_id}/download/",
                asset.get("size") or asset.get("blob_size"),
                str(asset.get("digest") or ""),
            ))
            if len(files) > MAX_DANDI_FILES:
                return {
                    "status": REVIEW,
                    "reason": f"DANDI listing exceeded {MAX_DANDI_FILES} assets",
                    "record_id": dandiset,
                    "files": [],
                    "file_count": 0,
                }
        pages += 1
        next_url = str(payload.get("next") or "")
    return {
        "status": VERIFIED if files else REVIEW,
        "reason": "" if files else "DANDI record contains no downloadable assets",
        "api_url": first_url,
        "record_id": dandiset,
        "version": version,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item.get("bytes", 0)) for item in files),
        "listing_pages": pages,
    }


def _checksum_with_algorithm(value: Any) -> str:
    checksum = str(value or "").strip()
    if re.fullmatch(r"[0-9a-fA-F]{32}", checksum):
        return f"md5:{checksum}"
    if re.fullmatch(r"[0-9a-fA-F]{40}", checksum):
        return f"sha1:{checksum}"
    if re.fullmatch(r"[0-9a-fA-F]{64}", checksum):
        return f"sha256:{checksum}"
    return checksum


def _verify_dryad(locator: dict[str, Any], timeout: int, get_json: Callable) -> dict[str, Any]:
    doi = str(locator["record_id"])
    dataset_url = "https://datadryad.org/api/v2/datasets/" + urllib.parse.quote(
        f"doi:{doi}", safe="",
    )
    status, resolved, dataset = get_json(dataset_url, timeout)
    if status != 200:
        return {"status": DEFERRED_NETWORK, "reason": f"Dryad metadata returned HTTP {status}"}

    version_link = (dataset.get("_links") or {}).get("stash:version") or {}
    version_url = str(version_link.get("href") or "") if isinstance(version_link, dict) else str(version_link or "")
    if not version_url:
        return {
            "status": REVIEW,
            "reason": "Dryad dataset did not expose its current version",
            "api_url": resolved,
            "record_id": doi,
            "files": [],
            "file_count": 0,
        }
    version_url = urllib.parse.urljoin("https://datadryad.org", version_url)
    version_status, _, version = get_json(version_url, timeout)
    if version_status != 200:
        return {"status": DEFERRED_NETWORK, "reason": f"Dryad version returned HTTP {version_status}"}
    files_link = (version.get("_links") or {}).get("stash:files") or {}
    next_url = str(files_link.get("href") or "") if isinstance(files_link, dict) else str(files_link or "")
    next_url = next_url or version_url.rstrip("/") + "/files"
    next_url = urllib.parse.urljoin("https://datadryad.org", next_url)

    files: list[dict[str, Any]] = []
    pages = 0
    while next_url:
        file_status, _, payload = get_json(next_url, timeout)
        if file_status != 200:
            return {"status": DEFERRED_NETWORK, "reason": f"Dryad files returned HTTP {file_status}"}
        rows = (payload.get("_embedded") or {}).get("stash:files") or payload.get("files") or []
        for item in rows:
            item_links = item.get("_links") or {}
            download = item_links.get("stash:download") or item_links.get("download") or {}
            download_url = str(download.get("href") or "") if isinstance(download, dict) else str(download or "")
            path = str(item.get("path") or item.get("name") or item.get("id") or "")
            if not path or not download_url:
                continue
            checksum = item.get("digest") or item.get("md5") or item.get("checksum") or ""
            files.append(_file_entry(
                path,
                urllib.parse.urljoin("https://datadryad.org", download_url),
                item.get("size"),
                _checksum_with_algorithm(checksum),
            ))
            if len(files) > MAX_EBI_FILES:
                return {
                    "status": REVIEW,
                    "reason": f"Dryad listing exceeded {MAX_EBI_FILES} files",
                    "record_id": doi,
                    "files": [],
                    "file_count": 0,
                }
        pages += 1
        next_link = (payload.get("_links") or {}).get("next") or {}
        next_url = str(next_link.get("href") or "") if isinstance(next_link, dict) else str(next_link or "")
        next_url = urllib.parse.urljoin("https://datadryad.org", next_url) if next_url else ""
    version_number = version.get("versionNumber") or version.get("version") or version.get("id") or ""
    return {
        "status": VERIFIED if files else REVIEW,
        "reason": "" if files else "Dryad version contains no downloadable files",
        "api_url": resolved,
        "record_id": doi,
        "version": str(version_number),
        "license": str(version.get("license") or dataset.get("license") or ""),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item.get("bytes", 0)) for item in files),
        "listing_pages": pages,
    }


def _verify_dataverse(locator: dict[str, Any], timeout: int, get_json: Callable) -> dict[str, Any]:
    doi = str(locator["record_id"])
    prefix_hosts = {
        "10.7910/DVN/": "https://dataverse.harvard.edu",
        "10.18738/T8/": "https://dataverse.tdl.org",
        "10.11588/DATA/": "https://heidata.uni-heidelberg.de",
    }
    base = next((host for prefix, host in prefix_hosts.items() if doi.upper().startswith(prefix)), "")
    if not base:
        return {"status": REVIEW, "reason": "unsupported Dataverse DOI authority"}
    api_url = base + "/api/datasets/:persistentId/?" + urllib.parse.urlencode({
        "persistentId": f"doi:{doi}",
    })
    status, resolved, payload = get_json(api_url, timeout)
    data = payload.get("data") or {}
    latest = data.get("latestVersion") or {}
    if status != 200 or not latest:
        return {"status": DEFERRED_NETWORK, "reason": f"Dataverse metadata returned HTTP {status}"}
    files: list[dict[str, Any]] = []
    restricted = 0
    for wrapper in latest.get("files") or []:
        item = wrapper.get("dataFile") or {}
        if wrapper.get("restricted") or item.get("restricted"):
            restricted += 1
            continue
        file_id = item.get("id")
        name = str(item.get("filename") or file_id or "")
        size = item.get("filesize")
        if not file_id or not name or not str(size or "").isdigit():
            continue
        directory = str(wrapper.get("directoryLabel") or "").strip("/")
        path = f"{directory}/{name}" if directory else name
        checksum_row = item.get("checksum") or {}
        algorithm = str(checksum_row.get("type") or "").casefold().replace("-", "")
        checksum_value = str(checksum_row.get("value") or "")
        checksum = f"{algorithm}:{checksum_value}" if algorithm and checksum_value else ""
        files.append(_file_entry(
            path,
            f"{base}/api/access/datafile/{file_id}",
            int(size),
            checksum,
        ))
    complete = bool(files) and restricted == 0
    return {
        "status": VERIFIED if complete else REVIEW,
        "reason": "" if complete else (
            f"Dataverse dataset includes {restricted} restricted files" if restricted
            else "Dataverse dataset contains no size-verified public files"
        ),
        "api_url": resolved,
        "record_id": doi,
        "version": str(latest.get("versionNumber") or ""),
        "license": str(latest.get("license") or latest.get("termsOfUse") or ""),
        "files": files if complete else [],
        "file_count": len(files) if complete else 0,
        "total_bytes": sum(int(item.get("bytes", 0)) for item in files) if complete else 0,
        "restricted_files": restricted,
    }


def _verify_pride(locator: dict[str, Any], timeout: int, get_json: Callable) -> dict[str, Any]:
    accession = str(locator["record_id"])
    api_url = f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{accession}/files"
    status, resolved, payload = get_json(api_url, timeout)
    files: list[dict[str, Any]] = []
    for item in payload if isinstance(payload, list) else []:
        locations = item.get("publicFileLocations") or []
        location = next((
            str(row.get("value") or "") for row in locations
            if "ftp" in str(row.get("name") or "").casefold()
            or str(row.get("value") or "").startswith("ftp://")
        ), "")
        if location.startswith("ftp://ftp.pride.ebi.ac.uk/"):
            location = "https://ftp.pride.ebi.ac.uk/" + location.removeprefix("ftp://ftp.pride.ebi.ac.uk/")
        if not location.startswith(("http://", "https://")):
            continue
        files.append(_file_entry(
            str(item.get("fileName") or Path(urllib.parse.urlsplit(location).path).name),
            location,
            item.get("fileSizeBytes"),
            _checksum_with_algorithm(item.get("checksum")),
        ))
        if len(files) > MAX_EBI_FILES:
            return {
                "status": REVIEW,
                "reason": f"PRIDE listing exceeded {MAX_EBI_FILES} files",
                "record_id": accession,
                "files": [],
                "file_count": 0,
            }
    return {
        "status": VERIFIED if status == 200 and files else REVIEW,
        "reason": "" if files else "PRIDE project contains no public files",
        "api_url": resolved,
        "record_id": accession,
        "version": "archive-current",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item.get("bytes", 0)) for item in files),
    }


def _verify_biostudies(locator: dict[str, Any], timeout: int, get_json: Callable) -> dict[str, Any]:
    accession = str(locator["record_id"])
    api_url = f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}"
    status, resolved, payload = get_json(api_url, timeout)
    found: dict[str, dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            path = str(value.get("path") or "")
            if path and value.get("type") == "file":
                safe = _safe_remote_file_path(path)
                found.setdefault(safe, _file_entry(
                    safe,
                    f"https://www.ebi.ac.uk/biostudies/files/{accession}/{urllib.parse.quote(safe, safe='/')}",
                    value.get("size"),
                ))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    files = list(found.values())
    if len(files) > MAX_EBI_FILES:
        return {
            "status": REVIEW,
            "reason": f"BioStudies listing exceeded {MAX_EBI_FILES} files",
            "record_id": accession,
            "files": [],
            "file_count": 0,
        }
    return {
        "status": VERIFIED if status == 200 and files else REVIEW,
        "reason": "" if files else "BioStudies record contains no public files",
        "api_url": resolved,
        "record_id": accession,
        "version": "api-current",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item.get("bytes", 0)) for item in files),
    }


def _verify_single_public_object(
    *, repository: str, record_id: str, url: str, filename: str, timeout: int, probe: Callable,
) -> dict[str, Any]:
    status, resolved, headers = probe(url, timeout)
    total_bytes = _response_total_bytes(headers)
    if status not in {200, 206} or total_bytes is None:
        return {
            "status": REVIEW if status in {200, 206} else REJECTED,
            "reason": "download endpoint did not disclose the complete object size",
            "record_id": record_id,
            "files": [],
            "file_count": 0,
        }
    return {
        "status": VERIFIED,
        "record_id": record_id,
        "version": "archive-current",
        "api_url": url,
        "files": [_file_entry(filename, resolved, total_bytes)],
        "file_count": 1,
        "total_bytes": total_bytes,
        "repository": repository,
    }


def _verify_emdb(locator: dict[str, Any], timeout: int, probe: Callable) -> dict[str, Any]:
    accession = str(locator["record_id"]).upper()
    number = accession.removeprefix("EMD-")
    filename = f"emd_{number}.map.gz"
    return _verify_single_public_object(
        repository="emdb",
        record_id=accession,
        url=f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/{accession}/map/{filename}",
        filename=filename,
        timeout=timeout,
        probe=probe,
    )


def _verify_rcsb(locator: dict[str, Any], timeout: int, probe: Callable) -> dict[str, Any]:
    accession = str(locator["record_id"]).upper()
    filename = f"{accession}.cif"
    return _verify_single_public_object(
        repository="rcsb",
        record_id=accession,
        url=f"https://files.rcsb.org/download/{filename}",
        filename=filename,
        timeout=timeout,
        probe=probe,
    )


def _verify_sasbdb(locator: dict[str, Any], timeout: int, probe: Callable) -> dict[str, Any]:
    accession = str(locator["record_id"]).upper()
    return _verify_single_public_object(
        repository="sasbdb",
        record_id=accession,
        url=f"https://www.sasbdb.org/media/zip_directories/{accession}.zip",
        filename=f"{accession}.zip",
        timeout=timeout,
        probe=probe,
    )


def _verify_ena(locator: dict[str, Any], timeout: int, get_json: Callable) -> dict[str, Any]:
    accession = str(locator["record_id"]).upper()
    fields = (
        "run_accession,fastq_ftp,fastq_bytes,fastq_md5,"
        "submitted_ftp,submitted_bytes,submitted_md5"
    )
    api_url = "https://www.ebi.ac.uk/ena/portal/api/filereport?" + urllib.parse.urlencode({
        "accession": accession,
        "result": "read_run",
        "fields": fields,
        "format": "json",
    })
    status, resolved, payload = get_json(api_url, timeout)
    files: dict[str, dict[str, Any]] = {}
    for run in payload if isinstance(payload, list) else []:
        prefix = "fastq" if run.get("fastq_ftp") else "submitted"
        urls = str(run.get(f"{prefix}_ftp") or "").split(";")
        sizes = str(run.get(f"{prefix}_bytes") or "").split(";")
        checksums = str(run.get(f"{prefix}_md5") or "").split(";")
        for index, raw_url in enumerate(urls):
            raw_url = raw_url.strip()
            if not raw_url:
                continue
            if raw_url.startswith("ftp://"):
                download_url = "https://" + raw_url.removeprefix("ftp://")
            elif raw_url.startswith(("http://", "https://")):
                download_url = raw_url
            else:
                download_url = "https://" + raw_url.lstrip("/")
            path = Path(urllib.parse.urlsplit(download_url).path).name
            size = sizes[index].strip() if index < len(sizes) else ""
            checksum = checksums[index].strip() if index < len(checksums) else ""
            if path:
                files.setdefault(path, _file_entry(
                    path,
                    download_url,
                    int(size) if size.isdigit() else None,
                    _checksum_with_algorithm(checksum),
                ))
            if len(files) > MAX_EBI_FILES:
                return {
                    "status": REVIEW,
                    "reason": f"ENA listing exceeded {MAX_EBI_FILES} files",
                    "record_id": accession,
                    "files": [],
                    "file_count": 0,
                }
    output = list(files.values())
    return {
        "status": VERIFIED if status == 200 and output else REVIEW,
        "reason": "" if output else "ENA accession contains no public read files",
        "api_url": resolved,
        "record_id": accession,
        "version": "ena-current",
        "files": output,
        "file_count": len(output),
        "total_bytes": sum(int(item.get("bytes", 0)) for item in output),
    }


def verify_locator(
    locator: dict[str, Any],
    *,
    timeout: int = 20,
    get_json: Callable[[str, int], tuple[int, str, Any]] = _http_json,
    get_text: Callable[[str, int], tuple[int, str, str]] = _http_text,
    probe: Callable[[str, int], tuple[int, str, dict[str, str]]] = _http_probe,
) -> dict[str, Any]:
    checked = {**locator, "checked_at_utc": _utc_now()}
    repository = str(locator.get("repository") or "generic")
    try:
        if repository == "figshare":
            verification = _verify_figshare(locator, timeout, get_json)
        elif repository == "zenodo":
            verification = _verify_zenodo(locator, timeout, get_json)
        elif repository == "github":
            verification = _verify_github(locator, timeout, get_json)
        elif repository == "gin":
            verification = _verify_gin(locator, timeout, get_json, get_text, probe)
        elif repository == "osf":
            verification = _verify_osf(locator, timeout, get_json)
        elif repository == "geo":
            verification = _verify_geo(locator, timeout, get_text, probe)
        elif repository == "openneuro":
            verification = _verify_openneuro(locator, timeout, get_text)
        elif repository == "neurovault":
            verification = _verify_neurovault(locator, timeout, get_json)
        elif repository == "encode":
            verification = _verify_encode(locator, timeout, get_json)
        elif repository == "dandi":
            verification = _verify_dandi(locator, timeout, get_json)
        elif repository == "dryad":
            verification = _verify_dryad(locator, timeout, get_json)
        elif repository == "dataverse":
            verification = _verify_dataverse(locator, timeout, get_json)
        elif repository == "pride":
            verification = _verify_pride(locator, timeout, get_json)
        elif repository == "biostudies":
            verification = _verify_biostudies(locator, timeout, get_json)
        elif repository == "emdb":
            verification = _verify_emdb(locator, timeout, probe)
        elif repository == "rcsb":
            verification = _verify_rcsb(locator, timeout, probe)
        elif repository == "sasbdb":
            verification = _verify_sasbdb(locator, timeout, probe)
        elif repository == "ena":
            verification = _verify_ena(locator, timeout, get_json)
        elif locator.get("locator_kind") == "file":
            status, resolved, headers = probe(str(locator["original_url"]), timeout)
            total_bytes = _response_total_bytes(headers)
            is_downloadable = status in {200, 206} and total_bytes is not None
            verification = {
                "status": VERIFIED if is_downloadable else (REVIEW if status in {200, 206} else REJECTED),
                "resolved_url": resolved,
                "http_status": status,
                "files": [
                    _file_entry(Path(urllib.parse.urlsplit(resolved).path).name or "download", resolved, total_bytes)
                ] if status in {200, 206} else [],
                "file_count": 1 if status in {200, 206} else 0,
            }
            if status in {200, 206} and total_bytes is None:
                verification["reason"] = "download endpoint did not disclose the complete object size"
        else:
            status, resolved, _ = probe(str(locator["original_url"]), timeout)
            verification = {
                "status": REVIEW,
                "resolved_url": resolved,
                "http_status": status,
                "reason": "landing page is reachable but concrete dataset files were not verified",
                "files": [],
                "file_count": 0,
            }
    except urllib.error.HTTPError as exc:
        verification = {
            "status": REJECTED if exc.code in {401, 403, 404, 410} else DEFERRED_NETWORK,
            "reason": f"HTTP {exc.code}: {exc.reason}",
            "files": [],
            "file_count": 0,
        }
    except (TimeoutError, urllib.error.URLError) as exc:
        verification = {
            "status": DEFERRED_NETWORK,
            "reason": f"{type(exc).__name__}: {exc}",
            "files": [],
            "file_count": 0,
        }
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        verification = {"status": REJECTED, "reason": f"{type(exc).__name__}: {exc}", "files": [], "file_count": 0}
    checked["verification"] = verification
    return checked


def verify_candidate_links(
    candidates_path: Path,
    output_path: Path,
    *,
    jobs: int = 8,
    timeout: int = 20,
    max_retries: int = 2,
    verifier: Callable[..., dict[str, Any]] = verify_locator,
) -> ValidationReport:
    rows = read_jsonl(candidates_path)
    report = ValidationReport()
    unique: dict[str, dict[str, Any]] = {}
    paper_keys: list[list[str]] = []
    for row in rows:
        keys: list[str] = []
        for locator in row.get("data_locators", []):
            key = json.dumps({
                "repository": locator.get("repository"),
                "record_id": locator.get("record_id"),
                "version": locator.get("version"),
                "url": locator.get("original_url"),
            }, sort_keys=True, ensure_ascii=False)
            unique.setdefault(key, locator)
            keys.append(key)
        paper_keys.append(keys)

    def verify_with_retry(locator: dict[str, Any]) -> dict[str, Any]:
        checked = verifier(locator, timeout=timeout)
        attempts = 1
        while checked.get("verification", {}).get("status") == DEFERRED_NETWORK and attempts <= max_retries:
            checked = verifier(locator, timeout=timeout)
            attempts += 1
        checked["verification"]["attempts"] = attempts
        return checked

    checked_by_key: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        future_keys = {pool.submit(verify_with_retry, locator): key for key, locator in unique.items()}
        for future in as_completed(future_keys):
            checked_by_key[future_keys[future]] = future.result()

    output: list[dict[str, Any]] = []
    for row, keys in zip(rows, paper_keys):
        locators = [checked_by_key[key] for key in keys]
        verified = [item for item in locators if item.get("verification", {}).get("status") == VERIFIED]
        review = [item for item in locators if item.get("verification", {}).get("status") == REVIEW]
        deferred = [item for item in locators if item.get("verification", {}).get("status") == DEFERRED_NETWORK]
        status = VERIFIED if verified else (REVIEW if review else (DEFERRED_NETWORK if deferred else REJECTED))
        output.append({
            **row,
            "data_locators": locators,
            "status": status,
            "verified_locator_count": len(verified),
        })
    write_jsonl(output_path, output)
    counts = {
        status: sum(row.get("status") == status for row in output)
        for status in (VERIFIED, REVIEW, DEFERRED_NETWORK, REJECTED)
    }
    report.stats.update({"papers": len(output), "unique_locators_checked": len(unique), **counts})
    write_json(output_path.with_suffix(".report.json"), report.to_dict())
    return report


def build_acquisition_handoff(
    verified_path: Path,
    output_root: Path,
    *,
    max_bytes: int | None = DEFAULT_REMOTE_MAX_BYTES,
) -> ValidationReport:
    """Build a deduplicated download queue without fetching dataset payloads."""
    rows = read_jsonl(verified_path)
    report = ValidationReport()
    datasets: dict[str, dict[str, Any]] = {}
    paper_mapping: dict[str, dict[str, Any]] = {}
    deferred_oversize = 0

    for paper in rows:
        if paper.get("status") != VERIFIED:
            continue
        paper_id = str(paper.get("paper_id") or "")
        data_ids: list[str] = []
        for locator in paper.get("data_locators", []):
            verification = locator.get("verification") or {}
            if verification.get("status") != VERIFIED:
                continue
            repository = str(locator.get("repository") or "generic")
            record_id = str(verification.get("record_id") or locator.get("record_id") or locator.get("original_url") or "")
            version = str(verification.get("version") or locator.get("version") or "unspecified")
            digest_id = stable_id("remote", repository, record_id, version)
            dataset_id = f"{slug(repository, 18)}--{digest_id[-12:]}"
            files = []
            for raw_file in verification.get("files") or []:
                item = dict(raw_file)
                try:
                    item["path"] = _safe_remote_file_path(item.get("path"))
                except ValueError as exc:
                    report.error("unsafe_remote_file_path", f"{paper_id}: {exc}")
                    continue
                files.append(item)
            files.sort(key=lambda item: (
                str(item.get("path") or ""), str(item.get("download_url") or ""),
            ))
            if not files:
                report.error("verified_dataset_has_no_safe_files", f"{paper_id}: {record_id}")
                continue
            total_bytes = int(
                verification.get("total_bytes")
                or sum(int(item.get("bytes", 0)) for item in files)
            )
            if max_bytes is not None and total_bytes > max_bytes:
                deferred_oversize += 1
                report.warn(
                    "remote_dataset_exceeds_size_limit",
                    f"{dataset_id}: {total_bytes} bytes > {max_bytes}",
                )
                continue
            record = {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": dataset_id,
                "acquisition_status": "awaiting_data_download",
                "repository": repository,
                "record_id": record_id,
                "version": version,
                "license": str(verification.get("license") or "unknown"),
                "landing_url": str(locator.get("original_url") or ""),
                "api_url": str(verification.get("api_url") or ""),
                "archive_url": str(verification.get("archive_url") or ""),
                "target_relative_path": f"data/{dataset_id}",
                "file_count": int(verification.get("file_count") or len(files)),
                "total_bytes": total_bytes,
                "files": files,
                "source_papers": [],
                "payload_downloaded": False,
                "payload_hash_verified": False,
            }
            existing = datasets.get(dataset_id)
            if existing is None:
                datasets[dataset_id] = record
                existing = record
            elif existing["files"] != record["files"]:
                report.error("remote_manifest_conflict", dataset_id)
                continue
            if paper_id not in existing["source_papers"]:
                existing["source_papers"].append(paper_id)
            if dataset_id not in data_ids:
                data_ids.append(dataset_id)
        if data_ids:
            paper_mapping[paper_id] = {
                "paper_id": paper_id,
                "doi": str(paper.get("doi") or ""),
                "title": str(paper.get("title") or ""),
                "article_url": str(paper.get("article_url") or ""),
                "dataset_ids": data_ids,
                "data_locations": [f"data/{dataset_id}" for dataset_id in data_ids],
                "status": "awaiting_data_download",
            }

    output_root.mkdir(parents=True, exist_ok=True)
    queue = sorted(datasets.values(), key=lambda item: item["dataset_id"])
    for record in queue:
        dataset_root = output_root / "datasets" / record["dataset_id"]
        write_json(dataset_root / "data_locator.json", record)
    for record in paper_mapping.values():
        write_json(output_root / "papers" / record["paper_id"] / "paper_data.json", record)
    write_jsonl(output_root / "acquisition_queue.jsonl", queue)
    write_json(output_root / "paper_data_mapping.json", {
        "schema_version": SCHEMA_VERSION,
        "papers": paper_mapping,
    })
    report.stats.update({
        "input_papers": len(rows),
        "handoff_papers": len(paper_mapping),
        "unique_remote_datasets": len(queue),
        "deferred_oversize": deferred_oversize,
        "max_bytes": max_bytes,
        "downloaded_payload_bytes": 0,
        "declared_remote_bytes": sum(int(item["total_bytes"]) for item in queue),
    })
    write_json(output_root / "handoff_report.json", report.to_dict())
    return report
