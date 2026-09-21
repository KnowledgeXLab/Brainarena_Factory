#!/usr/bin/env python3
"""Resumable, checksummed downloader for BrainArena remote acquisition queues."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe relative path: {value!r}")
    return Path(*pure.parts)


def collision_name(path: Path, url: str, ordinal: int) -> Path:
    suffixes = "".join(path.suffixes)
    base = path.name[:-len(suffixes)] if suffixes else path.name
    identity = hashlib.sha256(f"{ordinal}\0{url}".encode()).hexdigest()[:12]
    return path.with_name(f"{base}__remote_{identity}{suffixes}")


def planned_files(record: dict[str, Any]) -> list[dict[str, Any]]:
    seen: dict[Path, int] = {}
    output: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(record.get("files") or [], 1):
        item = dict(raw)
        relative = safe_relative(str(item.get("path") or "download"))
        seen[relative] = seen.get(relative, 0) + 1
        if seen[relative] > 1:
            relative = collision_name(relative, str(item.get("download_url") or ""), ordinal)
        item["resolved_path"] = relative.as_posix()
        output.append(item)
    return output


def parse_host_jobs(value: str) -> dict[str, int]:
    limits: dict[str, int] = {}
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        host, separator, count_text = token.partition("=")
        host = host.strip().casefold()
        if not separator or not host:
            raise ValueError(f"invalid --host-jobs entry: {token!r}")
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(f"invalid --host-jobs count: {token!r}") from exc
        if count < 1:
            raise ValueError(f"--host-jobs count must be positive: {token!r}")
        limits[host] = count
    return limits


def checksum_spec(value: Any) -> tuple[str, str] | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if ":" in text:
        algorithm, digest = text.split(":", 1)
        algorithm = algorithm.replace("-", "")
    else:
        digest = text
        algorithm = {32: "md5", 40: "sha1", 64: "sha256"}.get(len(digest), "")
    if algorithm not in hashlib.algorithms_available or not digest or any(c not in "0123456789abcdef" for c in digest):
        return None
    return algorithm, digest


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def git_blob_digest(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def valid_file(path: Path, item: dict[str, Any]) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    expected = int(item.get("bytes") or 0)
    actual = path.stat().st_size
    if expected and actual != expected:
        return False, f"size:{actual}!={expected}"
    spec = checksum_spec(item.get("checksum"))
    if spec:
        algorithm, expected_digest = spec
        host = urllib.parse.urlsplit(str(item.get("download_url") or "")).netloc.casefold()
        raw_checksum = str(item.get("checksum") or "")
        if algorithm == "sha1" and ":" not in raw_checksum and host == "raw.githubusercontent.com":
            # GitHub tree entries expose the Git blob object ID, not the raw
            # file SHA-1.  Verify the canonical ``blob <size>\0<content>`` hash.
            actual_digest = git_blob_digest(path)
        else:
            actual_digest = file_digest(path, algorithm)
        if actual_digest.lower() != expected_digest:
            return False, f"{algorithm}:{actual_digest}!={expected_digest}"
    return True, "verified"


class Downloader:
    def __init__(
        self,
        args: argparse.Namespace,
        records: list[dict[str, Any]],
        *,
        input_records: int | None = None,
        skipped_completed: int = 0,
    ) -> None:
        self.args = args
        self.records = records
        self.input_records = input_records if input_records is not None else len(records)
        self.skipped_completed = skipped_completed
        self.lock = threading.Lock()
        self.host_lock = threading.Lock()
        self.host_limits = parse_host_jobs(args.host_jobs)
        self.host_semaphores: dict[str, threading.BoundedSemaphore] = {}
        self.completed = 0
        self.failed = 0
        self.bytes_verified = 0
        self.started = utc_now()
        self.links = (
            self._link_map(args.package_root, records)
            if args.package_root and not args.dry_run
            else {}
        )

    def _host_slot(self, url: str):
        host = urllib.parse.urlsplit(url).netloc.casefold()
        limit = self.host_limits.get(host, self.args.per_host_jobs)
        if not host or limit <= 0:
            return contextlib.nullcontext()
        with self.host_lock:
            semaphore = self.host_semaphores.get(host)
            if semaphore is None:
                semaphore = threading.BoundedSemaphore(limit)
                self.host_semaphores[host] = semaphore
        return semaphore

    @staticmethod
    def _link_map(package_root: Path, records: list[dict[str, Any]]) -> dict[str, list[Path]]:
        """Build dataset links from the queue without reading every locator.

        Locator files live on object storage and opening hundreds of them makes
        downloader startup unnecessarily slow.  The acquisition queue already
        carries both the canonical target path and all source paper IDs.
        """
        links: dict[str, list[Path]] = {}
        paper_roots: dict[str, Path] = {}
        for paper_root in package_root.joinpath("data").iterdir():
            separator = "--"
            if separator in paper_root.name:
                paper_roots[f"paper_{paper_root.name.rsplit(separator, 1)[1]}"] = paper_root
        for record in records:
            dataset_id = str(record.get("dataset_id") or "")
            if not dataset_id:
                continue
            targets: list[Path] = []
            target_relative = str(record.get("target_relative_path") or "")
            if target_relative:
                targets.append(package_root / safe_relative(target_relative))
            for paper_id in record.get("source_papers") or []:
                paper_root = paper_roots.get(str(paper_id))
                if paper_root is not None:
                    targets.append(paper_root / dataset_id)
            links[dataset_id] = list(dict.fromkeys(targets))
        return links

    def _report(self) -> None:
        atomic_json(self.args.state_root / "RUN_REPORT.json", {
            "schema_version": 1,
            "started_at": self.started,
            "updated_at": utc_now(),
            "queue_datasets": len(self.records),
            "input_queue_datasets": self.input_records,
            "skipped_completed_datasets": self.skipped_completed,
            "completed_datasets_this_run": self.completed,
            "failed_datasets_this_run": self.failed,
            "verified_bytes_this_run": self.bytes_verified,
            "output_root": str(self.args.output_root),
            "package_root": str(self.args.package_root or ""),
        })

    def _download_file(self, item: dict[str, Any], dataset_root: Path) -> tuple[bool, str, int]:
        destination = dataset_root / safe_relative(str(item["resolved_path"]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        ok, reason = valid_file(destination, item)
        if ok:
            return True, "already_verified", destination.stat().st_size

        part = destination.with_name(destination.name + ".part")
        if destination.exists():
            quarantine = destination.with_name(destination.name + f".invalid.{int(datetime.now().timestamp())}")
            destination.rename(quarantine)
        part_ok, _ = valid_file(part, item)
        if part_ok:
            os.replace(part, destination)
            return True, "resumed_file_already_complete", destination.stat().st_size
        expected = int(item.get("bytes") or 0)
        if part.exists() and expected and part.stat().st_size > expected:
            part.unlink()
        url = str(item.get("download_url") or "")
        if not url.startswith(("https://", "http://")):
            return False, "unsupported_or_missing_url", 0

        command = [
            self.args.curl, "--location", "--fail", "--silent", "--show-error",
            "--connect-timeout", str(self.args.connect_timeout),
            "--max-time", str(self.args.file_timeout),
            "--retry", str(self.args.retries), "--retry-all-errors",
            "--continue-at", "-", "--output", str(part), "--url", url,
        ]
        if self.args.low_speed_limit > 0 and self.args.low_speed_time > 0:
            command[command.index("--continue-at"):command.index("--continue-at")] = [
                "--speed-limit", str(self.args.low_speed_limit),
                "--speed-time", str(self.args.low_speed_time),
            ]
        with self._host_slot(url):
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if result.returncode == 33 and part.exists():
            part.unlink()
            command[command.index("--continue-at") + 1] = "0"
            with self._host_slot(url):
                result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if result.returncode:
            message = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else f"curl_exit_{result.returncode}"
            return False, message[-500:], part.stat().st_size if part.exists() else 0

        ok, reason = valid_file(part, item)
        if not ok:
            return False, reason, part.stat().st_size if part.exists() else 0
        os.replace(part, destination)
        return True, "downloaded_and_verified", destination.stat().st_size

    def _link_dataset(self, dataset_id: str, dataset_root: Path) -> list[str]:
        made: list[str] = []
        for target in self.links.get(dataset_id, []):
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() and target.resolve() == dataset_root.resolve():
                made.append(str(target))
                continue
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"refusing to replace dataset link target: {target}")
            target.symlink_to(dataset_root, target_is_directory=True)
            made.append(str(target))
        return made

    def download_dataset(self, record: dict[str, Any]) -> dict[str, Any]:
        dataset_id = str(record.get("dataset_id") or "")
        if not dataset_id or "/" in dataset_id or dataset_id in {".", ".."}:
            raise ValueError(f"unsafe dataset_id: {dataset_id!r}")
        dataset_root = self.args.output_root / dataset_id
        dataset_root.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        failures = 0
        for item in planned_files(record):
            try:
                ok, status, size = self._download_file(item, dataset_root)
            except Exception as exc:  # preserve checkpoint and continue other files
                ok, status, size = False, f"{type(exc).__name__}: {exc}", 0
            results.append({
                "source_path": item.get("path"),
                "resolved_path": item.get("resolved_path"),
                "download_url": item.get("download_url"),
                "expected_bytes": int(item.get("bytes") or 0),
                "checksum": item.get("checksum") or "",
                "ok": ok,
                "status": status,
                "bytes_present": size,
            })
            if not ok:
                failures += 1
                if failures >= self.args.max_failures_per_dataset:
                    break

        complete = len(results) == len(record.get("files") or []) and all(row["ok"] for row in results)
        links: list[str] = []
        if complete and self.args.package_root:
            links = self._link_dataset(dataset_id, dataset_root)
        report = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "repository": record.get("repository"),
            "record_id": record.get("record_id"),
            "version": record.get("version"),
            "complete": complete,
            "updated_at": utc_now(),
            "expected_files": len(record.get("files") or []),
            "processed_files": len(results),
            "verified_files": sum(row["ok"] for row in results),
            "expected_bytes": int(record.get("total_bytes") or 0),
            "verified_bytes": sum(row["bytes_present"] for row in results if row["ok"]),
            "links": links,
            "files": results,
        }
        atomic_json(self.args.state_root / "datasets" / f"{dataset_id}.json", report)
        with self.lock:
            if complete:
                self.completed += 1
                self.bytes_verified += report["verified_bytes"]
            else:
                self.failed += 1
            self._report()
            print(json.dumps({
                "dataset_id": dataset_id, "complete": complete,
                "verified_files": report["verified_files"],
                "expected_files": report["expected_files"],
                "verified_bytes": report["verified_bytes"],
            }), flush=True)
        return report

    def run(self) -> int:
        if self.args.dry_run:
            print(json.dumps({
                "datasets": len(self.records),
                "files": sum(len(row.get("files") or []) for row in self.records),
                "bytes": sum(int(row.get("total_bytes") or 0) for row in self.records),
                "output_root": str(self.args.output_root),
            }, indent=2))
            return 0
        self.args.output_root.mkdir(parents=True, exist_ok=True)
        self.args.state_root.mkdir(parents=True, exist_ok=True)
        self._report()
        with ThreadPoolExecutor(max_workers=max(1, self.args.jobs)) as pool:
            futures = [pool.submit(self.download_dataset, row) for row in self.records]
            for future in as_completed(futures):
                future.result()
        return 0 if self.failed == 0 else 1


def select_records(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    repositories = {value.strip() for value in args.repositories.split(",") if value.strip()}
    if repositories:
        records = [row for row in records if str(row.get("repository") or "") in repositories]
    records.sort(key=lambda row: (int(row.get("total_bytes") or 0), str(row.get("dataset_id") or "")))
    selected: list[dict[str, Any]] = []
    total = 0
    for row in records:
        size = int(row.get("total_bytes") or 0)
        if args.max_dataset_bytes is not None and size > args.max_dataset_bytes:
            continue
        if args.max_total_bytes is not None and selected and total + size > args.max_total_bytes:
            continue
        selected.append(row)
        total += size
        if args.max_datasets is not None and len(selected) >= args.max_datasets:
            break
    return selected


def completed_report_matches(record: dict[str, Any], args: argparse.Namespace) -> bool:
    """Fast resume check for a dataset previously downloaded and verified.

    The prior atomic report proves file-size/checksum verification.  Trust it on
    restart instead of issuing thousands of object-store stat/read operations;
    a full payload audit remains a separate finalization step.
    """
    dataset_id = str(record.get("dataset_id") or "")
    report_path = args.state_root / "datasets" / f"{dataset_id}.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    files = planned_files(record)
    if not report.get("complete"):
        return False
    if int(report.get("expected_files") or -1) != len(files):
        return False
    if int(report.get("expected_bytes") or -1) != int(record.get("total_bytes") or 0):
        return False
    return int(report.get("verified_bytes") or -1) == int(record.get("total_bytes") or 0)


def completed_ids_from_log(path: Path) -> set[str]:
    """Recover the latest completed dataset IDs from an append-only run log."""
    latest: dict[str, bool] = {}
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                dataset_id = str(row.get("dataset_id") or "")
                if dataset_id and "complete" in row:
                    latest[dataset_id] = bool(row["complete"])
    except OSError:
        return set()
    return {dataset_id for dataset_id, complete in latest.items() if complete}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--repositories", default="")
    parser.add_argument("--max-datasets", type=int)
    parser.add_argument("--max-dataset-bytes", type=int)
    parser.add_argument("--max-total-bytes", type=int)
    parser.add_argument("--connect-timeout", type=int, default=30)
    parser.add_argument("--file-timeout", type=int, default=0, help="curl max time per attempt; 0 disables")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--max-failures-per-dataset", type=int, default=10)
    parser.add_argument(
        "--skip-completed-reports", action="store_true",
        help="skip datasets with a matching atomic report from an earlier verified run",
    )
    parser.add_argument(
        "--completed-log", type=Path, action="append", default=[],
        help="append-only JSONL run log used as a fast completed-dataset index; repeatable",
    )
    parser.add_argument(
        "--per-host-jobs", type=int, default=0,
        help="maximum concurrent transfers per download host; 0 disables the limit",
    )
    parser.add_argument(
        "--host-jobs", default="",
        help="comma-separated host-specific limits, for example datadryad.org=3,osf.io=4",
    )
    parser.add_argument("--low-speed-limit", type=int, default=0, help="curl low-speed threshold in bytes/s")
    parser.add_argument("--low-speed-time", type=int, default=0, help="seconds below low-speed threshold before retry")
    parser.add_argument("--curl", default=shutil.which("curl") or "curl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        parse_host_jobs(args.host_jobs)
    except ValueError as exc:
        parser.error(str(exc))
    records = select_records(read_jsonl(args.queue), args)
    if not records:
        parser.error("selection is empty")
    input_records = len(records)
    skipped_completed = 0
    if args.completed_log:
        completed_ids: set[str] = set()
        for completed_log in args.completed_log:
            completed_ids.update(completed_ids_from_log(completed_log))
        remaining = [record for record in records if str(record.get("dataset_id") or "") not in completed_ids]
        skipped_completed = len(records) - len(remaining)
        records = remaining
    elif args.skip_completed_reports:
        remaining = []
        for record in records:
            if completed_report_matches(record, args):
                skipped_completed += 1
            else:
                remaining.append(record)
        records = remaining
    if not records:
        print(json.dumps({
            "datasets": 0,
            "input_datasets": input_records,
            "skipped_completed_datasets": skipped_completed,
            "status": "all_selected_datasets_already_complete",
        }, indent=2))
        return 0
    return Downloader(
        args,
        records,
        input_records=input_records,
        skipped_completed=skipped_completed,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
