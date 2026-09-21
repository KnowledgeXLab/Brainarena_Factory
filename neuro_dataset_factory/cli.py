"""Command-line interface for the offline, deterministic construction core."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from neuro_dataset_factory.brainarena import audit_brainarena, materialize_brainarena
from neuro_dataset_factory.dedup import deduplicate_task_candidates
from neuro_dataset_factory.discovery import search_dataset_candidates, search_paper_candidates
from neuro_dataset_factory.llm_client import load_env_file
from neuro_dataset_factory.packages import build_packages, validate_task_package
from neuro_dataset_factory.reference_runner import run_reference_analyses
from neuro_dataset_factory.remote_data import (
    build_acquisition_handoff,
    extract_s3_paper_contexts,
    open_local_csv,
    open_s3_csv,
    scan_corpus_rows,
    select_remote_candidates,
    verify_candidate_links,
)
from neuro_dataset_factory.remote_tasks import (
    audit_provisional_remote_packages,
    build_provisional_remote_packages,
    generate_remote_task_candidates,
)
from neuro_dataset_factory.remote_brainarena import (
    audit_remote_brainarena,
    materialize_remote_brainarena,
)
from neuro_dataset_factory.paper_figures import (
    audit_gt_figures,
    extract_s3_paper_figures,
    map_tasks_to_paper_figures,
    materialize_gt_figures,
)
from neuro_dataset_factory.registry import profile_datasets, validate_paper_links
from neuro_dataset_factory.splits import assign_grouped_splits
from neuro_dataset_factory.storage import write_json
from neuro_dataset_factory.task_generation import generate_task_candidates

DEFAULT_MAX_BYTES = 50 * 1024 ** 3
DEFAULT_REMOTE_MAX_BYTES = 2 ** 63 - 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuro-dataset-factory",
        description="Dataset-centric neuroscience training-task construction.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    discover = sub.add_parser(
        "search-datasets",
        help="search unverified dataset candidates from question/capability requests",
    )
    discover.add_argument("--requests", type=Path, required=True)
    discover.add_argument("--out", type=Path, required=True)
    discover.add_argument("--api-key-env", default="SERPER_KEY")
    discover.add_argument("--results-per-query", type=int, default=10)
    discover.add_argument(
        "--enrich-with-jina", action="store_true",
        help="read dataset landing pages through Jina and extract DOI/download evidence",
    )
    discover.add_argument("--jina-key-env", default="JINA_API_KEY")
    discover.add_argument("--jina-max-chars", type=int, default=20000)
    discover.add_argument("--jina-timeout", type=int, default=20)
    discover.add_argument("--jina-jobs", type=int, default=8)

    profile = sub.add_parser("profile", help="profile local dataset seeds and build SHA-256 manifests")
    profile.add_argument("--seeds", type=Path, required=True, help="dataset seed JSONL")
    profile.add_argument("--out-root", type=Path, required=True, help="new or existing registry workspace")
    profile.add_argument(
        "--max-bytes", type=int, default=DEFAULT_MAX_BYTES,
        help="complete dataset size cap in bytes (default: 50 GiB)",
    )

    links = sub.add_parser("validate-links", help="normalize and validate dataset-to-paper links")
    links.add_argument("--registry", type=Path, required=True)
    links.add_argument("--links", type=Path, required=True)
    links.add_argument("--out", type=Path, required=True)

    papers = sub.add_parser("search-papers", help="search unverified papers for registered datasets")
    papers.add_argument("--registry", type=Path, required=True)
    papers.add_argument("--out", type=Path, required=True)
    papers.add_argument("--results-per-dataset", type=int, default=20)
    papers.add_argument("--mailto", default="")

    corpus = sub.add_parser(
        "scan-paper-corpus",
        help="scan paper CSV data_link fields without downloading dataset payloads",
    )
    source = corpus.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path)
    source.add_argument("--s3-uri")
    corpus.add_argument("--endpoint-url", default="")
    corpus.add_argument("--out", type=Path, required=True)
    corpus.add_argument("--max-papers", type=int)

    remote_select = sub.add_parser(
        "select-remote-candidates",
        help="select concrete repository/file locators for a verification batch",
    )
    remote_select.add_argument("--candidates", type=Path, required=True)
    remote_select.add_argument("--out", type=Path, required=True)
    remote_select.add_argument(
        "--repositories", default="figshare,zenodo,osf",
        help="comma-separated normalized repository names",
    )
    remote_select.add_argument("--include-direct-files", action="store_true")
    remote_select.add_argument("--exclude-papers", type=Path)
    remote_select.add_argument(
        "--statuses", default="",
        help="optional comma-separated input paper statuses (for retry batches)",
    )
    remote_select.add_argument("--max-papers", type=int)

    remote = sub.add_parser(
        "verify-remote-data",
        help="verify repository records and concrete downloadable files for paper data links",
    )
    remote.add_argument("--candidates", type=Path, required=True)
    remote.add_argument("--out", type=Path, required=True)
    remote.add_argument("--jobs", type=int, default=8)
    remote.add_argument("--timeout", type=int, default=20)
    remote.add_argument("--max-retries", type=int, default=2)

    handoff = sub.add_parser(
        "build-acquisition-handoff",
        help="build exact remote download locators for the data-acquisition team",
    )
    handoff.add_argument("--verified", type=Path, required=True)
    handoff.add_argument("--out-root", type=Path, required=True)
    handoff.add_argument(
        "--max-bytes", type=int, default=DEFAULT_REMOTE_MAX_BYTES,
        help="optional per-dataset size cap in bytes (default: no practical cap)",
    )

    contexts = sub.add_parser(
        "extract-paper-contexts",
        help="stream parsed S3 full text and retain only verified papers",
    )
    contexts.add_argument("--verified", type=Path, required=True)
    contexts.add_argument("--s3-uri", required=True)
    contexts.add_argument("--endpoint-url", required=True)
    contexts.add_argument("--out", type=Path, required=True)
    contexts.add_argument("--max-chars", type=int, default=60000)

    remote_generate = sub.add_parser(
        "generate-remote-candidates",
        help="generate provisional tasks from verified remote manifests and paper context",
    )
    remote_generate.add_argument("--handoff-root", type=Path, required=True)
    remote_generate.add_argument("--paper-contexts", type=Path, required=True)
    remote_generate.add_argument("--out", type=Path, required=True)
    remote_generate.add_argument("--cache-dir", type=Path, required=True)
    remote_generate.add_argument("--env-file", type=Path)
    remote_generate.add_argument("--model")
    remote_generate.add_argument("--tasks-per-paper", type=int, default=4)
    remote_generate.add_argument("--jobs", type=int, default=4)
    remote_generate.add_argument("--timeout", type=int, default=180)
    remote_generate.add_argument("--max-retries", type=int, default=3)
    remote_generate.add_argument("--max-tokens", type=int, default=12000)
    remote_generate.add_argument("--temperature", type=float, default=0.2)
    remote_generate.add_argument("--max-context-chars", type=int, default=30000)
    remote_generate.add_argument("--max-papers", type=int)
    remote_generate.add_argument("--no-resume", dest="resume", action="store_false")
    remote_generate.add_argument("--refresh-cache", action="store_true")
    remote_generate.add_argument("--use-env-proxy", action="store_true")

    provisional = sub.add_parser(
        "build-provisional-packages",
        help="build awaiting_data_download task packages from remote candidates",
    )
    provisional.add_argument("--candidates", type=Path, required=True)
    provisional.add_argument("--handoff-root", type=Path, required=True)
    provisional.add_argument("--out-root", type=Path, required=True)

    provisional_audit = sub.add_parser(
        "audit-provisional-packages",
        help="audit awaiting_data_download packages without treating locators as payload",
    )
    provisional_audit.add_argument("root", type=Path)

    remote_materialize = sub.add_parser(
        "materialize-remote-brainarena",
        help="export provisional remote tasks in the established data/ BrainArena layout",
    )
    remote_materialize.add_argument("--packages-root", type=Path, required=True)
    remote_materialize.add_argument("--out-root", type=Path, required=True)

    remote_format_audit = sub.add_parser(
        "audit-remote-brainarena",
        help="audit the established BrainArena layout before remote data download",
    )
    remote_format_audit.add_argument("root", type=Path)

    figure_extract = sub.add_parser("extract-paper-figures", help="extract source-paper figure objects and captions from S3 parsed full text")
    figure_extract.add_argument("--verified", type=Path, required=True)
    figure_extract.add_argument("--s3-uri", required=True)
    figure_extract.add_argument("--endpoint-url", required=True)
    figure_extract.add_argument("--out", type=Path, required=True)

    figure_map = sub.add_parser("map-task-figures", help="map remote tasks to exact source-paper figures")
    figure_map.add_argument("--candidates", type=Path, required=True)
    figure_map.add_argument("--figures", type=Path, required=True)
    figure_map.add_argument("--out", type=Path, required=True)
    figure_map.add_argument("--cache-dir", type=Path, required=True)
    figure_map.add_argument("--env-file", type=Path)
    figure_map.add_argument("--model")
    figure_map.add_argument("--jobs", type=int, default=4)
    figure_map.add_argument("--timeout", type=int, default=180)
    figure_map.add_argument("--max-retries", type=int, default=3)
    figure_map.add_argument("--max-tokens", type=int, default=6000)
    figure_map.add_argument("--max-papers", type=int)
    figure_map.add_argument("--refresh-cache", action="store_true")
    figure_map.add_argument("--use-env-proxy", action="store_true")

    figure_materialize = sub.add_parser("materialize-gt-figures", help="read matched paper figures from S3 into benchmark/gt/gt_figure")
    figure_materialize.add_argument("--mappings", type=Path, required=True)
    figure_materialize.add_argument("--brainarena-root", type=Path, required=True)
    figure_materialize.add_argument("--endpoint-url", required=True)

    figure_audit = sub.add_parser("audit-gt-figures", help="audit gt_figure files, hashes, and source provenance")
    figure_audit.add_argument("root", type=Path)

    generate = sub.add_parser(
        "generate-candidates",
        help="generate end-to-end task candidates through API_BASE/API_KEY",
    )
    generate.add_argument("--registry", type=Path, required=True)
    generate.add_argument("--links", type=Path, required=True)
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--cache-dir", type=Path, required=True)
    generate.add_argument("--paper-contexts", type=Path)
    generate.add_argument("--env-file", type=Path)
    generate.add_argument("--model")
    generate.add_argument("--tasks-per-link", type=int, default=4)
    generate.add_argument("--jobs", type=int, default=4)
    generate.add_argument("--timeout", type=int, default=180)
    generate.add_argument("--max-retries", type=int, default=3)
    generate.add_argument("--max-tokens", type=int, default=12000)
    generate.add_argument("--temperature", type=float, default=0.2)
    generate.add_argument("--max-manifest-files", type=int, default=500)
    generate.add_argument("--max-context-chars", type=int, default=20000)
    generate.add_argument("--no-resume", dest="resume", action="store_false")
    generate.add_argument("--refresh-cache", action="store_true")
    generate.add_argument(
        "--use-env-proxy", action="store_true",
        help="honor ambient HTTP(S)_PROXY for an external API_BASE; default is direct",
    )

    dedup = sub.add_parser(
        "deduplicate",
        help="remove lexical/structural near-duplicate task candidates within each dataset",
    )
    dedup.add_argument("--candidates", type=Path, required=True)
    dedup.add_argument("--out", type=Path, required=True)
    dedup.add_argument("--duplicates", type=Path, required=True)
    dedup.add_argument("--threshold", type=float, default=0.90)

    references = sub.add_parser(
        "run-references",
        help="execute reviewed reference commands and verify declared metrics/artifacts",
    )
    references.add_argument("--registry", type=Path, required=True)
    references.add_argument("--candidates", type=Path, required=True)
    references.add_argument("--out-candidates", type=Path, required=True)
    references.add_argument("--results", type=Path, required=True)
    references.add_argument("--out-root", type=Path, required=True)
    references.add_argument("--workspace-root", type=Path, required=True)
    references.add_argument("--jobs", type=int, default=2)
    references.add_argument("--timeout", type=int, default=1800)
    references.add_argument("--accept-observed-targets", action="store_true")
    references.add_argument(
        "--allow-execution", action="store_true",
        help="required safety acknowledgement after reviewing reference commands",
    )

    packages = sub.add_parser("build-packages", help="build canonical task packages from verified candidates")
    packages.add_argument("--registry", type=Path, required=True)
    packages.add_argument("--links", type=Path, required=True)
    packages.add_argument("--candidates", type=Path, required=True)
    packages.add_argument("--out-root", type=Path, required=True)
    packages.add_argument(
        "--max-bytes", type=int, default=DEFAULT_MAX_BYTES,
        help="complete shared dataset size cap in bytes (default: 50 GiB)",
    )
    packages.add_argument(
        "--allow-unverified", action="store_true",
        help="development only: build candidates that fail semantic quality gates",
    )

    validate = sub.add_parser("validate-package", help="validate one canonical task directory")
    validate.add_argument("task_root", type=Path, help="path ending in package/task")
    validate.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    validate.add_argument("--report", type=Path)

    materialize = sub.add_parser("materialize", help="convert validated packages to portable BrainArena layout")
    materialize.add_argument("--packages-root", type=Path, required=True)
    materialize.add_argument("--out-root", type=Path, required=True)

    audit = sub.add_parser("audit-brainarena", help="audit registry paths and score_100 rubrics")
    audit.add_argument("root", type=Path)
    audit.add_argument("--report", type=Path)

    splits = sub.add_parser(
        "assign-splits",
        help="assign leakage-safe splits to connected dataset/paper task groups",
    )
    splits.add_argument("--registry", type=Path, required=True)
    splits.add_argument("--out", type=Path, required=True)
    splits.add_argument("--manifest", type=Path, required=True)
    splits.add_argument("--train-ratio", type=float, default=0.8)
    splits.add_argument("--validation-ratio", type=float, default=0.1)
    splits.add_argument("--test-ratio", type=float, default=0.1)
    splits.add_argument("--seed", default="brainarena-v1")
    splits.add_argument("--group-fields", nargs="+", default=["dataset_id", "paper_id"])
    return parser


def _emit(report) -> int:
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.ok else 1


def _model_name(cli_value: str | None) -> str:
    model = (cli_value or os.environ.get("TASKGEN_MODEL") or "").strip()
    if not model:
        raise SystemExit("--model or TASKGEN_MODEL is required for LLM stages")
    return model


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "search-datasets":
        return _emit(search_dataset_candidates(
            args.requests,
            args.out,
            api_key_env=args.api_key_env,
            results_per_query=args.results_per_query,
            enrich_with_jina=args.enrich_with_jina,
            jina_key_env=args.jina_key_env,
            jina_max_chars=args.jina_max_chars,
            jina_timeout=args.jina_timeout,
            jina_jobs=args.jina_jobs,
        ))
    if args.command == "profile":
        _, report = profile_datasets(args.seeds, args.out_root, max_bytes=args.max_bytes)
        return _emit(report)
    if args.command == "validate-links":
        return _emit(validate_paper_links(args.registry, args.links, args.out))
    if args.command == "search-papers":
        return _emit(search_paper_candidates(
            args.registry,
            args.out,
            results_per_dataset=args.results_per_dataset,
            mailto=args.mailto,
        ))
    if args.command == "scan-paper-corpus":
        if args.csv:
            handle, rows = open_local_csv(args.csv)
            try:
                return _emit(scan_corpus_rows(
                    rows, args.out, source_uri=str(args.csv), max_papers=args.max_papers,
                ))
            finally:
                handle.close()
        if not args.endpoint_url:
            raise SystemExit("--endpoint-url is required with --s3-uri")
        with open_s3_csv(
            args.s3_uri,
            endpoint_url=args.endpoint_url,
            access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
            secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        ) as rows:
            return _emit(scan_corpus_rows(
                rows, args.out, source_uri=args.s3_uri, max_papers=args.max_papers,
            ))
    if args.command == "select-remote-candidates":
        repositories = {
            item.strip().casefold()
            for item in args.repositories.split(",")
            if item.strip()
        }
        return _emit(select_remote_candidates(
            args.candidates,
            args.out,
            repositories=repositories,
            include_direct_files=args.include_direct_files,
            exclude_papers_path=args.exclude_papers,
            statuses={item.strip() for item in args.statuses.split(",") if item.strip()},
            max_papers=args.max_papers,
        ))
    if args.command == "verify-remote-data":
        return _emit(verify_candidate_links(
            args.candidates, args.out, jobs=args.jobs, timeout=args.timeout,
            max_retries=args.max_retries,
        ))
    if args.command == "build-acquisition-handoff":
        return _emit(build_acquisition_handoff(
            args.verified, args.out_root, max_bytes=args.max_bytes,
        ))
    if args.command == "extract-paper-contexts":
        return _emit(extract_s3_paper_contexts(
            args.verified,
            args.out,
            s3_uri=args.s3_uri,
            endpoint_url=args.endpoint_url,
            access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
            secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            max_chars=args.max_chars,
        ))
    if args.command == "generate-remote-candidates":
        if args.env_file:
            load_env_file(args.env_file)
        model = _model_name(args.model)
        return _emit(generate_remote_task_candidates(
            args.handoff_root,
            args.paper_contexts,
            args.out,
            args.cache_dir,
            model=model,
            tasks_per_paper=args.tasks_per_paper,
            jobs=args.jobs,
            timeout=args.timeout,
            max_retries=args.max_retries,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_context_chars=args.max_context_chars,
            max_papers=args.max_papers,
            resume=args.resume,
            refresh_cache=args.refresh_cache,
            use_env_proxy=args.use_env_proxy,
        ))
    if args.command == "build-provisional-packages":
        return _emit(build_provisional_remote_packages(
            args.candidates, args.handoff_root, args.out_root,
        ))
    if args.command == "audit-provisional-packages":
        return _emit(audit_provisional_remote_packages(args.root))
    if args.command == "materialize-remote-brainarena":
        return _emit(materialize_remote_brainarena(args.packages_root, args.out_root))
    if args.command == "audit-remote-brainarena":
        return _emit(audit_remote_brainarena(args.root))
    if args.command == "extract-paper-figures":
        return _emit(extract_s3_paper_figures(
            args.verified, args.out,
            s3_uri=args.s3_uri, endpoint_url=args.endpoint_url,
            access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
            secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        ))
    if args.command == "map-task-figures":
        if args.env_file:
            load_env_file(args.env_file)
        model = _model_name(args.model)
        return _emit(map_tasks_to_paper_figures(
            args.candidates, args.figures, args.out, args.cache_dir,
            model=model, jobs=args.jobs, timeout=args.timeout,
            max_retries=args.max_retries, max_tokens=args.max_tokens,
            max_papers=args.max_papers,
            refresh_cache=args.refresh_cache,
            use_env_proxy=args.use_env_proxy,
        ))
    if args.command == "materialize-gt-figures":
        return _emit(materialize_gt_figures(
            args.mappings, args.brainarena_root,
            endpoint_url=args.endpoint_url,
            access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
            secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        ))
    if args.command == "audit-gt-figures":
        return _emit(audit_gt_figures(args.root))
    if args.command == "generate-candidates":
        if args.env_file:
            load_env_file(args.env_file)
        model = _model_name(args.model)
        return _emit(generate_task_candidates(
            args.registry,
            args.links,
            args.out,
            args.cache_dir,
            model=model,
            tasks_per_link=args.tasks_per_link,
            jobs=args.jobs,
            timeout=args.timeout,
            max_retries=args.max_retries,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_manifest_files=args.max_manifest_files,
            max_context_chars=args.max_context_chars,
            paper_contexts_path=args.paper_contexts,
            resume=args.resume,
            refresh_cache=args.refresh_cache,
            use_env_proxy=args.use_env_proxy,
        ))
    if args.command == "deduplicate":
        return _emit(deduplicate_task_candidates(
            args.candidates,
            args.out,
            args.duplicates,
            threshold=args.threshold,
        ))
    if args.command == "run-references":
        return _emit(run_reference_analyses(
            args.registry,
            args.candidates,
            args.out_candidates,
            args.results,
            args.out_root,
            workspace_root=args.workspace_root,
            allow_execution=args.allow_execution,
            accept_observed_targets=args.accept_observed_targets,
            jobs=args.jobs,
            default_timeout=args.timeout,
        ))
    if args.command == "build-packages":
        report = build_packages(
            args.registry,
            args.links,
            args.candidates,
            args.out_root,
            max_bytes=args.max_bytes,
            allow_unverified=args.allow_unverified,
        )
        return _emit(report)
    if args.command == "validate-package":
        report = validate_task_package(args.task_root, max_bytes=args.max_bytes)
        if args.report:
            write_json(args.report, report.to_dict())
        return _emit(report)
    if args.command == "materialize":
        return _emit(materialize_brainarena(args.packages_root, args.out_root))
    if args.command == "audit-brainarena":
        report = audit_brainarena(args.root)
        if args.report:
            write_json(args.report, report.to_dict())
        return _emit(report)
    if args.command == "assign-splits":
        return _emit(assign_grouped_splits(
            args.registry,
            args.out,
            args.manifest,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            group_fields=tuple(args.group_fields),
        ))
    raise AssertionError(args.command)
