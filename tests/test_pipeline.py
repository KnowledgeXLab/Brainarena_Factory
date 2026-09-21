from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from neuro_dataset_factory.brainarena import audit_brainarena, materialize_brainarena
from neuro_dataset_factory.cli import DEFAULT_MAX_BYTES, DEFAULT_REMOTE_MAX_BYTES
from neuro_dataset_factory.dedup import deduplicate_task_candidates
from neuro_dataset_factory.discovery import search_dataset_candidates, search_paper_candidates
from neuro_dataset_factory.manifest import build_manifest
from neuro_dataset_factory.packages import build_packages, validate_task_package
from neuro_dataset_factory.reference_runner import run_reference_analyses
from neuro_dataset_factory.remote_data import (
    DEFERRED_NETWORK,
    REJECTED,
    VERIFIED,
    build_acquisition_handoff,
    normalize_locator,
    scan_corpus_rows,
    select_remote_candidates,
    select_paper_context,
    verify_locator,
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
from neuro_dataset_factory.paper_figures import extract_figure_candidates_from_blocks
from neuro_dataset_factory.registry import profile_datasets, validate_paper_links
from neuro_dataset_factory.splits import assign_grouped_splits
from neuro_dataset_factory.storage import read_json, read_jsonl, write_json
from neuro_dataset_factory.task_generation import generate_task_candidates


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "source_dataset"
        self.dataset.mkdir()
        (self.dataset / "LICENSE").write_text("CC-BY-4.0\n", encoding="utf-8")
        (self.dataset / "recordings.csv").write_text("subject,stimulus,response\n1,A,3\n2,B,5\n", encoding="utf-8")
        self.seeds = self.root / "seeds.jsonl"
        write_jsonl(self.seeds, [{
            "dataset_id": "demo_v1",
            "name": "Demo neural recordings",
            "version": "v1",
            "local_path": str(self.dataset),
            "source_url": "https://example.org/demo",
            "license": "CC-BY-4.0",
            "modalities": ["electrophysiology"],
            "species": ["mouse"],
            "task_domains": ["perception"],
            "high_value_questions": ["How is stimulus identity represented?"],
        }])

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _profile_and_link(self, *, verified: bool = True) -> tuple[Path, Path, str]:
        registry_root = self.root / "registry"
        records, profile = profile_datasets(self.seeds, registry_root)
        self.assertTrue(profile.ok)
        self.assertEqual(records[0]["profile_status"], "valid")
        raw_links = self.root / "paper_links.jsonl"
        write_jsonl(raw_links, [{
            "dataset_id": "demo_v1",
            "paper_id": "paper_1",
            "title": "A demo neural result",
            "dataset_version_match": True,
            "scientific_question": "Can population activity decode stimulus identity?",
            "target_result": "Held-out decoding accuracy with uncertainty.",
            "required_files": ["recordings.csv"],
            "reference_code": "reference.py",
            "gt_type": "numeric",
            "reproducibility": "verified" if verified else "partial",
            "data_support": "yes",
            "leakage_risk": "none",
            "evidence": {
                "dataset_mention": "Methods names demo v1",
                "data_availability": "Public URL",
                "result_location": "Figure 2B",
            },
        }])
        normalized = self.root / "verified_links.jsonl"
        link_report = validate_paper_links(registry_root / "dataset_registry.jsonl", raw_links, normalized)
        self.assertTrue(link_report.ok)
        return registry_root / "dataset_registry.jsonl", normalized, read_jsonl(normalized)[0]["link_id"]

    def _candidate(
        self,
        link_id: str,
        *,
        reference_status: str = "verified",
        task_id: str = "NEURO_demo_001",
        task_tag: str = "Main",
    ) -> dict:
        return {
            "task_id": task_id,
            "task_tag": task_tag,
            "task_format": "end_to_end",
            "dataset_id": "demo_v1",
            "link_id": link_id,
            "query": "Use the provided recordings to estimate held-out stimulus decoding with grouped validation, uncertainty, and an auditable figure.",
            "deliverable": "conclusion.md + analyze.py + analyze.png",
            "required_files": ["recordings.csv"],
            "reference": {
                "status": reference_status,
                "command": "python reference.py",
                "result_summary": "completed",
                "metrics": [{"name": "accuracy", "target": 0.73, "tolerance": 0.03}],
            },
            "rubric": {
                "reason": "real data and valid analysis",
                "core_conclusion": "The population carries above-chance stimulus information under grouped held-out validation.",
                "scoring_items": [
                    {"point": 30, "criterion": "Uses the real provided recordings; synthetic replacement data receives zero.", "keywords": ["real data"]},
                    {"point": 25, "criterion": "Uses subject-aware grouped validation and a valid null.", "keywords": ["grouped"]},
                    {"point": 20, "criterion": "Reports effect size, sample size, and uncertainty.", "keywords": ["uncertainty"]},
                    {"point": 15, "criterion": "Provides runnable code and a matching result figure.", "keywords": ["runnable"]},
                    {"point": 10, "criterion": "Keeps the conclusion within the evidence.", "keywords": ["limitations"]},
                ],
            },
        }

    def test_multiple_end_to_end_tasks_share_one_dataset_asset(self) -> None:
        registry, links, link_id = self._profile_and_link()
        first = self._candidate(link_id, task_id="NEURO_demo_001", task_tag="Analysis_01")
        second = self._candidate(link_id, task_id="NEURO_demo_002", task_tag="Analysis_02")
        second["query"] = (
            "Use the provided recordings to run an end-to-end condition-level response analysis, "
            "including planning, executable code, uncertainty, a figure, and scientific interpretation."
        )
        candidates = self.root / "candidates.jsonl"
        write_jsonl(candidates, [first, second])
        packages = self.root / "packages"
        build_report = build_packages(registry, links, candidates, packages)
        self.assertTrue(build_report.ok, build_report.to_dict())
        self.assertEqual(build_report.stats["built"], 2)

        brainarena = self.root / "brainarena"
        materialized = materialize_brainarena(packages, brainarena)
        self.assertTrue(materialized.ok, materialized.to_dict())
        self.assertEqual(materialized.stats["tasks_materialized"], 2)
        self.assertEqual(materialized.stats["shared_dataset_assets"], 1)
        task_registry = read_json(brainarena / "benchmark" / "task_registry.json")
        self.assertEqual({row["paper_id"] for row in task_registry["tasks"]}, {"paper_1"})
        self.assertEqual({row["data_path"] for row in task_registry["tasks"]}, {"data/demo_v1"})
        self.assertTrue((brainarena / "benchmark/querys/paper_1/Analysis_01.md").is_file())
        self.assertTrue((brainarena / "benchmark/querys/paper_1/Analysis_02.md").is_file())
        self.assertEqual(
            len(list((brainarena / "data" / "demo_v1").glob("recordings.csv"))),
            1,
        )
        self.assertTrue((brainarena / "data" / "demo_v1" / "LICENSE").is_file())
        self.assertTrue(audit_brainarena(brainarena).ok)

    def test_candidate_dedup_and_grouped_splits(self) -> None:
        candidates = self.root / "candidates.jsonl"
        first = {
            "task_id": "task_1", "dataset_id": "ds_1",
            "query": "Use the provided neural data to fit a model and report a figure.",
            "required_files": ["data.csv"],
            "reference": {"metrics": [{"name": "accuracy"}]},
            "rubric": {"core_conclusion": "The model predicts neural responses."},
        }
        duplicate = {**first, "task_id": "task_2"}
        distinct = {
            **first,
            "task_id": "task_3",
            "query": "Estimate response latency by condition and interpret temporal differences.",
            "reference": {"metrics": [{"name": "latency"}]},
            "rubric": {"core_conclusion": "Response latency differs across conditions."},
        }
        write_jsonl(candidates, [first, duplicate, distinct])
        accepted = self.root / "accepted.jsonl"
        duplicates = self.root / "duplicates.jsonl"
        dedup = deduplicate_task_candidates(candidates, accepted, duplicates)
        self.assertTrue(dedup.ok)
        self.assertEqual(dedup.stats["accepted"], 2)
        self.assertEqual(read_jsonl(duplicates)[0]["duplicate_of"], "task_1")

        registry = self.root / "task_registry.json"
        registry.write_text(json.dumps({
            "schema_version": 2,
            "fields": ["task_id", "dataset_id", "paper_id"],
            "tasks": [
                {"task_id": "t1", "dataset_id": "ds1", "paper_id": "p1"},
                {"task_id": "t2", "dataset_id": "ds1", "paper_id": "p2"},
                {"task_id": "t3", "dataset_id": "ds2", "paper_id": "p2"},
                {"task_id": "t4", "dataset_id": "ds3", "paper_id": "p3"},
            ],
        }), encoding="utf-8")
        split_registry = self.root / "task_registry_split.json"
        split_manifest = self.root / "split_manifest.json"
        split_report = assign_grouped_splits(registry, split_registry, split_manifest)
        self.assertTrue(split_report.ok)
        split_tasks = read_json(split_registry)["tasks"]
        self.assertEqual(len({task["split"] for task in split_tasks[:3]}), 1)
        self.assertEqual(split_report.stats["groups"], 2)

    def test_default_dataset_limit_is_50_gib(self) -> None:
        self.assertEqual(DEFAULT_MAX_BYTES, 50 * 1024 ** 3)
        self.assertGreater(DEFAULT_REMOTE_MAX_BYTES, 50 * 1024 ** 3)

    def test_llm_batch_generator_with_injected_client(self) -> None:
        registry, links, _ = self._profile_and_link()

        class FakeClient:
            call_count = 1
            cache_hits = 0

            def chat_json(self, **kwargs):
                def item(tag: str, question: str) -> dict:
                    return {
                        "task_tag": tag,
                        "query": question,
                        "deliverable": "analysis.py, summary.json, figure.png, and report.md",
                        "required_files": ["recordings.csv"],
                        "construction_mode": "reproduce",
                        "reference": {
                            "status": "unverified",
                            "metrics_file": "summary.json",
                            "artifacts": ["summary.json", "figure.png"],
                            "metrics": [{"name": "effect", "tolerance": 0.01}],
                        },
                        "rubric": {
                            "reason": "grounded end-to-end analysis",
                            "core_conclusion": "The real recordings contain a condition-dependent neural effect.",
                            "scoring_items": [
                                {"point": 25, "criterion": "Uses the real provided data and never fabricates synthetic replacements.", "keywords": ["real data"]},
                                {"point": 20, "criterion": "Provides a justified analysis plan.", "keywords": ["plan"]},
                                {"point": 20, "criterion": "Provides runnable code and correct statistics.", "keywords": ["code"]},
                                {"point": 15, "criterion": "Produces an auditable figure.", "keywords": ["figure"]},
                                {"point": 20, "criterion": "Interprets the neuroscience result and limitations.", "keywords": ["interpretation"]},
                            ],
                        },
                    }

                return {
                    "candidates": [
                        item("Analysis_01", "Use the provided recordings to run an end-to-end decoding analysis with code, statistics, a figure, and interpretation."),
                        item("Analysis_02", "Use the provided recordings to quantify condition-level response differences with code, uncertainty, visualization, and interpretation."),
                    ]
                }, {"cache_key": "fake-cache", "cache_hit": False}

        output = self.root / "generated.jsonl"
        report = generate_task_candidates(
            registry,
            links,
            output,
            self.root / "llm_cache",
            model="test-model",
            tasks_per_link=2,
            client=FakeClient(),
        )
        self.assertTrue(report.ok, report.to_dict())
        generated = read_jsonl(output)
        self.assertEqual(len(generated), 2)
        self.assertEqual({row["task_tag"] for row in generated}, {"Analysis_01", "Analysis_02"})
        self.assertTrue(all(row["reference"]["status"] == "unverified" for row in generated))
        self.assertTrue(all(row["generation"]["model"] == "test-model" for row in generated))

    def test_reference_batch_execution_and_metric_verification(self) -> None:
        registry, _, link_id = self._profile_and_link()
        script = self.root / "reference.py"
        script.write_text(
            "import argparse, json\n"
            "from pathlib import Path\n"
            "p=argparse.ArgumentParser(); p.add_argument('--dataset-root'); p.add_argument('--out-dir'); a=p.parse_args()\n"
            "out=Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)\n"
            "(out/'summary.json').write_text(json.dumps({'accuracy': 0.73}))\n"
            "(out/'figure.png').write_bytes(b'png')\n",
            encoding="utf-8",
        )
        candidate = self._candidate(link_id)
        candidate["reference"].update({
            "command": [
                sys.executable, "reference.py", "--dataset-root", "<data>", "--out-dir", "<output>",
            ],
            "metrics_file": "summary.json",
            "artifacts": ["summary.json", "figure.png"],
        })
        candidates = self.root / "reference_candidates.jsonl"
        write_jsonl(candidates, [candidate])
        denied = run_reference_analyses(
            registry,
            candidates,
            self.root / "denied.jsonl",
            self.root / "denied_results.jsonl",
            self.root / "denied_outputs",
            workspace_root=self.root,
        )
        self.assertFalse(denied.ok)

        updated_path = self.root / "reference_verified.jsonl"
        results_path = self.root / "reference_results.jsonl"
        report = run_reference_analyses(
            registry,
            candidates,
            updated_path,
            results_path,
            self.root / "reference_outputs",
            workspace_root=self.root,
            allow_execution=True,
        )
        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(report.stats["verified"], 1)
        updated = read_jsonl(updated_path)[0]
        self.assertEqual(updated["reference"]["status"], "verified")
        self.assertEqual(updated["reference"]["metrics"][0]["observed"], 0.73)

    def test_manifest_is_deterministic_and_keeps_license(self) -> None:
        first, summary1, report1 = build_manifest(self.dataset)
        second, summary2, report2 = build_manifest(self.dataset)
        self.assertTrue(report1.ok and report2.ok)
        self.assertEqual(summary1["tree_sha256"], summary2["tree_sha256"])
        roles = {row["path"]: row["role"] for row in first}
        self.assertEqual(roles["LICENSE"], "license")
        self.assertEqual(first, second)

    def test_end_to_end_package_and_brainarena_paths(self) -> None:
        registry, links, link_id = self._profile_and_link()
        candidates = self.root / "candidates.jsonl"
        write_jsonl(candidates, [self._candidate(link_id)])
        packages = self.root / "packages"
        build_report = build_packages(registry, links, candidates, packages)
        self.assertTrue(build_report.ok, build_report.to_dict())
        task_root = packages / "NEURO_demo_001" / "task"
        self.assertTrue(validate_task_package(task_root).ok)
        public = read_json(task_root / "task_info.json")
        self.assertNotIn("0.73", public["query"])
        self.assertNotIn("core_conclusion", public)

        brainarena = self.root / "brainarena"
        materialized = materialize_brainarena(packages, brainarena)
        self.assertTrue(materialized.ok)
        audit = audit_brainarena(brainarena)
        self.assertTrue(audit.ok, audit.to_dict())
        task = read_json(brainarena / "benchmark" / "task_registry.json")["tasks"][0]
        self.assertFalse(Path(task["data_path"]).is_absolute())
        self.assertTrue((brainarena / task["data_path"] / "recordings.csv").is_file())

    def test_unverified_reference_is_gated(self) -> None:
        registry, links, link_id = self._profile_and_link(verified=False)
        candidates = self.root / "candidates.jsonl"
        write_jsonl(candidates, [self._candidate(link_id, reference_status="partial")])
        packages = self.root / "packages"
        report = build_packages(registry, links, candidates, packages)
        self.assertTrue(report.ok)
        self.assertEqual(report.stats["built"], 0)
        self.assertEqual(report.stats["skipped_by_gate"], 1)
        self.assertFalse((packages / "NEURO_demo_001").exists())

    def test_unknown_license_is_gated(self) -> None:
        seed = json.loads(self.seeds.read_text(encoding="utf-8"))
        seed["license"] = "unknown"
        write_jsonl(self.seeds, [seed])
        registry, links, link_id = self._profile_and_link()
        candidates = self.root / "candidates.jsonl"
        write_jsonl(candidates, [self._candidate(link_id)])
        packages = self.root / "packages"
        report = build_packages(registry, links, candidates, packages)
        self.assertEqual(report.stats["built"], 0)
        self.assertEqual(report.stats["skipped_by_gate"], 1)

    def test_discovery_outputs_remain_unverified(self) -> None:
        requests = self.root / "discovery.jsonl"
        write_jsonl(requests, [{
            "scientific_question": "How is stimulus identity encoded?",
            "capability_gap": ["Coding"],
            "modality": "electrophysiology",
            "species": "mouse",
        }])

        def fake_post(url, body, headers, timeout):
            return {"organic": [{"title": "Demo Dataset", "link": "https://example.org/data", "snippet": "open data"}]}

        datasets_out = self.root / "dataset_candidates.jsonl"
        report = search_dataset_candidates(requests, datasets_out, api_key="test", post_json=fake_post)
        self.assertTrue(report.ok)
        self.assertEqual(read_jsonl(datasets_out)[0]["status"], "unverified")

        registry_root = self.root / "registry"
        profile_datasets(self.seeds, registry_root)

        def fake_get(url, timeout):
            return {"results": [{
                "id": "https://openalex.org/W123",
                "doi": "https://doi.org/10.1/demo",
                "display_name": "Paper using Demo Dataset",
                "publication_year": 2025,
                "primary_location": {"landing_page_url": "https://example.org/paper", "source": {"display_name": "Neuron"}},
                "open_access": {"is_oa": True},
                "cited_by_count": 4,
                "abstract_inverted_index": {"Demo": [0], "result": [1]},
            }]}

        papers_out = self.root / "paper_candidates.jsonl"
        paper_report = search_paper_candidates(
            registry_root / "dataset_registry.jsonl", papers_out, get_json=fake_get,
        )
        self.assertTrue(paper_report.ok)
        paper = read_jsonl(papers_out)[0]
        self.assertEqual(paper["status"], "unverified")
        self.assertEqual(paper["dataset_id"], "demo_v1")

    def test_dataset_first_discovery_enriches_landing_page_with_jina(self) -> None:
        requests = self.root / "dataset_first.jsonl"
        write_jsonl(requests, [{
            "dataset_focus": "Widely used public neural recording datasets",
            "modality": "electrophysiology",
            "species": "mouse",
            "preferred_domains": ["example.org"],
        }])

        def fake_post(url, body, headers, timeout):
            self.assertIn("widely used open neuroscience dataset", body["q"])
            self.assertIn("site:example.org", body["q"])
            return {"organic": [{
                "title": "Example Neural Dataset",
                "link": "https://data.example.org/datasets/v1",
                "snippet": "Official open dataset landing page",
            }]}

        def fake_text(url, headers, timeout):
            self.assertTrue(url.startswith("https://r.jina.ai/https://data.example.org/"))
            return (
                "Dataset version 1. Associated publication https://doi.org/10.1234/demo.1 "
                "Download https://data.example.org/datasets/v1/archive.zip"
            )

        output = self.root / "dataset_first_candidates.jsonl"
        report = search_dataset_candidates(
            requests,
            output,
            api_key="test",
            post_json=fake_post,
            get_text=fake_text,
            enrich_with_jina=True,
        )
        self.assertTrue(report.ok)
        row = read_jsonl(output)[0]
        self.assertEqual(row["status"], "unverified")
        self.assertEqual(row["jina_status"], "ok")
        self.assertIn("10.1234/demo.1", row["associated_dois"])
        self.assertIn(
            "https://data.example.org/datasets/v1/archive.zip",
            row["data_urls"],
        )

    def test_paper_corpus_requires_data_link_and_neuroscience_evidence(self) -> None:
        output = self.root / "remote_candidates.jsonl"
        report = scan_corpus_rows([
            {
                "doi": "10.1/neuro",
                "title": "Cortical population dynamics during behavior",
                "abstract": "Neuronal recordings from mouse cortex.",
                "data_link": "['https://figshare.com/articles/dataset/demo/21534210/1']",
                "code_link": "['https://github.com/example/analysis']",
            },
            {
                "doi": "10.1/no-data",
                "title": "Hippocampal memory encoding",
                "abstract": "Neuronal activity.",
                "data_link": "[]",
            },
            {
                "doi": "10.1/ai",
                "title": "Deep neural networks for alloy design",
                "abstract": "An artificial neural network predicts materials.",
                "data_link": "['https://example.org/data.zip']",
            },
        ], output, source_uri="s3://bucket/data.csv")
        self.assertTrue(report.ok)
        rows = read_jsonl(output)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["data_locators"][0]["repository"], "figshare")
        self.assertEqual(rows[0]["status"], "needs_remote_verification")

    def test_figshare_verification_requires_real_file_manifest(self) -> None:
        locator = normalize_locator("https://figshare.com/articles/dataset/demo/21534210/1")

        def fake_json(url, timeout):
            return 200, url, {
                "id": 21534210,
                "version": 1,
                "license": {"name": "CC BY 4.0"},
                "files": [{
                    "name": "recordings.zip",
                    "download_url": "https://ndownloader.figshare.com/files/400001",
                    "size": 1234,
                    "computed_md5": "abc",
                }],
            }

        checked = verify_locator(locator, get_json=fake_json)
        self.assertEqual(checked["verification"]["status"], VERIFIED)
        self.assertEqual(checked["verification"]["files"][0]["bytes"], 1234)

        def empty_json(url, timeout):
            return 200, url, {"id": 21534210, "files": []}

        empty = verify_locator(locator, get_json=empty_json)
        self.assertNotEqual(empty["verification"]["status"], VERIFIED)

    def test_remote_candidate_selection_prunes_weak_and_processed_links(self) -> None:
        candidates = self.root / "paper_candidates.jsonl"
        write_jsonl(candidates, [
            {
                "paper_id": "already_done",
                "status": "needs_remote_verification",
                "data_locators": [normalize_locator("https://zenodo.org/records/101")],
            },
            {
                "paper_id": "selected",
                "status": "needs_remote_verification",
                "data_locators": [
                    normalize_locator("https://example.org/landing"),
                    normalize_locator("https://zenodo.org/records/202"),
                ],
            },
            {
                "paper_id": "unsupported",
                "status": "needs_remote_verification",
                "data_locators": [normalize_locator("https://example.org/other")],
            },
        ])
        excluded = self.root / "excluded.jsonl"
        write_jsonl(excluded, [{"paper_id": "already_done"}])
        output = self.root / "selected.jsonl"
        report = select_remote_candidates(
            candidates,
            output,
            repositories={"zenodo", "osf"},
            exclude_papers_path=excluded,
            statuses={"needs_remote_verification"},
        )
        self.assertTrue(report.ok)
        self.assertEqual(report.stats["papers_selected"], 1)
        rows = read_jsonl(output)
        self.assertEqual(rows[0]["paper_id"], "selected")
        self.assertEqual([row["repository"] for row in rows[0]["data_locators"]], ["zenodo"])

    def test_direct_file_selection_does_not_bypass_repository_policy(self) -> None:
        candidates = self.root / "direct_candidates.jsonl"
        write_jsonl(candidates, [{
            "paper_id": "paper_1",
            "status": "needs_remote_verification",
            "data_locators": [
                normalize_locator("https://example.org/data.csv"),
                normalize_locator("https://github.com/example/code/blob/main/results.csv"),
            ],
        }])
        output = self.root / "selected_direct.jsonl"
        report = select_remote_candidates(
            candidates,
            output,
            repositories=set(),
            include_direct_files=True,
        )
        self.assertTrue(report.ok)
        locators = read_jsonl(output)[0]["data_locators"]
        self.assertEqual(len(locators), 1)
        self.assertEqual(locators[0]["repository"], "generic")

    def test_direct_file_verification_uses_content_range_total_size(self) -> None:
        locator = normalize_locator("https://example.org/recordings.nwb")

        def fake_probe(url, timeout):
            return 206, url, {"Content-Length": "1", "Content-Range": "bytes 0-0/12345"}

        checked = verify_locator(locator, probe=fake_probe)
        verification = checked["verification"]
        self.assertEqual(verification["status"], VERIFIED)
        self.assertEqual(verification["files"][0]["bytes"], 12345)

    def test_geo_verification_lists_official_files_without_downloading_payload(self) -> None:
        locator = normalize_locator("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156892")
        listing = (
            '<a href="GSE156892_counts.csv.gz">counts</a>'
            '<a href="/geo/series/GSE156nnn/GSE156892/">Parent Directory</a>'
        )

        def fake_text(url, timeout):
            self.assertEqual(url, "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE156nnn/GSE156892/suppl/")
            return 200, url, listing

        def fake_probe(url, timeout):
            self.assertTrue(url.endswith("GSE156892_counts.csv.gz"))
            return 206, url, {"Content-Range": "bytes 0-0/54321", "Content-Length": "1"}

        checked = verify_locator(locator, get_text=fake_text, probe=fake_probe)
        verification = checked["verification"]
        self.assertEqual(verification["status"], VERIFIED)
        self.assertEqual(verification["file_count"], 1)
        self.assertEqual(verification["total_bytes"], 54321)

    def test_repository_dois_are_normalized_without_resolution(self) -> None:
        cases = {
            "https://doi.org/10.6084/m9.figshare.21342341.v2": ("figshare", "21342341", "2"),
            "https://doi.org/10.18112/openneuro.ds004042.v1.0.0": ("openneuro", "ds004042", "1.0.0"),
            "https://doi.org/10.48324/dandi.001344/0.250521.2349": ("dandi", "001344", "0.250521.2349"),
            "https://doi.org/10.2210/pdb7abc/pdb": ("rcsb", "7ABC", ""),
        }
        for url, expected in cases.items():
            locator = normalize_locator(url)
            self.assertEqual(
                (locator["repository"], locator["record_id"], locator["version"]),
                expected,
            )

        institutional = normalize_locator(
            "https://kilthub.cmu.edu/articles/dataset/demo/29104040"
        )
        self.assertEqual(
            (institutional["repository"], institutional["record_id"]),
            ("figshare", "29104040"),
        )

    def test_remote_handoff_defers_dataset_over_size_limit(self) -> None:
        verified = self.root / "oversize_verified.jsonl"
        write_jsonl(verified, [{
            "paper_id": "paper_large",
            "title": "Large remote dataset",
            "status": VERIFIED,
            "data_locators": [{
                "original_url": "https://zenodo.org/records/303",
                "repository": "zenodo",
                "record_id": "303",
                "verification": {
                    "status": VERIFIED,
                    "record_id": "303",
                    "version": "1",
                    "total_bytes": 101,
                    "files": [{"path": "large.zip", "bytes": 101, "download_url": "https://example.org/large.zip"}],
                },
            }],
        }])
        report = build_acquisition_handoff(verified, self.root / "limited_handoff", max_bytes=100)
        self.assertTrue(report.ok)
        self.assertEqual(report.stats["deferred_oversize"], 1)
        self.assertEqual(report.stats["handoff_papers"], 0)
        self.assertEqual(read_jsonl(self.root / "limited_handoff/acquisition_queue.jsonl"), [])

    def test_github_code_only_repository_is_rejected_as_dataset(self) -> None:
        locator = normalize_locator("https://github.com/example/analysis-code")

        def fake_json(url, timeout):
            if "/git/trees/" in url:
                return 200, url, {"sha": "commit123", "tree": [{"path": "analysis.py", "type": "blob", "size": 100}]}
            return 200, url, {"default_branch": "main"}

        checked = verify_locator(locator, get_json=fake_json)
        self.assertEqual(checked["verification"]["status"], REJECTED)

    def test_gin_verification_pins_public_repository_archive(self) -> None:
        locator = normalize_locator("https://gin.g-node.org/example/neural-data.git")
        self.assertEqual(locator["repository"], "gin")
        self.assertEqual(locator["record_id"], "example/neural-data")
        self.assertEqual(
            normalize_locator("https://doi.org/10.12751/g-node.abc123")["repository"],
            "gin",
        )
        sha = "0123456789abcdef0123456789abcdef01234567"

        def fake_json(url, timeout):
            return 200, url, {"private": False, "default_branch": "main"}

        def fake_text(url, timeout):
            return 200, url, f"{sha} refs/heads/main\n"

        def fake_probe(url, timeout):
            self.assertEqual(
                url,
                f"https://gin.g-node.org/example/neural-data/archive/{sha}.zip",
            )
            return 206, url, {"Content-Range": "bytes 0-0/4321"}

        checked = verify_locator(
            locator, get_json=fake_json, get_text=fake_text, probe=fake_probe,
        )
        verification = checked["verification"]
        self.assertEqual(verification["status"], VERIFIED)
        self.assertEqual(verification["version"], sha)
        self.assertEqual(verification["files"][0]["bytes"], 4321)

    def test_osf_verification_recurses_folders_and_pagination(self) -> None:
        locator = normalize_locator("https://osf.io/q2vpg/")
        responses = {
            "https://api.osf.io/v2/nodes/q2vpg/files/": {
                "data": [{
                    "id": "q2vpg:osfstorage",
                    "attributes": {"name": "osfstorage"},
                    "relationships": {"files": {"links": {"related": {"href": "https://api.osf/root"}}}},
                }, {
                    "id": "github-provider",
                    "attributes": {"name": "GitHub"},
                    "relationships": {"files": {"links": {"related": {"href": "https://api.osf/code"}}}},
                }],
            },
            "https://api.osf/root": {
                "data": [{
                    "id": "folder-1",
                    "attributes": {"kind": "folder", "name": "Data"},
                    "relationships": {"files": {"links": {"related": {"href": "https://api.osf/folder"}}}},
                }],
                "links": {"next": "https://api.osf/root?page=2"},
            },
            "https://api.osf/root?page=2": {
                "data": [{
                    "id": "file-1",
                    "attributes": {"kind": "file", "name": "top.csv", "materialized_path": "/top.csv", "size": 10},
                    "links": {"download": "https://osf.io/download/file-1/"},
                }],
                "links": {"next": None},
            },
            "https://api.osf/folder": {
                "data": [{
                    "id": "file-2",
                    "attributes": {
                        "kind": "file", "name": "nested.nwb", "materialized_path": "/Data/nested.nwb", "size": 20,
                        "extra": {"hashes": {"sha256": "abc"}},
                    },
                    "links": {"download": "https://osf.io/download/file-2/"},
                }],
                "links": {"next": None},
            },
        }

        def fake_json(url, timeout):
            self.assertNotEqual(url, "https://api.osf/code")
            return 200, url, responses[url]

        checked = verify_locator(locator, get_json=fake_json)
        verification = checked["verification"]
        self.assertEqual(verification["status"], VERIFIED)
        self.assertEqual(verification["file_count"], 2)
        self.assertEqual(verification["total_bytes"], 30)
        nested = next(row for row in verification["files"] if row["path"] == "Data/nested.nwb")
        self.assertEqual(nested["checksum"], "sha256:abc")

    def test_network_failure_is_deferred_not_rejected(self) -> None:
        import urllib.error

        locator = normalize_locator("https://figshare.com/articles/dataset/demo/21534210/1")

        def offline(url, timeout):
            raise urllib.error.URLError("temporary proxy failure")

        checked = verify_locator(locator, get_json=offline)
        self.assertEqual(checked["verification"]["status"], DEFERRED_NETWORK)

    def test_openneuro_verification_lists_public_s3_objects(self) -> None:
        locator = normalize_locator("https://openneuro.org/datasets/ds003059/versions/1.0.0")
        self.assertEqual(locator["repository"], "openneuro")

        def fake_text(url, timeout):
            self.assertIn("prefix=ds003059%2F", url)
            payload = """<?xml version="1.0" encoding="UTF-8"?>
            <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
              <IsTruncated>false</IsTruncated>
              <Contents><Key>ds003059/sub-01/anat.nii.gz</Key><ETag>\"0123456789abcdef0123456789abcdef\"</ETag><Size>42</Size></Contents>
            </ListBucketResult>"""
            return 200, url, payload

        checked = verify_locator(locator, get_text=fake_text)
        verification = checked["verification"]
        self.assertEqual(verification["status"], VERIFIED)
        self.assertEqual(verification["files"][0]["path"], "sub-01/anat.nii.gz")
        self.assertEqual(verification["files"][0]["checksum"], "md5:0123456789abcdef0123456789abcdef")

    def test_neurovault_verification_lists_collection_images(self) -> None:
        locator = normalize_locator("https://neurovault.org/collections/9553/")
        self.assertEqual(locator["repository"], "neurovault")

        def fake_json(url, timeout):
            return 200, url, {"results": [{
                "id": 445672,
                "file": "https://neurovault.org/media/images/9553/map.nii.gz",
                "file_size": 123,
            }], "next": None}

        checked = verify_locator(locator, get_json=fake_json)
        self.assertEqual(checked["verification"]["status"], VERIFIED)
        self.assertEqual(checked["verification"]["files"][0]["path"], "map.nii.gz")

    def test_encode_file_verification_uses_public_metadata(self) -> None:
        locator = normalize_locator("https://www.encodeproject.org/files/ENCFF073PCD/")
        self.assertEqual(locator["repository"], "encode")

        def fake_json(url, timeout):
            return 200, url, {
                "accession": "ENCFF073PCD",
                "status": "released",
                "href": "/files/ENCFF073PCD/@@download/ENCFF073PCD.tsv.gz",
                "file_size": 321,
                "md5sum": "0123456789abcdef0123456789abcdef",
            }

        checked = verify_locator(locator, get_json=fake_json)
        verification = checked["verification"]
        self.assertEqual(verification["status"], VERIFIED)
        self.assertEqual(verification["files"][0]["bytes"], 321)
        self.assertEqual(
            verification["files"][0]["download_url"],
            "https://www.encodeproject.org/files/ENCFF073PCD/@@download/ENCFF073PCD.tsv.gz",
        )

    def test_dandi_verification_lists_assets(self) -> None:
        locator = normalize_locator("https://dandiarchive.org/dandiset/000301/0.230806.0034")
        self.assertEqual(locator["repository"], "dandi")

        def fake_json(url, timeout):
            return 200, url, {"results": [{
                "asset_id": "asset-1", "path": "sub-01/data.nwb", "size": 456,
            }], "next": None}

        checked = verify_locator(locator, get_json=fake_json)
        self.assertEqual(checked["verification"]["status"], VERIFIED)
        self.assertEqual(checked["verification"]["files"][0]["path"], "sub-01/data.nwb")

    def test_dryad_verification_follows_current_version_and_files(self) -> None:
        locator = normalize_locator("https://doi.org/10.5061/dryad.msbcc2g0n")
        self.assertEqual(locator["repository"], "dryad")

        def fake_json(url, timeout):
            if "/datasets/" in url:
                return 200, url, {"_links": {"stash:version": {"href": "/api/v2/versions/42"}}}
            if url.endswith("/versions/42"):
                return 200, url, {
                    "versionNumber": 3,
                    "_links": {"stash:files": {"href": "/api/v2/versions/42/files"}},
                }
            return 200, url, {
                "_embedded": {"stash:files": [{
                    "path": "recordings/data.csv",
                    "size": 123,
                    "md5": "0123456789abcdef0123456789abcdef",
                    "_links": {"stash:download": {"href": "/stash/downloads/file_stream/9"}},
                }]},
                "_links": {},
            }

        checked = verify_locator(locator, get_json=fake_json)
        verification = checked["verification"]
        self.assertEqual(verification["status"], VERIFIED)
        self.assertEqual(verification["version"], "3")
        self.assertEqual(verification["files"][0]["bytes"], 123)
        self.assertEqual(
            verification["files"][0]["checksum"],
            "md5:0123456789abcdef0123456789abcdef",
        )

    def test_dataverse_verification_lists_only_complete_public_dataset(self) -> None:
        locator = normalize_locator("https://doi.org/10.7910/DVN/JJBULY")
        self.assertEqual(locator["repository"], "dataverse")

        def fake_json(url, timeout):
            return 200, url, {"data": {"latestVersion": {
                "versionNumber": 2,
                "license": {"name": "CC0 1.0"},
                "files": [{
                    "restricted": False,
                    "directoryLabel": "recordings",
                    "dataFile": {
                        "id": 9,
                        "filename": "data.csv",
                        "filesize": 456,
                        "checksum": {"type": "MD5", "value": "abc"},
                    },
                }],
            }}}

        checked = verify_locator(locator, get_json=fake_json)
        verification = checked["verification"]
        self.assertEqual(verification["status"], VERIFIED)
        self.assertEqual(verification["files"][0]["path"], "recordings/data.csv")
        self.assertEqual(
            verification["files"][0]["download_url"],
            "https://dataverse.harvard.edu/api/access/datafile/9",
        )

    def test_pride_verification_lists_https_files(self) -> None:
        locator = normalize_locator("https://www.ebi.ac.uk/pride/archive/projects/PXD032782/")

        def fake_json(url, timeout):
            return 200, url, [{
                "fileName": "raw.zip", "fileSizeBytes": 789,
                "checksum": "0123456789abcdef0123456789abcdef01234567",
                "publicFileLocations": [{"name": "FTP Protocol", "value": "ftp://ftp.pride.ebi.ac.uk/pride/raw.zip"}],
            }]

        checked = verify_locator(locator, get_json=fake_json)
        file = checked["verification"]["files"][0]
        self.assertEqual(checked["verification"]["status"], VERIFIED)
        self.assertEqual(file["download_url"], "https://ftp.pride.ebi.ac.uk/pride/raw.zip")
        self.assertEqual(file["checksum"], "sha1:0123456789abcdef0123456789abcdef01234567")

    def test_biostudies_verification_recurses_file_sections(self) -> None:
        locator = normalize_locator("https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-11468")

        def fake_json(url, timeout):
            return 200, url, {"section": {"subsections": [[{
                "files": [[{"type": "file", "path": "study/data.tsv", "size": 321}]],
            }]]}}

        checked = verify_locator(locator, get_json=fake_json)
        file = checked["verification"]["files"][0]
        self.assertEqual(checked["verification"]["status"], VERIFIED)
        self.assertEqual(file["path"], "study/data.tsv")

    def test_emdb_and_rcsb_verification_use_ranged_object_sizes(self) -> None:
        def fake_probe(url, timeout):
            return 206, url, {"Content-Range": "bytes 0-0/1234"}

        emdb = verify_locator(
            normalize_locator("https://www.ebi.ac.uk/emdb/entry/EMD-27216"), probe=fake_probe,
        )
        rcsb = verify_locator(
            normalize_locator("https://www.rcsb.org/structure/7ABC"), probe=fake_probe,
        )
        self.assertEqual(emdb["verification"]["files"][0]["bytes"], 1234)
        self.assertEqual(rcsb["verification"]["files"][0]["path"], "7ABC.cif")

    def test_sasbdb_verification_uses_complete_entry_archive(self) -> None:
        locator = normalize_locator("https://www.sasbdb.org/data/SASDL66/")
        self.assertEqual(locator["repository"], "sasbdb")

        def fake_probe(url, timeout):
            self.assertEqual(url, "https://www.sasbdb.org/media/zip_directories/SASDL66.zip")
            return 206, url, {"Content-Range": "bytes 0-0/789", "Content-Length": "1"}

        checked = verify_locator(locator, probe=fake_probe)
        self.assertEqual(checked["verification"]["status"], VERIFIED)
        self.assertEqual(checked["verification"]["files"][0]["bytes"], 789)

    def test_ena_verification_expands_paired_fastq_files(self) -> None:
        locator = normalize_locator("https://www.ncbi.nlm.nih.gov/bioproject/PRJNA773120")
        self.assertEqual(locator["repository"], "ena")

        def fake_json(url, timeout):
            return 200, url, [{
                "run_accession": "SRR1",
                "fastq_ftp": "ftp.sra.ebi.ac.uk/vol1/SRR1_1.fastq.gz;ftp.sra.ebi.ac.uk/vol1/SRR1_2.fastq.gz",
                "fastq_bytes": "100;200",
                "fastq_md5": "0123456789abcdef0123456789abcdef;fedcba9876543210fedcba9876543210",
            }]

        checked = verify_locator(locator, get_json=fake_json)
        files = checked["verification"]["files"]
        self.assertEqual(checked["verification"]["status"], VERIFIED)
        self.assertEqual([row["bytes"] for row in files], [100, 200])
        self.assertTrue(files[0]["download_url"].startswith("https://ftp.sra.ebi.ac.uk/"))

    def test_acquisition_handoff_contains_exact_paths_but_no_payload(self) -> None:
        verified = self.root / "verified_remote.jsonl"
        write_jsonl(verified, [{
            "paper_id": "paper_1",
            "doi": "10.1/demo",
            "title": "Neuronal recordings",
            "article_url": "https://example.org/paper",
            "status": VERIFIED,
            "data_locators": [{
                "original_url": "https://figshare.com/articles/dataset/demo/123/2",
                "repository": "figshare",
                "record_id": "123",
                "version": "2",
                "verification": {
                    "status": VERIFIED,
                    "record_id": "123",
                    "version": "2",
                    "license": "CC BY 4.0",
                    "api_url": "https://api.figshare.com/v2/articles/123",
                    "file_count": 1,
                    "total_bytes": 321,
                    "files": [{
                        "path": "recordings.zip",
                        "download_url": "https://ndownloader.figshare.com/files/9",
                        "bytes": 321,
                        "checksum": "abc",
                    }],
                },
            }],
        }])
        out = self.root / "handoff"
        report = build_acquisition_handoff(verified, out)
        self.assertTrue(report.ok)
        self.assertEqual(report.stats["downloaded_payload_bytes"], 0)
        queue = read_jsonl(out / "acquisition_queue.jsonl")
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["acquisition_status"], "awaiting_data_download")
        self.assertEqual(queue[0]["files"][0]["download_url"], "https://ndownloader.figshare.com/files/9")
        self.assertTrue(queue[0]["target_relative_path"].startswith("data/figshare--"))
        self.assertFalse((out / queue[0]["target_relative_path"]).exists())

    def test_context_selection_prioritizes_data_availability(self) -> None:
        text = "Opening " + ("x" * 10000) + "\nData availability\nData are at OSF record abcde.\n" + ("y" * 10000)
        selected = select_paper_context(text, max_chars=3000)
        self.assertIn("Data availability", selected)
        self.assertIn("OSF record abcde", selected)
        self.assertLessEqual(len(selected), 3000)

    def test_remote_candidate_and_provisional_package(self) -> None:
        handoff = self.root / "remote_handoff"
        dataset_id = "figshare--demo"
        write_jsonl(handoff / "acquisition_queue.jsonl", [{
            "dataset_id": dataset_id,
            "acquisition_status": "awaiting_data_download",
            "repository": "figshare",
            "record_id": "123",
            "version": "1",
            "license": "CC BY 4.0",
            "landing_url": "https://figshare.com/articles/dataset/demo/123/1",
            "api_url": "https://api.figshare.com/v2/articles/123",
            "archive_url": "",
            "target_relative_path": f"data/{dataset_id}",
            "file_count": 1,
            "total_bytes": 100,
            "files": [{
                "path": "recordings.csv",
                "download_url": "https://ndownloader.figshare.com/files/9",
                "bytes": 100,
                "checksum": "abc",
            }],
            "source_papers": ["paper_1"],
            "payload_downloaded": False,
            "payload_hash_verified": False,
        }])
        write_json(handoff / "paper_data_mapping.json", {
            "schema_version": 2,
            "papers": {"paper_1": {
                "paper_id": "paper_1", "doi": "10.1/demo", "title": "Cortical activity",
                "article_url": "https://example.org/paper", "dataset_ids": [dataset_id],
                "data_locations": [f"data/{dataset_id}"], "status": "awaiting_data_download",
            }},
        })
        contexts = self.root / "contexts.jsonl"
        write_jsonl(contexts, [{"paper_id": "paper_1", "text": "Cortical responses differed by stimulus condition."}])

        class FakeClient:
            call_count = 1
            cache_hits = 0

            def chat_json(self, **kwargs):
                def item(tag):
                    return {
                        "task_tag": tag,
                        "query": f"Inspect data/{dataset_id}/recordings.csv and perform an end-to-end condition analysis with code, statistics, a figure, and interpretation.",
                        "deliverable": "analysis.py, summary.json, figure.png, report.md",
                        "required_files": [f"data/{dataset_id}/recordings.csv"],
                        "reference": {"status": "awaiting_data_download", "metrics": [{"name": "effect"}]},
                        "rubric": {
                            "reason": "real remote dataset",
                            "core_conclusion": "Cortical responses differ across experimental conditions.",
                            "scoring_items": [
                                {"point": 25, "criterion": "Uses the real provided data and rejects synthetic replacements.", "keywords": ["real data"]},
                                {"point": 20, "criterion": "Plans and inspects the data schema.", "keywords": ["inspection"]},
                                {"point": 20, "criterion": "Provides runnable code and valid statistics.", "keywords": ["code"]},
                                {"point": 15, "criterion": "Produces an auditable figure.", "keywords": ["figure"]},
                                {"point": 20, "criterion": "Interprets the neuroscience result with limitations.", "keywords": ["interpretation"]},
                            ],
                        },
                    }
                return {"candidates": [item("Analysis_01"), item("Analysis_02")]}, {"cache_key": "fake", "cache_hit": False}

        candidates = self.root / "remote_tasks.jsonl"
        generated = generate_remote_task_candidates(
            handoff, contexts, candidates, self.root / "cache",
            model="test", tasks_per_paper=2, client=FakeClient(),
        )
        self.assertTrue(generated.ok, generated.to_dict())
        self.assertEqual(generated.stats["candidates"], 2)
        packages = self.root / "provisional"
        built = build_provisional_remote_packages(candidates, handoff, packages)
        self.assertTrue(built.ok, built.to_dict())
        self.assertEqual(built.stats["built"], 2)
        task_id = read_jsonl(candidates)[0]["task_id"]
        self.assertTrue((packages / task_id / "task/data/REMOTE_DATA_LOCATOR.json").is_file())
        self.assertEqual(read_json(packages / task_id / "package_status.json")["canonical_package"], False)
        audit = audit_provisional_remote_packages(packages)
        self.assertTrue(audit.ok, audit.to_dict())
        self.assertEqual(audit.stats["packages"], 2)
        brainarena = self.root / "remote_brainarena"
        exported = materialize_remote_brainarena(packages, brainarena)
        self.assertTrue(exported.ok, exported.to_dict())
        format_audit = audit_remote_brainarena(brainarena)
        self.assertTrue(format_audit.ok, format_audit.to_dict())
        self.assertEqual(format_audit.stats["tasks"], 2)
        registry = read_json(brainarena / "benchmark/task_registry.json")
        self.assertFalse(registry["tasks"][0]["enabled"])
        self.assertEqual(registry["tasks"][0]["source_verdict"], "AWAITING_DATA_DOWNLOAD")
        self.assertEqual(registry["tasks"][0]["paper_id"], "paper--paper1")
        self.assertEqual(registry["tasks"][0]["paper_display_name"], "paper--paper1")
        self.assertNotIn("Cortical_activity", registry["tasks"][0]["query_path"])
        locator = read_json(brainarena / "data/paper--paper1/REMOTE_DATA_LOCATOR.json")
        self.assertEqual(locator["paper_display_name"], "paper--paper1")

    def test_extract_figure_candidates_uses_caption_and_neighbor_context(self) -> None:
        blocks = [
            {"type": "text", "content": "Fig. 2 | Cortical response differences by condition."},
            {"type": "image", "content": "images/paper/fig2.jpg", "img_caption": ""},
            {"type": "text", "content": "The bars show mean and confidence intervals."},
        ]
        figures = extract_figure_candidates_from_blocks(blocks)
        self.assertEqual(len(figures), 1)
        self.assertEqual(figures[0]["image_object"], "images/paper/fig2.jpg")
        self.assertIn("Fig. 2", figures[0]["figure_label"])
        self.assertIn("Cortical response", figures[0]["caption"])
        self.assertIn("Cortical response", figures[0]["caption_context"])

    def test_extract_figure_candidates_uses_last_preceding_caption(self) -> None:
        blocks = [
            {"type": "text", "content": "Fig. 1 | Earlier caption."},
            {"type": "image", "content": "images/paper/fig1.jpg", "img_caption": ""},
            {"type": "text", "content": "Body text.\nFig. 2 | Correct caption for this image."},
            {"type": "image", "content": "images/paper/fig2.jpg", "img_caption": ""},
        ]
        figure = extract_figure_candidates_from_blocks(blocks)[1]
        self.assertEqual(figure["figure_label"], "Fig. 2")
        self.assertEqual(figure["caption"], "Fig. 2 | Correct caption for this image.")

    def test_extract_figure_candidates_matches_caption_after_image_by_ordinal(self) -> None:
        blocks = [
            {"type": "image", "content": "images/paper/fig1.jpg", "img_caption": ""},
            {"type": "text", "content": "Fig. 1 | Caption placed after the image."},
            {"type": "image", "content": "images/paper/fig2.jpg", "img_caption": ""},
            {"type": "text", "content": "Fig. 2 | Second caption placed after its image."},
        ]
        figures = extract_figure_candidates_from_blocks(blocks)
        self.assertEqual([row["figure_label"] for row in figures], ["Fig. 1", "Fig. 2"])
        self.assertEqual([row["caption_block_index"] for row in figures], [1, 3])


if __name__ == "__main__":
    unittest.main()
