from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from neuro_dataset_factory.ops import download_remote_payloads as download


class DownloadRemotePayloadsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def downloader_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            output_root=self.root / "datasets",
            state_root=self.root / "state",
            package_root=None,
            dry_run=False,
            curl="curl",
            connect_timeout=5,
            file_timeout=0,
            retries=4,
            low_speed_limit=1024,
            low_speed_time=300,
            per_host_jobs=2,
            host_jobs="example.org=1",
        )

    def test_parse_host_jobs(self) -> None:
        self.assertEqual(
            download.parse_host_jobs("example.org=2, osf.io=3"),
            {"example.org": 2, "osf.io": 3},
        )

    def test_completed_ids_from_log_uses_latest_status(self) -> None:
        log = self.root / "run.log"
        log.write_text(
            "\n".join(
                [
                    json.dumps({"dataset_id": "a", "complete": True}),
                    json.dumps({"dataset_id": "b", "complete": True}),
                    json.dumps({"dataset_id": "a", "complete": False}),
                    "not-json",
                ]
            ),
            encoding="utf-8",
        )
        self.assertEqual(download.completed_ids_from_log(log), {"b"})

    def test_download_uses_host_limit_and_low_speed_options(self) -> None:
        args = self.downloader_args()
        worker = download.Downloader(args, [])
        item = {
            "resolved_path": "payload.bin",
            "download_url": "https://example.org/payload.bin",
            "bytes": 4,
        }
        observed: list[str] = []

        def fake_run(command, **kwargs):
            observed.extend(command)
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"data")
            return SimpleNamespace(returncode=0, stderr="")

        with mock.patch.object(download.subprocess, "run", side_effect=fake_run):
            ok, status, size = worker._download_file(item, args.output_root / "dataset")

        self.assertEqual((ok, status, size), (True, "downloaded_and_verified", 4))
        self.assertEqual(observed[observed.index("--speed-limit") + 1], "1024")
        self.assertEqual(observed[observed.index("--speed-time") + 1], "300")
        self.assertNotIn("--retry-delay", observed)


if __name__ == "__main__":
    unittest.main()
