# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSWE benchmark loader tests: task discovery, provisioning, and the
ground-truth JSON payload the agentic runner parses."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from aiperf.accuracy.benchmarks import deepswe
from aiperf.accuracy.benchmarks.deepswe import DeepSWEBenchmark
from aiperf.common.environment import Environment
from aiperf.common.exceptions import ConfigurationError
from tests.unit.conftest import make_benchmark_run

TASK_TOML_TEMPLATE = """\
schema_version = "1.3"
[task]
name = "datacurve/{task_id}"
[metadata]
task_id = "{task_id}"
language = "python"
repository_url = "https://example.com/repo.git"
base_commit_hash = "abc123def"
[agent]
timeout_sec = 10800.0
[verifier]
timeout_sec = 1800.0
[environment]
docker_image = "public.ecr.aws/example/{task_id}:v1.1"
"""

INSTRUCTION_TEMPLATE = "Implement feature {task_id}.\n\nIMPORTANT: commit everything.\n"

TASK_IDS = ("alpha-feature", "beta-fix")


def make_tasks_tree(root: Path, task_ids: tuple[str, ...] = TASK_IDS) -> Path:
    """Create a fake deep-swe tasks tree: <root>/tasks/<id>/{task.toml,instruction.md}."""
    tasks_dir = root / "tasks"
    for task_id in task_ids:
        task_dir = tasks_dir / task_id
        task_dir.mkdir(parents=True)
        (task_dir / "task.toml").write_text(
            TASK_TOML_TEMPLATE.format(task_id=task_id), encoding="utf-8"
        )
        (task_dir / "instruction.md").write_text(
            INSTRUCTION_TEMPLATE.format(task_id=task_id), encoding="utf-8"
        )
    return tasks_dir


def use_tasks_dir(monkeypatch, tasks_dir: Path) -> None:
    monkeypatch.setattr(
        Environment.ACCURACY, "DEEPSWE_TASKS_DIR", str(tasks_dir.parent)
    )


class TestValidateTasksDir:
    def test_valid_dir_returns_sorted_task_ids(self, tmp_path: Path) -> None:
        tasks_dir = make_tasks_tree(tmp_path)
        assert deepswe.validate_tasks_dir(tasks_dir) == list(TASK_IDS)

    def test_missing_dir_raises_configuration_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="does not exist"):
            deepswe.validate_tasks_dir(tmp_path / "nope")

    def test_empty_dir_raises_configuration_error(self, tmp_path: Path) -> None:
        empty = tmp_path / "tasks"
        empty.mkdir()
        with pytest.raises(ConfigurationError, match="no task folders"):
            deepswe.validate_tasks_dir(empty)

    def test_find_tasks_dir_accepts_repo_root_and_direct_tasks_dir(
        self, tmp_path: Path
    ) -> None:
        tasks_dir = make_tasks_tree(tmp_path)
        # An extracted repo root (contains tasks/) resolves to the tasks dir.
        assert deepswe._find_tasks_dir(tmp_path) == tasks_dir
        # The tasks dir itself also resolves to itself.
        assert deepswe._find_tasks_dir(tasks_dir) == tasks_dir

    def test_find_tasks_dir_rejects_unrelated_dir(self, tmp_path: Path) -> None:
        (tmp_path / "unrelated").mkdir()
        with pytest.raises(ConfigurationError, match="tasks directory not found"):
            deepswe._find_tasks_dir(tmp_path / "unrelated")


class TestLoadProblems:
    @pytest.fixture
    def run(self, monkeypatch, tmp_path: Path):
        use_tasks_dir(monkeypatch, make_tasks_tree(tmp_path))
        return make_benchmark_run(accuracy={"benchmark": "deepswe"})

    async def test_load_problems_all_tasks(self, run) -> None:
        problems = await DeepSWEBenchmark(run=run).load_problems(None, 0, False)

        assert [p.task for p in problems] == list(TASK_IDS)
        for problem, task_id in zip(problems, TASK_IDS, strict=True):
            # The prompt is the canonical instruction verbatim.
            assert problem.prompt == INSTRUCTION_TEMPLATE.format(task_id=task_id)
            assert problem.metadata == {"agentic": True}

    async def test_load_problems_ground_truth_payload(self, run) -> None:
        import orjson

        problems = await DeepSWEBenchmark(run=run).load_problems(None, 0, False)

        info = orjson.loads(problems[0].ground_truth)
        assert info["task_id"] == "alpha-feature"
        assert info["image"] == "public.ecr.aws/example/alpha-feature:v1.1"
        assert info["base_commit"] == "abc123def"
        assert info["agent_timeout_s"] == 10800.0
        assert info["verifier_timeout_s"] == 1800.0
        assert info["language"] == "python"
        assert Path(info["task_dir"]).is_dir()

    async def test_load_problems_selected_tasks(self, run) -> None:
        problems = await DeepSWEBenchmark(run=run).load_problems(["beta-fix"], 0, False)
        assert [p.task for p in problems] == ["beta-fix"]

    async def test_load_problems_unknown_task_lists_available(self, run) -> None:
        with pytest.raises(ConfigurationError, match="alpha-feature"):
            await DeepSWEBenchmark(run=run).load_problems(["nope-task"], 0, False)

    async def test_load_problems_ignores_n_shots_and_cot(self, run) -> None:
        baseline = await DeepSWEBenchmark(run=run).load_problems(None, 0, False)
        with_shots = await DeepSWEBenchmark(run=run).load_problems(None, 5, True)
        assert [p.prompt for p in with_shots] == [p.prompt for p in baseline]


class TestProvisionTasks:
    def _tarball_bytes(self) -> bytes:
        """A minimal GitHub-style archive: deep-swe-<ref>/tasks/<id>/task.toml."""
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for task_id in TASK_IDS:
                content = TASK_TOML_TEMPLATE.format(task_id=task_id).encode()
                member = tarfile.TarInfo(f"deep-swe-ref/tasks/{task_id}/task.toml")
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        return buffer.getvalue()

    def _patch_cache(self, monkeypatch, tmp_path: Path) -> Path:
        cache_root = tmp_path / "cache"
        monkeypatch.setattr(deepswe, "DEEPSWE_CACHE_DIR", cache_root)
        return cache_root

    def _patch_download(
        self, monkeypatch, payload: bytes | Exception
    ) -> pytest.MonkeyPatch:
        async def fake_download(url: str, dest: Path) -> None:
            if isinstance(payload, Exception):
                raise payload
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)

        monkeypatch.setattr(deepswe, "_download_tarball", fake_download)
        return monkeypatch

    async def test_env_override_wins_without_download(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        tasks_dir = make_tasks_tree(tmp_path)
        monkeypatch.setattr(
            Environment.ACCURACY, "DEEPSWE_TASKS_DIR", str(tasks_dir.parent)
        )
        # Any download attempt would fail the test.
        self._patch_download(monkeypatch, RuntimeError("must not download"))

        assert await deepswe.provision_tasks() == tasks_dir

    async def test_download_failure_raises_configuration_error(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        import aiohttp

        self._patch_cache(monkeypatch, tmp_path)
        monkeypatch.setattr(Environment.ACCURACY, "DEEPSWE_TASKS_DIR", None)
        monkeypatch.setattr(Environment.ACCURACY, "DEEPSWE_PINNED_REF", "deadbeef")
        self._patch_download(monkeypatch, aiohttp.ClientError("no route to github"))

        with pytest.raises(ConfigurationError, match="Failed to download"):
            await deepswe.provision_tasks()

    async def test_downloads_extracts_and_caches(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        cache_root = self._patch_cache(monkeypatch, tmp_path)
        monkeypatch.setattr(Environment.ACCURACY, "DEEPSWE_TASKS_DIR", None)
        monkeypatch.setattr(Environment.ACCURACY, "DEEPSWE_PINNED_REF", "ref")
        self._patch_download(monkeypatch, self._tarball_bytes())

        tasks_dir = await deepswe.provision_tasks()

        assert tasks_dir == cache_root / "ref" / "tasks"
        assert deepswe.list_task_ids(tasks_dir) == list(TASK_IDS)
        # The tarball is consumed by extraction, leaving only the cache entry.
        assert not (cache_root / "ref.tar.gz").exists()

        # A second provisioning call is a cache hit: no download needed.
        self._patch_download(monkeypatch, RuntimeError("must not re-download"))
        assert await deepswe.provision_tasks() == tasks_dir

    async def test_stale_cache_entry_is_replaced(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        cache_root = self._patch_cache(monkeypatch, tmp_path)
        monkeypatch.setattr(Environment.ACCURACY, "DEEPSWE_TASKS_DIR", None)
        monkeypatch.setattr(Environment.ACCURACY, "DEEPSWE_PINNED_REF", "ref")
        # A leftover invalid entry (interrupted extraction).
        stale = cache_root / "ref" / "tasks"
        stale.mkdir(parents=True)
        self._patch_download(monkeypatch, self._tarball_bytes())

        tasks_dir = await deepswe.provision_tasks()

        assert deepswe.list_task_ids(tasks_dir) == list(TASK_IDS)

    def test_extract_tarball_strips_leading_component_and_skips_symlinks(
        self, tmp_path: Path
    ) -> None:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            content = b"hello"
            member = tarfile.TarInfo("deep-swe-ref/tasks/x/task.toml")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
            link = tarfile.TarInfo("deep-swe-ref/tasks/x/link")
            link.type = tarfile.SYMTYPE
            link.linkname = "task.toml"
            archive.addfile(link)

        tarball = tmp_path / "tb.tar.gz"
        tarball.write_bytes(buffer.getvalue())
        dest = tmp_path / "out"

        deepswe._extract_tarball(tarball, dest)

        assert (dest / "tasks" / "x" / "task.toml").read_bytes() == b"hello"
        assert not (dest / "tasks" / "x" / "link").exists()
