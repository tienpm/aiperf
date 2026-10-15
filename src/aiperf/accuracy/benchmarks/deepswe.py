# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSWE v1.1 agentic benchmark loader.

DeepSWE (``github.com/datacurve-ai/deep-swe``) measures coding agents on 113
original, long-horizon software-engineering tasks drawn from active
open-source repositories (TypeScript, Go, Python, JavaScript, Rust). Each task
is a Harbor-format directory — ``task.toml`` (repo URL, base commit, prebuilt
sandbox image, timeouts, resource limits), ``instruction.md``, ``tests/``
(program-based verifier), and a held-out ``solution/`` — graded by applying the
agent's committed work as a patch in a pristine verifier container
(``reward.json``, binary reward plus pass fractions).

Unlike AIPerf's single-request accuracy benchmarks, DeepSWE is *agentic*: the
model interactively explores the repository and runs commands inside the task
sandbox, so it cannot be expressed as pre-defined conversations in the
load-generation pipeline. This loader only discovers and provisions tasks; the
agent loop and verifier are delegated to Pier
(``pier run --agent mini-swe-agent``) by
:class:`aiperf.accuracy.deepswe_runner.DeepSWEAgenticRunner`, which
``run_benchmark`` dispatches to automatically because this benchmark's
plugins.yaml metadata declares ``agentic: true``.

Task provisioning resolves in this order:

1. ``AIPERF_ACCURACY_DEEPSWE_TASKS_DIR`` — a pre-provisioned local tasks
   directory (the dir holding the task folders, or an extracted repo root
   containing ``tasks/``). The supported path for hosts without GitHub access.
2. The pinned upstream tarball
   (``https://github.com/datacurve-ai/deep-swe/archive/<ref>.tar.gz``, ref
   from ``AIPERF_ACCURACY_DEEPSWE_PINNED_REF``), downloaded once and extracted
   into the dataset cache (``.cache/aiperf/datasets/deepswe/<ref>/``).

Each problem's ``ground_truth`` carries a JSON payload (task dir, image, base
commit, timeouts) that the runner parses — the same ground-truth-JSON pattern
the ``code_execution`` grader uses for LiveCodeBench test cases.
"""

from __future__ import annotations

import asyncio
import shutil
import tarfile
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp
import orjson

from aiperf.accuracy.models import BenchmarkProblem
from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.environment import Environment
from aiperf.common.exceptions import ConfigurationError
from aiperf.common.mixins import AIPerfLoggerMixin

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

GITHUB_REPO = "datacurve-ai/deep-swe"

# Dataset cache root the pinned tarball extracts into (one subdir per ref),
# alongside the other public-dataset caches under .cache/aiperf/datasets.
DEEPSWE_CACHE_DIR = Path(".cache/aiperf/datasets/deepswe")

# Seconds allowed for downloading the ~4 MB upstream tarball.
TARBALL_DOWNLOAD_TIMEOUT_S = 300.0

# Every task directory must carry a task.toml (the Harbor task manifest) and
# an instruction.md (the canonical prompt handed to the agent verbatim).
TASK_MANIFEST_FILENAME = "task.toml"
INSTRUCTION_FILENAME = "instruction.md"

_logger = AIPerfLogger(__name__)


def _tarball_url(ref: str) -> str:
    return f"https://github.com/{GITHUB_REPO}/archive/{ref}.tar.gz"


def _is_task_dir(path: Path) -> bool:
    return path.is_dir() and (path / TASK_MANIFEST_FILENAME).is_file()


def _find_tasks_dir(root: Path) -> Path:
    """Return the tasks directory at or under ``root``.

    Accepts either an extracted repo root (contains ``tasks/``) or a tasks
    directory itself (contains the task folders), so
    ``AIPERF_ACCURACY_DEEPSWE_TASKS_DIR`` can point at either shape.
    """
    tasks_dir = root / "tasks"
    if tasks_dir.is_dir():
        return tasks_dir
    if any(_is_task_dir(child) for child in root.iterdir()):
        return root
    raise ConfigurationError(
        f"DeepSWE tasks directory not found under '{root}': expected either "
        f"a 'tasks/' subdirectory (extracted repo root) or the task folders "
        f"themselves (each with a '{TASK_MANIFEST_FILENAME}')."
    )


def list_task_ids(tasks_dir: Path) -> list[str]:
    """Return the sorted task ids (directory names) under ``tasks_dir``."""
    return sorted(child.name for child in tasks_dir.iterdir() if _is_task_dir(child))


def validate_tasks_dir(tasks_dir: Path) -> list[str]:
    """Validate a tasks directory and return its sorted task ids.

    Raises:
        ConfigurationError: If the directory does not exist or contains no
            Harbor-format task directories (each needs a ``task.toml``).
    """
    if not tasks_dir.is_dir():
        raise ConfigurationError(
            f"DeepSWE tasks directory '{tasks_dir}' does not exist. Provide a "
            f"valid directory via AIPERF_ACCURACY_DEEPSWE_TASKS_DIR or allow "
            f"the benchmark to download the pinned task set from GitHub."
        )
    task_ids = list_task_ids(tasks_dir)
    if not task_ids:
        raise ConfigurationError(
            f"DeepSWE tasks directory '{tasks_dir}' contains no task folders "
            f"(directories with a '{TASK_MANIFEST_FILENAME}')."
        )
    return task_ids


def _is_valid_tasks_dir(tasks_dir: Path) -> bool:
    try:
        validate_tasks_dir(tasks_dir)
    except ConfigurationError:
        return False
    return True


async def _download_tarball(url: str, dest: Path) -> None:
    """Download the deep-swe tarball to ``dest`` as raw bytes.

    ``AioHttpClient.get_request`` text-decodes every non-media content type
    and GitHub serves the archive as ``application/x-gzip``, so the binary
    download goes through a plain aiohttp session — the same direct-aiohttp
    approach ``cli_commands/chat.py`` uses for non-pipeline HTTP.
    """
    timeout = aiohttp.ClientTimeout(total=TARBALL_DOWNLOAD_TIMEOUT_S)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.get(url) as response,
    ):
        response.raise_for_status()
        data = await response.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def _extract_tarball(tarball: Path, dest: Path) -> None:
    """Extract the tarball into ``dest``, stripping the leading
    ``deep-swe-<ref>/`` component GitHub archives always carry.

    Extraction lands in a ``<dest>.extracting`` sibling first and is renamed
    into place, so a half-extracted cache entry is never mistaken for a
    complete one.
    """
    staging = dest.parent / f"{dest.name}.extracting"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        with tarfile.open(tarball) as archive:
            for member in archive.getmembers():
                if not member.isfile() and not member.isdir():
                    continue
                parts = Path(member.name).parts
                if len(parts) < 2:
                    continue  # the leading "deep-swe-<ref>/" component itself
                target = staging.joinpath(*parts[1:])
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        archive.extractfile(member) as src,
                        open(target, "wb") as out,
                    ):
                        out.write(src.read())
        dest.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(dest)
    finally:
        tarball.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)


async def provision_tasks() -> Path:
    """Resolve the deep-swe tasks directory, provisioning it if needed.

    Returns the directory that directly contains the task folders. Uses
    ``AIPERF_ACCURACY_DEEPSWE_TASKS_DIR`` when set; otherwise downloads and
    extracts the pinned upstream tarball into the dataset cache (reusing an
    existing cache entry).
    """
    env_dir = Environment.ACCURACY.DEEPSWE_TASKS_DIR
    if env_dir:
        root = Path(env_dir).expanduser().resolve()
        tasks_dir = _find_tasks_dir(root)
        _logger.info(f"Using pre-provisioned DeepSWE tasks at '{tasks_dir}'")
        return tasks_dir

    ref = Environment.ACCURACY.DEEPSWE_PINNED_REF
    cached = DEEPSWE_CACHE_DIR / ref
    tasks_dir = cached / "tasks"
    if _is_valid_tasks_dir(tasks_dir):
        _logger.info(f"Using cached DeepSWE tasks at '{tasks_dir}' (ref {ref})")
        return tasks_dir

    # A leftover invalid cache entry (e.g. an interrupted extraction) is our
    # own cache dir; discard it before re-provisioning.
    shutil.rmtree(cached, ignore_errors=True)

    url = _tarball_url(ref)
    _logger.info(f"Downloading DeepSWE task set from {url}")
    tarball = cached.parent / f"{ref}.tar.gz"
    try:
        await _download_tarball(url, tarball)
    except (TimeoutError, aiohttp.ClientError) as e:
        raise ConfigurationError(
            f"Failed to download the DeepSWE task set from {url}: {e!r}. "
            f"Provide a pre-provisioned tasks directory via "
            f"AIPERF_ACCURACY_DEEPSWE_TASKS_DIR for hosts without GitHub "
            f"access."
        ) from e
    await asyncio.to_thread(_extract_tarball, tarball, cached)
    validate_tasks_dir(tasks_dir)
    return tasks_dir


def load_task_info(task_dir: Path) -> dict[str, Any]:
    """Parse a task's ``task.toml`` into the runner-facing info dict.

    Keeps only the fields the agentic runner needs; everything else stays in
    the manifest on disk.
    """
    with open(task_dir / TASK_MANIFEST_FILENAME, "rb") as f:
        cfg = tomllib.load(f)
    metadata = cfg.get("metadata", {})
    environment = cfg.get("environment", {})
    verifier = cfg.get("verifier", {})
    agent = cfg.get("agent", {})
    return {
        "task_id": task_dir.name,
        "task_dir": str(task_dir.resolve()),
        "image": environment.get("docker_image"),
        "base_commit": metadata.get("base_commit_hash"),
        "repository_url": metadata.get("repository_url"),
        "language": metadata.get("language"),
        "agent_timeout_s": agent.get("timeout_sec"),
        "verifier_timeout_s": verifier.get("timeout_sec"),
    }


def resolve_selected_tasks(
    tasks_dir: Path, requested: list[str] | None
) -> list[tuple[str, Path]]:
    """Resolve the requested task selection to (task_id, dir) pairs.

    Args:
        tasks_dir: Directory containing the task folders.
        requested: Task ids from ``--accuracy-tasks``; None selects all.

    Raises:
        ConfigurationError: If any requested task id is unknown, listing the
            available ids (the MMLU subjects pattern).
    """
    available = list_task_ids(tasks_dir)
    if requested is None:
        return [(task_id, tasks_dir / task_id) for task_id in available]
    unknown = [t for t in requested if t not in available]
    if unknown:
        raise ConfigurationError(
            f"Unknown DeepSWE task(s): {', '.join(unknown)}. "
            f"Available tasks: {', '.join(available)}."
        )
    return [(task_id, tasks_dir / task_id) for task_id in requested]


class DeepSWEBenchmark(AIPerfLoggerMixin):
    """DeepSWE v1.1 agentic benchmark loader.

    Discovers and provisions the Harbor-format task directories and emits one
    ``BenchmarkProblem`` per task whose ``ground_truth`` is the JSON runner
    payload (task dir, image, base commit, timeouts). The agent loop and
    program-based verifier are executed by
    :class:`aiperf.accuracy.deepswe_runner.DeepSWEAgenticRunner` via Pier —
    the problems are never dispatched as worker requests.
    """

    def __init__(self, run: BenchmarkRun, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run = run

    async def load_problems(
        self, tasks: list[str] | None, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        """Load DeepSWE problems for the selected tasks.

        Args:
            tasks: Task ids to evaluate (None selects all tasks).
            n_shots: Ignored — DeepSWE is agentic; the agent explores the
                repository interactively rather than from few-shot examples.
            enable_cot: Ignored — the agent decides how to reason; there is
                no single response to prepend CoT instructions to.

        Returns:
            One ``BenchmarkProblem`` per selected task, in sorted task-id
            order.
        """
        tasks_dir = await provision_tasks()
        selected = resolve_selected_tasks(tasks_dir, tasks)
        self.info(f"Selected {len(selected)} DeepSWE task(s) from '{tasks_dir}'")
        return await asyncio.to_thread(self._build_problems, selected)

    def _build_problems(
        self, selected: list[tuple[str, Path]]
    ) -> list[BenchmarkProblem]:
        problems: list[BenchmarkProblem] = []
        for task_id, task_dir in selected:
            info = load_task_info(task_dir)
            instruction_path = task_dir / INSTRUCTION_FILENAME
            try:
                instruction = instruction_path.read_text(encoding="utf-8")
            except OSError as e:
                raise ConfigurationError(
                    f"DeepSWE task '{task_id}' is missing its instruction "
                    f"file '{instruction_path}': {e!r}"
                ) from e
            problems.append(
                BenchmarkProblem(
                    # The canonical DeepSWE prompt: pier hands this
                    # instruction to the agent verbatim.
                    prompt=instruction,
                    # JSON payload parsed by DeepSWEAgenticRunner — the
                    # same ground-truth-JSON pattern the code_execution
                    # grader uses for LiveCodeBench.
                    ground_truth=orjson.dumps(info).decode(),
                    task=task_id,
                    metadata={"agentic": True},
                )
            )
        return problems
