# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSWE agentic accuracy runner — executes DeepSWE trials via the Pier CLI.

DeepSWE tasks are agentic: the model under test interactively explores the
repository and runs commands inside the task's Docker sandbox, so the trials
cannot be expressed as AIPerf worker requests. This runner delegates each task
to Pier (``pier run --agent mini-swe-agent``), which runs mini-swe-agent inside
the sandbox against the configured OpenAI-compatible endpoint, then applies the
agent's committed work as a patch in a pristine verifier container and writes
``result.json`` (verifier rewards: binary ``reward`` plus ``f2p``/``p2p``/
``partial`` fractions).

Each completed trial becomes one :class:`AccuracyRecordsData` on the fly, and
the same reporting stack a normal accuracy run uses is assembled in-process —
:class:`AccuracyAccumulator` (summary), :class:`AccuracyJSONLWriter`
(``accuracy_export.jsonl``), :class:`AccuracyConsoleExporter` (Rich table),
and :class:`AccuracyDataExporter` (``accuracy_results.csv``) — so a DeepSWE run
produces byte-familiar artifacts without the 10-service topology.

The runner is dispatched by ``run_benchmark`` when the selected accuracy
benchmark's plugins.yaml metadata declares ``agentic: true``
(see :class:`aiperf.accuracy.benchmarks.deepswe.DeepSWEBenchmark`).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson
from rich.console import Console

from aiperf.accuracy.accumulator import AccuracyAccumulator
from aiperf.accuracy.accuracy_console_exporter import AccuracyConsoleExporter
from aiperf.accuracy.accuracy_data_exporter import AccuracyDataExporter
from aiperf.accuracy.benchmark_loader import load_benchmark_problems
from aiperf.accuracy.jsonl_writer import AccuracyJSONLWriter
from aiperf.accuracy.models import (
    AccuracyRecordsData,
    AccuracySummary,
    BenchmarkProblem,
)
from aiperf.common.constants import IS_WINDOWS
from aiperf.common.enums import CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.exceptions import ConfigurationError
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models.record_models import ProfileResults
from aiperf.exporters.exporter_config import ExporterConfig

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

# The "grader" for DeepSWE is the task's program-based verifier (tests/test.sh
# -> grader.py -> reward.json), executed by Pier inside the task's verifier
# container — recorded under this name in accuracy records and summaries.
GRADER_NAME = "deepswe_verifier"

# Pier agent that runs the model under test inside the task sandbox.
AGENT_NAME = "mini-swe-agent"

# Workspace (pier ``--jobs-dir``) under the run's artifact directory; one pier
# job per task, named by task id: <workspace>/<task_id>/<trial>/result.json.
WORKSPACE_DIRNAME = "deepswe"

# Tail of a trial's pier output kept for failure explanations.
OUTPUT_TAIL_CHARS = 4096


def resolve_pier_bin() -> str:
    """Resolve the Pier CLI executable path.

    Raises:
        ConfigurationError: If the pier binary cannot be found, with the
            install hint (``uv tool install datacurve-pier``).
    """
    pier_bin = Environment.ACCURACY.DEEPSWE_PIER_BIN
    resolved = shutil.which(pier_bin)
    if resolved is None:
        raise ConfigurationError(
            f"Pier CLI '{pier_bin}' not found. The deepswe benchmark runs "
            f"its agentic trials via Pier; install it with "
            f"'uv tool install datacurve-pier' (or point "
            f"AIPERF_ACCURACY_DEEPSWE_PIER_BIN at the executable)."
        )
    return resolved


def _openai_base_url(url: str) -> str:
    """Normalize an endpoint URL to the OpenAI-style base URL mini-swe-agent's
    LiteLLM client expects (``.../v1``, never a full completions path).
    """
    base = url.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _primary_model_name(run: BenchmarkRun) -> str:
    """The primary model name from the run config (pier needs exactly one)."""
    items = run.cfg.models.items
    if not items:
        raise ConfigurationError(
            "The deepswe benchmark requires a model name (--model) to send "
            "the agent's LLM requests to the endpoint."
        )
    return items[0].name


def _endpoint_url(run: BenchmarkRun) -> str:
    urls = run.cfg.endpoint.urls
    if not urls:
        raise ConfigurationError(
            "The deepswe benchmark requires an endpoint URL (--url) for the "
            "agent's LLM requests."
        )
    return urls[0]


def _parse_rewards(rewards: dict[str, Any]) -> tuple[bool, float, str]:
    """Map a verifier rewards dict to (passed, confidence, explanation).

    DeepSWE's grader writes ``reward`` (binary 0/1: every fail-to-pass test
    passes and no pass-to-pass test regresses), ``f2p``/``p2p``/``partial``
    fractions, raw ``f2p_*``/``p2p_*`` counts, and ``apply_failed`` when the
    agent's patch did not apply.
    """
    reward = float(rewards.get("reward", 0.0) or 0.0)
    passed = reward >= 1.0
    confidence = _clamp01(
        float(rewards.get("partial", rewards.get("f2p", reward)) or 0.0)
    )
    parts = [f"reward={reward:g}"]
    if "f2p_passed" in rewards or "f2p_total" in rewards:
        parts.append(
            f"f2p={rewards.get('f2p_passed', 0)}/{rewards.get('f2p_total', 0)}"
        )
    if "p2p_passed" in rewards or "p2p_total" in rewards:
        parts.append(
            f"p2p={rewards.get('p2p_passed', 0)}/{rewards.get('p2p_total', 0)}"
        )
    if rewards.get("apply_failed"):
        parts.append("model patch did not apply")
    return passed, confidence, ", ".join(parts)


class DeepSWEAgenticRunner(AIPerfLoggerMixin):
    """Runs DeepSWE tasks as Pier trials and ingests the verifier rewards.

    One pier invocation per task (bounded by a semaphore — each trial runs a
    Docker sandbox plus a verifier container), one
    :class:`AccuracyRecordsData` per completed trial, then the standard
    accuracy reporting stack in-process.
    """

    def __init__(self, run: BenchmarkRun, pier_bin: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run = run
        self._pier_bin = pier_bin
        self._console = Console()
        # The controller's results console enforces a minimum width so the
        # accuracy table renders readably; match it for this headless path.
        if self._console.width < 100:
            self._console.width = 100
        self._live_procs: set[asyncio.subprocess.Process] = set()

    async def run_trials(
        self, problems: Sequence[BenchmarkProblem]
    ) -> AccuracySummary | None:
        """Execute every problem's trial and report the accuracy results.

        A pier crash, per-task timeout, or missing verifier result grades the
        task as failed (never aborts the run); Ctrl+C kills every live pier
        process group.
        """
        model_name = _primary_model_name(self.run)
        base_url = _openai_base_url(_endpoint_url(self.run))
        # mini-swe-agent refuses to start without an API key env var even
        # when the endpoint ignores auth, so fall back to a placeholder.
        api_key = self.run.cfg.endpoint.api_key or "aiperf-dummy-key"

        workspace = self.run.artifact_dir / WORKSPACE_DIRNAME
        workspace.mkdir(parents=True, exist_ok=True)
        concurrency = Environment.ACCURACY.DEEPSWE_CONCURRENCY
        semaphore = asyncio.Semaphore(concurrency)
        self.info(
            f"Running {len(problems)} DeepSWE trial(s) via '{self._pier_bin}' "
            f"with agent '{AGENT_NAME}', model 'openai/{model_name}', "
            f"endpoint '{base_url}', concurrency {concurrency}"
        )

        start_ns = time.perf_counter_ns()
        try:
            records = await asyncio.gather(
                *(
                    self._run_trial(
                        session_num=i,
                        problem=problem,
                        workspace=workspace,
                        model_name=model_name,
                        base_url=base_url,
                        api_key=api_key,
                        semaphore=semaphore,
                    )
                    for i, problem in enumerate(problems)
                )
            )
        except BaseException:
            # Cancellation (Ctrl+C) or an unexpected error: reap every live
            # pier process group so no sandbox outlives the run.
            await self._kill_live_procs()
            raise
        end_ns = time.perf_counter_ns()

        summary = await self._report(list(records), start_ns=start_ns, end_ns=end_ns)
        return summary

    async def _run_trial(
        self,
        session_num: int,
        problem: BenchmarkProblem,
        *,
        workspace: Path,
        model_name: str,
        base_url: str,
        api_key: str,
        semaphore: asyncio.Semaphore,
    ) -> AccuracyRecordsData:
        task_id = problem.task
        self._console.print(f"[cyan]deepswe[/cyan] {task_id}: running...")
        async with semaphore:
            outcome = await self._execute_trial(
                problem=problem,
                workspace=workspace,
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
            )

        status = "[green]PASSED[/green]" if outcome.passed else "[red]FAILED[/red]"
        self._console.print(
            f"[cyan]deepswe[/cyan] {task_id}: {status} ({outcome.actual})"
        )
        return AccuracyRecordsData(
            session_num=session_num,
            conversation_id=task_id,
            worker_id="deepswe_runner",
            benchmark_phase=CreditPhase.PROFILING,
            timestamp_ns=time.perf_counter_ns(),
            task=task_id,
            grader_name=GRADER_NAME,
            passed=outcome.passed,
            unparsed=False,
            confidence=outcome.confidence,
            expected="verifier pass (all f2p tests pass, no p2p regressions)",
            actual=outcome.actual,
            explanation=outcome.explanation,
            model_output=f"trial: {outcome.trial_dir}",
        )

    async def _execute_trial(
        self,
        *,
        problem: BenchmarkProblem,
        workspace: Path,
        model_name: str,
        base_url: str,
        api_key: str,
    ) -> _TrialOutcome:
        """Run one pier trial and resolve its outcome.

        The ``ground_truth`` JSON payload (built by ``DeepSWEBenchmark``)
        carries the task dir and the task.toml timeouts.
        """
        try:
            info = orjson.loads(problem.ground_truth)
        except orjson.JSONDecodeError as e:
            return _TrialOutcome.failed(
                f"invalid ground-truth payload: {e!r}", trial_dir=str(workspace)
            )
        task_dir = Path(info["task_dir"])
        task_id = info["task_id"]

        args = [
            self._pier_bin,
            "run",
            "--path",
            str(task_dir),
            "--agent",
            AGENT_NAME,
            "--model",
            f"openai/{model_name}",
            "--jobs-dir",
            str(workspace),
            "--job-name",
            task_id,
            "--n-concurrent",
            "1",
            "--n-attempts",
            "1",
            "--max-retries",
            "0",
            "--quiet",
            "--yes",
            "--agent-env",
            f"OPENAI_BASE_URL={base_url}",
            "--agent-env",
            f"OPENAI_API_BASE={base_url}",
            "--agent-env",
            f"OPENAI_API_KEY={api_key}",
        ]

        # The optional hard wall-clock ceiling (DEEPSWE_TASK_TIMEOUT_S) bounds
        # the whole trial — build, agent run, verification — beyond Pier's own
        # per-phase timeouts from task.toml (which govern when unset).
        timeout_s = Environment.ACCURACY.DEEPSWE_TASK_TIMEOUT_S
        exit_code, output, timed_out = await self._spawn(args, timeout_s=timeout_s)

        trial_dir, result = _find_trial_result(workspace, task_id)
        if result is None:
            if timed_out:
                detail = output[-OUTPUT_TAIL_CHARS:].strip()
                return _TrialOutcome.failed(
                    f"pier trial exceeded the hard timeout of {timeout_s}s and "
                    f"was killed" + (f"; output tail: {detail}" if detail else ""),
                    trial_dir=str(trial_dir or workspace / task_id),
                )
            detail = output[-OUTPUT_TAIL_CHARS:].strip() or f"exit code {exit_code}"
            return _TrialOutcome.failed(
                f"pier produced no trial result (exit code {exit_code}); "
                f"output tail: {detail}",
                trial_dir=str(trial_dir or workspace / task_id),
            )
        return _outcome_from_result(result, trial_dir=trial_dir)

    async def _spawn(
        self, args: list[str], timeout_s: float | None
    ) -> tuple[int, str, bool]:
        """Spawn pier in its own process group and collect its output.

        Returns (exit_code, combined_output, timed_out); on timeout the
        process group is killed, the exit code reports -9, and the output
        collected so far is returned.
        """
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=not IS_WINDOWS,
        )
        self._live_procs.add(proc)
        try:
            try:
                output_bytes, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_s
                )
                return (
                    proc.returncode or 0,
                    output_bytes.decode(errors="replace"),
                    False,
                )
            except TimeoutError:
                self.warning(
                    f"Pier trial exceeded the hard timeout of {timeout_s}s; "
                    f"killing process group"
                )
                _kill_process_group(proc)
                try:
                    output_bytes, _ = await proc.communicate()
                except Exception:
                    output_bytes = b""
                return -9, output_bytes.decode(errors="replace"), True
        finally:
            self._live_procs.discard(proc)

    async def _kill_live_procs(self) -> None:
        """Kill every live pier process group (cancellation path)."""
        for proc in list(self._live_procs):
            _kill_process_group(proc)
        for proc in list(self._live_procs):
            with contextlib.suppress(Exception):
                await proc.wait()

    async def _report(
        self, records: list[AccuracyRecordsData], *, start_ns: int, end_ns: int
    ) -> AccuracySummary | None:
        """Assemble the standard accuracy artifacts from the trial records.

        Reuses the normal accuracy run's components in-process: the
        accumulator for the summary, the JSONL writer for
        ``accuracy_export.jsonl``, and the console/CSV exporters for the Rich
        table and ``accuracy_results.csv``.
        """
        accumulator = AccuracyAccumulator(run=self.run)
        jsonl_writer = AccuracyJSONLWriter(run=self.run)
        # The writer's file handle opens in its @on_init lifecycle hook (the
        # same initialize_and_start the records manager performs) and closes
        # on stop; the accumulator is plain in-memory rollup.
        await jsonl_writer.initialize_and_start()
        try:
            for record in records:
                await accumulator.process_record(record)
                await jsonl_writer.process_record(record)
            await jsonl_writer.finalize()
        finally:
            await jsonl_writer.stop()

        summary = await accumulator.summarize()
        if summary is None:
            self.error("No DeepSWE trial records were produced")
            return None

        profile_results = ProfileResults(
            records=summary.to_metric_results(),
            completed=len(records),
            start_ns=start_ns,
            end_ns=end_ns,
        )
        exporter_config = ExporterConfig(
            results=profile_results,
            cfg=self.run.cfg,
            telemetry_results=None,
            run=self.run,
        )
        await AccuracyConsoleExporter(exporter_config).export(console=self._console)
        await AccuracyDataExporter(exporter_config).export()
        return summary


@dataclass(slots=True)
class _TrialOutcome:
    """Resolution of one DeepSWE trial (verifier reward or failure)."""

    passed: bool
    confidence: float
    actual: str
    explanation: str
    trial_dir: str

    @classmethod
    def failed(cls, reason: str, trial_dir: str) -> _TrialOutcome:
        return cls(
            passed=False,
            confidence=0.0,
            actual="no verifier result",
            explanation=reason,
            trial_dir=trial_dir,
        )


def _outcome_from_result(result: dict[str, Any], trial_dir: Path) -> _TrialOutcome:
    """Map a pier ``result.json`` to a trial outcome.

    Prefers the verifier rewards; falls back to the trial's exception info
    when verification never produced rewards.
    """
    verifier = result.get("verifier_result") or {}
    rewards = verifier.get("rewards")
    if rewards:
        passed, confidence, summary = _parse_rewards(rewards)
        return _TrialOutcome(
            passed=passed,
            confidence=confidence,
            actual=summary,
            explanation=f"verifier rewards: {summary}",
            trial_dir=str(trial_dir),
        )
    exception = result.get("exception_info") or {}
    if exception:
        exc_type = exception.get("exception_type", "Exception")
        exc_msg = exception.get("exception_message", "")
        return _TrialOutcome.failed(
            f"trial raised {exc_type}: {exc_msg}", trial_dir=str(trial_dir)
        )
    return _TrialOutcome.failed(
        "verifier produced no rewards and no exception was recorded",
        trial_dir=str(trial_dir),
    )


def _find_trial_result(
    workspace: Path, job_name: str
) -> tuple[Path | None, dict[str, Any] | None]:
    """Locate the newest trial ``result.json`` for a pier job.

    Pier lays out ``<jobs-dir>/<job-name>/<trial>/result.json``; the newest
    (highest mtime) wins so a re-run into the same job dir still resolves the
    latest attempt.
    """
    job_dir = workspace / job_name
    if not job_dir.is_dir():
        return None, None
    candidates = sorted(
        job_dir.glob("*/result.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        return None, None
    result_path = candidates[-1]
    try:
        return result_path.parent, orjson.loads(result_path.read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return result_path.parent, None


def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """Kill a pier process and (on POSIX) its whole process group."""
    if IS_WINDOWS:
        with contextlib.suppress(Exception):
            proc.kill()
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)


async def run_agentic_accuracy_benchmark(run: BenchmarkRun) -> AccuracySummary | None:
    """Run an agentic accuracy benchmark (DeepSWE) end to end.

    Resolves the pier CLI, loads the benchmark's problems (which provisions
    the task set), executes every trial, and produces the standard accuracy
    artifacts. Returns the summary, or None when no records were produced.
    """
    pier_bin = resolve_pier_bin()
    problems = await load_benchmark_problems(run)
    runner = DeepSWEAgenticRunner(run=run, pier_bin=pier_bin)
    return await runner.run_trials(problems)
