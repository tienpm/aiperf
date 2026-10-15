# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSWE agentic runner tests: reward parsing, trial-result discovery, and
end-to-end trial execution against a fake pier executable (the fake-worker-
script pattern from test_codegen_worker_client.py)."""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import pytest

from aiperf.accuracy.deepswe_runner import (
    GRADER_NAME,
    DeepSWEAgenticRunner,
    _find_trial_result,
    _openai_base_url,
    _outcome_from_result,
    _parse_rewards,
    resolve_pier_bin,
)
from aiperf.accuracy.models import BenchmarkProblem
from aiperf.common.environment import Environment
from aiperf.common.exceptions import ConfigurationError
from tests.unit.conftest import make_benchmark_run

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

FAKE_PIER_PASS = """\
#!/usr/bin/env bash
# Fake pier: writes a passing trial result.json under --jobs-dir/--job-name.
jobs_dir=""; job_name=""; prev=""
for a in "$@"; do
  case "$prev" in
    --jobs-dir) jobs_dir="$a" ;;
    --job-name) job_name="$a" ;;
  esac
  prev="$a"
done
trial="$jobs_dir/$job_name/trial-1"
mkdir -p "$trial/verifier"
cat > "$trial/result.json" <<'EOF'
{
  "task_name": "fake",
  "verifier_result": {"rewards": {"reward": 1, "f2p_total": 3, "f2p_passed": 3,
    "p2p_total": 5, "p2p_passed": 5, "f2p": 1.0, "p2p": 1.0, "partial": 1.0}},
  "exception_info": null
}
EOF
exit 0
"""

FAKE_PIER_FAIL_NO_RESULT = """\
#!/usr/bin/env bash
echo "pier: infrastructure exploded" >&2
exit 3
"""

FAKE_PIER_SLEEP = """\
#!/usr/bin/env bash
sleep 60
"""


def write_executable(path: Path, body: str) -> str:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


def make_problem(task_id: str) -> BenchmarkProblem:
    return BenchmarkProblem(
        prompt=f"instruction for {task_id}",
        ground_truth=json.dumps({"task_id": task_id, "task_dir": "/tmp/whatever"}),
        task=task_id,
        metadata={"agentic": True},
    )


def make_run(tmp_path: Path):
    return make_benchmark_run(
        model_names=["test-model"],
        accuracy={"benchmark": "deepswe"},
        extra={"artifacts": {"dir": str(tmp_path / "artifacts")}},
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestOpenaiBaseUrl:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("http://h:30000", "http://h:30000/v1"),
            ("http://h:30000/", "http://h:30000/v1"),
            ("http://h:30000/v1", "http://h:30000/v1"),
            ("http://h:30000/v1/", "http://h:30000/v1"),
            ("http://h:30000/v1/chat/completions", "http://h:30000/v1"),
        ],
    )
    def test_normalization(self, url: str, expected: str) -> None:
        assert _openai_base_url(url) == expected


class TestParseRewards:
    def test_full_pass(self) -> None:
        passed, confidence, summary = _parse_rewards(
            {
                "reward": 1,
                "f2p_total": 3,
                "f2p_passed": 3,
                "p2p_total": 5,
                "p2p_passed": 5,
                "f2p": 1.0,
                "p2p": 1.0,
                "partial": 1.0,
            }
        )
        assert passed is True
        assert confidence == 1.0
        assert "reward=1" in summary and "f2p=3/3" in summary and "p2p=5/5" in summary

    def test_partial_progress_is_failure(self) -> None:
        passed, confidence, _ = _parse_rewards(
            {"reward": 0, "f2p_total": 4, "f2p_passed": 2, "partial": 0.5}
        )
        assert passed is False
        assert confidence == 0.5

    def test_apply_failed_flagged(self) -> None:
        passed, _, summary = _parse_rewards({"reward": 0, "apply_failed": 1})
        assert passed is False
        assert "did not apply" in summary

    def test_empty_rewards(self) -> None:
        assert _parse_rewards({}) == (False, 0.0, "reward=0")


class TestOutcomeFromResult:
    def test_rewards_present(self, tmp_path: Path) -> None:
        outcome = _outcome_from_result(
            {
                "verifier_result": {"rewards": {"reward": 1, "partial": 1.0}},
                "exception_info": None,
            },
            trial_dir=tmp_path,
        )
        assert outcome.passed is True
        assert outcome.confidence == 1.0

    def test_exception_info(self, tmp_path: Path) -> None:
        outcome = _outcome_from_result(
            {
                "verifier_result": None,
                "exception_info": {
                    "exception_type": "TimeoutError",
                    "exception_message": "agent timed out",
                },
            },
            trial_dir=tmp_path,
        )
        assert outcome.passed is False
        assert "TimeoutError" in outcome.explanation
        assert "agent timed out" in outcome.explanation

    def test_neither_rewards_nor_exception(self, tmp_path: Path) -> None:
        outcome = _outcome_from_result(
            {"verifier_result": {}, "exception_info": None}, trial_dir=tmp_path
        )
        assert outcome.passed is False
        assert "no rewards" in outcome.explanation


class TestFindTrialResult:
    def test_newest_result_wins(self, tmp_path: Path) -> None:
        workspace = tmp_path / "jobs"
        old = workspace / "task-a" / "trial-1"
        new = workspace / "task-a" / "trial-2"
        for trial, payload in ((old, {"n": 1}), (new, {"n": 2})):
            trial.mkdir(parents=True)
            (trial / "result.json").write_text(json.dumps(payload))
        os.utime(old / "result.json", (1000, 1000))
        time.sleep(0.01)

        trial_dir, result = _find_trial_result(workspace, "task-a")

        assert trial_dir == new
        assert result == {"n": 2}

    def test_missing_job_dir(self, tmp_path: Path) -> None:
        assert _find_trial_result(tmp_path, "nope") == (None, None)

    def test_corrupt_result_json(self, tmp_path: Path) -> None:
        trial = tmp_path / "jobs" / "task-a" / "trial-1"
        trial.mkdir(parents=True)
        (trial / "result.json").write_text("{not json")
        _, result = _find_trial_result(tmp_path / "jobs", "task-a")
        assert result is None


class TestResolvePierBin:
    def test_missing_pier_raises_with_hint(self, monkeypatch) -> None:
        monkeypatch.setattr(
            Environment.ACCURACY, "DEEPSWE_PIER_BIN", "not-a-real-pier-binary"
        )
        with pytest.raises(ConfigurationError, match="datacurve-pier"):
            resolve_pier_bin()

    def test_resolves_explicit_path(self, tmp_path: Path, monkeypatch) -> None:
        pier = tmp_path / "pier"
        pier.write_text("#!/bin/sh\n")
        pier.chmod(0o755)
        monkeypatch.setattr(Environment.ACCURACY, "DEEPSWE_PIER_BIN", str(pier))
        assert resolve_pier_bin() == str(pier)


# ---------------------------------------------------------------------------
# Trial execution against a fake pier
# ---------------------------------------------------------------------------


class TestRunnerTrials:
    async def test_passing_trial_produces_passed_record_and_artifacts(
        self, tmp_path: Path
    ) -> None:
        pier_bin = write_executable(tmp_path / "pier", FAKE_PIER_PASS)
        run = make_run(tmp_path)
        runner = DeepSWEAgenticRunner(run=run, pier_bin=pier_bin)

        summary = await runner.run_trials([make_problem("task-a")])

        assert summary is not None
        assert summary.total_evaluated == 1
        assert summary.total_passed == 1
        assert summary.grader_name == GRADER_NAME

        # The standard accuracy artifacts land in the artifact dir.
        artifact_dir = Path(run.cfg.artifacts.dir)
        assert (artifact_dir / "accuracy_results.csv").is_file()
        jsonl = artifact_dir / "accuracy_export.jsonl"
        assert jsonl.is_file()
        record = json.loads(jsonl.read_text().splitlines()[0])
        assert record["task"] == "task-a"
        assert record["grader_name"] == GRADER_NAME
        assert record["passed"] is True

    async def test_pier_crash_without_result_grades_failed(
        self, tmp_path: Path
    ) -> None:
        pier_bin = write_executable(tmp_path / "pier", FAKE_PIER_FAIL_NO_RESULT)
        run = make_run(tmp_path)
        runner = DeepSWEAgenticRunner(run=run, pier_bin=pier_bin)

        summary = await runner.run_trials([make_problem("task-a")])

        assert summary is not None
        assert summary.total_passed == 0
        jsonl = artifact_jsonl(run)
        assert jsonl[0]["passed"] is False
        assert "infrastructure exploded" in jsonl[0]["explanation"]

    async def test_hard_timeout_kills_pier_and_grades_failed(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        pier_bin = write_executable(tmp_path / "pier", FAKE_PIER_SLEEP)
        monkeypatch.setattr(Environment.ACCURACY, "DEEPSWE_TASK_TIMEOUT_S", 0.5)
        run = make_run(tmp_path)
        runner = DeepSWEAgenticRunner(run=run, pier_bin=pier_bin)

        start = time.monotonic()
        summary = await runner.run_trials([make_problem("task-a")])
        elapsed = time.monotonic() - start

        assert elapsed < 30  # the 60s sleeper was killed, not awaited
        assert summary is not None
        assert summary.total_passed == 0
        assert "timeout" in artifact_jsonl(run)[0]["explanation"].lower()

    async def test_concurrency_one_serializes_trials(self, tmp_path: Path) -> None:
        # Each fake pier appends start/end timestamps; with concurrency 1 the
        # second trial must not start before the first ends.
        body = FAKE_PIER_PASS.replace(
            "exit 0",
            'echo "start $(date +%s.%N)" >> "$jobs_dir/order.log"\n'
            "sleep 0.3\n"
            'echo "end $(date +%s.%N)" >> "$jobs_dir/order.log"\n'
            "exit 0",
        )
        pier_bin = write_executable(tmp_path / "pier", body)
        run = make_run(tmp_path)
        runner = DeepSWEAgenticRunner(run=run, pier_bin=pier_bin)

        await runner.run_trials([make_problem("task-a"), make_problem("task-b")])

        order = (Path(run.artifact_dir) / "deepswe" / "order.log").read_text().split()
        # Each line emits two tokens (event + timestamp): the sequence must be
        # start,end,start,end — never start,start,end,end.
        assert [order[i] for i in (0, 2, 4, 6)] == ["start", "end", "start", "end"]

    async def test_no_records_reports_none(self, tmp_path: Path) -> None:
        pier_bin = write_executable(tmp_path / "pier", FAKE_PIER_PASS)
        run = make_run(tmp_path)
        runner = DeepSWEAgenticRunner(run=run, pier_bin=pier_bin)

        assert await runner.run_trials([]) is None


def artifact_jsonl(run) -> list[dict]:
    path = Path(run.cfg.artifacts.dir) / "accuracy_export.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


class TestSpawnCommand:
    async def test_pier_command_and_env_wiring(self, tmp_path: Path) -> None:
        """The pier invocation must carry the task dir, agent, openai/<model>,
        workspace, and the endpoint env vars the agent needs."""
        captured: dict = {}

        async def fake_spawn(
            args: list[str], timeout_s: float | None
        ) -> tuple[int, str, bool]:
            captured["args"] = args
            captured["timeout_s"] = timeout_s
            return 0, "", False

        run = make_run(tmp_path)
        runner = DeepSWEAgenticRunner(run=run, pier_bin="/fake/pier")
        runner._spawn = fake_spawn  # noqa: SLF001 - unit under test

        # A real problem payload so task_dir/job-name resolution is exercised.
        task_dir = tmp_path / "tasks" / "task-a"
        task_dir.mkdir(parents=True)
        problem = BenchmarkProblem(
            prompt="instr",
            ground_truth=json.dumps({"task_id": "task-a", "task_dir": str(task_dir)}),
            task="task-a",
            metadata={"agentic": True},
        )

        await runner._execute_trial(  # noqa: SLF001 - unit under test
            problem=problem,
            workspace=Path(run.artifact_dir) / "deepswe",
            model_name="test-model",
            base_url="http://h:30000/v1",
            api_key="secret",
        )

        args = captured["args"]
        assert args[0] == "/fake/pier"
        assert args[args.index("--path") + 1] == str(task_dir)
        assert args[args.index("--agent") + 1] == "mini-swe-agent"
        assert args[args.index("--model") + 1] == "openai/test-model"
        assert args[args.index("--job-name") + 1] == "task-a"
        assert "--quiet" in args and "--yes" in args
        env_args = [args[i + 1] for i, a in enumerate(args) if a == "--agent-env"]
        assert "OPENAI_BASE_URL=http://h:30000/v1" in env_args
        assert "OPENAI_API_BASE=http://h:30000/v1" in env_args
        assert "OPENAI_API_KEY=secret" in env_args
        assert captured["timeout_s"] is None
