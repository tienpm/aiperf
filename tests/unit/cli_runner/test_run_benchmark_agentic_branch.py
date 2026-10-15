# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""run_benchmark's agentic-accuracy dispatch: agentic benchmarks (``agentic:
true`` plugins.yaml metadata, e.g. deepswe) bypass the service topology and
run in-process via ``_run_agentic_accuracy_benchmark``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import aiperf.cli_runner as cli_runner
from aiperf.cli_runner import _is_agentic_accuracy_run, run_benchmark


def _plan(
    *,
    accuracy: object,
    agentic: bool,
    single_run: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        use_adaptive=False,
        trials=1,
        is_single_run=single_run,
        variations=[SimpleNamespace()],
        variables={},
        configs=[
            SimpleNamespace(
                accuracy=accuracy,
                artifacts=SimpleNamespace(auto_plot=False),
            )
        ],
    )


def _acc(enabled: bool, benchmark: str = "deepswe"):
    return SimpleNamespace(enabled=enabled, benchmark=benchmark)


def _patch_registry(monkeypatch, *, agentic: bool) -> None:
    monkeypatch.setattr(
        "aiperf.plugin.plugins.get_metadata",
        lambda _type, _name: {"agentic": agentic},
    )


class TestIsAgenticAccuracyRun:
    def test_agentic_benchmark_detected(self, monkeypatch) -> None:
        _patch_registry(monkeypatch, agentic=True)
        assert _is_agentic_accuracy_run(_plan(accuracy=_acc(True), agentic=True))

    def test_non_agentic_benchmark_not_detected(self, monkeypatch) -> None:
        _patch_registry(monkeypatch, agentic=False)
        assert not _is_agentic_accuracy_run(_plan(accuracy=_acc(True), agentic=False))

    def test_accuracy_disabled_not_detected(self, monkeypatch) -> None:
        _patch_registry(monkeypatch, agentic=True)
        assert not _is_agentic_accuracy_run(_plan(accuracy=_acc(False), agentic=True))

    def test_missing_accuracy_config_not_detected(self, monkeypatch) -> None:
        _patch_registry(monkeypatch, agentic=True)
        plan = SimpleNamespace(configs=[SimpleNamespace(accuracy=None)])
        assert not _is_agentic_accuracy_run(plan)


class TestRunBenchmarkAgenticDispatch:
    @pytest.fixture
    def stubs(self, monkeypatch):
        """Neutralize every preflight and collaborator run_benchmark touches."""
        for name in (
            "_preflight_artifact_dir",
            "_preflight_accuracy_deps",
            "_preflight_fd_limit",
            "_preflight_dataset_access",
            "_preflight_endpoint_ready",
            "_preflight_dataset_materialize",
        ):
            monkeypatch.setattr(cli_runner, name, lambda *a, **kw: None)
        monkeypatch.setattr(
            "aiperf.common.bootstrap.register_sigusr1_faulthandler",
            lambda: None,
        )
        monkeypatch.setattr(
            "aiperf.orchestrator.orchestrator.resolve_run_seed",
            lambda plan, variation: 7,
        )

    def test_agentic_single_run_dispatches_in_process(self, monkeypatch, stubs) -> None:
        _patch_registry(monkeypatch, agentic=True)
        calls: list = []
        monkeypatch.setattr(
            cli_runner,
            "_make_benchmark_run",
            lambda config, **kwargs: SimpleNamespace(
                cfg=config, artifact_dir="/tmp/fake"
            ),
        )
        monkeypatch.setattr(
            cli_runner,
            "_run_agentic_accuracy_benchmark",
            lambda run: calls.append(run),
        )

        run_benchmark(_plan(accuracy=_acc(True), agentic=True))

        assert len(calls) == 1

    def test_agentic_multi_run_rejected(self, monkeypatch, stubs) -> None:
        _patch_registry(monkeypatch, agentic=True)
        monkeypatch.setattr(
            cli_runner,
            "_run_agentic_accuracy_benchmark",
            lambda run: None,
        )

        with pytest.raises(ValueError, match="single runs"):
            run_benchmark(_plan(accuracy=_acc(True), agentic=True, single_run=False))
