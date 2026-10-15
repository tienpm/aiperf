# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agentic-accuracy execution path for aiperf.cli_runner.

One ``_run_agentic_accuracy_benchmark`` call runs an agentic accuracy
benchmark (e.g. DeepSWE) in-process: the resolver chain prepares the run, then
``run_agentic_accuracy_benchmark`` executes the trials (Pier subprocesses) and
assembles the standard accuracy artifacts. No service topology is spawned —
agentic benchmarks produce no worker requests, so the 10-service pipeline
would only idle.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from aiperf.cli_utils import raise_startup_error_and_exit
from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.exceptions import ConfigurationError as AiperfConfigurationError
from aiperf.config.loader.errors import ConfigurationError

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun

_logger = AIPerfLogger(__name__)


def _run_agentic_accuracy_benchmark(run: BenchmarkRun) -> None:
    """Run an agentic accuracy benchmark (DeepSWE) in-process.

    Mirrors ``_run_single_benchmark``'s resolver-chain error handling, then
    executes ``run_agentic_accuracy_benchmark`` on a fresh event loop. Pier
    subprocess failures, per-task timeouts, and verifier failures grade the
    affected task as failed; only configuration/infrastructure setup errors
    exit non-zero.
    """
    from aiperf.config.resolution.resolvers import build_default_resolver_chain

    try:
        chain = build_default_resolver_chain()
        chain.resolve_all(run)
    except Exception as e:  # resolver chain wraps every user-input error type
        # ``logger.error`` over ``.exception``: user-input errors carry their
        # own context; tracebacks trip chaos-harness crash heuristics.
        _logger.error(f"Configuration resolution failed: {e}")
        raise_startup_error_and_exit(
            f"Configuration resolution failed: {e}",
            title="Configuration Error",
        )

    from aiperf.accuracy.deepswe_runner import run_agentic_accuracy_benchmark

    try:
        asyncio.run(run_agentic_accuracy_benchmark(run))
    except AiperfConfigurationError as e:
        # Keep agentic setup failures on the quiet ConfigurationError panel
        # path (pier/docker/tasks provisioning), never a raw traceback.
        raise ConfigurationError(str(e)) from e
