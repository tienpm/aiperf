# DeepSWE (deepswe) Runbook

[DeepSWE v1.1](https://github.com/datacurve-ai/deep-swe) measures coding
agents on 113 original, long-horizon software-engineering tasks drawn from
active open-source repositories (TypeScript, Go, Python, JavaScript, Rust).
Each task is a [Harbor-format](https://docs.harborframework.com/tasks/overview)
directory — `task.toml` (repo URL, base commit, prebuilt sandbox image,
timeouts, resource limits), `instruction.md`, `tests/` (the program-based
verifier), and a held-out `solution/` — graded by applying the agent's
committed work as a patch in a pristine verifier container. The verifier's
`reward.json` carries a binary `reward` (1 iff every fail-to-pass test passes
and no pass-to-pass test regresses) plus `f2p`/`p2p`/`partial` fractions.

AIPerf runs DeepSWE **agentically**: `--accuracy-benchmark deepswe` bypasses
the 10-service load-generation topology entirely and dispatches to the
in-process `DeepSWEAgenticRunner`
(`src/aiperf/accuracy/deepswe_runner.py`), which executes one
[Pier](https://github.com/datacurve-ai/pier) trial per task
(`pier run --agent mini-swe-agent`) and ingests the verifier rewards as
accuracy records.

## Quick start

```bash
# Prerequisites: Docker running, and the Pier CLI installed once per host:
uv tool install datacurve-pier

# One task against an OpenAI-compatible endpoint (must be on port 80/443 —
# see "Model and endpoint wiring" below):
uv run aiperf profile \
  --url http://<endpoint-host> \
  --model <model-name> \
  --accuracy-benchmark deepswe \
  --accuracy-tasks bandit-incremental-cache-control
```

Output artifacts (same as any accuracy run):

- Console: the per-task Rich table with Correct/Total/Unparsed/Accuracy and
  an OVERALL row.
- `<artifact_dir>/accuracy_results.csv` — per-task summary.
- `<artifact_dir>/accuracy_export.jsonl` — one record per trial
  (pass/fail, confidence, verifier reward summary, trial dir).
- `<artifact_dir>/deepswe/<task_id>/` — the raw Pier job: trial
  `result.json`, `verifier/reward.json`, `ctrf.json`, `test-stdout.txt`,
  `trial.log`, and the agent trajectory.

There are **no performance metrics**: the agent's LLM requests are issued by
mini-swe-agent inside the task sandbox (they do not flow through AIPerf's
workers), so the run reports accuracy only.

## Requirements

| Requirement | Why | Check |
|---|---|---|
| Docker daemon | Each task runs in its own sandbox container (image pinned in `task.toml`, e.g. `public.ecr.aws/d3j8x8q7/swe-bench-202605:<id>-v1.1`) plus a separate verifier container | `docker info` |
| Pier CLI (> 0.3.0) | Drives mini-swe-agent in the sandbox and the verifier | `uv tool install datacurve-pier` |
| Deep-swe task set | Task manifests, instructions, and tests | Auto-downloaded (see below) |
| Network egress from containers | The sandbox pulls the mini-swe-agent install (apt/uv/pypi) and calls the LLM endpoint; Pier's per-agent network allowlist covers both (the endpoint host is auto-allowlisted from `OPENAI_BASE_URL`) | First trial acts as the check |
| ~2 CPUs / 8 GiB free per trial | `task.toml` resource limits per sandbox and verifier | Mind `AIPERF_ACCURACY_DEEPSWE_CONCURRENCY` |

AIPerf checks the pier binary and the Docker daemon at preflight and exits
with a clean `ConfigurationError` panel when either is missing.

## Task-set provisioning

Provisioning resolves in this order:

1. `AIPERF_ACCURACY_DEEPSWE_TASKS_DIR` — a pre-provisioned local directory:
   either the extracted repo root (contains `tasks/`) or the tasks directory
   itself (contains the task folders). Use this on hosts without GitHub
   access (e.g. rsync the task set over once).
2. Otherwise the pinned upstream tarball
   `https://github.com/datacurve-ai/deep-swe/archive/<ref>.tar.gz` is
   downloaded once (ref from `AIPERF_ACCURACY_DEEPSWE_PINNED_REF`, default
   the v1.1 task-set commit `0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea`) and
   extracted into the dataset cache
   (`.cache/aiperf/datasets/deepswe/<ref>/`).

Task selection uses the standard accuracy flag: `--accuracy-tasks
<task-id>[,<task-id>...]` (task ids are the directory names, e.g.
`bandit-incremental-cache-control`, `fastapi-deprecation-response-headers`);
an unknown id fails with the full list of available tasks.

## Model and endpoint wiring

The runner takes the primary model (`--model`) and the first endpoint URL
(`--url`) from the resolved config and hands them to Pier as
`--model openai/<model-name>` with agent env `OPENAI_BASE_URL` /
`OPENAI_API_BASE` (both set to the endpoint URL normalized to `.../v1`) and
`OPENAI_API_KEY` (the configured `--api-key`, or a placeholder when the
endpoint ignores auth — mini-swe-agent refuses to start without one). The
endpoint host is automatically added to Pier's per-agent network allowlist.

**Endpoint port must be 80 or 443.** The task sandboxes run with
`network_mode = "no-network"` and reach the outside world only through
Pier's egress proxy (Squid), whose policy hard-codes `Safe_ports 80 443` and
denies every other port *before* the domain allowlist is consulted. An
endpoint served on another port (e.g. `http://host:30000`) fails every trial
with `litellm.APIError` wrapping a Squid `ERR_ACCESS_DENIED` page. Expose the
endpoint on port 80/443, or front it with a reverse proxy on one of those
ports (e.g. nginx/socat forwarding 443 → the model server's port) and point
`--url` at that.

## Operational knobs

| Env var | Default | Meaning |
|---|---|---|
| `AIPERF_ACCURACY_DEEPSWE_TASKS_DIR` | *(unset)* | Pre-provisioned tasks directory (skips the GitHub download) |
| `AIPERF_ACCURACY_DEEPSWE_PINNED_REF` | `0b9fabbb…` (v1.1 commit) | Git ref of the tarball to download and pin |
| `AIPERF_ACCURACY_DEEPSWE_PIER_BIN` | `pier` | Pier executable name or path |
| `AIPERF_ACCURACY_DEEPSWE_CONCURRENCY` | `1` | Concurrent trials (each needs ~2 CPUs / 8 GiB; a full agent run can take hours) |
| `AIPERF_ACCURACY_DEEPSWE_TASK_TIMEOUT_S` | *(unset)* | Optional hard wall-clock ceiling per trial (kills the pier process group); unset = task.toml timeouts govern |

## Runtime expectations

- A single trial = image pull (multi-GB, once per task) + agent install +
  agent run (up to the task's agent timeout, commonly 3 h) + verification
  (verifier timeout, commonly 30 min). Start with one or two tasks; the full
  113-task suite is a multi-day, single-concurrency endeavor.
- A crashed trial, a killed timeout, or a missing verifier result grades the
  task as **failed** with the pier output tail in the record's explanation —
  it never aborts the run.
- `Ctrl+C` kills every live pier process group.

## Calibration (no LLM): the oracle agent

To validate the host's Docker + Pier + task-set + verifier plumbing without
the endpoint, run Pier's `oracle` agent on one task — it applies the
reference solution, so the verifier must reward 1:

```bash
pier run -p .cache/aiperf/datasets/deepswe/<ref>/tasks/<task-id> \
  --agent oracle --jobs-dir /tmp/deepswe-calibration
```

Then check `/tmp/deepswe-calibration/<job>/<trial>/verifier/reward.json` for
`"reward": 1`. If the oracle fails, fix the host before spending
endpoint-hours on the model.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Preflight: `Pier CLI 'pier' not found` | `uv tool install datacurve-pier`, or set `AIPERF_ACCURACY_DEEPSWE_PIER_BIN` |
| Preflight: `Docker is installed but the daemon is not reachable` | Start Docker |
| `Failed to download the DeepSWE task set` | No GitHub egress — rsync the task set over and set `AIPERF_ACCURACY_DEEPSWE_TASKS_DIR` |
| Trial fails with agent install errors inside the sandbox | Container network egress; Pier's allowlist covers `astral.sh`, pypi, and the endpoint host — check the trial log under `<artifact_dir>/deepswe/` |
| Every task fails with LLM API errors | Endpoint URL/key wrong or endpoint unreachable *from inside the container network*; the endpoint host is allowlisted from `OPENAI_BASE_URL`, so verify `--url` |
| Every task fails with `litellm.APIError` wrapping a Squid `ERR_ACCESS_DENIED` HTML page | The endpoint is on a port other than 80/443 — Pier's egress proxy denies non-standard ports before the allowlist. Front the endpoint on 80/443 (see [Model and endpoint wiring](#model-and-endpoint-wiring)) |
| `model patch did not apply` in the explanation | The agent finished but its diff did not apply to the base commit — a legitimate graded failure (reward 0), not an infrastructure error |
