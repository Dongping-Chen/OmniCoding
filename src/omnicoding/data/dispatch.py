"""On-demand single-item dispatcher for the kira → SFT pipeline.

Goal: keep at most ``--max_concurrent`` slurm jobs running, each doing
exactly one item. As one finishes, launch the next pending item. Resumes
from on-disk state on restart, so killing the dispatcher is safe.

Usage::

    tmux new-session -d -s kira-dispatch \\
      'OmniCoding/.venv_harness/bin/python \\
       OmniCoding/local_model/scripts/dispatch_synthetic.py \\
         --items_file <items.json> \\
         --dataset_root <staged> \\
         --out <out_dir> \\
         --bench omnigaia \\
         --max_concurrent 4 \\
         --slot_spec acct-B=1'

Concurrency rationale
---------------------
Each codex-router slot caps in-flight upstream calls at ``DEFAULT_INFLIGHT
= 4`` (`pool.py`). With one slurm job per item and one in-flight call
per slurm job, ``max_concurrent`` should be ``DEFAULT_INFLIGHT *
n_accounts``. Default 4 (single-account acct-B). Crank to 8 only when
both accounts are below 80 % weekly quota.

Per-job result files
--------------------
Each sbatch writes ``<out>/results.<source_index>.json`` (single-row
file) so concurrent jobs don't race on the master ``results.json``.
The dispatcher merges them periodically into ``<out>/results.json`` for
downstream filter/convert.

Restart safety
--------------
Restart-safe via:
  - Per-job result files persist across restarts (auto-resume in run_bench_kira).
  - Dispatcher polls slurm via ``squeue -j <jid>`` for active jobs;
    state is the squeue snapshot, no separate ledger to keep in sync.
  - Items already accepted by ``filter_correct_trajectories`` (or any
    successful per-job result) are skipped on the next dispatcher start.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

LOG = logging.getLogger("dispatch_synthetic")

SBATCH_SUBMIT_RE = re.compile(r"Submitted batch job (\d+)")


def _setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "dispatch.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stderr)],
    )


def _write_pidfile(out_dir: Path) -> None:
    """Drop our PID where the watchdog can find it. The watchdog uses
    this to kill us on stall + restart with the same CLI."""
    (out_dir / "dispatcher.pid").write_text(str(os.getpid()), encoding="utf-8")


def _load_items(items_file: Path) -> list[dict]:
    """Read items.json and stamp ``__source_index__`` exactly like
    ``common.spec.load_json_items`` does (so dispatcher source_index ==
    run_bench_kira's source_index for the same row)."""
    raw = json.loads(items_file.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for key in ("data", "items", "results", "questions"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    items = [it for it in raw if isinstance(it, dict)]
    for idx, it in enumerate(items):
        if "__source_index__" not in it:
            it["__source_index__"] = idx
    return items


def _result_path(out_dir: Path, si: int) -> Path:
    return out_dir / f"results.{si}.json"


def _is_done(out_dir: Path, si: int, min_attempt: int = 1) -> bool:
    """Per-job result exists, has no error, AND was produced at attempt
    ≥ ``min_attempt``. The attempt gate is what makes escalation
    resume-safe: pass-1 results have ``attempt: 1``; when pass 2 calls
    this with ``min_attempt=2`` it returns False for a pass-1 row, so
    the dispatcher requeues the item. Any partial completion of pass 2
    is also resumed correctly because successful attempt-2 rows have
    ``attempt: 2`` and the gate then returns True."""
    p = _result_path(out_dir, si)
    if not p.exists():
        return False
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return any(
        isinstance(r, dict)
        and r.get("source_index") == si
        and not r.get("error")
        and int(r.get("attempt", 1)) >= min_attempt
        for r in rows
    )


def _probe_tavily_keys(probe_script: Path, timeout_s: int = 60) -> dict | None:
    """Run the Tavily-key probe script and return its JSON report. Returns
    None on missing-script (so the dispatcher can run on a node where
    Tavily isn't configured at all). Returning a report with ``n_alive==0``
    is the dispatcher-side signal to abort the run."""
    if not probe_script.exists():
        LOG.warning("dispatch tavily probe script missing: %s", probe_script)
        return None
    try:
        proc = subprocess.run(
            [sys.executable, str(probe_script)],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        LOG.warning("dispatch tavily probe timeout after %ds", timeout_s)
        return None
    if proc.returncode not in (0, 2):
        LOG.warning("dispatch tavily probe rc=%d stderr=%s",
                    proc.returncode, proc.stderr.strip()[:300])
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        LOG.warning("dispatch tavily probe: unparseable stdout=%r", proc.stdout[:200])
        return None


def _fetch_quota(router_url: str, timeout: float = 5.0) -> list[dict]:
    """GET ``/accounts`` and return the per-account list. Empty list on
    error (treat as "no signal" — caller falls through to launching)."""
    try:
        with urllib.request.urlopen(f"{router_url.rstrip('/')}/accounts",
                                    timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        LOG.warning("dispatch quota fetch failed: %s", exc)
        return []
    return data.get("accounts") or []


def _quota_should_pause(accounts: list[dict], pause_at: float, resume_at: float,
                         currently_paused: bool) -> tuple[bool, float | None]:
    """Decide whether the dispatcher should pause new launches.

    Looks at ``secondary_used_percent`` (the weekly bucket — what we burn
    on real traffic). If we have no quota signal yet (fresh router, no
    upstream calls landed), don't pause. Returns ``(should_pause, max_pct)``.

    Hysteresis: pause when MIN across accounts crosses ``pause_at`` (every
    account is hot — no failover headroom left). Resume only when MIN
    drops below ``resume_at`` so we don't oscillate at the boundary.
    """
    pcts: list[float] = []
    for a in accounts:
        q = a.get("quota") or {}
        v = q.get("secondary_used_percent")
        if isinstance(v, (int, float)):
            pcts.append(float(v))
    if not pcts:
        return False, None
    min_pct = min(pcts)
    if currently_paused:
        return (min_pct >= resume_at), min_pct
    return (min_pct >= pause_at), min_pct


def _squeue_running(jids: list[int]) -> set[int]:
    """Return the subset of ``jids`` slurm currently shows as running/queued."""
    if not jids:
        return set()
    out = subprocess.run(
        ["squeue", "-h", "-o", "%i", "-j", ",".join(str(j) for j in jids)],
        capture_output=True, text=True,
    ).stdout
    return {int(line.strip()) for line in out.splitlines() if line.strip().isdigit()}


def _launch_one(args: argparse.Namespace, si: int, item: dict, attempt: int = 1,
                 prev_effort: str = "") -> int:
    """Submit one sbatch for ``si``. Routes to GPU vs CPU sbatch based
    on ``effort_policy.pick_cpu_only(item)`` — ~20% of items train the
    model on CPU-only sandboxes. ``prev_effort`` (only meaningful for
    attempt ≥ 2) drives effort_policy's monotonic escalation: the new
    effort is strictly higher than the prior tier."""
    from omnicoding.harnesses.effort_policy import pick_cpu_only

    cpu_only = pick_cpu_only(item) if args.cpu_only_mix else False
    sbatch_script = args.sbatch_script_cpu if cpu_only else args.sbatch_script

    env = os.environ.copy()
    env.update({
        "INPUT": str(args.items_file.resolve()),
        "DATASET_ROOT": str(args.dataset_root.resolve()),
        "OUT": str(args.out.resolve()),
        "SOURCE_INDEX": str(si),
        "BENCH": args.bench,
        "SLOT_SPEC": args.slot_spec,
        "ROUTER_URL": args.router_url,
        "KIRA_BLOCK_TIMEOUT": str(args.block_timeout),
        "KIRA_REQUEST_TIMEOUT": str(args.request_timeout),
        "KIRA_STEP_LIMIT": str(args.step_limit),
        "KIRA_MODEL_NAME": args.model_name,
        "KIRA_REASONING_EFFORT": args.reasoning_effort,
        "KIRA_EFFORT_STRATEGY": args.effort_strategy,
        "KIRA_ATTEMPT": str(attempt),
        "KIRA_PREV_EFFORT": prev_effort,
        "KIRA_CPU_ONLY": "1" if cpu_only else "0",
    })
    suffix = "-cpu" if cpu_only else ""
    proc = subprocess.run(
        ["sbatch",
         "--export=ALL,SLOT_SPEC,ROUTER_URL,BENCH,SOURCE_INDEX,INPUT,DATASET_ROOT,"
         "OUT,KIRA_MODEL_NAME,KIRA_REASONING_EFFORT,KIRA_BLOCK_TIMEOUT,"
         "KIRA_REQUEST_TIMEOUT,KIRA_STEP_LIMIT,KIRA_EFFORT_STRATEGY,KIRA_ATTEMPT,"
         "KIRA_PREV_EFFORT,KIRA_CPU_ONLY",
         "--job-name", f"kira-si{si}-a{attempt}{suffix}", str(sbatch_script)],
        env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        LOG.error("sbatch failed for si=%d rc=%d stderr=%s",
                  si, proc.returncode, proc.stderr.strip())
        return -1
    m = SBATCH_SUBMIT_RE.search(proc.stdout)
    if not m:
        LOG.error("sbatch stdout unparseable for si=%d: %r", si, proc.stdout)
        return -1
    jid = int(m.group(1))
    LOG.info("dispatch launched jid=%d for si=%d", jid, si)
    return jid


def _merge_results(out_dir: Path) -> int:
    """Concat all ``results.*.json`` under out_dir into ``results.json``.
    Returns the row count. Idempotent."""
    merged: dict[int, dict] = {}
    for p in sorted(out_dir.glob("results.*.json")):
        if p.name == "results.json":
            continue
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOG.warning("dispatch skip malformed %s", p)
            continue
        for r in rows or []:
            si = r.get("source_index") if isinstance(r, dict) else None
            if isinstance(si, int):
                merged[si] = r
    rows = [merged[k] for k in sorted(merged.keys())]
    target = out_dir / "results.json"
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)
    return len(rows)


def _pick_pending(items: list[dict], out_dir: Path, in_flight: set[int],
                  attempt: int = 1) -> list[int]:
    """SIs that are neither in flight nor already at-success-for-this-attempt."""
    out: list[int] = []
    for it in items:
        si = it.get("__source_index__")
        if not isinstance(si, int):
            continue
        if si in in_flight:
            continue
        if _is_done(out_dir, si, min_attempt=attempt):
            continue
        out.append(si)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--items_file", required=True, type=Path)
    p.add_argument("--dataset_root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--bench", default="omnigaia")
    p.add_argument("--max_concurrent", type=int, default=12,
                   help="Max slurm jobs in flight. Default 12 = DEFAULT_INFLIGHT (6) "
                        "× 2 accounts. Adjust if you change the router's inflight cap "
                        "or run a single-account dispatch.")
    p.add_argument("--slot_spec", default="acct-A=1|acct-B=1",
                   help="codex-router slot spec, pipe-separated. Default splits load "
                        "evenly across both accounts.")
    p.add_argument("--router_url", default="http://localhost:8765")
    p.add_argument("--sbatch_script", required=True, type=Path,
                   help="GPU sbatch template. See infra/slurm/harness_item.sbatch.")
    p.add_argument("--sbatch_script_cpu", type=Path,
                   default=None,
                   help="CPU-only sbatch template — used when an item is "
                        "selected for CPU-only by ``effort_policy.pick_cpu_only`` "
                        "(~20%% by default).")
    p.add_argument("--cpu_only_mix", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Route ~20%% of items to CPU-only sbatch (no --gres=gpu) "
                        "so the trained model sees both GPU and CPU sandbox "
                        "behaviour. ``--no-cpu_only_mix`` disables and routes "
                        "every item to the GPU template.")
    p.add_argument("--block_timeout", type=int, default=1200)
    p.add_argument("--request_timeout", type=int, default=1500)
    p.add_argument("--step_limit", type=int, default=120)
    p.add_argument("--model_name", default="openai/gpt-5.5")
    p.add_argument("--reasoning_effort", default="high",
                   help="Used as fallback when --effort_strategy=fixed. Ignored "
                        "when --effort_strategy=auto (effort_policy decides per-item).")
    p.add_argument("--effort_strategy", default="auto", choices=["fixed", "auto"],
                   help="Per-item reasoning_effort policy. ``auto``: low/medium/high "
                        "from item['Level']; random low/medium for unlabeled items "
                        "on attempt 1; random high/xhigh on attempt ≥ 2 escalation.")
    p.add_argument("--escalate_failures", action="store_true",
                   help="After the first pass, identify failures (no task_complete "
                        "or wrong answer per filter logic) and re-launch them with "
                        "attempt=2 — effort_policy will pick high/xhigh randomly.")
    p.add_argument("--escalation_random_high_xhigh", action="store_true",
                   help="On --escalate_failures, force every retry to pick "
                        "randomly from {high, xhigh} regardless of prev_effort. "
                        "Without this flag, effort_policy's strict-monotone "
                        "escalation kicks in: low → random{medium,high,xhigh} "
                        "(some get medium, not what you want when most prev "
                        "items ran at low). Use this when the round-1 was all "
                        "low/medium auto and you want round-2 retries to land "
                        "directly in the high/xhigh tier without any medium "
                        "regression.")
    p.add_argument("--poll_s", type=int, default=60,
                   help="How often to reap finished jobs and launch new ones.")
    p.add_argument("--merge_every_n_polls", type=int, default=5,
                   help="Merge per-job results into master results.json every N polls.")
    p.add_argument("--quota_pause_at", type=float, default=95.0,
                   help="Pause new sbatch launches when MIN secondary-used-percent "
                        "across accounts crosses this threshold. Already-running "
                        "jobs are NOT cancelled; we just stop adding more. 0 disables.")
    p.add_argument("--quota_resume_at", type=float, default=80.0,
                   help="Resume launches once MIN drops below this. Hysteresis "
                        "vs. --quota_pause_at avoids oscillation at the boundary.")
    p.add_argument("--max_item_runtime_s", type=int, default=7200,
                   help="Per-item walltime cap (default 2h). The dispatcher polls "
                        "sacct/squeue start-time; jobs exceeding this are scancelled "
                        "and re-queued via the escalation path. 0 disables.")
    p.add_argument("--tavily_probe_script", type=Path, default=None,
                   help="Tavily key health probe — runs at dispatcher start to "
                        "warn of dead keys. Aborts the run if zero keys alive.")
    p.add_argument("--no_tavily_probe", action="store_true",
                   help="Skip the startup Tavily key check. Useful for offline "
                        "smoke tests where web search isn't exercised.")
    p.add_argument("--tavily_recheck_polls", type=int, default=60,
                   help="Re-run the Tavily probe every N polls (default 60 ≈ "
                        "60 minutes at default poll_s=60). Dead keys mid-run "
                        "are logged but the dispatcher keeps going as long as "
                        "≥1 key is alive (mid-run abort would lose work).")
    args = p.parse_args(argv)

    if args.cpu_only_mix and args.sbatch_script_cpu is None:
        p.error("--sbatch_script_cpu is required unless --no-cpu_only_mix is set")

    _setup_logging(args.out)
    _write_pidfile(args.out)
    items = _load_items(args.items_file)
    LOG.info(
        "dispatch start out=%s bench=%s items=%d max_concurrent=%d slot=%s "
        "effort_strategy=%s escalate=%s",
        args.out, args.bench, len(items), args.max_concurrent, args.slot_spec,
        args.effort_strategy, args.escalate_failures,
    )

    if not args.no_tavily_probe and args.tavily_probe_script is not None:
        report = _probe_tavily_keys(args.tavily_probe_script)
        if report is not None:
            LOG.info("dispatch tavily probe: alive=%d/%d dead=%s",
                     report.get("n_alive", 0), report.get("n_total", 0),
                     report.get("dead", []))
            (args.out / "tavily_probe.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if report.get("n_alive", 0) == 0:
                LOG.error("dispatch ABORT: zero Tavily keys alive — refresh keys "
                          "and rerun. Pass --no_tavily_probe to override.")
                return 2

    _run_pass(args, items, attempt=1)

    if args.escalate_failures:
        failed = _identify_failures(args.out, items)
        if failed:
            # Lookup prev_effort for each failure + filter out items
            # whose attempt-1 effort was already at the top of the
            # ladder (xhigh) — those have nowhere to escalate to, so
            # we skip them rather than burn compute re-running at the
            # same tier.
            from omnicoding.harnesses.effort_policy import can_escalate
            prev_efforts = _read_prev_efforts(args.out, failed)
            escalatable = [
                si for si in failed
                if can_escalate(prev_efforts.get(si, ""))
            ]
            terminal = [si for si in failed if si not in escalatable]
            if terminal:
                LOG.warning(
                    "dispatch escalation: %d items already at top tier "
                    "(can't escalate further): %s",
                    len(terminal), terminal,
                )
            if escalatable:
                LOG.info(
                    "dispatch escalating %d items to attempt=2 "
                    "(prev_efforts=%s): %s",
                    len(escalatable),
                    {si: prev_efforts.get(si) for si in escalatable},
                    escalatable,
                )
                # NO wipe — _run_pass(attempt=2) plus the attempt-aware
                # ``_is_done(min_attempt=2)`` gate makes pass-1 rows look
                # "not done at attempt 2" so they get re-queued. After
                # the new run finishes, _atomic_write_results overwrites
                # the per-job results.<si>.json with the attempt-2 row.
                # Restart-safe: a kill mid-pass-2 lets a fresh dispatcher
                # pick up where we left off.
                # When --escalation_random_high_xhigh is set, override
                # prev_efforts to empty so effort_policy.pick_reasoning_effort
                # falls into its legacy ``prev_effort=None`` branch —
                # uniform random pick from ESCALATION_CHOICES = {high,
                # xhigh}. Otherwise keep the strictly-monotone (one-tier-up)
                # default which can land low → medium for ~33% of low items.
                effective_prev = (
                    {} if args.escalation_random_high_xhigh else prev_efforts
                )
                _run_pass(
                    args,
                    [it for it in items
                     if it.get("__source_index__") in escalatable],
                    attempt=2, prev_efforts=effective_prev,
                )
        else:
            LOG.info("dispatch escalation: no failures from attempt=1")

    n = _merge_results(args.out)
    LOG.info("dispatch all done — final merged rows=%d", n)
    return 0


def _run_pass(args: argparse.Namespace, items: list[dict], attempt: int,
              prev_efforts: dict[int, str] | None = None) -> None:
    """Single dispatch pass: launch sbatch jobs for any pending item, poll
    squeue, refill, until empty. Items already at-success-for-this-attempt
    on disk are skipped automatically by ``_pick_pending``. ``prev_efforts``
    (optional, only used in escalation pass=2): si → attempt-1 effort,
    forwarded via ``KIRA_PREV_EFFORT`` env so effort_policy can pick a
    strictly-higher tier.
    """
    items_by_si = {it.get("__source_index__"): it for it in items
                   if isinstance(it.get("__source_index__"), int)}
    prev_efforts = prev_efforts or {}
    active: dict[int, int] = {}
    job_started: dict[int, float] = {}
    quota_paused = False
    poll_count = 0
    LOG.info("dispatch pass attempt=%d start", attempt)
    while True:
        running = _squeue_running(list(active.keys()))
        for jid in list(active):
            if jid not in running:
                LOG.info("dispatch reap jid=%d si=%d (attempt=%d)",
                         jid, active[jid], attempt)
                del active[jid]
                job_started.pop(jid, None)
        if args.max_item_runtime_s > 0:
            _scancel_overrun_jobs(active, job_started, args.max_item_runtime_s)
        in_flight = set(active.values())
        pending = _pick_pending(items, args.out, in_flight, attempt=attempt)
        quota_paused = _update_quota_state(args, quota_paused)
        if not quota_paused:
            _launch_until_full(args, items_by_si, pending, active, job_started,
                               attempt, prev_efforts)
        if not pending and not active:
            LOG.info("dispatch pass attempt=%d done", attempt)
            return
        poll_count += 1
        if poll_count % args.merge_every_n_polls == 0:
            n = _merge_results(args.out)
            LOG.info("dispatch tick active=%d pending=%d merged=%d (attempt=%d) "
                     "quota_paused=%s",
                     len(active), len(pending), n, attempt, quota_paused)
            _write_metrics(args.out, attempt, len(active), len(pending), n,
                           quota_paused)
        if (not args.no_tavily_probe and args.tavily_recheck_polls > 0
                and poll_count % args.tavily_recheck_polls == 0):
            report = _probe_tavily_keys(args.tavily_probe_script)
            if report is not None and report.get("n_dead"):
                LOG.warning("dispatch tavily mid-run: alive=%d/%d dead=%s",
                            report.get("n_alive", 0), report.get("n_total", 0),
                            report.get("dead", []))
                (args.out / "tavily_probe.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        time.sleep(args.poll_s)


def _update_quota_state(args: argparse.Namespace, currently_paused: bool) -> bool:
    """Poll the router /accounts endpoint and apply hysteresis pause/resume.
    Returns the new pause state. Disabled when ``--quota_pause_at <= 0``."""
    if args.quota_pause_at <= 0:
        return currently_paused
    accts = _fetch_quota(args.router_url)
    should_pause, pct = _quota_should_pause(
        accts, args.quota_pause_at, args.quota_resume_at, currently_paused,
    )
    if should_pause and not currently_paused:
        LOG.warning(
            "dispatch quota PAUSE — secondary used %% min=%.1f ≥ %.1f; "
            "holding launches until min < %.1f",
            pct or 0.0, args.quota_pause_at, args.quota_resume_at,
        )
        return True
    if not should_pause and currently_paused:
        LOG.info("dispatch quota RESUME — secondary used %% min=%.1f < %.1f",
                 pct or 0.0, args.quota_resume_at)
        return False
    return currently_paused


def _launch_until_full(args: argparse.Namespace, items_by_si: dict[int, dict],
                        pending: list[int], active: dict[int, int],
                        job_started: dict[int, float], attempt: int,
                        prev_efforts: dict[int, str]) -> None:
    """Launch sbatch for pending sis until ``max_concurrent`` is reached.
    Mutates ``pending``/``active``/``job_started`` in place."""
    while pending and len(active) < args.max_concurrent:
        si = pending.pop(0)
        item = items_by_si.get(si)
        if item is None:
            LOG.warning("dispatch skip si=%d (no item)", si)
            continue
        jid = _launch_one(args, si, item, attempt=attempt,
                          prev_effort=prev_efforts.get(si, ""))
        if jid > 0:
            active[jid] = si
            job_started[jid] = time.time()


def _scancel_overrun_jobs(active: dict[int, int], job_started: dict[int, float],
                           max_runtime_s: int) -> None:
    """Cancel jobs whose dispatcher-side wall-clock has crossed
    ``max_runtime_s``. We use dispatcher-side timing instead of sacct
    StartTime because sacct lags + costs an exec on every poll. Cancelled
    jobs are removed from ``active``; their per-job results.<si>.json is
    left untouched, so the next pass (or escalation pass) re-queues them
    via the standard ``_pick_pending``/``_is_done`` gate.
    """
    now = time.time()
    for jid, si in list(active.items()):
        started = job_started.get(jid)
        if started is None:
            continue
        if now - started < max_runtime_s:
            continue
        LOG.warning("dispatch scancel overrun jid=%d si=%d ran=%ds > %ds",
                    jid, si, int(now - started), max_runtime_s)
        try:
            subprocess.run(["scancel", str(jid)], check=False,
                            capture_output=True, timeout=10)
        except (subprocess.SubprocessError, OSError) as exc:
            LOG.warning("dispatch scancel failed jid=%d: %s", jid, exc)
        active.pop(jid, None)
        job_started.pop(jid, None)


def _write_metrics(out_dir: Path, attempt: int, active: int, pending: int,
                    merged: int, quota_paused: bool) -> None:
    """Write a snapshot of dispatcher state for live monitoring. Atomic:
    write tmp + rename so a tail-er never reads a half-written file.

    Bonus: aggregates per-row latency/tokens from any merged results so
    operators can grep the file instead of replaying logs.
    """
    target = out_dir / "dispatcher.metrics.json"
    snap: dict = {
        "ts": time.time(),
        "attempt": attempt,
        "active_jobs": active,
        "pending_jobs": pending,
        "merged_rows": merged,
        "quota_paused": quota_paused,
    }
    # Per-row aggregates from merged results (cheap — small N).
    rj = out_dir / "results.json"
    if rj.exists():
        try:
            rows = json.loads(rj.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            rows = []
        latencies = [r.get("elapsed_s") for r in rows
                     if isinstance(r, dict) and isinstance(r.get("elapsed_s"), (int, float))]
        tcs = [r.get("tool_call_num") for r in rows
               if isinstance(r, dict) and isinstance(r.get("tool_call_num"), int)]
        n_err = sum(1 for r in rows if isinstance(r, dict) and r.get("error"))
        snap["agg"] = {
            "n_rows": len(rows),
            "n_errors": n_err,
            "median_elapsed_s": sorted(latencies)[len(latencies)//2] if latencies else None,
            "median_tool_calls": sorted(tcs)[len(tcs)//2] if tcs else None,
        }
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)


def _read_prev_efforts(out_dir: Path, source_indices: list[int]) -> dict[int, str]:
    """Read each per-job ``results.<si>.json`` and return the
    ``reasoning_effort`` recorded on the row. Used by escalation to
    feed effort_policy's ``prev_effort`` so attempt-2 picks a strictly
    higher tier (no more wasted same-tier reruns)."""
    out: dict[int, str] = {}
    for si in source_indices:
        p = _result_path(out_dir, si)
        if not p.exists():
            continue
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        match = next((r for r in rows or []
                      if isinstance(r, dict) and r.get("source_index") == si), None)
        eff = match.get("reasoning_effort") if match else ""
        if eff:
            out[si] = eff
    return out


def _identify_failures(out_dir: Path, items: list[dict]) -> list[int]:
    """Return source_indices that failed — either errored, or whose
    predicted_answer doesn't match any ground_truth.

    Round-17.11 fix (2026-04-30): the prior version bailed on
    ``exit_reason != "task_complete"`` BEFORE checking predicted_answer.
    But the FINAL_ANSWER_PROTOCOL nudges the model to emit
    ``<answer>X</answer>`` as plain assistant text and skip the
    explicit ``task_complete`` call — kira then exits with
    ``exit_reason="no_tool_calls"`` cleanly. Those rows DO have a
    valid ``predicted_answer`` and are routinely correct, so the old
    rule re-escalated them as failures and burned ~3x the quota.

    New rule: failure iff (error set) OR (answer doesn't match GT).
    Exit reason isn't a signal of correctness any more — the harness
    has multiple legitimate exit shapes under the new protocol.

    Re-uses ``filter_correct_trajectories.is_correct`` to keep a
    single correctness rule across pipelines.
    """
    from omnicoding.data.filtering import is_correct

    items_by_si = {it.get("__source_index__"): it for it in items}
    failed: list[int] = []
    for p in sorted(out_dir.glob("results.*.json")):
        if p.name == "results.json":
            continue
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        # Pick the LATEST attempt row per item — escalation may have
        # written a fresh row over an older one, and the freshest
        # state is what we score.
        latest_per_si: dict[int, dict] = {}
        for r in rows or []:
            if not isinstance(r, dict):
                continue
            si = r.get("source_index")
            if not isinstance(si, int):
                continue
            prev = latest_per_si.get(si)
            if prev is None or int(r.get("attempt", 1)) >= int(prev.get("attempt", 1)):
                latest_per_si[si] = r
        for si, r in latest_per_si.items():
            if r.get("error"):
                failed.append(si)
                continue
            it = items_by_si.get(si) or {}
            ok, _ = is_correct(
                answer_type=(it.get("answer_type") or "").lower(),
                predicted=(r.get("predicted_answer") or "").strip(),
                ground_truths=it.get("ground_truth") or [],
            )
            if not ok:
                failed.append(si)
    return sorted(set(failed))


if __name__ == "__main__":
    raise SystemExit(main())
