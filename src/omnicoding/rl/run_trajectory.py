"""Per-trajectory runner — invoked from inside an sbatch job by
``sbatch_one_job.sh``.

Reads a request JSON, runs ONE kira trajectory, writes a ``Trajectory``
JSON to the result path. Always exits 0 — failures land in the result file
as a failed-Trajectory shape so the coordinator can read uniformly.

CLI::

    python run_trajectory.py --request <req.json> --result <res.json>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import traceback
from dataclasses import fields
from pathlib import Path

LOG = logging.getLogger("run_trajectory")


def _build_record(payload: dict):
    """Re-instantiate ``coordinator.dataset.Record`` from a JSON dict.
    The dispatcher serialized via ``dataclasses.asdict``; here we filter
    keys to known fields (defensive against schema drift)."""
    from omnicoding.rl.coordinator.dataset import Record  # noqa: PLC0415

    accepted = {f.name for f in fields(Record)}
    values = {k: v for k, v in payload.items() if k in accepted}
    # Gold answers never cross into a worker request. The field remains on the
    # shared Record type for coordinator-side grading only.
    values["ground_truth"] = []
    return Record(**values)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True, type=Path)
    ap.add_argument("--result", required=True, type=Path)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )

    payload = json.loads(args.request.read_text())
    sample_idx = int(payload["sample_index"])

    # Lazy imports — keep startup time low for sbatch wrapper diagnostics.
    from omnicoding.rl.coordinator.worker import _failed_trajectory, run_one_trajectory  # noqa: PLC0415
    from omnicoding.rl.schemas import RolloutRequest  # noqa: PLC0415

    try:
        record = _build_record(payload["record"])
        req = RolloutRequest.model_validate(payload["request"])
        workspace = Path(payload["workspace"])
        staged_media = [str(value) for value in payload.get("staged_media", [])]
        workspace.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        LOG.error("payload parse failed: %s\n%s", exc, traceback.format_exc())
        traj = _failed_trajectory(sample_idx, "error", f"payload parse: {exc}")
        args.result.write_text(traj.model_dump_json())
        return 0

    LOG.info(
        "run_trajectory start sample=%d task=%s ws=%s",
        sample_idx, record.id, workspace,
    )
    try:
        traj = asyncio.run(
            run_one_trajectory(
                record=record,
                sample_index=sample_idx,
                req=req,
                workspace=workspace,
                staged_media=staged_media,
            )
        )
    except Exception as exc:  # noqa: BLE001
        LOG.error("kira run crashed: %s\n%s", exc, traceback.format_exc())
        traj = _failed_trajectory(sample_idx, "error", f"kira: {exc}")

    args.result.write_text(traj.model_dump_json())
    LOG.info(
        "run_trajectory done sample=%d exit=%s reward=%.3f steps=%d",
        sample_idx, traj.exit_reason, traj.reward, traj.n_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
