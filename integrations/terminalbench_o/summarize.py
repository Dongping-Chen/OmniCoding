#!/usr/bin/env python3
"""Summarize versioned TerminalBench-O grade.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
TASK_IDS = tuple(
    line.strip()
    for line in (HERE / "task_ids.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
)


def load_grade(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def case_task(case_dir: Path) -> str | None:
    return next((task for task in TASK_IDS if f"-{task}-" in case_dir.name), None)


def build_summary(runs_dir: Path, grade_name: str) -> dict[str, Any]:
    rows = []
    for case_dir in sorted(path for path in runs_dir.glob("newbench-*") if path.is_dir()):
        task = case_task(case_dir)
        if not task:
            continue
        grade_path = case_dir / "grades" / grade_name / "grade.json"
        grade = load_grade(grade_path)
        parsed = grade.get("parsed") if isinstance(grade, dict) else None
        rows.append(
            {
                "task": task,
                "case_dir": str(case_dir),
                "grade_path": str(grade_path),
                "exists": grade_path.exists(),
                "status": grade.get("status") if grade else "missing",
                "returncode": grade.get("returncode") if grade else None,
                "passed": (
                    parsed.get("pass")
                    if isinstance(parsed, dict) and "pass" in parsed
                    else grade.get("passed")
                    if grade
                    else None
                ),
                "score": (
                    parsed.get("score", parsed.get("score_so_far"))
                    if isinstance(parsed, dict)
                    else grade.get("score")
                    if grade
                    else None
                ),
                "missing_flags": grade.get("missing_flags") if grade else None,
            }
        )

    scores = [row["score"] for row in rows if isinstance(row["score"], (int, float))]
    errors = [
        row
        for row in rows
        if row["exists"]
        and (
            row["status"] != "completed"
            or (row["returncode"] not in (0, None) and row["score"] is None)
        )
    ]
    return {
        "total": len(rows),
        "graded": sum(1 for row in rows if row["exists"]),
        "passed": sum(1 for row in rows if row["passed"] is True),
        "failed": sum(1 for row in rows if row["passed"] is False),
        "errors": len(errors),
        "mean_score": sum(scores) / len(scores) if scores else None,
        "rows": rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--grade-name", default="terminalbench_o_v1")
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_summary(args.runs_dir.expanduser().resolve(), args.grade_name)
    print(
        "total={total} graded={graded} passed={passed} failed={failed} "
        "errors={errors} mean_score={mean_score}".format(**summary)
    )
    for row in summary["rows"]:
        print(
            f"{row['task']}\tpass={row['passed']}\t"
            f"score={row['score']}\tstatus={row['status']}\t"
            f"returncode={row['returncode']}"
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"summary_json={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
