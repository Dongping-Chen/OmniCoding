#!/usr/bin/env python3
"""Run one released TerminalBench-O grader against an existing case output."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
TASKS_DIR = HERE / "tasks"
LIB_DIR = HERE / "_lib"
DEFAULT_GRADE_NAME = "terminalbench_o_v1"
TASK_IDS = tuple(
    line.strip()
    for line in (HERE / "task_ids.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
)


def _iter_add_argument_calls(text: str) -> Iterable[str]:
    marker = "add_argument("
    pos = 0
    while True:
        start = text.find(marker, pos)
        if start == -1:
            return
        index = start + len(marker)
        depth = 1
        in_string: str | None = None
        escaped = False
        while index < len(text):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == in_string:
                    in_string = None
            else:
                if char in ("'", '"'):
                    in_string = char
                elif char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        yield text[start : index + 1]
                        pos = index + 1
                        break
            index += 1
        else:
            return


def parse_grader_args(grader: Path) -> list[str]:
    text = grader.read_text(encoding="utf-8", errors="ignore")
    flags: list[str] = []
    seen: set[str] = set()
    for call in _iter_add_argument_calls(text):
        match = re.search(r"add_argument\(\s*[\"'](--[^\"']+)", call)
        if not match:
            continue
        flag = match.group(1)
        if flag in seen:
            continue
        if re.search(r"action\s*=\s*[\"']store_(?:true|false)[\"']", call):
            continue
        seen.add(flag)
        flags.append(flag)
    return flags


def candidate_files(base: Path, stem: str) -> list[Path]:
    stems = [stem, stem.replace("-", "_"), stem.replace("_", "-")]
    extensions = [
        "",
        ".json",
        ".csv",
        ".txt",
        ".md",
        ".pdf",
        ".mp4",
        ".mov",
        ".mp3",
        ".wav",
        ".flac",
        ".png",
        ".jpg",
        ".jpeg",
    ]
    return [base / f"{candidate}{extension}" for candidate in stems for extension in extensions]


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def infer_gt(reference: Path) -> Path | None:
    preferred = [
        "gt.json",
        "player_events_gt.json",
        "room_labels_gt.json",
        "page_meta_gt.json",
        "key_numbers.json",
        "risks_gt.json",
        "key_facts_gt.json",
        "check_windows_gt.json",
        "timeline_gt.json",
        "all_events_gt.json",
        "selling_points_gt.json",
        "ledger_gt.json",
        "speakers_gt.json",
        "turns_gt.json",
        "receipts_gt.json",
        "matches_gt.json",
        "quote_gt.json",
        "incidents_gt.json",
        "uploads_gt.json",
        "keynote_gt.json",
    ]
    found = first_existing(reference / name for name in preferred)
    if found:
        return found
    gt_files = sorted(reference.glob("*gt*.json"))
    if len(gt_files) == 1:
        return gt_files[0]
    json_files = sorted(reference.glob("*.json"))
    return json_files[0] if len(json_files) == 1 else None


def infer_arg_path(
    flag: str,
    *,
    data_task_dir: Path,
    output_dir: Path,
    grade_dir: Path,
) -> Path | None:
    name = flag[2:]
    reference = data_task_dir / "reference"
    workspace_fixtures = output_dir.parent / "fixtures"
    fixtures = workspace_fixtures if workspace_fixtures.exists() else data_task_dir / "fixtures"

    aliases = {
        "audio": ["audio", "interview"],
        "cases": ["cases"],
        "chapters_gt": ["chapters_gt"],
        "clips_dir": ["cam", "clips", "clips_dir"],
        "diar_gt": ["diarization_gt", "diar_gt"],
        "fixtures-forms": ["fixtures"],
        "glossary": ["glossary"],
        "ingr_gt": ["ingredients_gt"],
        "keywords_gt": ["keywords_gt"],
        "label_defs": ["label_definitions"],
        "segments": ["segments"],
        "sources": ["sources"],
        "steps_gt": ["steps_gt"],
        "transcript": ["transcript"],
        "uploads": ["uploads", "uploads_gt"],
        "video": ["review", "input", "video"],
        "covers": ["oracle_covers_gt"],
        "specs": ["platform_specs"],
        "marginalia_gt": ["marginalia_phrases"],
        "meta": ["meta", "papers_meta"],
        "paper_specific_gt": ["paper_specific_gt"],
        "claims_gt": ["claims_gt"],
        "timeline_gt": ["timeline_gt"],
        "override_gt": ["addendum_overrides_gt"],
    }

    if name == "src":
        found = first_existing(
            candidate_files(fixtures, "camA")
            + candidate_files(fixtures, "source")
            + candidate_files(fixtures, "input")
            + candidate_files(fixtures, "match")
            + candidate_files(fixtures, "review")
            + candidate_files(fixtures, "video")
        )
        return found if found else fixtures

    explicit: dict[str, Path | None] = {
        "output": output_dir,
        "output-dir": output_dir,
        "fixtures": fixtures,
        "fixtures-dir": fixtures,
        "fixtures-forms": fixtures,
        "reference-dir": reference,
        "report": grade_dir / "grader_report.json",
        "gt": infer_gt(reference),
        "target-meta": reference / "target_meta.json",
    }
    if name in explicit:
        path = explicit[name]
        if path is not None and (
            path.exists() or name in {"output", "output-dir", "report"}
        ):
            return path
        return None

    for alias in aliases.get(name, []):
        found = first_existing(
            candidate
            for base in (reference, fixtures)
            for candidate in candidate_files(base, alias)
        )
        if found:
            return found

    return first_existing(
        candidate
        for base in (reference, fixtures)
        for candidate in candidate_files(base, name)
    )


def build_grader_command(
    task: str,
    *,
    benchmark_root: Path,
    output_dir: Path,
    grade_dir: Path,
    grader_python: str,
) -> tuple[list[str], list[str]]:
    grader = TASKS_DIR / task / "grader.py"
    if not grader.is_file():
        raise FileNotFoundError(f"missing released grader: {grader}")
    data_task_dir = benchmark_root / "tasks" / task
    if not data_task_dir.is_dir():
        raise FileNotFoundError(f"missing private task data: {data_task_dir}")

    flags = parse_grader_args(grader)
    output_flag = "--output-dir" if "--output-dir" in flags and "--output" not in flags else "--output"
    command = [grader_python, str(grader), output_flag, str(output_dir)]
    missing_flags: list[str] = []
    for flag in flags:
        if flag in {"--output", "--output-dir"}:
            continue
        path = infer_arg_path(
            flag,
            data_task_dir=data_task_dir,
            output_dir=output_dir,
            grade_dir=grade_dir,
        )
        if path is None:
            missing_flags.append(flag)
            continue
        command.extend([flag, str(path)])
    return command, missing_flags


def parse_stdout_json(stdout: str) -> dict[str, Any] | None:
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]+\}", text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def parse_grader_pass(stdout: str, stderr: str) -> bool | None:
    parsed = parse_stdout_json(stdout)
    if parsed and isinstance(parsed.get("pass"), bool):
        return parsed["pass"]
    blob = f"{stdout or ''}\n{stderr or ''}"
    if "Overall: PASS" in blob:
        return True
    if "Overall: FAIL" in blob:
        return False
    return None


def write_grade(
    grade_dir: Path,
    *,
    command: list[str],
    returncode: int | None,
    stdout: str,
    stderr: str,
    passed: bool | None,
    missing_flags: list[str],
    status: str,
) -> None:
    grade_dir.mkdir(parents=True, exist_ok=True)
    parsed = parse_stdout_json(stdout)
    if parsed and isinstance(parsed.get("pass"), bool):
        passed = parsed["pass"]
    payload: dict[str, Any] = {
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "command": command,
        "passed": passed,
        "missing_flags": missing_flags,
        "status": status,
    }
    if parsed:
        payload["parsed"] = parsed
        payload["score"] = parsed.get("score", parsed.get("score_so_far"))
    (grade_dir / "grade.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (grade_dir / "grade.log").write_text(
        "+ " + shlex.join(command) + "\n" + stdout + stderr,
        encoding="utf-8",
        errors="replace",
    )


def find_case_dir(runs_dir: Path, task: str) -> Path:
    matches = sorted(
        path
        for path in runs_dir.glob(f"newbench-*-{task}-*")
        if path.is_dir()
    )
    if not matches:
        raise FileNotFoundError(f"no case run for {task} under {runs_dir}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def build_environment(*, allow_proxy: bool) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CLAW_BENCH_LIB_DIR"] = str(LIB_DIR)
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(LIB_DIR), environment.get("PYTHONPATH", ""))
        if part
    )
    if not allow_proxy:
        for key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "http_proxy",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
            "CLAW_YTDLP_PROXY",
        ):
            environment.pop(key, None)
    return environment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_IDS)
    parser.add_argument("--list", action="store_true", help="List released task IDs")
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--grade-dir", type=Path)
    parser.add_argument("--grade-name", default=DEFAULT_GRADE_NAME)
    parser.add_argument("--grader-python", default=os.environ.get("CLAW_BENCH_GRADER_PYTHON", sys.executable))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--allow-proxy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        print("\n".join(TASK_IDS))
        return 0
    if not args.task:
        raise SystemExit("--task is required unless --list is used")
    if not args.benchmark_root:
        raise SystemExit("--benchmark-root is required")

    benchmark_root = args.benchmark_root.expanduser().resolve()
    case_dir = args.case_dir.expanduser().resolve() if args.case_dir else None
    if case_dir is None and args.runs_dir:
        case_dir = find_case_dir(args.runs_dir.expanduser().resolve(), args.task)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else case_dir / "workspace" / "output"
        if case_dir
        else None
    )
    if output_dir is None:
        raise SystemExit("provide --case-dir, --runs-dir, or --output-dir")
    if not output_dir.is_dir():
        raise SystemExit(f"missing output directory: {output_dir}")

    grade_dir = (
        args.grade_dir.expanduser().resolve()
        if args.grade_dir
        else case_dir / "grades" / args.grade_name
        if case_dir
        else output_dir.parent / "grades" / args.grade_name
    )
    # Several graders write --report before they print their final JSON. The
    # runner must create this directory before launching the subprocess.
    grade_dir.mkdir(parents=True, exist_ok=True)
    command, missing_flags = build_grader_command(
        args.task,
        benchmark_root=benchmark_root,
        output_dir=output_dir,
        grade_dir=grade_dir,
        grader_python=args.grader_python,
    )
    print(f"[terminalbench-o] task={args.task}")
    print(f"[terminalbench-o] output_dir={output_dir}")
    print(f"[terminalbench-o] grade_dir={grade_dir}")
    print("+", shlex.join(command), flush=True)
    if missing_flags:
        print(
            "unresolved required grader arguments: " + ", ".join(missing_flags),
            file=sys.stderr,
        )
        return 2
    if args.dry_run:
        return 0

    environment = build_environment(allow_proxy=args.allow_proxy)
    try:
        process = subprocess.run(
            command,
            cwd=TASKS_DIR / args.task,
            text=True,
            capture_output=True,
            env=environment,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        write_grade(
            grade_dir,
            command=command,
            returncode=None,
            stdout=error.stdout or "",
            stderr=error.stderr or f"timeout after {args.timeout}s",
            passed=False,
            missing_flags=[],
            status="timeout",
        )
        print(f"grader timeout after {args.timeout}s", file=sys.stderr)
        return 124

    passed = parse_grader_pass(process.stdout, process.stderr)
    if passed is None and process.returncode != 0:
        passed = False
    write_grade(
        grade_dir,
        command=command,
        returncode=process.returncode,
        stdout=process.stdout,
        stderr=process.stderr,
        passed=passed,
        missing_flags=[],
        status="completed",
    )
    if process.stdout:
        print(process.stdout, end="")
    if process.stderr:
        print(process.stderr, end="", file=sys.stderr)
    print(f"[terminalbench-o] saved={grade_dir / 'grade.json'}")
    print(f"[terminalbench-o] passed={passed} returncode={process.returncode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
