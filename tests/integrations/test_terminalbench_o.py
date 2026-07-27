from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path("integrations/terminalbench_o")
TOKEN_RE = re.compile(
    r"(?:hf_[A-Za-z0-9]{20,}|ms-[A-Za-z0-9-]{20,}|sk-[A-Za-z0-9_-]{20,})"
)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evaluator() -> ModuleType:
    return _load_module("terminalbench_o_evaluate", ROOT / "evaluate.py")


def test_release_has_exactly_50_graders_and_no_private_artifacts(
    evaluator: ModuleType,
) -> None:
    task_ids = evaluator.TASK_IDS
    graders = sorted(ROOT.glob("tasks/*/grader.py"))
    assert len(task_ids) == 50
    assert len(set(task_ids)) == 50
    assert [path.parent.name for path in graders] == sorted(task_ids)

    for path in ROOT.rglob("*"):
        assert not path.is_symlink()
        assert path.name != ".env"
        assert "cookie" not in path.name.lower()
        if not path.is_file() or path.suffix == ".pyc":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        assert TOKEN_RE.search(text) is None
        for private_prefix in (
            "/fs/cml-projects/",
            "/nfshomes/dongping/",
            "/media/sata3/",
        ):
            assert private_prefix not in text

    openrouter_client = (ROOT / "_lib/openrouter_client.py").read_text(
        encoding="utf-8"
    )
    assert "_load_dotenv" not in openrouter_client
    assert 'parent / ".env"' not in openrouter_client


def test_argument_parser_handles_multiline_flags_and_boolean_switches(
    evaluator: ModuleType,
    tmp_path: Path,
) -> None:
    grader = tmp_path / "grader.py"
    grader.write_text(
        """
parser.add_argument(
    "--reference-dir",
    type=Path,
    required=True,
)
parser.add_argument("--strict", action="store_true")
parser.add_argument("--output", required=True)
""",
        encoding="utf-8",
    )
    assert evaluator.parse_grader_args(grader) == ["--reference-dir", "--output"]


def test_build_command_keeps_public_grader_separate_from_private_data(
    evaluator: ModuleType,
    tmp_path: Path,
) -> None:
    benchmark_root = tmp_path / "private-benchmark"
    task_dir = benchmark_root / "tasks" / "T01a_soccer_highlights"
    fixtures = task_dir / "fixtures"
    reference = task_dir / "reference"
    output = tmp_path / "case" / "workspace" / "output"
    grade_dir = tmp_path / "case" / "grades" / "release-check"
    fixtures.mkdir(parents=True)
    reference.mkdir()
    output.mkdir(parents=True)
    (fixtures / "source.mp4").write_bytes(b"fixture")
    (reference / "gt.json").write_text("{}", encoding="utf-8")
    (reference / "target_meta.json").write_text("{}", encoding="utf-8")

    command, missing = evaluator.build_grader_command(
        "T01a_soccer_highlights",
        benchmark_root=benchmark_root,
        output_dir=output,
        grade_dir=grade_dir,
        grader_python="python",
    )

    assert missing == []
    assert command[0] == "python"
    assert Path(command[1]) == ROOT.resolve() / "tasks/T01a_soccer_highlights/grader.py"
    assert command[command.index("--output") + 1] == str(output)
    assert command[command.index("--src") + 1] == str(fixtures / "source.mp4")
    assert command[command.index("--gt") + 1] == str(reference / "gt.json")
    assert command[command.index("--target-meta") + 1] == str(
        reference / "target_meta.json"
    )

    assert evaluator.main(
        [
            "--task",
            "T01a_soccer_highlights",
            "--benchmark-root",
            str(benchmark_root),
            "--output-dir",
            str(output),
            "--grade-dir",
            str(grade_dir),
            "--grader-python",
            "python",
            "--dry-run",
        ]
    ) == 0
    assert grade_dir.is_dir()


def test_all_released_grader_flags_have_a_resolution_strategy(
    evaluator: ModuleType,
) -> None:
    path_flags = {
        "--audio",
        "--cases",
        "--chapters_gt",
        "--clips_dir",
        "--covers",
        "--diar_gt",
        "--fixtures",
        "--fixtures-dir",
        "--fixtures-forms",
        "--glossary",
        "--gt",
        "--ingr_gt",
        "--keywords_gt",
        "--label_defs",
        "--output",
        "--output-dir",
        "--reference-dir",
        "--report",
        "--segments",
        "--sources",
        "--src",
        "--steps_gt",
        "--target-meta",
        "--transcript",
        "--uploads",
        "--video",
    }
    seen = set()
    for task_id in evaluator.TASK_IDS:
        flags = evaluator.parse_grader_args(
            ROOT / "tasks" / task_id / "grader.py"
        )
        seen.update(flags)
        assert set(flags) <= path_flags
    assert "--output" in seen
    assert "--output-dir" in seen


def test_parse_pass_and_summary_are_stable(
    evaluator: ModuleType,
    tmp_path: Path,
) -> None:
    assert evaluator.parse_grader_pass('{"pass": true, "score": 0.8}', "") is True
    assert evaluator.parse_grader_pass("", "Overall: FAIL") is False
    assert evaluator.parse_grader_pass("not-json", "") is None

    summary_module = _load_module(
        "terminalbench_o_summarize",
        ROOT / "summarize.py",
    )
    case = tmp_path / "newbench-codex-T01a_soccer_highlights-run"
    grade_dir = case / "grades" / "release-check"
    grade_dir.mkdir(parents=True)
    (grade_dir / "grade.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "returncode": 0,
                "passed": True,
                "score": 0.8,
            }
        ),
        encoding="utf-8",
    )
    summary = summary_module.build_summary(tmp_path, "release-check")
    assert summary["total"] == 1
    assert summary["graded"] == 1
    assert summary["passed"] == 1
    assert summary["failed"] == 0
    assert summary["errors"] == 0
    assert summary["mean_score"] == pytest.approx(0.8)


def test_metadata_export_is_metadata_only(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    exporter = _load_module(
        "terminalbench_o_export_metadata",
        ROOT / "export_metadata.py",
    )
    source_root = tmp_path / "private-source"
    for index, task_id in enumerate(exporter.EXPECTED_TASK_IDS):
        task_dir = source_root / "tasks" / task_id
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text(
            f"id: {task_id}\ntitle: Test {index}\nprompt: Do task {index}\n",
            encoding="utf-8",
        )
    first_task = source_root / "tasks" / exporter.EXPECTED_TASK_IDS[0]
    (first_task / "fixtures").mkdir()
    (first_task / "fixtures/input.mp4").write_bytes(b"raw media")
    (first_task / "reference").mkdir()
    (first_task / "reference/gt.json").write_text("{}", encoding="utf-8")

    output_dir = tmp_path / "public-dataset"
    manifest = exporter.build_dataset(source_root, output_dir)

    assert manifest["task_count"] == 50
    assert manifest["fixture_file_count"] == 1
    assert manifest["reference_file_count"] == 1
    assert manifest["contains_raw_media"] is False
    assert manifest["contains_hidden_references"] is False
    assert len((output_dir / "data/tasks.jsonl").read_text().splitlines()) == 50
    assert len(list((output_dir / "tasks").glob("*/task.yaml"))) == 50
    assert not list(output_dir.rglob("*.mp4"))
    assert not list(output_dir.rglob("gt.json"))
    assert yaml.safe_load(
        (output_dir / "tasks" / exporter.EXPECTED_TASK_IDS[0] / "task.yaml")
        .read_text(encoding="utf-8")
    )["id"] == exporter.EXPECTED_TASK_IDS[0]

    unsafe_output = tmp_path / "unsafe-output"
    unsafe_output.mkdir()
    (unsafe_output / "stale-private-media.mp4").write_bytes(b"do not publish")
    with pytest.raises(ValueError, match="output directory must be empty"):
        exporter.build_dataset(source_root, unsafe_output)


def test_slurm_copy_resolves_integration_from_submit_directory(
    tmp_path: Path,
) -> None:
    spooled_script = tmp_path / "slurm_script"
    shutil.copyfile(ROOT / "eval_array.sbatch", spooled_script)
    fake_python = tmp_path / "python"
    capture = tmp_path / "arguments.json"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "open(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    subprocess.run(
        ["bash", str(spooled_script)],
        check=True,
        env={
            **os.environ,
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_SUBMIT_DIR": str(Path.cwd()),
            "BENCHMARK_ROOT": str(tmp_path / "benchmark"),
            "RUNS_DIR": str(tmp_path / "runs"),
            "GRADER_PYTHON": str(fake_python),
            "CAPTURE": str(capture),
        },
    )

    arguments = json.loads(capture.read_text(encoding="utf-8"))
    assert Path(arguments[0]) == ROOT.resolve() / "evaluate.py"
    assert arguments[arguments.index("--task") + 1] == "T01a_soccer_highlights"
