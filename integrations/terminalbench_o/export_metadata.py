#!/usr/bin/env python3
"""Build the metadata-only Hugging Face artifact for TerminalBench-O."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
EXPECTED_TASK_IDS = tuple(
    line.strip()
    for line in (HERE / "task_ids.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
)
PRIVATE_PATH_RE = re.compile(r"(?:^|[\s\"'])(?:/fs/|/nfshomes/|/home/|/root/|/media/sata)")
SECRET_VALUE_RE = re.compile(
    r"(?:hf_[A-Za-z0-9]{20,}|ms-[A-Za-z0-9-]{20,}|sk-[A-Za-z0-9_-]{20,})"
)
SECRET_KEY_RE = re.compile(
    r"(?:api[_-]?key|access[_-]?token|secret|password|credential|cookie)",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_public_value(value: Any, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if SECRET_KEY_RE.search(str(key)):
                raise ValueError(f"credential-shaped metadata key at {path}.{key}")
            validate_public_value(child, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            validate_public_value(child, path=f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return
    if SECRET_VALUE_RE.search(value):
        raise ValueError(f"credential-shaped metadata value at {path}")
    if PRIVATE_PATH_RE.search(value):
        raise ValueError(f"private absolute path at {path}")


def public_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlink is not allowed in metadata inventory: {path}")
        if path.is_file():
            files.append(path)
    return sorted(files)


def file_inventory(root: Path) -> dict[str, Any]:
    files = public_files(root)
    extensions = Counter(path.suffix.lower() or "[none]" for path in files)
    return {
        "count": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "extensions": dict(sorted(extensions.items())),
    }


def normalized_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def task_row(
    task_dir: Path,
    *,
    metadata_task_dir: Path | None = None,
    released_grader: Path,
    yaml_module: Any,
) -> tuple[dict[str, Any], str]:
    task_yaml = (metadata_task_dir or task_dir) / "task.yaml"
    if not task_yaml.is_file():
        raise FileNotFoundError(task_yaml)
    raw_yaml = task_yaml.read_text(encoding="utf-8")
    metadata = yaml_module.safe_load(raw_yaml)
    if not isinstance(metadata, dict):
        raise ValueError(f"{task_yaml} must contain a YAML mapping")

    task_id = task_dir.name
    declared_id = metadata.get("id")
    source_declared_id = str(declared_id) if declared_id is not None else ""
    if source_declared_id != task_id:
        metadata["id"] = task_id
        if re.search(r"(?m)^id:\s*.*$", raw_yaml):
            raw_yaml = re.sub(
                r"(?m)^id:\s*.*$",
                f"id: {task_id}",
                raw_yaml,
                count=1,
            )
        else:
            raw_yaml = f"id: {task_id}\n{raw_yaml}"
    validate_public_value(metadata, path=task_id)
    title = (
        metadata.get("title")
        or metadata.get("name")
        or task_id.split("_", 1)[-1].replace("_", " ").title()
    )
    categories = normalized_list(
        metadata.get("category")
        or metadata.get("categories")
        or metadata.get("domain")
        or metadata.get("task_type")
    )
    modality = normalized_list(metadata.get("modality"))
    timeout_seconds = metadata.get("timeout_seconds")
    if timeout_seconds is None and metadata.get("timeout_minutes") is not None:
        timeout_seconds = int(metadata["timeout_minutes"]) * 60
    if timeout_seconds is None and metadata.get("estimated_time_minutes") is not None:
        timeout_seconds = int(metadata["estimated_time_minutes"]) * 60

    fixture_inventory = file_inventory(task_dir / "fixtures")
    reference_inventory = file_inventory(task_dir / "reference")
    row = {
        "task_id": task_id,
        "source_declared_id": source_declared_id,
        "title": str(title),
        "category": categories,
        "modality": modality,
        "language": str(metadata.get("language") or ""),
        "difficulty": str(metadata.get("difficulty") or ""),
        "timeout_seconds": timeout_seconds,
        "max_turns": metadata.get("max_turns"),
        "prompt": str(metadata.get("prompt") or metadata.get("description") or ""),
        "data_source_json": json.dumps(
            metadata.get("data_source") or {},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        "fixture_file_count": fixture_inventory["count"],
        "fixture_total_bytes": fixture_inventory["bytes"],
        "fixture_extensions_json": json.dumps(
            fixture_inventory["extensions"], sort_keys=True
        ),
        "reference_file_count": reference_inventory["count"],
        "reference_total_bytes": reference_inventory["bytes"],
        "reference_extensions_json": json.dumps(
            reference_inventory["extensions"], sort_keys=True
        ),
        "task_yaml_sha256": sha256_text(raw_yaml),
        "source_task_yaml_sha256": sha256(task_yaml),
        "grader_sha256": sha256(released_grader),
        "release_version": "terminalbench_o_v1_20260507",
    }
    return row, raw_yaml


def dataset_card() -> str:
    return """---
license: other
pretty_name: TerminalBench-O
task_categories:
- other
language:
- en
tags:
- benchmark
- multimodal
- terminal-agent
- agent-evaluation
- audio
- video
configs:
- config_name: default
  data_files:
  - split: test
    path: data/tasks.jsonl
---

# TerminalBench-O

TerminalBench-O is a metadata-only release of 50 multimodal terminal-agent
tasks spanning video, audio, image, document, geospatial, and media-production
workflows.

This repository contains:

- `data/tasks.jsonl`: one normalized metadata row per task;
- `tasks/<task_id>/task.yaml`: the complete public task definition;
- `task_ids.txt`: canonical task order;
- `manifest.json`: hashes and aggregate counts for release verification.

It intentionally does **not** contain raw media, fixtures, hidden references,
ground truth, model outputs, API keys, cookies, or evaluation logs. Aggregate
fixture/reference file counts and byte sizes are included only to make local
packaging audits possible.

The canonical task-directory name is written to each YAML `id` field. The
original declaration, including missing or legacy short IDs, remains recorded
in `source_declared_id`.

The released evaluator and all 50 graders are in
[Dongping-Chen/OmniCoding](https://github.com/Dongping-Chen/OmniCoding/tree/main/integrations/terminalbench_o).
Evaluation requires an authorized private benchmark root containing each
task's `fixtures/` and `reference/` directories.

## License and data notice

The metadata and evaluator are released under the terms of the OmniCoding
repository. Underlying media comes from multiple upstream sources with
different terms and is not redistributed here. Repository availability does
not grant rights to obtain or redistribute the source media.
"""


def build_dataset(
    source_root: Path,
    output_dir: Path,
    *,
    metadata_root: Path | None = None,
) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise SystemExit("PyYAML is required: pip install PyYAML") from error

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            "output directory must be empty to prevent stale private files "
            f"from entering the release: {output_dir}"
        )

    source_tasks = source_root / "tasks"
    discovered = tuple(
        path.name
        for path in sorted(source_tasks.iterdir())
        if path.is_dir() and path.name.startswith("T")
    )
    if discovered != EXPECTED_TASK_IDS:
        raise ValueError(
            "task set mismatch: "
            f"expected={list(EXPECTED_TASK_IDS)!r} discovered={list(discovered)!r}"
        )
    metadata_tasks = (metadata_root or source_root) / "tasks"
    metadata_discovered = tuple(
        path.name
        for path in sorted(metadata_tasks.iterdir())
        if path.is_dir() and path.name.startswith("T")
    )
    if metadata_discovered != EXPECTED_TASK_IDS:
        raise ValueError(
            "metadata task set mismatch: "
            f"expected={list(EXPECTED_TASK_IDS)!r} "
            f"discovered={list(metadata_discovered)!r}"
        )

    data_dir = output_dir / "data"
    tasks_dir = output_dir / "tasks"
    data_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for task_id in EXPECTED_TASK_IDS:
        source_task = source_tasks / task_id
        released_grader = HERE / "tasks" / task_id / "grader.py"
        if not released_grader.is_file():
            raise FileNotFoundError(released_grader)
        row, raw_yaml = task_row(
            source_task,
            metadata_task_dir=metadata_tasks / task_id,
            released_grader=released_grader,
            yaml_module=yaml,
        )
        rows.append(row)
        destination = tasks_dir / task_id
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "task.yaml").write_text(raw_yaml, encoding="utf-8")

    jsonl = data_dir / "tasks.jsonl"
    jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_dir / "task_ids.txt").write_text(
        "".join(f"{task_id}\n" for task_id in EXPECTED_TASK_IDS),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(dataset_card(), encoding="utf-8")
    manifest = {
        "dataset": "shuaishuaicdp/TerminalBench-O",
        "release_version": "terminalbench_o_v1_20260507",
        "task_count": len(rows),
        "tasks_jsonl_sha256": sha256(jsonl),
        "fixture_file_count": sum(row["fixture_file_count"] for row in rows),
        "fixture_total_bytes": sum(row["fixture_total_bytes"] for row in rows),
        "reference_file_count": sum(row["reference_file_count"] for row in rows),
        "reference_total_bytes": sum(row["reference_total_bytes"] for row in rows),
        "contains_raw_media": False,
        "contains_hidden_references": False,
        "contains_credentials": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_public_value(
        {
            path.relative_to(output_dir).as_posix(): path.read_text(
                encoding="utf-8", errors="replace"
            )
            for path in output_dir.rglob("*")
            if path.is_file()
        },
        path="export",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--metadata-root",
        type=Path,
        help="Optional root containing syntax-repaired public task YAML files",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_root = args.source_root.expanduser().resolve()
    metadata_root = (
        args.metadata_root.expanduser().resolve()
        if args.metadata_root
        else None
    )
    output_dir = args.output_dir.expanduser().resolve()
    if (
        source_root == output_dir
        or source_root in output_dir.parents
        or metadata_root == output_dir
        or (metadata_root is not None and metadata_root in output_dir.parents)
    ):
        raise SystemExit("output directory must be outside the private source root")
    manifest = build_dataset(
        source_root,
        output_dir,
        metadata_root=metadata_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
