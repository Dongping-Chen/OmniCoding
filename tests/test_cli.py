from __future__ import annotations

from omnicoding.cli import build_command, parse_args


def test_unified_launcher_builds_kira_command() -> None:
    args = parse_args(
        [
            "--harness",
            "kira",
            "--bench",
            "omnigaia",
            "--model",
            "openai/test-model",
            "--input-file",
            "items.json",
            "--dataset-root",
            "media",
            "--output-dir",
            "outputs/smoke",
            "--",
            "--max_items",
            "8",
        ]
    )

    command = build_command(args)
    assert command[1:3] == ["-m", "omnicoding.harnesses.kira"]
    assert command[-2:] == ["--max_items", "8"]
    assert command[command.index("--bench") + 1] == "omnigaia"


def test_unified_launcher_supports_every_public_harness() -> None:
    for harness in ("claude", "codex", "kira"):
        args = parse_args(
            [
                "--harness",
                harness,
                "--bench",
                "videozerobench",
                "--model",
                "model",
                "--input-file",
                "items.json",
                "--dataset-root",
                "media",
                "--output-dir",
                "outputs",
            ]
        )
        assert harness in build_command(args)[2]
