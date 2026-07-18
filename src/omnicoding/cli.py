"""Unified launcher for the public benchmark harnesses."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

from omnicoding.benchmarks import specs

HARNESS_MODULES = {
    "claude": "omnicoding.harnesses.claude",
    "codex": "omnicoding.harnesses.codex",
    "kira": "omnicoding.harnesses.kira",
}


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        HARNESS_MODULES[args.harness],
        "--bench",
        args.bench,
        "--model_name",
        args.model,
        "--input_file",
        args.input_file,
        "--dataset_root",
        args.dataset_root,
        "--output_dir",
        args.output_dir,
    ]
    extra = list(args.harness_args)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    return command + extra


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, choices=sorted(HARNESS_MODULES))
    parser.add_argument("--bench", required=True, choices=specs.names())
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without running the harness.",
    )
    parser.add_argument(
        "harness_args",
        nargs=argparse.REMAINDER,
        help="Additional harness-specific flags after a `--` separator.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_command(args)
    if args.dry_run:
        print(shlex.join(command))
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
