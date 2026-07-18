#!/usr/bin/env python3
"""Rebuild the exact benchmark subsets used by OmniCoding evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _load_recipes(path: Any) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recipes = payload.get("subsets")
    if not isinstance(recipes, dict):
        raise ValueError(f"{path} must contain an object named 'subsets'")
    return recipes


def _default_recipes():
    return resources.files("omnicoding.benchmarks").joinpath("eval_subsets.json")


def _rows(payload: Any, wrapper_field: str | None) -> list[dict[str, Any]]:
    rows = payload.get(wrapper_field) if wrapper_field else payload
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        location = f"['{wrapper_field}']" if wrapper_field else ""
        raise ValueError(f"input JSON{location} must be a list of objects")
    return rows


def _render_subset(payload: Any, recipe: dict[str, Any]) -> tuple[bytes, int]:
    wrapper_field = recipe.get("wrapper_field")
    rows = _rows(payload, wrapper_field)
    expected_source_count = recipe.get("expected_source_count")
    if expected_source_count is not None and len(rows) != expected_source_count:
        raise ValueError(
            f"source row count is {len(rows)}, expected {expected_source_count}"
        )

    selection = recipe.get("selection")
    if selection == "all":
        selected = rows
    elif selection == "first_n":
        count = recipe.get("count")
        if not isinstance(count, int) or count < 1:
            raise ValueError("first_n selection requires a positive integer 'count'")
        selected = rows[:count]
    else:
        raise ValueError(f"unsupported selection: {selection!r}")

    expected_count = recipe.get("expected_count")
    if expected_count is not None and len(selected) != expected_count:
        raise ValueError(
            f"selected row count is {len(selected)}, expected {expected_count}"
        )

    id_field = recipe.get("id_field")
    if id_field:
        identifiers = [row.get(id_field) for row in selected]
        if any(identifier is None for identifier in identifiers):
            raise ValueError(f"selected rows must all contain {id_field!r}")
        if len(set(map(str, identifiers))) != len(identifiers):
            raise ValueError(f"selected {id_field!r} values must be unique")

    if wrapper_field:
        output_payload = dict(payload)
        output_payload[wrapper_field] = selected
        count_field = recipe.get("count_field")
        if count_field:
            output_payload[count_field] = len(selected)
    else:
        output_payload = selected

    rendered = json.dumps(
        output_payload, ensure_ascii=False, indent=2
    ).encode("utf-8")
    return rendered, len(selected)


def prepare_subset(
    recipes_path: Any,
    name: str,
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    recipes = _load_recipes(recipes_path)
    if name not in recipes:
        raise ValueError(f"unknown subset {name!r}; choose from {sorted(recipes)}")
    recipe = recipes[name]

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.resolve() == input_path.resolve():
        raise ValueError("output must differ from the upstream input file")
    source_sha256 = sha256_file(input_path)
    expected_source_sha256 = recipe.get("source_sha256")
    if expected_source_sha256 and source_sha256 != expected_source_sha256:
        raise ValueError(
            f"source SHA256 is {source_sha256}, expected {expected_source_sha256}"
        )

    source_bytes = input_path.read_bytes()
    payload = json.loads(source_bytes.decode("utf-8"))
    rendered, count = _render_subset(payload, recipe)
    if (
        recipe.get("selection") == "all"
        and not recipe.get("wrapper_field")
        and not recipe.get("count_field")
    ):
        # A fixed official evaluation file is already the desired manifest.
        # Preserve its bytes instead of introducing formatting-only drift.
        rendered = source_bytes
    output_sha256 = sha256_bytes(rendered)
    expected_output_sha256 = recipe.get("output_sha256")
    if expected_output_sha256 and output_sha256 != expected_output_sha256:
        raise ValueError(
            f"output SHA256 is {output_sha256}, expected {expected_output_sha256}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(rendered)
    return {
        "name": name,
        "rows": count,
        "source_sha256": source_sha256,
        "output_sha256": output_sha256,
        "output": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipes",
        type=Path,
        help="Subset recipe file; defaults to the copy packaged with OmniCoding",
    )
    parser.add_argument("--name", help="Subset recipe name")
    parser.add_argument("--input", type=Path, help="Downloaded upstream metadata file")
    parser.add_argument("--output", type=Path, help="Where to write the exact subset")
    parser.add_argument("--list", action="store_true", help="List available subset recipes")
    args = parser.parse_args()
    recipes_path = args.recipes or _default_recipes()

    if args.list:
        for name, recipe in sorted(_load_recipes(recipes_path).items()):
            print(f"{name}\t{recipe['repo_id']}@{recipe['revision']}:{recipe['source_file']}")
        return 0
    if not args.name or not args.input or not args.output:
        parser.error("--name, --input, and --output are required unless --list is used")

    try:
        report = prepare_subset(recipes_path, args.name, args.input, args.output)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
