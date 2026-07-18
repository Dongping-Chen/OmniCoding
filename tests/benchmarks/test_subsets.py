from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from omnicoding.benchmarks.subsets import _default_recipes, main, prepare_subset


def _write_recipes(path: Path, recipe: dict) -> None:
    path.write_text(json.dumps({"subsets": {"test": recipe}}), encoding="utf-8")


def test_prepare_first_n_list_and_verify_digest(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps([{"id": 1}, {"id": 2}, {"id": 3}]), encoding="utf-8")
    expected = json.dumps([{"id": 1}, {"id": 2}], indent=2).encode()
    recipes = tmp_path / "recipes.json"
    _write_recipes(
        recipes,
        {
            "selection": "first_n",
            "count": 2,
            "expected_source_count": 3,
            "expected_count": 2,
            "id_field": "id",
            "output_sha256": hashlib.sha256(expected).hexdigest(),
        },
    )

    output = tmp_path / "nested" / "subset.json"
    report = prepare_subset(recipes, "test", source, output)

    assert output.read_bytes() == expected
    assert report["rows"] == 2


def test_prepare_wrapped_subset_updates_count(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps({"name": "x", "total": 3, "data": [{"qid": 1}, {"qid": 2}, {"qid": 3}]}),
        encoding="utf-8",
    )
    recipes = tmp_path / "recipes.json"
    _write_recipes(
        recipes,
        {
            "selection": "first_n",
            "count": 2,
            "wrapper_field": "data",
            "count_field": "total",
            "expected_count": 2,
            "id_field": "qid",
        },
    )

    output = tmp_path / "subset.json"
    prepare_subset(recipes, "test", source, output)

    assert json.loads(output.read_text()) == {
        "name": "x",
        "total": 2,
        "data": [{"qid": 1}, {"qid": 2}],
    }


def test_prepare_rejects_source_digest_mismatch_before_write(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('[{"id": 1}]', encoding="utf-8")
    recipes = tmp_path / "recipes.json"
    _write_recipes(
        recipes,
        {
            "selection": "all",
            "id_field": "id",
            "source_sha256": "0" * 64,
        },
    )
    output = tmp_path / "subset.json"

    with pytest.raises(ValueError, match="source SHA256"):
        prepare_subset(recipes, "test", source, output)

    assert not output.exists()


def test_prepare_rejects_duplicate_ids(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('[{"id": 1}, {"id": 1}]', encoding="utf-8")
    recipes = tmp_path / "recipes.json"
    _write_recipes(recipes, {"selection": "all", "id_field": "id"})

    with pytest.raises(ValueError, match="unique"):
        prepare_subset(recipes, "test", source, tmp_path / "subset.json")


def test_prepare_all_preserves_official_file_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('[ { "id" : 1 } ]\n', encoding="utf-8")
    recipes = tmp_path / "recipes.json"
    _write_recipes(recipes, {"selection": "all", "id_field": "id"})
    output = tmp_path / "subset.json"

    prepare_subset(recipes, "test", source, output)

    assert output.read_bytes() == source.read_bytes()


def test_prepare_refuses_to_overwrite_upstream_input(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('[{"id": 1}]', encoding="utf-8")
    recipes = tmp_path / "recipes.json"
    _write_recipes(recipes, {"selection": "all", "id_field": "id"})

    with pytest.raises(ValueError, match="must differ"):
        prepare_subset(recipes, "test", source, source)


def test_packaged_default_recipe_is_cwd_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    release_recipe = Path(__file__).parents[2] / "recipes" / "eval_subsets.json"
    assert _default_recipes().read_text(encoding="utf-8") == release_recipe.read_text(
        encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["omnicoding-bench-subset", "--list"])

    assert main() == 0
    assert "omnigaia_full360" in capsys.readouterr().out
