"""Shared format-tolerance helpers for Claw-Bench graders.

Why this module exists:
  Original graders did literal-string matching on filenames, CSV headers, and
  cross-doc identifiers (`'character_name' in cols`, `OPERATOR 1 == OPERATOR 1`).
  Any agent that wrote `Voice-Actor-Guess` instead of `voice_actor_guess`,
  saved `Production-Notes.MD` instead of `production_notes.md`, or used
  `Operator #1` instead of `OPERATOR 1` would be marked as a content error
  even when the underlying answer was correct.

  These helpers normalize identifiers (`_norm`), do fuzzy-but-safe filename
  resolution (`find_file`, `resolve_required`), and load CSVs with normalized
  headers (`load_csv_norm`). Use them at the gating + evaluation boundaries.

Design notes:
  - `_norm` is intentionally aggressive (drops all non-word chars after
    space/hyphen → underscore) so identifiers like 'Operator #1', 'operator 1',
    'OPERATOR-1', 'operator_1' all collapse to 'operator_1'. This is
    appropriate for entity-name comparisons but NOT for free-form prose.
  - `find_file` only treats the .md / .markdown / .txt extension family as
    interchangeable; binary extensions (.mp4 / .png / .csv / .json) must
    match exactly. Otherwise a stray .json could match a request for .csv.
  - `load_csv_norm` returns dicts with normalized keys, so consumer code
    should also use normalized keys when calling `.get()`.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

# Extension families considered interchangeable when fuzzy-matching filenames.
_TEXT_EXTS = {"md", "markdown", "txt"}


def _norm(s) -> str:
    """Canonicalize an identifier: lowercase, strip, hyphen/space -> underscore,
    drop punctuation. Returns "" for None/empty input.

    Examples:
        _norm("Voice-Actor-Guess") == "voice_actor_guess"
        _norm("OPERATOR 1") == "operator_1"
        _norm("Operator #1") == "operator_1"
        _norm("MRS. ADAMS") == "mrs_adams"
    """
    s = str(s or "").strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^\w]+", "", s)
    return s


def find_file(out_dir: Path, target: str) -> Path | None:
    """Locate `target` under `out_dir` with case-insensitive + normalized-stem
    fallback. Returns the actual on-disk Path or None if no match.

    Lookup order:
      1. Exact path: `out_dir / target`.
      2. Same normalized stem AND same extension (case-insensitive).
      3. Same normalized stem AND extension is in the same family
         (.md / .markdown / .txt are interchangeable).
    """
    direct = out_dir / target
    if direct.exists():
        return direct
    if "." in target:
        t_stem, t_ext = target.rsplit(".", 1)
    else:
        t_stem, t_ext = target, ""
    t_ext = t_ext.lower()
    t_stem_n = _norm(t_stem)
    if not out_dir.exists() or not out_dir.is_dir():
        return None
    for p in out_dir.iterdir():
        if not p.is_file():
            continue
        if "." in p.name:
            p_stem, p_ext = p.name.rsplit(".", 1)
        else:
            p_stem, p_ext = p.name, ""
        if _norm(p_stem) != t_stem_n:
            continue
        p_ext = p_ext.lower()
        if p_ext == t_ext:
            return p
        if t_ext in _TEXT_EXTS and p_ext in _TEXT_EXTS:
            return p
    return None


def resolve_required(out_dir: Path, required: list[str]
                     ) -> tuple[list[str], dict[str, Path]]:
    """For each filename in `required`, locate it via `find_file` and check it
    is non-empty. Returns (errors, resolved) where `resolved` maps each
    requested name to its on-disk Path (only for entries with no errors).

    Drop-in replacement for the boilerplate `for f in REQUIRED: if not (out / f).exists(): ...`
    pattern that ~all graders open with.
    """
    errs: list[str] = []
    resolved: dict[str, Path] = {}
    for name in required:
        p = find_file(out_dir, name)
        if p is None:
            errs.append(f"missing: {name}")
        elif p.stat().st_size == 0:
            errs.append(f"empty: {name}")
        else:
            resolved[name] = p
    return errs, resolved


def check_columns(path: Path, required_cols: set[str]) -> list[str]:
    """Return list of normalized column names from `required_cols` that are
    NOT present in `path`'s CSV header (after header normalization). Empty list
    means all required columns are accounted for under some header variant.

    `required_cols` itself is normalized internally so callers can declare
    them in either canonical form (`{"voice_actor_guess"}`).
    """
    with path.open(newline="") as fh:
        cols_norm = {_norm(c) for c in
                     (csv.DictReader(fh).fieldnames or [])}
    return sorted(c for c in required_cols if _norm(c) not in cols_norm)


def errors_to_ratio(n_errors: int, *, max_errors: int) -> float:
    """Convert an integer error count into a continuous score in [0, 1].

    Use this in graders that previously did `1.0 if len(errs)==0 else 0.0`
    on a multi-item validation dim (closed-vocab gates, schema checks).
    The binary form caused ranking inversions where a strong model with one
    OOV mistake scored worse than a weak model with no submission at all.

    `max_errors` is the soft scale at which the dim becomes 0.0 — pick it
    based on the number of independent error sources in the dim (e.g.
    `max(8, len(items)*4)` if each item can independently emit several
    errors). 0 errors → 1.0, `max_errors` → 0.0, linearly interpolated.
    """
    if n_errors <= 0:
        return 1.0
    me = max(1, int(max_errors))
    return max(0.0, 1.0 - n_errors / me)


def load_csv_norm(p: Path) -> list[dict]:
    """Load CSV and normalize every header key. Downstream lookups should use
    normalized keys (`row.get("voice_actor_guess")`) regardless of how the
    agent capitalized / hyphenated them in the source file.

    Empty / unparseable rows are returned as `{}` (caller decides whether to
    drop them); this matches `csv.DictReader` semantics minus the case fragility.
    """
    with p.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        key_map = {f: _norm(f) for f in fields}
        return [{key_map[k]: v for k, v in row.items() if k in key_map}
                for row in reader]
