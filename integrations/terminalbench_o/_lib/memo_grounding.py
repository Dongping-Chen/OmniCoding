"""Hard-fact regex grounding for free-form memo deliverables.

Why this module exists:
  Memo / summary / report .md files are graded by `axes_yes_no_score`,
  which sends the memo text + GROUND_DATA to an LLM judge. The LLM is
  lenient about numeric mismatches: a memo can claim "the total burn
  area is 320 km²" when the agent's CSV says 47 km², and the judge
  still answers "yes" on the "total cited within ±5%" axis if the
  prose looks plausible.

  This caused the H<L ranking inversions on T79a / T51a / T104b: a
  worse model wrote a more confident memo and won the prose dim.

  This module gives graders a deterministic counterpart: pull numbers
  out of the memo by regex, check them against the agent's own
  structured output (CSV row counts, total_km2 from a CSV, etc.).
  Failed claims drop the dim score; passed claims keep it at 1.

Public API:
  - extract_first_number(text, pattern, group=1) -> float | None
  - check_numeric_claim(text, *, pattern, expected, tolerance, ...) -> dict
  - memo_factual_consistency(text, claims) -> dict
        Aggregate a list of claims into {n_pass, n_total, ratio, ok,
        details, n_unmatched}. Drop-in for a grader dim.

Conventions:
  - Failure modes ("memo doesn't mention the number at all" vs "memo
    states a wrong number") are reported separately. Both reduce the
    dim score, but the second is the strong negative signal.
  - tolerance_pct = 0.05 means within ±5% of expected.
  - Use absolute=True for non-percentage tolerances (e.g. word counts
    where ±10 is fine but ±5% would be too tight).
"""
from __future__ import annotations

import re
from typing import Iterable

# Match any number including k/M/km/km^2/km² suffixes — caller's regex
# pattern decides what context anchors it. We just provide the building
# blocks.
_NUMBER_RE = re.compile(
    r"[-+]?\d{1,3}(?:[, ]\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?|"
    r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
)


def _to_float(s: str) -> float | None:
    s = s.replace(",", "").replace(" ", "").strip()
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def extract_first_number(text: str, pattern: str,
                         group: int = 1) -> float | None:
    """Run `pattern` (must include a numeric capture group) against `text`
    case-insensitively across multilines. Return the first match's group
    converted to float, or None if no match / unparseable.

    Example pattern for "total burn area: 47.3 km^2":
        r'total\\s+burn\\s+area[^0-9]{0,30}([0-9][\\d.,]*)'
    """
    if not text or not pattern:
        return None
    try:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    except re.error:
        return None
    if not m:
        return None
    try:
        raw = m.group(group)
    except IndexError:
        return None
    return _to_float(raw)


def check_numeric_claim(
    text: str, *,
    pattern: str,
    expected: float | int | None,
    tolerance: float = 0.05,
    absolute: bool = False,
    group: int = 1,
) -> dict:
    """Verify a single numeric claim in a memo.

    Returns:
        {pass: bool, found: float|None, expected: float|None,
         delta: float|None, reason: str}

    `pass` is True iff a number was extracted AND it lies within
    tolerance of expected. If expected is None (caller has no GT
    number), `pass` is True iff a number was extracted at all (used
    for "memo must mention X" checks where the value can't be verified).
    """
    if expected is not None and not isinstance(expected, (int, float)):
        return {"pass": False, "found": None, "expected": expected,
                "reason": "expected is not numeric"}
    found = extract_first_number(text or "", pattern, group=group)
    if found is None:
        return {"pass": False, "found": None, "expected": expected,
                "delta": None,
                "reason": "no numeric match in memo for pattern"}
    if expected is None:
        return {"pass": True, "found": found, "expected": None,
                "delta": None, "reason": "value extracted (no GT to compare)"}
    delta = found - float(expected)
    if absolute:
        ok = abs(delta) <= float(tolerance)
        rel = None
    else:
        denom = max(1e-9, abs(float(expected)))
        rel = abs(delta) / denom
        ok = rel <= float(tolerance)
    return {
        "pass": bool(ok),
        "found": found,
        "expected": float(expected),
        "delta": round(delta, 4),
        "rel_error": (round(rel, 4) if rel is not None else None),
        "reason": ("within tolerance" if ok
                   else f"|Δ|={abs(delta):.4g} exceeds tol={tolerance}"
                        + ("" if absolute else " (relative)")),
    }


def memo_factual_consistency(
    text: str,
    claims: Iterable[dict],
) -> dict:
    """Run a list of numeric claim checks, aggregate to a dim result.

    Each `claim` is a dict with keys:
        name           required, label for the claim
        pattern        required, regex with at least one numeric group
        expected       required for hard-check; pass None for "must
                       mention any number for X" semantics
        tolerance      default 0.05
        absolute       default False (relative tolerance)
        group          default 1
        weight         default 1.0 (claim contribution weight)

    Returns:
        {ratio, score, ok, n_total, n_pass, n_no_match, details:
         [...per-claim records]}
    `score` is the weighted-pass ratio. `ok` is True iff every claim
    with a non-None expected passes (strict).
    """
    claims = list(claims)
    if not claims:
        return {"ratio": 1.0, "score": 1.0, "ok": True,
                "n_total": 0, "n_pass": 0, "n_no_match": 0, "details": []}

    details: list[dict] = []
    total_w = 0.0
    pass_w = 0.0
    n_no_match = 0
    n_strict_total = 0
    n_strict_pass = 0
    for claim in claims:
        name = str(claim.get("name", "?"))
        weight = float(claim.get("weight", 1.0))
        rec = check_numeric_claim(
            text,
            pattern=str(claim.get("pattern", "")),
            expected=claim.get("expected"),
            tolerance=float(claim.get("tolerance", 0.05)),
            absolute=bool(claim.get("absolute", False)),
            group=int(claim.get("group", 1)),
        )
        rec["name"] = name
        rec["weight"] = weight
        details.append(rec)
        total_w += weight
        if rec["pass"]:
            pass_w += weight
        else:
            if rec.get("found") is None:
                n_no_match += 1
        # strictness counter (only claims with a real expected value)
        if claim.get("expected") is not None:
            n_strict_total += 1
            if rec["pass"]:
                n_strict_pass += 1

    ratio = pass_w / max(1e-9, total_w)
    return {
        "ratio": round(ratio, 3),
        "score": round(ratio, 3),
        "ok": (n_strict_total == 0 or n_strict_pass == n_strict_total),
        "n_total": len(claims),
        "n_pass": sum(1 for d in details if d["pass"]),
        "n_no_match": n_no_match,
        "details": details,
    }
