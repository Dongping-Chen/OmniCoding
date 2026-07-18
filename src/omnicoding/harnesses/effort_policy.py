"""Per-item reasoning_effort policy for the kira → SFT pipeline.

Goal: stop wasting ``high`` reasoning tokens on easy items, and give
hard items the headroom they need. Two-stage strategy:

  Attempt 1 (first run for an item):
    - ``item['Level']`` set explicitly → map by ``LEVEL_TO_EFFORT``.
    - No ``Level`` annotation → random pick from ``UNLABELED_FIRST``
      ({low, medium}). Random choice is seeded by the item's stable id
      so reruns are reproducible.

  Attempt ≥ 2 (failure escalation, dispatcher re-queues failures):
    - Random pick from ``ESCALATION_CHOICES`` ({high, xhigh}). Same
      seeding rule (different attempt seeds → different choice).

Why "random low/medium" for unlabeled instead of always low: the
training signal stays diverse across the SFT dataset; if the model
trains only on "low effort succeeds" trajectories it never sees
medium-effort patterns. Random split forces both into the corpus.

This module is import-safe — no side effects, no I/O. Tested by
the effort-policy regression tests.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any

LEVEL_TO_EFFORT: dict[str, str] = {
    "easy": "low",
    "medium": "medium",
    "hard": "high",
}

UNLABELED_FIRST: tuple[str, ...] = ("low", "medium")
ESCALATION_CHOICES: tuple[str, ...] = ("high", "xhigh")

# Monotonic effort ladder. Used by ``pick_reasoning_effort`` on
# attempt ≥ 2 to guarantee strictly-higher escalation: if attempt-1 was
# ``high``, attempt-2 picks ONLY from {xhigh}; if attempt-1 was already
# ``xhigh``, ``can_escalate`` returns False and the dispatcher should
# skip the item entirely (no point re-running at the same tier).
EFFORT_LADDER: tuple[str, ...] = ("low", "medium", "high", "xhigh")


def can_escalate(prev_effort: str) -> bool:
    """True iff there's a strictly-higher tier above ``prev_effort``.
    Unknown efforts default to True (be permissive — caller will get a
    sensible random choice from ESCALATION_CHOICES)."""
    if prev_effort not in EFFORT_LADDER:
        return True
    return EFFORT_LADDER.index(prev_effort) < len(EFFORT_LADDER) - 1

# Step-limit budget per effort tier. Lighter efforts get fewer steps to
# stop them rambling; heavy efforts get extra runway. Tunable via the
# constants below — exported as a function so callers can override per
# request without monkey-patching the dict.
EFFORT_TO_STEP_LIMIT: dict[str, int] = {
    "low": 40,
    "medium": 80,
    "high": 100,
    "xhigh": 100,
}

# Probability that an item runs in CPU-only mode (sbatch without GPU).
# Trains the model to handle sandboxes that may or may not have GPU,
# matching real production-deploy heterogeneity.
CPU_ONLY_RATE: float = 0.2


def step_limit_for_effort(effort: str, fallback: int = 80) -> int:
    """Step budget for a given effort tier — see EFFORT_TO_STEP_LIMIT."""
    return EFFORT_TO_STEP_LIMIT.get(effort, fallback)


def pick_cpu_only(item: dict[str, Any], rate: float = CPU_ONLY_RATE) -> bool:
    """Decide if this item runs CPU-only (no GPU on sbatch). Seeded by
    item id so a given item flips deterministically across reruns —
    keeps the "20% CPU" mix reproducible."""
    item_id = str(item.get("id") or item.get("__source_index__") or "?")
    seed = int(hashlib.blake2b(
        f"cpu_only|{item_id}".encode("utf-8"), digest_size=8,
    ).hexdigest(), 16)
    return random.Random(seed).random() < rate


def _seeded_rng(item_id: str, attempt: int) -> random.Random:
    """Deterministic RNG keyed by (item_id, attempt). Same seed →
    same effort choice, so dispatcher reruns of the same item resolve
    identically until ``attempt`` advances."""
    seed_bytes = hashlib.blake2b(
        f"{item_id}|{attempt}".encode("utf-8"), digest_size=8,
    ).digest()
    return random.Random(int.from_bytes(seed_bytes, "big"))


def pick_reasoning_effort(
    item: dict[str, Any],
    attempt: int = 1,
    prev_effort: str | None = None,
) -> str:
    """Return the reasoning_effort string ({low, medium, high, xhigh})
    for this (item, attempt).

    On attempt ≥ 2 we **strictly escalate**: pick uniformly at random from
    the tiers above ``prev_effort`` in the monotonic ``EFFORT_LADDER``.
    This avoids the round-17.7 pitfall where a Hard-Level item that
    failed at ``high`` had a 50% chance of getting ``high`` again on
    retry — pure waste. If ``prev_effort`` is already ``xhigh`` (top of
    ladder), there is nothing to escalate to and we return ``xhigh``;
    callers should pre-filter via ``can_escalate(prev_effort)`` and skip
    the rerun entirely.

    When ``prev_effort=None`` we fall back to legacy random-pick from
    ``ESCALATION_CHOICES`` ({high, xhigh}) so older callers/tests keep
    working.
    """
    item_id = str(item.get("id") or item.get("__source_index__") or "?")
    rng = _seeded_rng(item_id, attempt)
    if attempt >= 2:
        if prev_effort is None:
            return rng.choice(ESCALATION_CHOICES)
        if prev_effort not in EFFORT_LADDER:
            return rng.choice(ESCALATION_CHOICES)
        idx = EFFORT_LADDER.index(prev_effort)
        higher = EFFORT_LADDER[idx + 1:]
        if not higher:
            return prev_effort
        return rng.choice(higher)
    level = (item.get("Level") or "").strip().lower()
    if level in LEVEL_TO_EFFORT:
        return LEVEL_TO_EFFORT[level]
    return rng.choice(UNLABELED_FIRST)
