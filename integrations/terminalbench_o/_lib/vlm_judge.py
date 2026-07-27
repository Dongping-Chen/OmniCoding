"""Generic VLM-as-judge helper for Claw-Bench graders.

Returns a dict { "score": 0..1, "reasons": "..." }.
The judge model defaults to CLAW_BENCH_JUDGE_MODEL or anthropic/claude-haiku-4-5
(multimodal, cheap, no BYOK dependency).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from openrouter_client import chat_json, vision_json

JUDGE_MODEL = os.environ.get("CLAW_BENCH_JUDGE_MODEL", "anthropic/claude-haiku-4-5")
JUDGE_API_KEY = os.environ.get("CLAW_BENCH_JUDGE_API_KEY")
JUDGE_BASE_URL = os.environ.get("CLAW_BENCH_JUDGE_BASE_URL")


def _judge_client_kwargs() -> dict:
    kwargs = {}
    if JUDGE_API_KEY:
        kwargs["api_key"] = JUDGE_API_KEY
    if JUDGE_BASE_URL:
        kwargs["base_url"] = JUDGE_BASE_URL
    return kwargs


def judge_text(rubric: str, content: str, max_tokens: int = 600) -> dict:
    """Judge a piece of text content against a rubric. Returns {score, reasons}."""
    prompt = (
        "You are an evaluation judge. Score the CONTENT against the RUBRIC.\n"
        "Return strict JSON: {\"score\": <0..1 float>, \"reasons\": \"<1-3 sentences>\"}.\n\n"
        f"RUBRIC:\n{rubric}\n\n"
        f"CONTENT:\n{content}\n"
    )
    return chat_json(
        [{"role": "user", "content": prompt}],
        model=JUDGE_MODEL,
        max_tokens=max_tokens,
        temperature=0.0,
        **_judge_client_kwargs(),
    )


def judge_memo_grounded(
    memo_path,
    rubric: str,
    ground_data: str,
    threshold: float = 0.55,
    memo_char_limit: int = 4000,
    ground_char_limit: int = 4000,
) -> dict:
    """Generic helper for "memo grounded against task data" graders.

    Reads `memo_path`, packages it with `ground_data` (typically a JSON dump
    of the agent's CSVs / GT structure to anchor the rubric on facts), sends
    to the LLM judge, and returns a dict ready to drop into a grader's
    dimension result:
        {"score": float 0..1, "reasons": str, "ok": bool}.

    Honours `CLAW_BENCH_SKIP_VLM=1` to short-circuit during dev. Failures
    return `{"skipped": True, ..., "ok": False}` so the grader still produces
    a structured result rather than crashing.
    """
    import os as _os
    from pathlib import Path as _Path

    if _os.environ.get("CLAW_BENCH_SKIP_VLM") == "1":
        return {"skipped": True, "score": 0.0, "ok": False,
                "reasons": "CLAW_BENCH_SKIP_VLM=1"}
    p = _Path(memo_path)
    if not p.exists():
        return {"skipped": True, "score": 0.0, "ok": False,
                "reasons": f"missing {p.name}"}
    try:
        memo = p.read_text()
    except Exception as e:
        return {"skipped": True, "score": 0.0, "ok": False,
                "reasons": f"read failed: {e}"}
    memo = memo[:memo_char_limit]
    ground_blob = (ground_data or "")[:ground_char_limit]
    content = (
        f"GROUND_DATA (from agent's structured output / fixtures — treat as facts):\n"
        f"{ground_blob}\n\n"
        f"MEMO ({p.name}):\n{memo}\n"
    )
    full_rubric = (
        f"{rubric}\n\n"
        "Anti-cheat: if the memo is mostly placeholder / lorem ipsum / vague "
        "filler / contradicts the GROUND_DATA / ignores task-specific facts, "
        "score ≤ 0.30. If sections are simply missing, deduct ~0.20 each."
    )
    try:
        res = judge_text(full_rubric, content, max_tokens=400)
    except Exception as e:
        return {"skipped": True, "score": 0.0, "ok": False,
                "reasons": f"judge call failed: {str(e)[:200]}"}
    score = float(res.get("score", 0.0))
    return {
        "score": round(score, 3),
        "reasons": str(res.get("reasons", ""))[:400],
        "ok": score >= threshold,
        "threshold": threshold,
    }


def judge_image(rubric: str, image_path: str | Path, max_tokens: int = 600) -> dict:
    prompt = (
        "You are an evaluation judge. Score the IMAGE against the RUBRIC.\n"
        "Return strict JSON: {\"score\": <0..1 float>, \"reasons\": \"<1-3 sentences>\"}.\n\n"
        f"RUBRIC:\n{rubric}\n"
    )
    return vision_json(
        prompt,
        [image_path],
        model=JUDGE_MODEL,
        max_tokens=max_tokens,
        temperature=0.0,
        **_judge_client_kwargs(),
    )


def judge_text_with_images(
    rubric: str,
    text: str,
    images: Iterable[str | Path],
    max_tokens: int = 600,
) -> dict:
    prompt = (
        "You are an evaluation judge. Score the CONTENT (text below + attached images) "
        "against the RUBRIC.\n"
        "Return strict JSON: {\"score\": <0..1 float>, \"reasons\": \"<1-3 sentences>\"}.\n\n"
        f"RUBRIC:\n{rubric}\n\n"
        f"CONTENT TEXT:\n{text}\n"
    )
    return vision_json(
        prompt,
        images,
        model=JUDGE_MODEL,
        max_tokens=max_tokens,
        temperature=0.0,
        **_judge_client_kwargs(),
    )
