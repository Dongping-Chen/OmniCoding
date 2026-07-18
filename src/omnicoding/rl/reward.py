"""Reward grader for the multimodal terminal agent.

The production training path grades in two stages:

1. Coordinator computes per-trajectory components: exact-match raw accuracy,
   tool-use statistics, format penalty, modality penalty, bad-tool penalty, and
   timeout masking.
2. Relax group RM applies the group-level weighted-step length penalty before
   dynamic filtering and reward normalization.

Video/audio tool sets are intentionally config-driven and default to empty while
we audit the exact tool universe. Set comma-separated env vars
``RELAX_ROUTER_VIDEO_TOOLS`` / ``RELAX_ROUTER_AUDIO_TOOLS`` /
``RELAX_ROUTER_IMAGE_TOOLS`` to enable modality satisfaction.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Greedy + non-greedy combo: capture the LAST <answer>...</answer> in the text.
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")
# Lowercase + collapse common separator variants used in open-ended answers
# (e.g. "Atlanta, Georgia" / "Atlanta; Georgia" / "Atlanta and Georgia"). The
# MCQ ground_truth from coding-agent-rl/scripts/refine/pass_mcq.py does not
# enumerate these — collapse is a pragmatic pre-filter for open-ended grading.
_SEP_RE = re.compile(r"\s*[,;]\s*|\s+and\s+|\s+")
_RAW_JSON_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_RAW_TOOL_TAG_RE = re.compile(r"<tool_call\b", re.IGNORECASE)
_SYNTAX_FAIL_RE = re.compile(
    r"SyntaxError|invalid syntax|unexpected EOF|parse error|unexpected token|"
    r"missing operand|unterminated string",
    re.IGNORECASE,
)
_NETWORK_RE = re.compile(r"\b(curl|wget|ssh|scp|rsync|nc|telnet)\b|https?://", re.IGNORECASE)
_GOLD_RE = re.compile(
    r"ground_truth|answer_key|gold_answer|\.gold|all_final\.jsonl|"
    r"rl_train\.jsonl|sft_train\.jsonl",
    re.IGNORECASE,
)

DEFAULT_ALLOWED_TOOLS = frozenset({"execute_commands", "image_read", "task_complete"})

# Modality tool universes — substring keywords scanned in execute_commands
# arguments to detect whether the agent actually consumed video / audio / image
# data via shell helpers. Used by ``_detect_configured_subtools`` to attribute
# usage of helpers wrapped inside ``execute_commands(keystrokes="...")``.
#
# Empty defaults are a footgun: with the empty set, ``check_modality`` returns
# False for every video/audio task → gated correctness collapses to 0 →
# ~95% of the dataset gets no positive reward. The lists below are the
# canonical universe (audited 2026-05-04); env vars override per-experiment.
DEFAULT_VIDEO_TOOLS = frozenset({
    "ffmpeg", "ffprobe", "ffplay", "yt-dlp", "mpv", "vlc",
    "VideoCapture", "VideoWriter", "VideoFileClip", "moviepy",
    "decord", "VideoReader", "imageio", "mediainfo",
})
DEFAULT_AUDIO_TOOLS = frozenset({
    "whisper", "faster_whisper", "librosa", "torchaudio",
    "soundfile", "sf.read", "pydub", "AudioSegment",
    "wave.open", "sox", "demucs", "spleeter", "lame",
})
DEFAULT_IMAGE_TOOLS = frozenset({
    "PIL.Image.open", "Image.open", "cv2.imread", "cv2.imwrite",
    "tesseract", "pytesseract", "convert", "magick", "identify", "mogrify",
    "easyocr", "paddleocr", "pdftotext", "pdftoppm", "exiftool", "image_read",
})

FAIL_CATEGORIES = ("unparseable", "disallowed", "escape", "syntax-fail")
TIMEOUT_EXITS = frozenset({"timeout", "step_limit", "context_overflow"})

# Format reward: positive if the trajectory cleanly exits with an answer wrapper,
# negative otherwise. The positive branch (FORMAT_BONUS) gives credit for
# format compliance even when the answer is wrong — keeps gradient signal alive
# on "tried correctly but wrong" trajectories instead of collapsing them to 0.
FORMAT_BONUS = 0.2
FORMAT_PENALTY = -0.2
# Heavier modality miss penalty (was -0.1, bumped 2026-05-04). Video/audio
# tasks that don't use the corresponding tool universe get punished more —
# the model needs to actually consume the modality, not guess from text.
MODALITY_PENALTY = -0.3
BAD_TOOL_WEIGHT = 0.5
LEN_ALPHA = 0.2
LEN_F_MIN = 0.5


@dataclass(slots=True)
class ToolCall:
    name: str | None
    arguments: Any
    malformed: bool = False
    tool_call_id: str | None = None


# ─── extraction ──────────────────────────────────────────────────────────────


def _walk_assistant_content(messages: list[dict[str, Any]]) -> str:
    """Concatenate text from assistant turns only (``content`` +
    ``reasoning_content``). Tool replies are skipped — they may include
    ``cat``-ed prompt-file content that contains stray ``<answer>`` patterns
    we should not treat as the model's answer.
    """
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content)
        elif isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    text = c.get("text") or ""
                    if text.strip():
                        parts.append(text)
        rc = msg.get("reasoning_content")
        if isinstance(rc, str) and rc.strip():
            parts.append(rc)
    return "\n\n".join(parts)


def extract_answer(final_text: str) -> str | None:
    """Pull the LAST ``<answer>...</answer>`` from a flat text blob. Useful for
    standalone strings (e.g. unit tests). Production callers should prefer
    ``extract_answer_from_messages`` to avoid tool-reply pollution."""
    if not final_text:
        return None
    matches = _ANSWER_RE.findall(final_text)
    if not matches:
        return None
    return matches[-1].strip()


def extract_answer_from_messages(messages: list[dict[str, Any]]) -> str | None:
    """Pull the LAST ``<answer>...</answer>`` from the model's assistant turns
    only (skipping tool replies). This is the production extractor."""
    return extract_answer(_walk_assistant_content(messages))


# ─── normalisation ───────────────────────────────────────────────────────────


def normalize(s: str) -> str:
    """Lowercase + collapse separator variants + collapse whitespace."""
    if not s:
        return ""
    s = s.strip().lower()
    s = _SEP_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# ─── tool parsing ────────────────────────────────────────────────────────────


def _split_env_set(name: str, default: set[str] | frozenset[str] | None = None) -> set[str]:
    raw = os.environ.get(name)
    if raw is None:
        return set(default or ())
    return {part.strip() for part in raw.split(",") if part.strip()}


def _allowed_tools() -> set[str]:
    return _split_env_set("RELAX_ROUTER_ALLOWED_TOOLS", DEFAULT_ALLOWED_TOOLS)


def _video_tools() -> set[str]:
    return _split_env_set("RELAX_ROUTER_VIDEO_TOOLS", DEFAULT_VIDEO_TOOLS)


def _audio_tools() -> set[str]:
    return _split_env_set("RELAX_ROUTER_AUDIO_TOOLS", DEFAULT_AUDIO_TOOLS)


def _image_tools() -> set[str]:
    return _split_env_set("RELAX_ROUTER_IMAGE_TOOLS", DEFAULT_IMAGE_TOOLS)


def _parse_arguments(raw: Any) -> tuple[Any, bool]:
    if raw is None:
        return {}, False
    if isinstance(raw, dict):
        return raw, False
    if isinstance(raw, str):
        if not raw.strip():
            return {}, False
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return raw, True
        return decoded, not isinstance(decoded, dict)
    return raw, not isinstance(raw, dict)


def _tool_call_from_payload(payload: dict[str, Any], tool_call_id: str | None = None) -> ToolCall:
    function = payload.get("function") if isinstance(payload.get("function"), dict) else {}
    name = payload.get("name") or function.get("name")
    raw_args = payload.get("arguments")
    if raw_args is None:
        raw_args = function.get("arguments")
    arguments, malformed = _parse_arguments(raw_args)
    return ToolCall(name=name, arguments=arguments, malformed=malformed, tool_call_id=tool_call_id)


def _extract_tool_calls_from_messages(messages: list[dict[str, Any]]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue

        structured = msg.get("tool_calls") or []
        if structured:
            for item in structured:
                if not isinstance(item, dict):
                    calls.append(ToolCall(name=None, arguments=None, malformed=True))
                    continue
                calls.append(_tool_call_from_payload(item, tool_call_id=item.get("id")))
            continue

        content = msg.get("content")
        if not isinstance(content, str) or "<tool_call" not in content:
            continue
        parsed_any = False
        for match in _RAW_JSON_TOOL_RE.finditer(content):
            parsed_any = True
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                calls.append(ToolCall(name=None, arguments=None, malformed=True))
                continue
            calls.append(_tool_call_from_payload(payload))
        if not parsed_any and _RAW_TOOL_TAG_RE.search(content):
            calls.append(ToolCall(name=None, arguments=None, malformed=True))
    return calls


def _flatten_string_values(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            out.extend(_flatten_string_values(value))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            out.extend(_flatten_string_values(value))
    return out


def _string_tokens(value: str) -> list[str]:
    try:
        return shlex.split(value)
    except ValueError:
        return value.split()


def _path_escapes(value: str, workspace: Path | None) -> bool:
    if _GOLD_RE.search(value):
        return True
    if workspace is None:
        return False
    ws = workspace.resolve()
    for token in _string_tokens(value):
        if not token.startswith("/"):
            continue
        try:
            resolved = Path(token).resolve()
        except OSError:
            return True
        if not str(resolved).startswith(str(ws)):
            return True
    return False


def _detect_configured_subtools(arguments: Any) -> set[str]:
    configured = _video_tools() | _audio_tools() | _image_tools()
    if not configured:
        return set()
    blob = " ".join(_flatten_string_values(arguments)).lower()
    return {tool for tool in configured if tool.lower() in blob}


def _tool_outputs_have_syntax_failure(messages: list[dict[str, Any]]) -> int:
    count = 0
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = " ".join(str(item) for item in content)
        if isinstance(content, str) and _SYNTAX_FAIL_RE.search(content):
            count += 1
    return count


def summarize_tool_calls(
    messages: list[dict[str, Any]],
    *,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    calls = _extract_tool_calls_from_messages(messages)
    counts = {cat: 0 for cat in FAIL_CATEGORIES}
    names_used: set[str] = set()
    ws = Path(workspace) if workspace else None
    allowed = _allowed_tools()

    for call in calls:
        if call.name:
            names_used.add(call.name)
        names_used.update(_detect_configured_subtools(call.arguments))

        if call.malformed or not call.name or not isinstance(call.arguments, dict):
            counts["unparseable"] += 1
            continue
        if call.name not in allowed:
            counts["disallowed"] += 1
        flattened = _flatten_string_values(call.arguments)
        if any(_path_escapes(value, ws) for value in flattened):
            counts["escape"] += 1
        if any(_NETWORK_RE.search(value) for value in flattened):
            counts["disallowed"] += 1

    counts["syntax-fail"] += _tool_outputs_have_syntax_failure(messages)
    n_tool = len(calls)
    n_fail = sum(counts.values())
    return {
        "n_tool": n_tool,
        "p_bad_tool": n_fail / (1 + n_tool),
        "names_used": sorted(names_used),
        "fail_counts": counts,
    }


def check_modality(media: dict[str, list[str]] | None, names_used: list[str] | set[str]) -> tuple[bool, bool, bool]:
    media = media or {}
    has_video = bool(media.get("videos"))
    has_audio = bool(media.get("audios"))
    names = set(names_used)
    used_video = bool(names & (_video_tools() | _image_tools()))
    used_audio = bool(names & _audio_tools())

    ok = True
    if has_video and not used_video:
        ok = False
    if has_audio and not used_audio:
        ok = False
    return ok, has_video, has_audio


# ─── grading ─────────────────────────────────────────────────────────────────


def grade_outcome(prediction: str | None, ground_truth: list[str], answer_type: str) -> float:
    """Outcome reward: 1.0 on any normalized exact-match, else 0.0."""
    del answer_type  # reserved for future LLM-judge fallback on `open`
    if prediction is None:
        return 0.0
    p = normalize(prediction)
    if not p:
        return 0.0
    for gt in ground_truth:
        if p == normalize(gt):
            return 1.0
    return 0.0


def grade_format(exit_reason: str | None, prediction: str | None) -> float:
    """Format component: ``FORMAT_BONUS`` (+0.2) when the trajectory exits
    cleanly via ``task_complete`` with a non-empty <answer> wrapper, else
    ``FORMAT_PENALTY`` (-0.2). The bonus does NOT stack with correctness —
    ``grade_trajectory`` picks ``base = 1.0`` when correctness=1 and
    ``base = format_value`` only when correctness=0, so max score is 1.0
    (not 1.2). The bonus exists ONLY to give "tried correctly but answered
    wrong" trajectories a positive +0.2 floor (vs the -0.2 floor of bad
    exits), preserving gradient signal on this band so groups don't collapse
    to all-zero and get filtered as zero-std."""
    if exit_reason == "task_complete" and prediction is not None and prediction.strip():
        return FORMAT_BONUS
    return FORMAT_PENALTY


def should_remove_trajectory(exit_reason: str | None, prediction: str | None) -> bool:
    return prediction is None and exit_reason in TIMEOUT_EXITS


def grade_trajectory(
    messages: list[dict[str, Any]],
    ground_truth: list[str],
    answer_type: str,
    *,
    exit_reason: str | None,
    media: dict[str, list[str]] | None = None,
    workspace: str | Path | None = None,
    observed_n_tool_calls: int | None = None,
) -> dict[str, Any]:
    prediction = extract_answer_from_messages(messages)
    raw_acc = grade_outcome(prediction, ground_truth, answer_type)
    tool = summarize_tool_calls(messages, workspace=workspace)
    if observed_n_tool_calls is not None and observed_n_tool_calls > tool["n_tool"]:
        n_fail = sum(tool["fail_counts"].values())
        tool = {
            **tool,
            "n_tool": observed_n_tool_calls,
            "p_bad_tool": n_fail / (1 + observed_n_tool_calls),
        }
    modality_ok, has_video, has_audio = check_modality(media, tool["names_used"])
    fail_counts = tool["fail_counts"]

    correctness = 1.0 if raw_acc > 0.5 and modality_ok and tool["n_tool"] > 0 else 0.0
    if fail_counts["escape"] > 0:
        correctness = 0.0

    removed = should_remove_trajectory(exit_reason, prediction)
    format_value = grade_format(exit_reason, prediction)
    modality_penalty = MODALITY_PENALTY if (has_video or has_audio) and not modality_ok else 0.0
    bad_tool_penalty = -BAD_TOOL_WEIGHT * tool["p_bad_tool"]
    # Three-tier reward base — correctness wins; otherwise format decides:
    #   - correctness=1                        → base 1.0
    #   - correctness=0 + format=good          → base 0.2 (FORMAT_BONUS)
    #   - correctness=0 + format=bad           → base -0.2 (FORMAT_PENALTY)
    # Then modality + bad_tool penalties subtract from base. The format
    # +0.2/-0.2 is mutually exclusive with the +1.0 correctness signal so
    # the maximum score remains 1.0 (not 1.2).
    if correctness >= 1.0:
        base = correctness
    else:
        base = format_value
    score = base + modality_penalty + bad_tool_penalty
    if removed:
        score = 0.0

    return {
        "score": float(score),
        "correctness": float(correctness),
        "raw_acc": float(raw_acc),
        "format": float(format_value),
        # 1 iff format penalty (negative) applied; 0 when bonus (positive) given
        "format_error": float(format_value < 0),
        "modality_penalty": float(modality_penalty),
        "modality_match": float(modality_ok),
        "has_video": float(has_video),
        "has_audio": float(has_audio),
        "bad_tool_penalty": float(bad_tool_penalty),
        "p_bad_tool": float(tool["p_bad_tool"]),
        "n_tool": float(tool["n_tool"]),
        "n_unparseable": float(fail_counts["unparseable"]),
        "n_disallowed": float(fail_counts["disallowed"]),
        "n_escape": float(fail_counts["escape"]),
        "n_syntax_fail": float(fail_counts["syntax-fail"]),
        "removed": float(removed),
        # Per-sample contribution to the effective training batch. Wandb's
        # mean(active) × (rollout_batch_size × n_samples_per_prompt) gives the
        # per-step active sample count after timeout/no-answer trajectories
        # are masked out. Cross-reference with rollout/dynamic_filter/
        # drop_zero_std_* (group-level drops) to spot a too-hard data slice
        # bleeding the gradient signal.
        "active": float(not removed),
        "extracted_answer": prediction,
        "prediction_normalized": normalize(prediction) if prediction else None,
        "tool_names_used": tool["names_used"],
    }


def _result_from_sample(sample: Any) -> dict[str, Any]:
    if isinstance(getattr(sample, "reward", None), dict):
        return dict(sample.reward)
    md = getattr(sample, "metadata", None) or {}
    if isinstance(md.get("rollout_reward_components"), dict):
        return dict(md["rollout_reward_components"])
    score = float(getattr(sample, "reward", 0.0) or 0.0)
    return {"score": score, "correctness": score, "modality_match": 1.0, "n_tool": 0.0}


def apply_length_penalty(results: list[dict[str, Any]]) -> None:
    refs = [
        r for r in results
        if not r.get("removed") and r.get("correctness", 0.0) >= 1.0
        and r.get("modality_match", 0.0) >= 1.0 and r.get("n_tool", 0.0) > 0
    ]
    if not refs:
        for r in results:
            r["length_factor"] = 1.0
            r["has_reference"] = 0.0
        return

    s_star = max(min(float(r.get("num_steps") or 1.0) for r in refs), 1.0)
    for r in results:
        r["has_reference"] = 1.0
        if not r.get("removed") and r.get("correctness", 0.0) >= 1.0:
            steps = max(float(r.get("num_steps") or 1.0), 1.0)
            factor = max(LEN_F_MIN, 1.0 - LEN_ALPHA * (steps - s_star) / s_star)
            r["score"] = float(r["score"]) * factor
            r["length_factor"] = factor
        else:
            r["length_factor"] = 1.0


async def reward_func_group(args: Any, samples: list[Any], **kwargs: Any) -> list[dict[str, Any]] | dict[str, Any]:
    del args, kwargs
    if not isinstance(samples, list):
        result = _result_from_sample(samples)
        result.setdefault("length_factor", 1.0)
        result.setdefault("has_reference", 0.0)
        samples.remove_sample = bool(result.get("removed"))
        samples.reward = result
        return result

    results = [_result_from_sample(sample) for sample in samples]
    apply_length_penalty(results)
    for sample, result in zip(samples, results, strict=False):
        sample.remove_sample = bool(result.get("removed"))
        sample.reward = result
        md = getattr(sample, "metadata", None)
        if isinstance(md, dict):
            md["rollout_reward_components"] = result
    return results


def check_active_reward_nonzero_std(args: Any, samples: list[Any], **kwargs: Any):
    try:
        from relax.engine.filters.base_types import DynamicFilterOutput  # noqa: PLC0415
    except ImportError:  # keep reward unit tests independent of the optional Relax install
        @dataclass
        class DynamicFilterOutput:  # type: ignore[no-redef]
            keep: bool
            reason: str | None = None

    del kwargs
    active_rewards = [
        float(sample.get_reward_value(args))
        for sample in samples
        if not getattr(sample, "remove_sample", False)
    ]
    if len(active_rewards) <= 1:
        return DynamicFilterOutput(keep=False, reason=f"active_{len(active_rewards)}")
    keep = max(active_rewards) - min(active_rewards) > 1e-8
    reason = None if keep else f"zero_std_{round(active_rewards[0], 3)}"
    return DynamicFilterOutput(keep=keep, reason=reason)


def reward_post_process(args: Any, samples: list[Any]) -> tuple[list[float], list[float]]:
    raw_rewards = [float(sample.get_reward_value(args)) for sample in samples]
    train_rewards = [0.0 for _ in samples]
    normalize_rewards = (
        args.advantage_estimator in ["grpo", "gspo", "sapo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    )

    group_size = max(int(getattr(args, "n_samples_per_prompt", 1) or 1), 1)
    for start in range(0, len(samples), group_size):
        end = min(start + group_size, len(samples))
        indices = list(range(start, end))
        active = [i for i in indices if not getattr(samples[i], "remove_sample", False)]
        if not active:
            continue
        if not normalize_rewards:
            for i in active:
                train_rewards[i] = raw_rewards[i]
            continue

        values = [raw_rewards[i] for i in active]
        mean = sum(values) / len(values)
        rewards = [value - mean for value in values]
        if args.advantage_estimator in ["grpo", "gspo", "sapo"] and args.grpo_std_normalization:
            variance = sum(reward * reward for reward in rewards) / len(rewards)
            std = math.sqrt(variance)
            rewards = [reward / (std + 1e-6) for reward in rewards]
        for i, reward in zip(active, rewards, strict=False):
            train_rewards[i] = reward
    return raw_rewards, train_rewards
