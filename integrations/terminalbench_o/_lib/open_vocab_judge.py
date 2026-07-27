"""Shared helpers for open-vocabulary, capability-focused grading.

Exposes:
  - chat_judge(prompt, max_tokens=300, model=None) -> dict | None
        Direct chat_json call using the judge model (not the cheaper agent
        default). Caller fully controls the prompt + expected JSON shape.

  - llm_yes_no(question, max_tokens=300) -> bool | None
        Strict yes/no question answered by the judge LLM.

  - llm_semantic_pairs(pairs, kind, examples=None, max_tokens=1200) -> list[bool]
        Batched semantic-equivalence judge for a list of (pred, gt) string
        pairs. Lenient on synonyms / hyponyms, strict on category drift.

  - vlm_grounding(samples, photos_dir, kind="label", max_tokens=200) -> list[dict]
        Per-sample VLM check that the predicted label is visually supported
        by the image. Each sample is {file, pred_label[, kind]}.

  - axes_yes_no_score(payload, axes, threshold=0.70) -> dict
        Run a list of (axis_key, question) pairs as independent yes/no calls
        and aggregate into {axes, score, ok}. Score = mean of yes-fraction.

All helpers honour CLAW_BENCH_SKIP_VLM=1 to short-circuit. They never raise
on infrastructure errors — they degrade to safe defaults so the grader
returns a structured result.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from _concurrent import parallel_map


def _judge_model() -> str | None:
    try:
        from vlm_judge import JUDGE_MODEL  # type: ignore
        return JUDGE_MODEL
    except Exception:
        return None


def _judge_client_kwargs() -> dict:
    kwargs = {}
    api_key = os.environ.get("CLAW_BENCH_JUDGE_API_KEY")
    base_url = os.environ.get("CLAW_BENCH_JUDGE_BASE_URL")
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def chat_judge(prompt: str, max_tokens: int = 300,
               model: str | None = None) -> dict | None:
    if os.environ.get("CLAW_BENCH_SKIP_VLM") == "1":
        return None
    try:
        from openrouter_client import chat_json  # type: ignore
    except Exception:
        return None
    if model is None:
        model = _judge_model()
    try:
        return chat_json(
            [{"role": "user", "content": prompt}],
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            **_judge_client_kwargs(),
        )
    except Exception:
        return None


def llm_yes_no(question: str, max_tokens: int = 300) -> bool | None:
    """Strict yes/no question. Returns True/False/None (None = infra
    failure or unparseable response)."""
    prompt = (
        f"{question}\n\n"
        "Answer STRICTLY as a JSON object: "
        '{"answer": "yes" | "no"}. Pick the closer of yes/no.'
    )
    r = chat_judge(prompt, max_tokens=max_tokens)
    if r is None:
        return None
    a = str(r.get("answer", "")).strip().lower()
    if a in ("yes", "y", "true"):
        return True
    if a in ("no", "n", "false"):
        return False
    s = r.get("score")
    try:
        s = float(s)
        if s >= 0.5:
            return True
        if 0.0 <= s < 0.5:
            return False
    except (TypeError, ValueError):
        pass
    return None


def _normalize(s: str) -> str:
    return str(s or "").strip().lower().replace("-", "_").replace(" ", "_")


def llm_semantic_pairs(pairs: list[tuple[str, str]],
                       kind: str = "label",
                       examples: list[tuple[str, str, bool]] | None = None,
                       max_tokens: int = 1500,
                       batch_size: int = 50) -> list[bool]:
    """For a list of (pred, gt) descriptor pairs, return per-pair bool
    (semantically equivalent or close-enough).

    Engineering hardening:
    1. **Identity pre-check** — pairs whose normalised strings match are
       auto-equivalent without an LLM call. This fixes the "chicken
       vs chicken" silent-False bug we hit on T08b.
    2. **Batched LLM calls** — large pair lists are split into
       `batch_size` chunks so each chunk fits in the token window; a
       failed chunk only zeros itself out, not the whole list.

    `examples` pins domain conventions:
        [("scratch", "linear_abrasion", True),
         ("scratch", "hole",            False), ...]
    """
    if not pairs:
        return []
    out = [False] * len(pairs)
    # Identity pre-check (no LLM cost).
    needs_llm: list[int] = []
    for i, (p, g) in enumerate(pairs):
        if _normalize(p) and _normalize(p) == _normalize(g):
            out[i] = True
        else:
            needs_llm.append(i)
    if not needs_llm:
        return out

    if examples:
        ex_lines = "\n".join(
            f"  - '{p}' vs '{g}'  -> equiv: {'true' if eq else 'false'}"
            for p, g, eq in examples
        )
        ex_block = f"Examples:\n{ex_lines}\n\n"
    else:
        ex_block = ""

    chunks = [needs_llm[s:s + batch_size]
              for s in range(0, len(needs_llm), batch_size)]

    def _run_chunk(chunk: list[int]):
        pairs_payload = [{"i": i, "pred": str(pairs[i][0]), "gt": str(pairs[i][1])}
                         for i in chunk]
        prompt = (
            f"You are evaluating whether two short {kind} descriptors refer "
            "to the same underlying phenomenon. Be lenient on wording "
            "(synonyms, hyper/hyponyms) but strict on category drift.\n"
            f"{ex_block}"
            f"PAIRS (one entry per input):\n"
            f"{json.dumps(pairs_payload, ensure_ascii=False)}\n\n"
            "Return STRICT JSON in the EXACT shape: "
            '{"results": [{"i": <int>, "equiv": true | false}, ...]} '
            "with one entry per input pair, preserving the i field."
        )
        return chat_judge(prompt, max_tokens=max_tokens)

    chunk_responses = parallel_map(_run_chunk, chunks)
    for r in chunk_responses:
        if r is None:
            continue
        results = r.get("results") if isinstance(r, dict) else None
        if not isinstance(results, list):
            results = r if isinstance(r, list) else None
        if results is None:
            continue
        for entry in results:
            try:
                if not isinstance(entry, dict):
                    continue
                i = int(entry.get("i"))
                if 0 <= i < len(pairs):
                    out[i] = bool(entry.get("equiv"))
            except Exception:
                pass
    return out


def _judge_image_one(rubric: str, image_path: str,
                     max_tokens: int = 200) -> dict | None:
    try:
        from vlm_judge import judge_image  # type: ignore
    except Exception:
        return None
    try:
        r = judge_image(rubric, image_path, max_tokens=max_tokens)
    except Exception:
        return None
    if not isinstance(r, dict):
        return None
    return r


def vlm_grounding(samples: list[dict], photos_dir: Path,
                  kind: str = "label",
                  max_tokens: int = 250,
                  frames_per_sample: int = 1,
                  ensemble: bool = False) -> list[dict]:
    """For each sample, ask the VLM whether the label is supported by
    the image(s). Returns per-sample dict with score/answer/ok keys.

    Sample shape:
      {"file": "img.png", ...}                — single-frame
      {"files": ["a.jpg","b.jpg","c.jpg"], ...} — multi-frame, this dim
                                                  is the median over
                                                  frames.

    Engineering hardening:
    - **Multi-frame**: pass a list of files per sample and the median
      score across frames is reported. Reduces single-frame VLM noise.
    - **Identity** still calls the VLM (no cheap shortcut for
      grounding) but median over multiple frames suppresses outliers.
    - **ensemble=True**: each frame is judged by 2 different judge
      models (the default JUDGE_MODEL plus a fallback) and the per-
      frame score is the mean. Costs ×2 but cuts model-bias noise.

    Caller is expected to extract frames before calling this — write
    them to `photos_dir/<filename>` and pass the basenames in
    `file` / `files`.
    """
    out: list[dict] = []
    if not samples:
        return out
    if os.environ.get("CLAW_BENCH_SKIP_VLM") == "1":
        return [{"file": s.get("file") or (s.get("files") or [None])[0],
                 "skipped": True, "ok": False}
                for s in samples]

    rubric_template = (
        "You are inspecting an image. The agent labelled this image "
        "with the __KIND__ '__LABEL__'. Is that label CONCRETELY "
        "supported by what is visible in the image?\n"
        'Return STRICT JSON: '
        '{"answer": "yes" | "no", "score": <0..1 float>, '
        '"reasons": "<one short sentence>"}. '
        "Score 1.0 if clearly visible, 0.5 if ambiguously visible, "
        "0.0 if not present at all."
    )
    photos_dir = Path(photos_dir)
    fallback_models: list[str | None] = [None]  # default judge
    if ensemble:
        fallback_models.append("openai/gpt-4o-mini")

    def _judge_one_sample(s: dict) -> dict:
        files = s.get("files")
        if not files:
            f = s.get("file")
            files = [f] if f else []
        if not files:
            return {"err": "no file in sample"}
        rubric = (rubric_template
                  .replace("__KIND__", s.get("kind", kind))
                  .replace("__LABEL__", str(s.get("pred_label", ""))[:80]))
        frame_scores: list[float] = []
        frame_answers: list[str] = []
        frame_reasons: list[str] = []
        used = 0
        for fname in files[:frames_per_sample] or files:
            path = photos_dir / fname
            if not path.exists():
                continue
            ensemble_scores: list[float] = []
            ensemble_answers: list[str] = []
            r = None
            for _model in fallback_models:
                r = _judge_image_one(rubric, str(path), max_tokens=max_tokens)
                if r is None:
                    continue
                ans = str(r.get("answer", "")).strip().lower()
                try:
                    sc = float(r.get("score", 0.0))
                except (TypeError, ValueError):
                    sc = 0.0
                ensemble_scores.append(sc)
                ensemble_answers.append(ans)
                if not ensemble:
                    break  # only one model
            if not ensemble_scores:
                continue
            frame_scores.append(sum(ensemble_scores) / len(ensemble_scores))
            frame_answers.append(ensemble_answers[0] if ensemble_answers else "")
            frame_reasons.append(str((r or {}).get("reasons", ""))[:160])
            used += 1
        if not frame_scores:
            return {"file": files[0], "err": "no informative frames"}
        sorted_scores = sorted(frame_scores)
        mid = len(sorted_scores) // 2
        if len(sorted_scores) % 2 == 0 and len(sorted_scores) >= 2:
            score = (sorted_scores[mid - 1] + sorted_scores[mid]) / 2
        else:
            score = sorted_scores[mid]
        yes_count = sum(1 for a in frame_answers if a == "yes")
        no_count = sum(1 for a in frame_answers if a == "no")
        if yes_count > no_count:
            ans = "yes"
        elif no_count > yes_count:
            ans = "no"
        else:
            ans = "yes" if score >= 0.5 else "no"
        ok = (ans == "yes") or (score >= 0.5)
        return {
            "file": files[0],
            "n_frames": used,
            "pred_label": s.get("pred_label"),
            "frame_scores": [round(x, 2) for x in frame_scores],
            "score": round(score, 2),
            "answer": ans,
            "ok": ok,
            "reasons": frame_reasons[0] if frame_reasons else "",
        }

    return parallel_map(_judge_one_sample, samples)


def axes_yes_no_score(payload: str, axes: list[tuple[str, str]],
                      threshold: float = 0.70,
                      max_tokens: int = 300) -> dict:
    """Run independent yes/no questions across the given axes, with the
    same payload prepended. Returns aggregate dict."""
    if os.environ.get("CLAW_BENCH_SKIP_VLM") == "1":
        return {"skipped": True, "score": 0.0, "ok": False,
                "reasons": "CLAW_BENCH_SKIP_VLM=1",
                "axes": [{"axis": k, "yes": False} for k, _ in axes]}
    def _ask(axis: tuple[str, str]):
        key, q = axis
        full_q = (
            f"{payload}\n\n"
            f"Question for axis '{key}': {q} "
            "Answer STRICTLY yes or no."
        )
        return key, llm_yes_no(full_q, max_tokens=max_tokens)

    answers = parallel_map(_ask, axes)
    yeses = []
    details = []
    n_unanswered = 0
    for key, ans in answers:
        if ans is None:
            n_unanswered += 1
            details.append({
                "axis": key,
                "yes": False,
                "skipped": True,
                "reasons": "judge unavailable or response unparseable",
            })
            yeses.append(0.0)
            continue
        is_yes = bool(ans)
        yeses.append(1.0 if is_yes else 0.0)
        details.append({"axis": key, "yes": is_yes})
    mean = sum(yeses) / max(1, len(yeses))
    result = {
        "axes": details,
        "yes_fraction": round(mean, 3),
        "score": round(mean, 3),
        "ok": mean >= threshold,
        "threshold": threshold,
    }
    if n_unanswered:
        result["n_unanswered"] = n_unanswered
    if axes and n_unanswered == len(axes):
        result["skipped"] = True
        result["reasons"] = "judge unavailable or all axis responses unparseable"
    return result


# ---------------------------------------------------------------------- #
# MLLM-as-judge for multimodal generative outputs
# (videos / images / audio without a fixed string GT).
# ---------------------------------------------------------------------- #
def _extract_frames_ffmpeg(video_path: Path, n_frames: int,
                           out_dir: Path) -> list[Path]:
    """Extract n_frames evenly-spaced JPGs from `video_path` into
    `out_dir`. Returns list of paths to extracted frames."""
    import subprocess
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Probe duration first.
    try:
        dur_s = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(video_path)],
            stderr=subprocess.DEVNULL,
        ).strip()
        dur = float(dur_s)
    except Exception:
        return []
    if dur <= 0:
        return []
    paths: list[Path] = []
    for k in range(1, n_frames + 1):
        t = dur * (k / (n_frames + 1))
        out_path = out_dir / f"f_{k:02d}.jpg"
        try:
            subprocess.check_call(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-ss", f"{t:.2f}",
                 "-i", str(video_path),
                 "-frames:v", "1", str(out_path)],
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            continue
        if out_path.exists():
            paths.append(out_path)
    return paths


def mllm_image_axes(image_path: str | Path, axes: list[tuple[str, str]],
                   threshold: float = 0.70,
                   max_tokens: int = 250) -> dict:
    """For a single image, ask the VLM each axis question (yes/no) and
    aggregate. Drop-in companion to `axes_yes_no_score` for image
    output evaluation."""
    if os.environ.get("CLAW_BENCH_SKIP_VLM") == "1":
        return {"skipped": True, "score": 0.0, "ok": False,
                "reasons": "CLAW_BENCH_SKIP_VLM=1",
                "axes": [{"axis": k, "yes": False} for k, _ in axes]}
    image_path = Path(image_path)
    if not image_path.exists():
        return {"skipped": True, "score": 0.0, "ok": False,
                "reasons": f"missing {image_path.name}",
                "axes": [{"axis": k, "yes": False} for k, _ in axes]}
    try:
        from vlm_judge import judge_image  # type: ignore
    except Exception:
        return {"skipped": True, "score": 0.0, "ok": False,
                "reasons": "vlm_judge import failed",
                "axes": [{"axis": k, "yes": False} for k, _ in axes]}
    def _ask(axis: tuple[str, str]):
        key, q = axis
        rubric = (
            f"{q}\n\n"
            'Return STRICT JSON: {"answer": "yes" | "no", '
            '"score": <0..1 float>, "reasons": "<one short sentence>"}.'
        )
        try:
            r = judge_image(rubric, str(image_path), max_tokens=max_tokens)
            err = ""
        except Exception as exc:
            r = None
            err = str(exc)[:200]
        return key, r, err

    answers = parallel_map(_ask, axes)
    yeses = []
    details = []
    n_unanswered = 0
    last_err = ""
    for key, r, err in answers:
        if err:
            last_err = err
        if not isinstance(r, dict) or not r:
            n_unanswered += 1
            yeses.append(0.0)
            details.append({
                "axis": key,
                "yes": False,
                "skipped": True,
                "reasons": "judge unavailable or response unparseable",
            })
            continue
        ans = str(r.get("answer", "")).strip().lower()
        try:
            sc = float(r.get("score", 0.0))
        except (TypeError, ValueError):
            sc = 0.0
        is_yes = (ans == "yes") or (ans not in ("no", "n") and sc >= 0.5)
        yeses.append(1.0 if is_yes else 0.0)
        details.append({"axis": key, "yes": is_yes,
                        "score": round(sc, 2),
                        "reasons": str(r.get("reasons", ""))[:160]})
    mean = sum(yeses) / max(1, len(yeses))
    result = {
        "axes": details,
        "yes_fraction": round(mean, 3),
        "score": round(mean, 3),
        "ok": mean >= threshold,
        "threshold": threshold,
    }
    if n_unanswered:
        result["n_unanswered"] = n_unanswered
    if axes and n_unanswered == len(axes):
        result["skipped"] = True
        result["reasons"] = (
            "judge unavailable or all axis responses unparseable"
            + (f": {last_err}" if last_err else "")
        )
    return result


def mllm_video_axes(video_path: str | Path, axes: list[tuple[str, str]],
                    n_frames: int = 4,
                    threshold: float = 0.70,
                    max_tokens: int = 250) -> dict:
    """For a video file, extract `n_frames` evenly-spaced frames via
    ffmpeg, then ask each axis question on each frame and aggregate
    by **majority vote** across frames per axis. Reduces single-frame
    judge noise for content-quality evaluation of generated videos."""
    import tempfile
    if os.environ.get("CLAW_BENCH_SKIP_VLM") == "1":
        return {"skipped": True, "score": 0.0, "ok": False,
                "reasons": "CLAW_BENCH_SKIP_VLM=1",
                "axes": [{"axis": k, "yes": False} for k, _ in axes]}
    video_path = Path(video_path)
    if not video_path.exists():
        return {"skipped": True, "score": 0.0, "ok": False,
                "reasons": f"missing {video_path.name}",
                "axes": [{"axis": k, "yes": False} for k, _ in axes]}
    try:
        from vlm_judge import judge_image  # type: ignore
    except Exception:
        return {"skipped": True, "score": 0.0, "ok": False,
                "reasons": "vlm_judge import failed",
                "axes": [{"axis": k, "yes": False} for k, _ in axes]}
    with tempfile.TemporaryDirectory() as td:
        frames = _extract_frames_ffmpeg(video_path, n_frames, Path(td))
        if not frames:
            return {"skipped": True, "score": 0.0, "ok": False,
                    "reasons": "no frames extracted",
                    "axes": [{"axis": k, "yes": False} for k, _ in axes]}
        # axes × frames matrix of yes counts; flatten the cross-product
        # and run all frame×axis judge calls concurrently.
        per_axis_yes = {k: 0 for k, _ in axes}
        per_axis_total = {k: 0 for k, _ in axes}
        per_axis_reasons = {k: "" for k, _ in axes}
        n_unanswered = 0
        last_err = ""

        jobs: list[tuple[Path, str, str]] = [
            (fp, key, q) for fp in frames for key, q in axes
        ]

        def _judge_job(job):
            fp, key, q = job
            rubric = (
                f"{q}\n\n"
                'Return STRICT JSON: {"answer": "yes" | "no", '
                '"score": <0..1 float>, "reasons": "<one short sentence>"}.'
            )
            try:
                r = judge_image(rubric, str(fp), max_tokens=max_tokens)
                err = ""
            except Exception as exc:
                r = None
                err = str(exc)[:200]
            return key, r, err

        for key, r, err in parallel_map(_judge_job, jobs):
            if err:
                n_unanswered += 1
                last_err = err
                continue
            if not isinstance(r, dict) or not r:
                n_unanswered += 1
                continue
            ans = str(r.get("answer", "")).strip().lower()
            try:
                sc = float(r.get("score", 0.0))
            except (TypeError, ValueError):
                sc = 0.0
            is_yes = (ans == "yes") or (ans not in ("no", "n") and sc >= 0.5)
            per_axis_total[key] += 1
            if is_yes:
                per_axis_yes[key] += 1
            if not per_axis_reasons[key]:
                per_axis_reasons[key] = str(r.get("reasons", ""))[:120]
    yeses = []
    details = []
    for key, _ in axes:
        n = per_axis_total[key]
        ratio = per_axis_yes[key] / max(1, n)
        # majority across frames
        is_yes = ratio >= 0.5 and n >= 1
        yeses.append(1.0 if is_yes else 0.0)
        details.append({
            "axis": key,
            "yes": is_yes,
            "yes_frames": per_axis_yes[key],
            "total_frames": n,
            "reasons": per_axis_reasons[key],
        })
    mean = sum(yeses) / max(1, len(yeses))
    result = {
        "axes": details,
        "n_frames_extracted": len(frames),
        "yes_fraction": round(mean, 3),
        "score": round(mean, 3),
        "ok": mean >= threshold,
        "threshold": threshold,
    }
    n_responses = sum(per_axis_total.values())
    if n_unanswered:
        result["n_unanswered"] = n_unanswered
    if n_responses == 0:
        result["skipped"] = True
        result["reasons"] = (
            "judge unavailable or all frame-axis responses unparseable"
            + (f": {last_err}" if last_err else "")
        )
    return result
