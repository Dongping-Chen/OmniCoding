"""Grader for T01a Soccer Player Highlights (v3 deterministic).

v3 redesign:
  - Removed `highlight_visual` VLM grounding dim entirely; GT events are
    authoritative for whether the agent built a real reel.
  - event_recall, clip_window, caption_check remain deterministic.
  - event_type closed-vocab strict normalised match against GT.
  - Reweighted: deterministic dims sum to 1.00; no LLM.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))
from open_vocab_judge import mllm_video_axes  # noqa: E402

EVENT_VOCAB = {"goal", "shot", "key_pass", "assist"}

# Multimodal checklist for `highlights.mp4` content quality. Sample 4
# evenly-spaced frames; majority vote per axis across frames.
HIGHLIGHTS_CONTENT_AXES = [
    ("real_match_footage_not_placeholder",
     "Is this frame natural soccer-match footage (broadcast camera "
     "view of a real game) — NOT a black screen, color bars, test "
     "pattern, or single-color placeholder?"),
    ("has_pitch_or_players_visible",
     "Is at least one of a soccer pitch, goal, or player visible "
     "in this frame?"),
    ("looks_like_action_highlights_not_static",
     "Does this frame look like it was sampled from continuous "
     "match-action highlight footage rather than a static title "
     "card / scoreboard graphic / single-text caption?"),
    ("broadcast_quality_not_lowres_blur",
     "Does the frame look like 1080p broadcast-quality video "
     "(reasonable sharpness, lighting, framing) rather than a "
     "heavily blurred / heavily pixelated / corrupted frame?"),
]


def _norm(s):
    return str(s or "").strip().lower().replace("-", "_").replace(" ", "_")


def ffprobe_dur(p):
    try:
        return float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(p)]).strip())
    except Exception:
        return None


def gating(out: Path):
    errs = []
    for f in ["highlights.mp4", "event_log.json", "caption.txt"]:
        if not (out / f).exists():
            errs.append(f"missing {f}")
    if (out / "highlights.mp4").exists():
        d = ffprobe_dur(out / "highlights.mp4")
        if d is None or not (60 <= d <= 90):
            errs.append(f"highlights {d}s not in [60,90]")
    if (out / "caption.txt").exists():
        c = (out / "caption.txt").read_text().strip()
        if len(c) > 200:
            errs.append(f"caption {len(c)} > 200 chars")
    if (out / "event_log.json").exists():
        try:
            log = json.load((out / "event_log.json").open())
            assert isinstance(log, list) and log, "event_log empty"
            last = -1
            for r in log:
                for k in ("index", "event_type", "match_time_sec",
                          "clip_start_sec", "clip_end_sec"):
                    assert k in r, f"missing field {k}"
                if r["match_time_sec"] < last - 1e-3:
                    errs.append("event_log not monotonic"); break
                last = r["match_time_sec"]
        except Exception as e:
            errs.append(f"event_log: {e}")
    return len(errs) == 0, errs


def event_recall(out: Path, gt: list) -> dict:
    log = json.load((out / "event_log.json").open())
    matched, unmatched = [], []
    for g in gt:
        gt_t = g["match_time_sec"]
        tol = g.get("tolerance_sec", 90.0)
        hit = False
        for c in log:
            cs, ce = c["clip_start_sec"], c["clip_end_sec"]
            center = (cs + ce) / 2
            if (cs - tol) <= gt_t <= (ce + tol) or abs(center - gt_t) <= tol:
                hit = True; break
        (matched if hit else unmatched).append(g)
    return {
        "recall": len(matched) / max(1, len(gt)),
        "matched": len(matched),
        "total": len(gt),
        "unmatched_events": [{"t": g["match_time_sec"], "type": g["event_type"]}
                             for g in unmatched],
        "ok": (len(matched) / max(1, len(gt))) >= 0.60,
    }


def clip_window_quality(out: Path) -> dict:
    log = json.load((out / "event_log.json").open())
    n_total = len(log); n_ok = 0
    for c in log:
        cs, ce, mt = c["clip_start_sec"], c["clip_end_sec"], c["match_time_sec"]
        dur = ce - cs
        if 6.0 <= dur <= 10.0 and cs <= mt <= ce:
            n_ok += 1
    ratio = n_ok / max(1, n_total)
    return {"in_spec": n_ok, "total": n_total,
            "ratio": ratio, "ok": ratio >= 0.80}


def event_type_vocab_score(out: Path) -> dict:
    """Closed-vocab check on event_type field."""
    log = json.load((out / "event_log.json").open())
    n = 0; n_ok = 0
    for c in log:
        n += 1
        if _norm(c.get("event_type")) in EVENT_VOCAB:
            n_ok += 1
    ratio = n_ok / max(1, n)
    return {"in_vocab": n_ok, "total": n, "ratio": ratio,
            "ok": ratio >= 0.95}


def caption_check(out: Path, target_meta: dict) -> dict:
    cap = (out / "caption.txt").read_text().strip()
    cap_low = cap.lower()
    name = target_meta["player_name"].lower()
    jersey = str(target_meta["jersey"])
    has_name = name in cap_low or any(part in cap_low for part in name.split() if len(part) > 2)
    has_jersey = bool(re.search(rf"#?{jersey}\b", cap)) or f"jersey {jersey}" in cap_low
    has_count = bool(re.search(r"\b\d+\s*(goal|goals|score)", cap_low))
    return {"has_player_name": has_name, "has_jersey": has_jersey,
            "has_goal_count": has_count,
            "ok": has_name and has_jersey and has_count}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--target-meta", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    ok, errs = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs},
                         indent=2)); return
    gt = json.load(open(args.gt))
    target_meta = json.load(open(args.target_meta))
    er = event_recall(out, gt)
    cw = clip_window_quality(out)
    et = event_type_vocab_score(out)
    cc = caption_check(out, target_meta)
    hl_content = mllm_video_axes(out / "highlights.mp4",
                                  HIGHLIGHTS_CONTENT_AXES,
                                  n_frames=4, threshold=0.70)
    hl_ok = bool(hl_content.get("ok", False))
    score = (
        0.35 * er["recall"]
        + 0.20 * cw["ratio"]
        + 0.10 * et["ratio"]
        + 0.20 * (1 if cc["ok"] else 0)
        + 0.15 * hl_content.get("score", 0.0)
    )
    passed = (er["ok"] and cw["ok"] and et["ok"] and cc["ok"] and hl_ok)
    print(json.dumps({
        "pass": passed,
        "event_recall": er,
        "clip_window": cw,
        "event_type_vocab": et,
        "caption": cc,
        "highlights_video_axes": hl_content,
        "highlights_video_axes_ok": hl_ok,
        "score": round(score, 3),
        "weights": {
            "event_recall": 0.35,
            "clip_window": 0.20,
            "event_type_vocab": 0.10,
            "caption": 0.20,
            "highlights_video_axes": 0.15,
        },
        "thresholds": {"event_recall": 0.6, "clip_window_ratio": 0.8,
                        "event_type_vocab_ratio": 0.95, "caption_ok": True,
                        "highlights_video_axes_min": 0.70},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
