"""Grader for T44a Podcast Post-Production (v3).

v3 pattern:
  - Closed-vocab + numeric dims (LUFS, keyword recall, speaker
    accuracy, chapter tIoU, transcript ASR grounding) are
    deterministic — GT-driven, no LLM.
  - show_notes.md is genuinely free-form prose; replaced single
    `judge_text` continuous-score with `axes_yes_no_score` over 5
    independent axes.
  - Reweighted: deterministic dims ~0.85, axes-LLM dim 0.15.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))

from open_vocab_judge import axes_yes_no_score  # noqa: E402


def gating(out: Path):
    errs = []
    for f in ["episode.mp3", "transcript.srt", "chapters.json", "show_notes.md"]:
        if not (out / f).exists():
            errs.append(f"missing {f}")
    if (out / "episode.mp3").exists():
        try:
            d = float(subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", str(out / "episode.mp3")]).strip())
            if d < 25 * 60:
                errs.append(f"episode {d:.0f}s < 25 min")
        except Exception as e:
            errs.append(f"mp3 probe: {e}")
    if (out / "chapters.json").exists():
        try:
            chs = json.load((out / "chapters.json").open())
            assert isinstance(chs, list)
            n = len(chs)
            if not (4 <= n <= 12):
                errs.append(f"chapters {n} not in [4,12]")
            last = -1.0
            for c in chs:
                for k in ("index", "title", "start_sec", "end_sec"):
                    assert k in c, f"chapter missing {k}"
                if c["start_sec"] < last - 1e-3:
                    errs.append(f"chapter {c['index']} overlaps")
                last = c["end_sec"]
        except Exception as e:
            errs.append(f"chapters.json: {e}")
    return len(errs) == 0, errs


def lufs_in_band(out: Path, lo=-16.5, hi=-15.5) -> dict:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-i", str(out / "episode.mp3"), "-filter:a", "ebur128",
             "-f", "null", "-"],
            stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, timeout=180,
        )
        log = proc.stderr.decode("utf-8", errors="ignore")
        # ebur128 emits a stream of progressive I: values then a final summary.
        # We want the last (integrated) value.
        matches = re.findall(r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", log)
        lufs = float(matches[-1]) if matches else None
    except Exception as e:
        return {"lufs": None, "error": str(e), "in_band": False}
    in_band = lufs is not None and lo <= lufs <= hi
    return {"lufs": lufs, "in_band": bool(in_band), "band": [lo, hi]}


def parse_srt(path: Path):
    text = path.read_text()
    blocks = re.split(r"\n\s*\n", text.strip())
    segs = []
    for b in blocks:
        lines = b.strip().splitlines()
        if len(lines) < 3: continue
        m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)",
                     lines[1])
        if not m: continue
        h, mn, s, ms = map(int, m.group(1, 2, 3, 4))
        start = h*3600 + mn*60 + s + ms/1000
        h, mn, s, ms = map(int, m.group(5, 6, 7, 8))
        end = h*3600 + mn*60 + s + ms/1000
        segs.append({"start_sec": start, "end_sec": end,
                     "text": " ".join(lines[2:]).strip()})
    return segs


def keyword_recall(srt_segs, kwgt: dict) -> dict:
    """Per-host keyword recall, then mean. Strip speaker prefix first."""
    full = " ".join(re.sub(r"\s*HOST_[AB]\s*:", "",
                            s["text"], flags=re.I) for s in srt_segs).lower()
    per_host = {}
    for host, words in kwgt.items():
        if not words: continue
        found = sum(1 for w in words if w.lower() in full)
        per_host[host] = {"found": found, "total": len(words),
                          "ratio": found / len(words)}
    if not per_host:
        return {"mean_recall": 0.0, "per_host": {}}
    mean = sum(v["ratio"] for v in per_host.values()) / len(per_host)
    return {"mean_recall": mean, "per_host": per_host}


def speaker_match(srt_segs, diar_gt) -> dict:
    matched = total = 0
    for seg in srt_segs:
        text = seg["text"]
        m = re.match(r"\s*(HOST_[AB])\s*:", text, re.I)
        if not m: continue
        spk = m.group(1).upper()
        mid = (seg["start_sec"] + seg["end_sec"]) / 2
        for g in diar_gt:
            if g["start_sec"] <= mid <= g["end_sec"]:
                total += 1
                if spk == g["speaker"]: matched += 1
                break
    return {"acc": matched / total if total else 0.0,
            "matched": matched, "total": total}


def chapter_tiou(pred_chapters, gt_chapters) -> dict:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return {"mean_tiou": 0.0, "error": "scipy missing"}
    if not pred_chapters or not gt_chapters:
        return {"mean_tiou": 0.0, "n_pairs": 0}
    P, G = len(pred_chapters), len(gt_chapters)
    cost = [[0.0]*G for _ in range(P)]
    for i, p in enumerate(pred_chapters):
        for j, g in enumerate(gt_chapters):
            inter = max(0, min(p["end_sec"], g["end_sec"]) -
                            max(p["start_sec"], g["start_sec"]))
            union = max(p["end_sec"], g["end_sec"]) - \
                    min(p["start_sec"], g["start_sec"])
            iou = inter / union if union > 0 else 0.0
            cost[i][j] = -iou  # maximise iou
    rows, cols = linear_sum_assignment(cost)
    ious = [-cost[r][c] for r, c in zip(rows, cols)]
    mean = sum(ious) / max(P, G)
    return {"mean_tiou": float(mean), "matched_pairs": len(ious),
            "n_pred": P, "n_gt": G, "ious": [round(i, 3) for i in ious]}


# --------------------------------------------------------------------------- #
# Show-notes axes-yes/no judge (5 independent yes/no axes, no continuous score)
# --------------------------------------------------------------------------- #
SHOW_NOTES_AXES = [
    ("h1_title_present",
     "Does SHOW_NOTES.MD start with an H1 episode title (a single `#` header) "
     "that is not a generic placeholder like 'Episode Title'?"),
    ("summary_three_paragraphs",
     "Does SHOW_NOTES.MD have a `## Summary` section containing at least "
     "three distinct, non-placeholder paragraphs of podcast prose?"),
    ("chapters_section_with_timestamps",
     "Does SHOW_NOTES.MD have a `## Chapters` section listing every chapter "
     "from CHAPTERS_JSON with an mm:ss timestamp and a non-trivial title "
     "(not 'Chapter 1', 'Chapter 2', ...)?"),
    ("mentioned_items_present",
     "Does SHOW_NOTES.MD have a `## Mentioned` (or equivalent links / books / "
     "quotes / names) section enumerating at least 2 concrete items?"),
    ("coherent_no_fabrication",
     "Is SHOW_NOTES.MD fluent English self-consistent with the supplied "
     "CHAPTERS_JSON, with no fabricated chapter titles or contradictions?"),
]


def show_notes_axes(out: Path, chapters_pred: list) -> dict:
    notes = (out / "show_notes.md").read_text()
    payload = (
        f"CHAPTERS_JSON:\n{json.dumps(chapters_pred, ensure_ascii=False)}\n\n"
        f"SHOW_NOTES.MD:\n{notes[:6000]}\n"
    )
    return axes_yes_no_score(payload, SHOW_NOTES_AXES, threshold=0.60)


# --------------------------------------------------------------------------- #
# Transcript grounding: sample Whisper ASR vs agent's transcript.srt
# --------------------------------------------------------------------------- #
def transcript_grounding(out: Path, srt_segs: list, n_windows: int = 3,
                         win_sec: float = 30.0) -> dict:
    """ASR random windows of episode.mp3 and compare to agent's SRT in the
    same time range. Catches fabricated transcripts.
    """
    if os.environ.get("CLAW_BENCH_SKIP_ASR") == "1":
        return {"skipped": True, "ratio": 0.0, "ok": False,
                "reasons": "CLAW_BENCH_SKIP_ASR=1"}
    try:
        from asr_judge import transcribe, wer  # type: ignore
    except Exception as e:
        return {"skipped": True, "ratio": 0.0, "ok": False,
                "reasons": f"asr_judge import failed: {e}"}

    mp3 = out / "episode.mp3"
    try:
        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(mp3)]).strip())
    except Exception as e:
        return {"skipped": True, "ratio": 0.0, "ok": False,
                "reasons": f"ffprobe: {e}"}

    rng = random.Random(0xC1AB1E)  # deterministic windows for reproducibility
    samples = []
    n_pass = 0
    werr_total = 0.0
    for _ in range(n_windows):
        start = rng.uniform(60, max(60, dur - win_sec - 60))
        end = start + win_sec
        clip = out / f"_grading_clip_{int(start)}.wav"
        try:
            subprocess.check_call([
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(mp3),
                "-ss", f"{start:.2f}", "-t", f"{win_sec:.2f}",
                "-ar", "16000", "-ac", "1", str(clip),
            ], timeout=120)
            asr = transcribe(clip, language="en")
        except Exception as e:
            samples.append({"start": start, "err": str(e)[:120]})
            continue
        finally:
            clip.unlink(missing_ok=True)

        # Concatenate agent SRT cues whose midpoint is in [start, end].
        agent_text = " ".join(
            re.sub(r"\s*HOST_[AB]\s*:", "", s["text"], flags=re.I)
            for s in srt_segs
            if start <= (s["start_sec"] + s["end_sec"]) / 2 <= end
        ).strip()
        ref = asr["text"].strip()
        if len(ref) < 20:
            # ASR hit silence / music; window not informative.
            samples.append({"start": round(start, 1), "skipped_no_speech": True})
            continue
        w = wer(ref, agent_text, language="en")
        werr_total += w
        passed = w <= 0.50
        if passed:
            n_pass += 1
        samples.append({
            "start": round(start, 1),
            "wer": round(w, 3),
            "passed": passed,
            "asr_preview": ref[:120],
            "agent_preview": agent_text[:120],
        })
    informative = sum(1 for s in samples if "wer" in s)
    ratio = n_pass / max(1, informative)
    mean_wer = werr_total / max(1, informative)
    return {
        "informative_windows": informative,
        "passed_windows": n_pass,
        "ratio": round(ratio, 3),
        "mean_wer": round(mean_wer, 3),
        "samples": samples,
        "ok": informative >= 1 and ratio >= 0.50,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--keywords_gt", required=True)
    ap.add_argument("--diar_gt", required=True)
    ap.add_argument("--chapters_gt", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    ok, errs = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs}, indent=2))
        return

    lufs = lufs_in_band(out)
    segs = parse_srt(out / "transcript.srt")
    kwgt = json.load(open(args.keywords_gt))
    diar = json.load(open(args.diar_gt))
    cgt = json.load(open(args.chapters_gt))
    pred_ch = json.load((out / "chapters.json").open())

    kr = keyword_recall(segs, kwgt)
    sp = speaker_match(segs, diar)
    ct = chapter_tiou(pred_ch, cgt)
    sn = show_notes_axes(out, pred_ch)
    tg = transcript_grounding(out, segs)

    lufs_ok = lufs.get("in_band", False)
    kr_ok = kr["mean_recall"] >= 0.6
    sp_ok = sp["acc"] >= 0.8
    ct_ok = ct["mean_tiou"] >= 0.5
    sn_ok = bool(sn.get("ok", False))
    tg_ok = tg["ok"]
    passed = lufs_ok and kr_ok and sp_ok and ct_ok and sn_ok and tg_ok

    # v3: deterministic dims dominate (~0.85). Only show_notes uses
    # axes-yes/no LLM (~0.15).
    score = (0.15 * (1 if lufs_ok else 0)
             + 0.25 * kr["mean_recall"]
             + 0.20 * sp["acc"]
             + 0.10 * max(0.0, ct["mean_tiou"])
             + 0.15 * sn.get("score", 0.0)
             + 0.15 * tg["ratio"])

    print(json.dumps({
        "pass": passed,
        "checks": {"lufs_ok": lufs_ok, "kw_recall_ok": kr_ok,
                   "speaker_ok": sp_ok, "chapter_tiou_ok": ct_ok,
                   "show_notes_axes_ok": sn_ok,
                   "transcript_grounding_ok": tg_ok},
        "thresholds": {"lufs": [-16.5, -15.5], "kw_recall": 0.6,
                       "speaker_acc": 0.8, "chapter_tiou": 0.5,
                       "show_notes_axes": 0.60, "transcript_wer_pass": 0.50},
        "weights": {
            "loudness": 0.15, "keyword_recall": 0.25,
            "speaker_match": 0.20, "chapter_tiou": 0.10,
            "show_notes_axes": 0.15, "transcript_grounding": 0.15,
        },
        "loudness": lufs,
        "keyword_recall": kr,
        "speaker_match": sp,
        "chapter_tiou": ct,
        "show_notes_axes": sn,
        "transcript_grounding": tg,
        "score": round(score, 3),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
