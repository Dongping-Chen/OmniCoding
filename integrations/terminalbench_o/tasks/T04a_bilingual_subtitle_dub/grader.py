"""Grader for T04a Bilingual Subtitle Dub (v3 deterministic).

v3 redesign:
  - Replaced single-LLM `summary_quality` and `translation_quality` with
    5-axis yes/no (`summary_axes`).
  - Kept ASR-based `dub_content_score` (lightweight content verification).
  - Reweighted: deterministic dims ~0.60, ASR ~0.25, axes ~0.15.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# _lib/ lives at repo-root/_lib/ — make it importable regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))
from open_vocab_judge import axes_yes_no_score


SRT_RE = re.compile(
    r"(\d+)\s+"
    r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*"
    r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s+"
    r"(.+?)(?=\n\n|\Z)", re.DOTALL)


def parse_srt_time(s: str) -> float:
    s = s.replace(",", ".")
    h, m, rest = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def parse_srt(text: str):
    cues = []
    for m in SRT_RE.finditer(text):
        try:
            start = parse_srt_time(m.group(2))
            end = parse_srt_time(m.group(3))
            text_body = m.group(4).strip()
            cues.append({"idx": int(m.group(1)),
                          "start": start, "end": end,
                          "text": text_body})
        except Exception:
            continue
    return cues


def gating(out: Path):
    errs = []
    if not (out / "zh.srt").exists():
        errs.append("missing zh.srt")
    if not (out / "terms.csv").exists():
        errs.append("missing terms.csv")
    if not (out / "summary.md").exists():
        errs.append("missing summary.md")
    if not (out / "dub").is_dir():
        errs.append("missing dub/")
    return len(errs) == 0, errs


def probe_dur(p: Path) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of",
             "default=noprint_wrappers=1:nokey=1", str(p)],
            text=True)
        return float(out.strip())
    except Exception:
        return 0.0


def _find_wav(out: Path, seg_no: int) -> Path | None:
    for cand in (out / "dub" / f"seg_{seg_no}.wav",
                 out / "dub" / f"seg_{seg_no:02d}.wav",
                 out / "dub" / f"seg_{seg_no:03d}.wav"):
        if cand.exists():
            return cand
    return None


def _score_dub_content(out: Path, segments, cues) -> dict:
    """ASR each dub WAV in Chinese and compare to the matching cue text.

    A segment passes when normalised CER <= 0.40, i.e. roughly "the spoken
    audio matches the subtitle". Silence / English-only / wrong-language
    cheats land at CER ~ 1.0 and fail the dim.
    """
    if os.environ.get("CLAW_BENCH_SKIP_ASR") == "1":
        return {
            "skipped": True,
            "ratio": 0.0,
            "ok": False,
            "errors": ["CLAW_BENCH_SKIP_ASR=1 - dub_content_score not run"],
        }
    try:
        from asr_judge import asr_score  # type: ignore
    except Exception as e:
        return {
            "skipped": True,
            "ratio": 0.0,
            "ok": False,
            "errors": [f"asr_judge import failed: {e}"],
        }

    cue_by_idx = {i + 1: c["text"] for i, c in enumerate(cues)}
    per_seg = []
    pass_count = 0
    cer_sum = 0.0
    n_scored = 0
    PASS_CER = 0.40
    for seg in segments:
        ref = cue_by_idx.get(seg["seg_no"], "").strip()
        wav = _find_wav(out, seg["seg_no"])
        if wav is None:
            per_seg.append({"seg": seg["seg_no"], "missing": True})
            continue
        if not ref:
            per_seg.append({"seg": seg["seg_no"], "missing_cue": True})
            continue
        try:
            r = asr_score(wav, ref, language="zh", metric="cer")
        except Exception as e:
            per_seg.append({"seg": seg["seg_no"], "asr_err": str(e)[:120]})
            continue
        cer_sum += r["cer"]
        n_scored += 1
        passed = r["cer"] <= PASS_CER
        if passed:
            pass_count += 1
        per_seg.append({
            "seg": seg["seg_no"],
            "cer": round(r["cer"], 3),
            "passed": passed,
        })
    ratio = pass_count / max(1, len(segments))
    mean_cer = cer_sum / max(1, n_scored)
    return {
        "matched": pass_count,
        "total": len(segments),
        "n_scored": n_scored,
        "mean_cer": round(mean_cer, 3),
        "ratio": round(ratio, 3),
        "ok": ratio >= 0.80,
        "per_segment": per_seg[:20],
    }


SUMMARY_AXES = [
    ("translation_choices_section",
     "Does the memo include a `## Translation choices` section explaining "
     "the approach to Chinese translation?"),
    ("term_compliance_section",
     "Does the memo include a `## Term compliance` section discussing "
     "glossary term usage?"),
    ("specific_terms_cited",
     "Does the memo mention >=3 specific glossary terms (en_term or zh_term) "
     "from the supplied GROUND_DATA glossary_terms list?"),
    ("word_count_compliant",
     "Is the memo <=250 words (strict word count per GROUND_DATA word_count)?"),
    ("coherent_workflow",
     "Does the memo read as coherent prose explaining dub workflow and "
     "segment timing considerations (not lorem-ipsum / placeholder text)?"),
]


def summary_axes_score(out: Path, glossary: list) -> dict:
    summary = (out / "summary.md").read_text()
    word_count = len(summary.split())
    ground = {
        "glossary_terms": [{"en": en, "zh": zh} for en, zh in glossary],
        "word_count": word_count,
        "word_limit": 250,
    }
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground, ensure_ascii=False)}\n\n"
        f"SUMMARY.MD ({word_count} words):\n{summary[:6000]}\n"
    )
    return axes_yes_no_score(payload, SUMMARY_AXES, threshold=0.70)


def evaluate(out: Path, segments_csv: Path, glossary_csv: Path):
    segments = []
    with open(segments_csv) as f:
        for row in csv.DictReader(f):
            segments.append({
                "seg_no": int(row["seg_no"]),
                "start": float(row["start"]),
                "end": float(row["end"]),
                "en_text": row["en_text"],
            })
    glossary = []
    with open(glossary_csv) as f:
        for row in csv.DictReader(f):
            glossary.append((row["en_term"].strip(),
                              row["zh_term"].strip()))

    cues = parse_srt((out / "zh.srt").read_text())
    summary = (out / "summary.md").read_text()
    terms_rows = list(csv.DictReader((out / "terms.csv").open()))

    # ---- subtitle_timing_acc (tightened: +/- 0.10 s) ----
    st_total = len(segments) * 2; st_ok = 0
    if len(cues) == len(segments):
        for cue, seg in zip(cues, segments):
            if abs(cue["start"] - seg["start"]) <= 0.10:
                st_ok += 1
            if abs(cue["end"] - seg["end"]) <= 0.10:
                st_ok += 1
    st_ratio = st_ok / max(1, st_total)

    # ---- dub_length_match (tightened: +/- 0.5 s) ----
    dub_total = len(segments); dub_ok = 0
    for seg in segments:
        wav = _find_wav(out, seg["seg_no"])
        if wav is None:
            continue
        d = probe_dur(wav)
        if abs(d - (seg["end"] - seg["start"])) <= 0.5:
            dub_ok += 1
    dub_ratio = dub_ok / max(1, dub_total)

    # ---- dub_content_score (ASR-grounded, NEW) ----
    dub_content = _score_dub_content(out, segments, cues)

    # ---- glossary_coverage ----
    gc_total = len(glossary)
    seen = set()
    for r in terms_rows:
        en = (r.get("en_term") or "").strip()
        zh = (r.get("zh_term") or "").strip()
        for (g_en, g_zh) in glossary:
            if en == g_en and zh == g_zh:
                seen.add(g_en)
    gc_ratio = len(seen) / max(1, gc_total)

    # ---- segment_count_match ----
    cm_ok = (len(cues) == len(segments))
    n_wavs = len(list((out / "dub").glob("*.wav")))
    cm_ok = cm_ok and (n_wavs == len(segments))

    # ---- cross_doc_consistency ----
    cdc_ok = True; cdc_err = []
    for r in terms_rows:
        try:
            seg_no = int(r.get("first_used_segment_no") or 0)
        except Exception:
            seg_no = 0
        zh = (r.get("zh_term") or "").strip()
        if seg_no >= 1 and seg_no <= len(cues):
            if zh and zh not in cues[seg_no - 1]["text"]:
                cdc_ok = False
                cdc_err.append(f"{r.get('en_term')} not in cue {seg_no}")
    # Summary mentions ≥3 glossary zh terms
    mentioned = sum(1 for (en, zh) in glossary if zh in summary)
    if mentioned < 3:
        cdc_ok = False
        cdc_err.append(f"summary mentions only {mentioned} glossary terms")

    return {
        "subtitle_timing_acc": {
            "matched": st_ok, "total": st_total,
            "ratio": round(st_ratio, 3),
            "tolerance_sec": 0.10,
            "ok": st_ratio >= 0.95,
        },
        "dub_length_match": {
            "matched": dub_ok, "total": dub_total,
            "ratio": round(dub_ratio, 3),
            "tolerance_sec": 0.5,
            "ok": dub_ratio >= 0.90,
        },
        "dub_content_score": dub_content,
        "glossary_coverage": {
            "matched": len(seen), "total": gc_total,
            "ratio": round(gc_ratio, 3),
            "ok": gc_ratio >= 1.0,
        },
        "segment_count_match": {
            "cues": len(cues), "segs": len(segments),
            "wavs": n_wavs,
            "ok": cm_ok,
        },
        "cross_doc_consistency": {
            "errors": cdc_err[:5],
            "ok": cdc_ok,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--glossary", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    ok, errs = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs},
                         indent=2))
        return
    res = evaluate(out, Path(args.segments), Path(args.glossary))

    # Extract glossary for summary axes
    glossary = []
    try:
        with open(args.glossary) as f:
            for row in csv.DictReader(f):
                glossary.append((row["en_term"].strip(),
                                  row["zh_term"].strip()))
    except Exception:
        pass

    sq = summary_axes_score(out, glossary)
    res["summary_axes"] = {
        "axes": sq.get("axes", []),
        "score": sq.get("score", 0.0),
        "ok": bool(sq.get("ok", False)),
        "threshold": 0.70,
    }

    all_ok = all(v["ok"] for v in res.values())
    # v3: deterministic dims ~0.60, ASR ~0.25, axes ~0.15
    score = (
        0.15 * res["subtitle_timing_acc"]["ratio"]
        + 0.10 * res["dub_length_match"]["ratio"]
        + 0.25 * res["dub_content_score"].get("ratio", 0.0)
        + 0.10 * res["glossary_coverage"]["ratio"]
        + 0.05 * (1 if res["segment_count_match"]["ok"] else 0)
        + 0.05 * (1 if res["cross_doc_consistency"]["ok"] else 0)
        + 0.15 * res["summary_axes"]["score"]
        + 0.15 * 0.0  # translation_quality removed (was VLM, now deterministic via ASR)
    )
    print(json.dumps({
        "pass": all_ok,
        "checks": {k: v["ok"] for k, v in res.items()},
        **res,
        "score_so_far": round(score, 3),
        "weights": {
            "subtitle_timing_acc": 0.15,
            "dub_length_match": 0.10,
            "dub_content_score": 0.25,
            "glossary_coverage": 0.10,
            "segment_count_match": 0.05,
            "cross_doc_consistency": 0.05,
            "summary_axes": 0.15,
            "translation_quality_removed": 0.15,
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
