"""Grader for T02a Interview Quote Pack (v3 deterministic).

v3 redesign:
  - Replaced single-LLM `summary_quality` with 5-axis yes/no (`summary_axes`).
  - Kept ASR-based `quote_audio_grounding` (lightweight content verification).
  - Reweighted: deterministic dims ~0.70, ASR ~0.15, axes ~0.15.

Pass criteria:
  - quote_provenance      == 1.0   (quote text must be contiguous within a
                                     single speaker's tokens AND inside the
                                     stated [start_sec, end_sec] window)
  - speaker_attribution   >= 0.90
  - first_appearance_acc  >= 0.85  (window tightened to +/- 1.0 s)
  - quote_count_in_range  == 1.0   (6 <= N <= 10, words 15..35,
                                     >= 3 quotes per speaker)
  - quote_diversity       == 1.0   (pairwise Jaccard token sim < 0.40)
  - cross_doc_consistency == 1.0
  - quote_audio_grounding (ASR)  ratio >= 0.80   (WER <= 0.35)
  - summary_axes          (LLM)  score >= 0.70
"""
import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))
from open_vocab_judge import axes_yes_no_score


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def gating(out: Path):
    errs = []
    if not (out / "quote_sheet.csv").exists():
        errs.append("missing quote_sheet.csv")
    if not (out / "lower_thirds.json").exists():
        errs.append("missing lower_thirds.json")
    if not (out / "broadcast_pack").is_dir():
        errs.append("missing broadcast_pack/")
    if not (out / "summary.md").exists():
        errs.append("missing summary.md")
    return len(errs) == 0, errs


def normalize(s):
    return re.sub(r"\s+", " ", (s or "").lower().strip())


def _tokenset(s):
    return set(re.findall(r"[a-z0-9']+", (s or "").lower()))


def _find_single_speaker_span(words, ts, te, q_norm):
    """Return the speaker_id (S1/S2) whose contiguous tokens inside
    [ts, te] (with small slack) form q_norm exactly. If no such span
    exists, return None.
    """
    SLACK = 0.15
    candidate = [w for w in words
                 if w["start"] >= ts - SLACK and w["end"] <= te + SLACK]
    if not candidate:
        return None
    # Try each contiguous slice with consistent speaker
    n = len(candidate)
    for i in range(n):
        spk = candidate[i]["speaker"]
        toks = []
        for j in range(i, n):
            if candidate[j]["speaker"] != spk:
                break
            toks.append(candidate[j]["text"])
            joined = normalize(" ".join(toks))
            if joined == q_norm:
                return spk
            if len(joined) > len(q_norm) + 5:
                break
    return None


# --------------------------------------------------------------------------- #
# Core grading
# --------------------------------------------------------------------------- #
def evaluate(out: Path, transcript, gt):
    rows = list(csv.DictReader((out / "quote_sheet.csv").open()))
    _lt_doc = json.load((out / "lower_thirds.json").open())
    lt = _lt_doc.get("lower_thirds") or _lt_doc.get("speakers") or []

    words = transcript["words"]
    gt_map = {s["id"]: s["name"] for s in gt["speakers"]}

    # quote_provenance: per-row, single-speaker contiguous span
    prov_total = len(rows)
    prov_ok = 0
    row_speakers = []  # parallel to rows: ground-truth S1/S2 per row, or None
    for r in rows:
        q = normalize(r.get("quote_text", ""))
        try:
            ts = float(r.get("start_sec", 0))
            te = float(r.get("end_sec", 0))
        except (ValueError, TypeError):
            row_speakers.append(None)
            continue
        spk = _find_single_speaker_span(words, ts, te, q) if q else None
        row_speakers.append(spk)
        if spk is not None:
            prov_ok += 1
    prov_ratio = prov_ok / max(1, prov_total)

    # speaker_attribution: agent's speaker_name vs GT[row_speakers[i]]
    attr_total, attr_ok = 0, 0
    for r, spk in zip(rows, row_speakers):
        if spk is None:
            continue
        attr_total += 1
        gt_name = gt_map.get(spk, "")
        if normalize(r.get("speaker_name", "")) == normalize(gt_name):
            attr_ok += 1
    attr_ratio = attr_ok / max(1, attr_total)

    # quote_count_in_range: N, word band, per-speaker minimum
    n = len(rows)
    valid_count = 6 <= n <= 10
    valid_lengths = all(15 <= len((r.get("quote_text") or "").split()) <= 35
                        for r in rows)
    per_spk = {}
    for spk in row_speakers:
        if spk:
            per_spk[spk] = per_spk.get(spk, 0) + 1
    per_spk_ok = (len(per_spk) >= 2
                  and all(c >= 3 for c in per_spk.values()))
    cnt_ok = valid_count and valid_lengths and per_spk_ok

    # quote_diversity: pairwise Jaccard < 0.40
    sims = []
    bad_pairs = []
    sets = [_tokenset(r.get("quote_text", "")) for r in rows]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            if not a or not b:
                continue
            jac = len(a & b) / max(1, len(a | b))
            sims.append(jac)
            if jac >= 0.40:
                bad_pairs.append({
                    "i": rows[i].get("quote_id"),
                    "j": rows[j].get("quote_id"),
                    "jaccard": round(jac, 3),
                })
    diversity_ok = len(bad_pairs) == 0
    max_jac = round(max(sims), 3) if sims else 0.0

    # first_appearance_acc within +/- 1.0 s
    fa_total, fa_ok = 0, 0
    for s in gt["speakers"]:
        gt_name = s["name"]
        gt_t = s["first_appearance_sec"]
        match = next((x for x in lt
                      if normalize(x.get("speaker_name", "")) == normalize(gt_name)),
                     None)
        if match is None:
            continue
        fa_total += 1
        try:
            t = float(match.get("first_appearance_sec", -100))
            if abs(t - gt_t) <= 1.0:
                fa_ok += 1
        except (ValueError, TypeError):
            pass
    fa_ratio = fa_ok / max(1, len(gt["speakers"]))

    # cross_doc_consistency
    cdc_ok = True
    cdc_err = []
    summary = (out / "summary.md").read_text() if (out / "summary.md").exists() else ""
    seen_ids = set()
    for r in rows:
        qid = (r.get("quote_id") or "").strip()
        if not qid:
            cdc_ok = False; cdc_err.append("empty quote_id"); continue
        if qid in seen_ids:
            cdc_ok = False; cdc_err.append(f"duplicate quote_id {qid}")
        seen_ids.add(qid)
        if not (out / "broadcast_pack" / f"quote_{qid}.txt").exists():
            cdc_ok = False; cdc_err.append(f"missing broadcast txt for {qid}")
        chyron = normalize(r.get("broadcast_chyron", ""))
        lt_label = normalize(r.get("lower_third_label", ""))
        if not chyron or not lt_label:
            cdc_ok = False; cdc_err.append(f"empty chyron/lower_third for {qid}")
        if qid not in summary:
            cdc_ok = False; cdc_err.append(f"summary missing {qid}")

    return {
        "quote_provenance": {
            "verbatim_contiguous_single_speaker": prov_ok,
            "total": prov_total,
            "ratio": round(prov_ratio, 3),
            "ok": prov_ratio >= 1.0,
        },
        "speaker_attribution": {
            "matched": attr_ok, "total": attr_total,
            "ratio": round(attr_ratio, 3),
            "ok": attr_ratio >= 0.90,
        },
        "first_appearance_acc": {
            "ok_count": fa_ok, "total": len(gt["speakers"]),
            "ratio": round(fa_ratio, 3),
            "tolerance_sec": 1.0,
            "ok": fa_ratio >= 0.85,
        },
        "quote_count_in_range": {
            "n_rows": n,
            "valid_count": valid_count,
            "valid_lengths": valid_lengths,
            "per_speaker": per_spk,
            "per_speaker_min_3_each": per_spk_ok,
            "ok": cnt_ok,
        },
        "quote_diversity": {
            "max_jaccard": max_jac,
            "violating_pairs": bad_pairs[:5],
            "ok": diversity_ok,
        },
        "cross_doc_consistency": {
            "errors": cdc_err[:5],
            "ok": cdc_ok,
        },
    }


# --------------------------------------------------------------------------- #
# Quote audio grounding (tightened: WER <= 0.35, ratio >= 0.80)
# --------------------------------------------------------------------------- #
def quote_audio_grounding(audio_path: Path, rows: list,
                          max_samples: int = 5,
                          wer_threshold: float = 0.35,
                          ratio_threshold: float = 0.80) -> dict:
    if os.environ.get("CLAW_BENCH_SKIP_ASR") == "1":
        return {"skipped": True, "ratio": 0.0, "ok": False,
                "reasons": "CLAW_BENCH_SKIP_ASR=1"}
    if not audio_path.exists():
        return {"skipped": True, "ratio": 0.0, "ok": False,
                "reasons": f"missing {audio_path}"}
    if not rows:
        return {"skipped": True, "ratio": 0.0, "ok": False,
                "reasons": "no quote rows"}
    try:
        from asr_judge import transcribe, wer  # type: ignore
    except ImportError as e:
        return {"skipped": True, "ratio": 0.0, "ok": False,
                "reasons": f"asr_judge import failed: {e}"}

    rng = random.Random(0xCAFE)
    pool = list(range(len(rows)))
    rng.shuffle(pool)
    chosen_idx = pool[:max_samples]

    samples = []
    n_pass = 0
    werr_total = 0.0
    with tempfile.TemporaryDirectory() as td:
        for idx in chosen_idx:
            r = rows[idx]
            try:
                ts = float(r.get("start_sec", 0))
                te = float(r.get("end_sec", 0))
            except (ValueError, TypeError):
                samples.append({"idx": idx, "err": "bad timestamps"})
                continue
            dur = te - ts
            if dur <= 0.5 or dur > 60.0:
                samples.append({"idx": idx, "err": f"bad dur={dur}"})
                continue
            quote_text = (r.get("quote_text") or "").strip()
            if not quote_text:
                samples.append({"idx": idx, "err": "empty quote_text"})
                continue
            wav = Path(td) / f"q_{idx}.wav"
            try:
                subprocess.check_call([
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", f"{ts:.2f}", "-t", f"{dur:.2f}",
                    "-i", str(audio_path),
                    "-ar", "16000", "-ac", "1", str(wav),
                ], timeout=120)
                asr = transcribe(wav, language="en")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
                samples.append({"idx": idx, "err": str(e)[:120]})
                continue
            hyp = (asr.get("text") or "").strip()
            if len(hyp) < 5:
                samples.append({"idx": idx, "skipped_no_speech": True})
                continue
            w = wer(hyp, quote_text, language="en")
            werr_total += w
            passed = w <= wer_threshold
            if passed:
                n_pass += 1
            samples.append({
                "idx": idx,
                "start": round(ts, 1),
                "wer": round(w, 3),
                "passed": passed,
                "asr_preview": hyp[:120],
                "quote_preview": quote_text[:120],
            })
    informative = sum(1 for s in samples if "wer" in s)
    ratio = n_pass / max(1, informative)
    mean_wer = werr_total / max(1, informative)
    return {
        "informative_quotes": informative,
        "passed_quotes": n_pass,
        "ratio": round(ratio, 3),
        "mean_wer": round(mean_wer, 3),
        "wer_threshold": wer_threshold,
        "ratio_threshold": ratio_threshold,
        "samples": samples,
        "ok": informative >= 1 and ratio >= ratio_threshold,
    }


SUMMARY_AXES = [
    ("speakers_listed",
     "Does the memo list every speaker from GROUND_DATA with role + "
     "organisation (not just bare names)?"),
    ("quote_pitches_present",
     "Does the memo include a 1-sentence editorial pitch per quote, "
     "referencing quote_ids from GROUND_DATA (every quote_id should be "
     "mentioned)?"),
    ("fluent_prose",
     "Does the memo read as fluent prose <= 400 words, not lorem ipsum / "
     "placeholder / pure heading dump / ID-list-without-prose?"),
    ("plausible_topics",
     "Are the pitches self-consistent — do they refer to topics that are "
     "plausible given the speaker name and organisation context from "
     "GROUND_DATA?"),
    ("no_fabricated_ids",
     "Is the memo free of fabricated quote_ids that aren't in the supplied "
     "GROUND_DATA quote_ids list?"),
]


def summary_axes_score(out: Path, rows: list, gt: dict) -> dict:
    summary = (out / "summary.md").read_text()
    ground = {
        "speakers": [
            {"name": s.get("name"), "role": s.get("role"),
             "org": s.get("org")}
            for s in (gt.get("speakers") or [])
        ],
        "quote_ids": [r.get("quote_id") for r in rows[:10]],
        "n_quotes": len(rows),
    }
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground, ensure_ascii=False)}\n\n"
        f"SUMMARY.MD:\n{summary[:6000]}\n"
    )
    return axes_yes_no_score(payload, SUMMARY_AXES, threshold=0.70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--audio", default="fixtures/interview.wav")
    args = ap.parse_args()
    out = Path(args.output)
    ok, errs = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs},
                         indent=2))
        return
    transcript = json.load(open(args.transcript))
    gt = json.load(open(args.gt))
    res = evaluate(out, transcript, gt)

    audio_path = Path(args.audio)
    if not audio_path.is_absolute() and not audio_path.exists():
        cand = Path(__file__).resolve().parent / "fixtures" / "interview.wav"
        if cand.exists():
            audio_path = cand
    rows = list(csv.DictReader((out / "quote_sheet.csv").open()))
    qag = quote_audio_grounding(audio_path, rows)
    sq = summary_axes_score(out, rows, gt)
    res["quote_audio_grounding"] = {**qag, "ok": qag.get("ok", False)}
    res["summary_axes"] = {
        "axes": sq.get("axes", []),
        "score": sq.get("score", 0.0),
        "ok": bool(sq.get("ok", False)),
        "threshold": 0.70,
    }

    all_ok = all(v["ok"] for v in res.values())
    # v3: deterministic dims ~0.70, ASR ~0.15, axes ~0.15
    score = (
        0.15 * (1.0 if res["quote_provenance"]["ok"] else
                res["quote_provenance"]["ratio"])
        + 0.20 * res["speaker_attribution"]["ratio"]
        + 0.10 * res["first_appearance_acc"]["ratio"]
        + 0.10 * (1 if res["quote_count_in_range"]["ok"] else 0)
        + 0.10 * (1 if res["quote_diversity"]["ok"] else 0)
        + 0.05 * (1 if res["cross_doc_consistency"]["ok"] else 0)
        + 0.15 * res["quote_audio_grounding"].get("ratio", 0.0)
        + 0.15 * res["summary_axes"]["score"]
    )
    print(json.dumps({
        "pass": all_ok,
        "checks": {k: v["ok"] for k, v in res.items()},
        **res,
        "score_so_far": round(score, 3),
        "weights": {
            "quote_provenance": 0.15,
            "speaker_attribution": 0.20,
            "first_appearance_acc": 0.10,
            "quote_count_in_range": 0.10,
            "quote_diversity": 0.10,
            "cross_doc_consistency": 0.05,
            "quote_audio_grounding": 0.15,
            "summary_axes": 0.15,
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
