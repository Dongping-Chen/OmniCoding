"""Grader for T33a CEO Interview Topic Index (v3 deterministic).

v3 redesign:
  - Bucket labels are closed-vocab {STRATEGY, PRODUCT, ORG, FILLER};
    sentence-level macro-F1 stays deterministic.
  - Restored `clip_audio_grounding` ASR dim: samples up to 3 topics,
    ASRs the agent's clip in English and compares vs the concatenated
    transcript sentences inside the topic span (WER). Catches silence
    / wrong-content / shifted-window clips that pass duration checks.
  - Added `summary_axes` (4 yes/no axes) for the free-form prose.
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))
from open_vocab_judge import axes_yes_no_score


def gating(out: Path):
    errs = []
    if not (out / "topic_index.json").exists():
        errs.append("missing topic_index.json")
    if not (out / "coverage.csv").exists():
        errs.append("missing coverage.csv")
    if not (out / "summary.md").exists():
        errs.append("missing summary.md")
    if not (out / "clips").is_dir():
        errs.append("missing clips/")
    return len(errs) == 0, errs


def probe(p: Path) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of",
             "default=noprint_wrappers=1:nokey=1", str(p)],
            text=True)
        return float(out.strip())
    except Exception:
        return 0.0


SUMMARY_AXES = [
    ("strategy_section",
     "Does the SUMMARY have a `## strategy` (or `## Strategy`) section "
     "with >=2 substantive sentences synthesizing long-term direction "
     "(mission, market, geography, framing of why)?"),
    ("product_section",
     "Does the SUMMARY have a `## product` (or `## Product`) section "
     "that names >=2 concrete product features (e.g. gamification, "
     "AI capabilities, language proficiency, streaks, lessons)?"),
    ("organization_section",
     "Does the SUMMARY have a `## organization` (or `## organisation` "
     "/ `## Org`) section covering hiring, culture, or leadership?"),
    ("real_prose_no_placeholder",
     "Each section reads as real summary prose (>=2 substantive "
     "sentences, NOT lorem ipsum, NOT placeholder, NOT a copy of the "
     "bucket name); summary contradicts no GROUND_DATA bucket counts."),
]


def clip_audio_grounding(out: Path, topics: list,
                         transcript_csv: Path,
                         n_sample: int = 3,
                         wer_max: float = 0.40,
                         pass_ratio_min: float = 0.65) -> dict:
    """Sample up to `n_sample` topics, ASR each agent clip in English
    and compare to the concatenated transcript text inside that topic
    span.  Returns:
      - per-sample WER + transcript preview
      - n_informative / n_passed / ratio
      - ok = (n_informative >= 2 AND ratio >= pass_ratio_min)
    """
    if os.environ.get("CLAW_BENCH_SKIP_ASR") == "1":
        return {"skipped": True, "ratio": 0.0,
                "ok": False, "reasons": "CLAW_BENCH_SKIP_ASR=1"}
    try:
        from asr_judge import asr_score  # type: ignore
    except Exception as e:
        return {"skipped": True, "ratio": 0.0,
                "ok": False, "reasons": f"asr_judge import failed: {e}"}

    transcript = list(csv.DictReader(transcript_csv.open()))
    sent_rows = []
    for r in transcript:
        try:
            sent_rows.append({
                "t_start": float(r["t_start"]),
                "t_end": float(r["t_end"]),
                "text": (r.get("text") or "").strip(),
            })
        except Exception:
            continue

    clips_dir = out / "clips"
    eligible = []
    for t in topics:
        try:
            ts = float(t.get("t_start"))
            te = float(t.get("t_end"))
        except Exception:
            continue
        if (te - ts) < 5.0:
            continue
        bucket = (t.get("bucket") or "").strip()
        if bucket.upper() == "FILLER":
            continue
        tid = t.get("id")
        if not tid:
            continue
        clip = clips_dir / bucket / f"{tid}.wav"
        if not clip.exists():
            for cand in clips_dir.rglob(f"{tid}.wav"):
                clip = cand
                break
        if not clip.exists():
            continue
        ref_text = " ".join(s["text"] for s in sent_rows
                              if s["t_start"] >= ts - 0.5 and
                                 s["t_end"] <= te + 0.5).strip()
        if len(ref_text.split()) < 8:
            continue
        eligible.append({"tid": tid, "bucket": bucket, "clip": clip,
                          "span": [ts, te], "ref": ref_text})

    if not eligible:
        return {"skipped": True, "ratio": 0.0, "ok": False,
                "reasons": "no eligible topic clips"}

    rng = random.Random(0xC2EE)
    rng.shuffle(eligible)
    samples = []
    n_informative = 0
    n_passed = 0
    for item in eligible[:n_sample]:
        try:
            r = asr_score(item["clip"], item["ref"],
                            language="en", metric="wer")
        except Exception as e:
            samples.append({"tid": item["tid"],
                              "asr_err": str(e)[:120]})
            continue
        n_informative += 1
        wer = r.get("wer")
        ok = wer is not None and wer <= wer_max
        if ok:
            n_passed += 1
        samples.append({
            "tid": item["tid"],
            "bucket": item["bucket"],
            "span": item["span"],
            "ref_preview": item["ref"][:120],
            "asr_preview": (r.get("transcript") or "")[:120],
            "wer": round(wer, 3) if wer is not None else None,
            "ok": bool(ok),
        })

    ratio = n_passed / max(1, n_informative) if n_informative else 0.0
    return {
        "samples": samples,
        "n_informative": n_informative,
        "n_passed": n_passed,
        "ratio": round(ratio, 3),
        "wer_max": wer_max,
        "pass_ratio_min": pass_ratio_min,
        "ok": n_informative >= 2 and ratio >= pass_ratio_min,
    }


def summary_axes_score(out: Path, topics: list) -> dict:
    summary = (out / "summary.md").read_text()
    counts = defaultdict(int)
    for t in topics:
        counts[(t.get("bucket") or "").upper()] += 1
    payload = (
        "GROUND_DATA:\n" + json.dumps({
            "bucket_counts": dict(counts),
            "topic_titles": [t.get("title") for t in topics][:20],
        }, ensure_ascii=False) + "\n\n"
        f"SUMMARY.MD:\n{summary[:6000]}\n"
    )
    return axes_yes_no_score(payload, SUMMARY_AXES, threshold=0.70)


def evaluate(out: Path, gt, transcript_csv: Path):
    topics = json.load((out / "topic_index.json").open())\
        .get("topics", [])
    cov = list(csv.DictReader((out / "coverage.csv").open()))
    summary = (out / "summary.md").read_text()
    transcript = list(csv.DictReader(transcript_csv.open()))

    sent_truth = {s["sentence_no"]: s for s in gt["sentence_truth"]}
    filler_set = set(gt["filler_sentence_nos"])
    bucket_set = set(gt["buckets"])

    sent_starts = {int(r["sentence_no"]): float(r["t_start"])
                    for r in transcript}
    sent_ends = {int(r["sentence_no"]): float(r["t_end"])
                  for r in transcript}

    cov_by_sno = {}
    for r in cov:
        try:
            sno = int(r.get("sentence_no"))
        except Exception:
            continue
        cov_by_sno[sno] = r

    pred_bucket_by_sno = {}
    for sno, r in cov_by_sno.items():
        b = (r.get("bucket") or "").strip().upper()
        if b in bucket_set:
            pred_bucket_by_sno[sno] = b

    f1s = []
    for b in bucket_set:
        true_pos = pred_pos = total_true = 0
        for sno, st in sent_truth.items():
            true_b = st["bucket"]
            pred_b = pred_bucket_by_sno.get(sno)
            if true_b == b: total_true += 1
            if pred_b == b: pred_pos += 1
            if true_b == b and pred_b == b: true_pos += 1
        prec = true_pos / pred_pos if pred_pos else 0.0
        rec = true_pos / total_true if total_true else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    ba_total = 0; ba_ok = 0
    for t in topics:
        try:
            ts = float(t.get("t_start"))
            te = float(t.get("t_end"))
        except Exception:
            continue
        ba_total += 2
        if any(abs(ts - v) <= 0.2 for v in sent_starts.values()):
            ba_ok += 1
        if any(abs(te - v) <= 0.2 for v in sent_ends.values()):
            ba_ok += 1
    ba_ratio = ba_ok / max(1, ba_total) if ba_total else 0.0

    fe_total = len(filler_set); fe_ok = 0
    for sno in filler_set:
        r = cov_by_sno.get(sno)
        if not r:
            continue
        tid = (r.get("topic_id") or "").strip().lower()
        b = (r.get("bucket") or "").strip().upper()
        if tid == "filler" or b == "FILLER":
            fe_ok += 1
    fe_ratio = fe_ok / max(1, fe_total) if fe_total else 1.0

    cdm_ok = True; cdm_err = []
    clips_dir = out / "clips"
    for t in topics:
        tid = t.get("id")
        bucket = (t.get("bucket") or "").strip()
        clip = (clips_dir / bucket / f"{tid}.wav")
        if not clip.exists():
            for cand in clips_dir.rglob(f"{tid}.wav"):
                clip = cand; break
        if not clip.exists():
            cdm_ok = False; cdm_err.append(f"{tid} missing")
            continue
        d = probe(clip)
        try:
            span = float(t.get("t_end")) - float(t.get("t_start"))
        except Exception:
            cdm_ok = False; cdm_err.append(f"{tid} bad span")
            continue
        if d > 90.5:
            cdm_ok = False; cdm_err.append(f"{tid} clip {d:.1f}s > 90")
        elif abs(d - span) > 2.0:
            cdm_ok = False
            cdm_err.append(f"{tid} clip {d:.1f} vs span {span:.1f}")

    cdc_ok = True; cdc_err = []
    if len(topics) < 9:
        cdc_ok = False; cdc_err.append(f"only {len(topics)} topics")
    bucket_counts = defaultdict(int)
    for t in topics:
        bucket_counts[(t.get("bucket") or "").strip().upper()] += 1
    for b in bucket_set:
        if bucket_counts[b] < 2:
            cdc_ok = False
            cdc_err.append(f"bucket {b} only {bucket_counts[b]}")
    # Out-of-vocab buckets in coverage.csv are flagged here too.
    cov_buckets = {(r.get("bucket") or "").strip().upper()
                   for r in cov if (r.get("bucket") or "").strip()}
    bad_buckets = cov_buckets - bucket_set
    if bad_buckets:
        cdc_ok = False
        cdc_err.append(f"coverage out-of-vocab buckets {sorted(bad_buckets)[:3]}")
    expected = set(sent_truth.keys())
    got = set(cov_by_sno.keys())
    if got != expected:
        cdc_ok = False
        miss = expected - got
        cdc_err.append(f"coverage missing {sorted(miss)[:5]}")
    low = summary.lower()
    for h in ("strategy", "product", "organization"):
        if f"## {h}" not in low:
            cdc_ok = False; cdc_err.append(f"summary missing ## {h}")

    return {
        "bucket_macro_f1": {
            "value": round(macro_f1, 3),
            "ok": macro_f1 >= 0.70,
        },
        "boundary_alignment": {
            "matched": ba_ok, "total": ba_total,
            "ratio": round(ba_ratio, 3),
            "ok": ba_ratio >= 0.90,
        },
        "filler_exclusion": {
            "matched": fe_ok, "total": fe_total,
            "ratio": round(fe_ratio, 3),
            "ok": fe_ratio >= 0.85,
        },
        "clip_duration_match": {
            "errors": cdm_err[:5],
            "ok": cdm_ok,
        },
        "cross_doc_consistency": {
            "errors": cdc_err[:5],
            "ok": cdc_ok,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--audio", default="fixtures/interview.wav")
    args = ap.parse_args()
    out = Path(args.output)
    ok, errs = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs},
                         indent=2))
        return
    gt = json.load(open(args.gt))
    res = evaluate(out, gt, Path(args.transcript))

    topics = json.load((out / "topic_index.json").open()).get("topics", [])
    sa = summary_axes_score(out, topics)
    res["summary_axes"] = {
        "axes": sa.get("axes", []),
        "score": sa.get("score", 0.0),
        "ok": bool(sa.get("ok", False)),
        "threshold": 0.70,
    }
    cag = clip_audio_grounding(out, topics, Path(args.transcript))
    res["clip_audio_grounding"] = {
        **cag,
        "ok": bool(cag.get("ok", False)),
    }

    all_ok = all(v["ok"] for v in res.values())
    score = (
        0.20 * res["bucket_macro_f1"]["value"]
        + 0.18 * res["boundary_alignment"]["ratio"]
        + 0.13 * res["filler_exclusion"]["ratio"]
        + 0.10 * (1 if res["clip_duration_match"]["ok"] else 0)
        + 0.12 * (1 if res["cross_doc_consistency"]["ok"] else 0)
        + 0.13 * res["summary_axes"]["score"]
        + 0.14 * res["clip_audio_grounding"].get("ratio", 0.0)
    )
    print(json.dumps({
        "pass": all_ok,
        "checks": {k: v["ok"] for k, v in res.items()},
        **res,
        "score_so_far": round(score, 3),
        "weights": {
            "bucket_macro_f1": 0.20,
            "boundary_alignment": 0.18,
            "filler_exclusion": 0.13,
            "clip_duration_match": 0.10,
            "cross_doc_consistency": 0.12,
            "summary_axes": 0.13,
            "clip_audio_grounding": 0.14,
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
