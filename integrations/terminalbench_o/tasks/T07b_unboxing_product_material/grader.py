"""Grader for T07b Unboxing -> Product Material (v3 deterministic).

v3 redesign:
  - Selling-point title matching is strict normalised string compare
    against the GT mentioned/decoy lists (closed vocab w/ decoys).
  - Removed `transcript_grounding` ASR dim and the single-LLM
    `summary_quality` dim.
  - `summary_axes` (4 yes/no axes) replaces the LLM rubric judge.
  - Reweighted: deterministic dims ~0.80, axes ~0.20.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))
from open_vocab_judge import axes_yes_no_score, mllm_video_axes

# Multimodal checklist for one short-form clip (shorts/short_01.mp4 by
# convention). Sample 4 evenly-spaced frames; majority vote per axis.
SHORT_CONTENT_AXES = [
    ("real_unboxing_or_product_footage",
     "Is this frame natural unboxing or product-review footage — NOT "
     "a black screen, color bars, test pattern, or single-color "
     "placeholder?"),
    ("product_or_person_visible",
     "Is at least one of a product (e.g. an Apple Pencil, gadget, "
     "package) or a person (hands / face / reviewer) visible in "
     "this frame?"),
    ("continuous_video_not_slideshow",
     "Does this frame look like it was sampled from continuous video "
     "footage rather than a static slideshow / single text-only title "
     "card / motionless thumbnail?"),
    ("vertical_or_short_form_aesthetic_ok",
     "Does the framing / composition look acceptable for a short-form "
     "(vertical or square) clip — not heavily blurred, not pure "
     "black bars, not a corrupted frame?"),
]


def gating(out: Path):
    errs = []
    if not (out / "selling_points.json").exists():
        errs.append("missing selling_points.json")
    if not (out / "shorts").is_dir():
        errs.append("missing shorts/")
    if not (out / "cover_pack").is_dir():
        errs.append("missing cover_pack/")
    if not (out / "summary.md").exists():
        errs.append("missing summary.md")
    return len(errs) == 0, errs


def normalize(s):
    return re.sub(r"\s+", " ", (s or "").lower().strip())


_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(s):
    return _TOKEN_RE.findall((s or "").lower())


def fuzzy_quote_match(quote, transcript_tokens, threshold=0.85):
    """ASR-friendly contiguous-span match.

    Returns True if any sliding window of `transcript_tokens` of length
    len(q_tokens) shares a token-set Jaccard ratio >= threshold with the
    quote's tokens. Tolerates ASR substitution / casing / punctuation.
    """
    q_tokens = _tokens(quote)
    if not q_tokens or len(q_tokens) > len(transcript_tokens):
        return False
    qset = set(q_tokens)
    qlen = len(q_tokens)
    best = 0.0
    for i in range(len(transcript_tokens) - qlen + 1):
        window = set(transcript_tokens[i:i + qlen])
        inter = len(qset & window)
        union = len(qset | window)
        if union == 0:
            continue
        ratio = inter / union
        if ratio > best:
            best = ratio
            if best >= threshold:
                return True
    return False


def _norm_title(s):
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


def ffprobe_dur(path):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                              "format=duration", "-of",
                              "default=noprint_wrappers=1:nokey=1",
                              str(path)],
                            capture_output=True, text=True, timeout=15)
        return float((r.stdout or "0").strip())
    except Exception:
        return -1.0


def evaluate(out: Path, transcript, gt):
    sp = json.load((out / "selling_points.json").open()).get("selling_points", [])
    summary = (out / "summary.md").read_text()

    gt_mentioned_norm = {_norm_title(g["title"]) for g in gt["mentioned"]}
    gt_decoy_norm = {_norm_title(d["title"]) for d in gt["decoys"]}
    pred_norm = {_norm_title(s.get("title", "")) for s in sp if s.get("title")}

    # selling_point_recall — strict normalised match against GT mentioned
    matched_sp = len(pred_norm & gt_mentioned_norm)
    sp_recall = matched_sp / max(1, len(gt_mentioned_norm))

    # decoy_rejection_acc — strict normalised match against decoys
    decoy_listed = len(pred_norm & gt_decoy_norm)
    decoy_reject = 1.0 - decoy_listed / max(1, len(gt_decoy_norm))

    # quote_provenance — fuzzy contiguous span match against held-out
    # reference transcript (ASR-friendly: token-set ratio >= 0.85 over a
    # sliding window of the quote's token length).
    transcript_tokens = _tokens(" ".join(w["text"] for w in transcript["words"]))
    prov_total, prov_ok = 0, 0
    for s in sp:
        prov_total += 1
        q = s.get("verbatim_quote", "")
        if q and fuzzy_quote_match(q, transcript_tokens, threshold=0.85):
            prov_ok += 1
    prov_ratio = prov_ok / max(1, prov_total)

    # shorts_count_and_length
    shorts = sorted((out / "shorts").glob("*.mp4"))
    n_shorts = len(shorts)
    durs_ok = 0
    for s in shorts:
        d = ffprobe_dur(s)
        if 12.0 <= d <= 25.0:
            durs_ok += 1
    sc_ok = (n_shorts == 3) and (durs_ok == 3)

    # cross_product_consistency
    cdc_ok = True
    cdc_err = []
    have_covers = {p.name for p in (out / "cover_pack").glob("*.png")}
    for s in sp:
        short = (s.get("supporting_short") or "")
        cov = (s.get("supporting_cover") or "")
        if not short or not (out / short).exists():
            cdc_ok = False; cdc_err.append(f"missing short {short}")
        if cov:
            if not any(cov in n for n in have_covers):
                cdc_ok = False; cdc_err.append(f"cover {cov} not in cover_pack/")
    decoys_named = sum(1 for d in gt["decoys"]
                        if normalize(d["title"]) in normalize(summary))
    if decoys_named < 2:
        cdc_ok = False; cdc_err.append(f"summary names only {decoys_named} decoys")

    return {
        "selling_point_recall": {
            "matched": matched_sp, "n_gt": len(gt_mentioned_norm),
            "ratio": round(sp_recall, 3),
            "ok": sp_recall >= 0.70,
        },
        "decoy_rejection_acc": {
            "listed_decoys": decoy_listed,
            "n_decoys": len(gt_decoy_norm),
            "ratio": round(decoy_reject, 3),
            "ok": decoy_reject >= 0.80,
        },
        "quote_provenance": {
            "verbatim": prov_ok, "total": prov_total,
            "ratio": round(prov_ratio, 3),
            "ok": prov_ratio >= 0.80,
        },
        "shorts_count_and_length": {
            "n": n_shorts, "len_ok": durs_ok,
            "ok": sc_ok,
        },
        "cross_product_consistency": {
            "errors": cdc_err[:5],
            "ok": cdc_ok,
        },
    }


SUMMARY_AXES = [
    ("identifies_product",
     "Does the SUMMARY identify the product as the Apple Pencil 2nd-gen "
     "(comparison vs USB-C variant)?"),
    ("lists_real_selling_points",
     "Does the SUMMARY mention >=3 of the genuine selling points "
     "(magnetic side charging, double-tap to switch tool, pressure "
     "sensitivity, customizable double-tap actions)?"),
    ("decoys_section_present",
     "Does the SUMMARY include a `## Decoys avoided` section that names "
     ">=2 candidate selling points NOT actually in the review?"),
    ("coherent_prose",
     "Is the SUMMARY coherent prose under 500 words (not lorem ipsum, "
     "placeholder, pure heading dump, or copy-pasted from the brief)?"),
]


def summary_axes(out: Path, sp_predicted, gt):
    summary = (out / "summary.md").read_text()
    payload = (
        "GROUND_DATA:\n" + json.dumps({
            "mentioned_titles": [g.get("title") for g in gt.get("mentioned", [])],
            "decoy_titles": [d.get("title") for d in gt.get("decoys", [])],
            "agent_sp_titles": [s.get("title") for s in sp_predicted][:10],
        }, ensure_ascii=False) + "\n\n"
        f"SUMMARY.MD:\n{summary[:6000]}\n"
    )
    return axes_yes_no_score(payload, SUMMARY_AXES, threshold=0.70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--video", default="fixtures/review.mp4")
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

    sp_predicted = json.load((out / "selling_points.json").open()).get(
        "selling_points", [])
    sa = summary_axes(out, sp_predicted, gt)
    res["summary_axes"] = {
        "axes": sa.get("axes", []),
        "score": sa.get("score", 0.0),
        "ok": bool(sa.get("ok", False)),
        "threshold": 0.70,
    }

    short_path = out / "shorts" / "short_01.mp4"
    sv = mllm_video_axes(short_path, SHORT_CONTENT_AXES,
                          n_frames=4, threshold=0.70)
    res["short_video_axes"] = {
        "axes": sv.get("axes", []),
        "score": sv.get("score", 0.0),
        "ok": bool(sv.get("ok", False)),
        "threshold": 0.70,
        "sampled_clip": "shorts/short_01.mp4",
    }

    all_ok = all(v["ok"] for v in res.values())
    score = (
        0.20 * res["selling_point_recall"]["ratio"]
        + 0.20 * res["decoy_rejection_acc"]["ratio"]
        + 0.10 * res["quote_provenance"]["ratio"]
        + 0.10 * (1 if res["shorts_count_and_length"]["ok"] else 0)
        + 0.10 * (1 if res["cross_product_consistency"]["ok"] else 0)
        + 0.15 * res["summary_axes"]["score"]
        + 0.15 * res["short_video_axes"]["score"]
    )
    print(json.dumps({
        "pass": all_ok,
        "checks": {k: v["ok"] for k, v in res.items()},
        **res,
        "score_so_far": round(score, 3),
        "weights": {
            "selling_point_recall": 0.20,
            "decoy_rejection_acc": 0.20,
            "quote_provenance": 0.10,
            "shorts_count_and_length": 0.10,
            "cross_product_consistency": 0.10,
            "summary_axes": 0.15,
            "short_video_axes": 0.15,
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
