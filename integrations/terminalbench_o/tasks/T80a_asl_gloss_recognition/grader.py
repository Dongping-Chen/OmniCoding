"""Grader for T80a ASL Continuous Sign Language Gloss Recognition (v3).

v3 redesign:
  - Closed-vocab fields (gloss in 200-word vocab, non_manual in 5-enum,
    decoy_type in 4-enum) remain strict normalised match — no LLM, no
    VLM. Decoy categories already act as decoys vs the gloss vocab.
  - translation.md is genuinely free-form prose; replaced single VLM
    judge_memo_grounded with 5-axis yes/no `translation_quality_axes`.
  - Reweighted: deterministic dims dominate (~0.85); only the prose
    translation uses LLM axes (0.15).

6 dimensions (weights sum 1.0):
  gloss_recall              >= 0.55  (0.23)  hungarian on (IoU>0.3 + gloss==)
  temporal_iou              >= 0.45  (0.18)  mean IoU over hits only
  decoy_rejection           >= 0.75  (0.13)  IoU>0.3 + decoy_type match
  translation_blue          >= 0.30  (0.10)  sacrebleu corpus_bleu / 100
  cross_doc_consistency      = 1.0   (0.21)  closed vocab + monotonic + counts
  translation_quality_axes  >= 0.70  (0.15)  5-axis yes/no on translation.md
"""
import argparse, csv, json, os, re, sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))

from open_vocab_judge import axes_yes_no_score  # noqa: E402

REQUIRED_FILES = ["gloss_timeline.csv", "decoy_rejections.json",
                  "translation.md", "fluency_review.md"]
CSV_FIELDS = ["idx", "start_sec", "end_sec", "gloss", "non_manual", "confidence"]
NM_ENUM = {"wh_question", "yn_question", "negation", "topic", "rhetorical", ""}


def gating(out: Path):
    errs = []
    for f in REQUIRED_FILES:
        p = out / f
        if not p.exists():
            errs.append(f"missing: {f}")
        elif p.stat().st_size == 0:
            errs.append(f"empty: {f}")
    if errs:
        return False, errs
    with open(out / "gloss_timeline.csv") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or set(r.fieldnames) < set(CSV_FIELDS):
            errs.append(f"gloss_timeline.csv missing fields, got {r.fieldnames}")
            return False, errs
        rows = list(r)
    if len(rows) < 200:
        errs.append(f"gloss_timeline.csv has only {len(rows)} rows (<200)")
    try:
        decoys = json.load(open(out / "decoy_rejections.json"))
        if not isinstance(decoys, list) or len(decoys) < 8:
            errs.append(f"decoy_rejections.json must be a list of >=8 entries, "
                        f"got {type(decoys).__name__} len={len(decoys) if isinstance(decoys, list) else 'NA'}")
    except Exception as e:
        errs.append(f"decoy_rejections.json parse: {e}")
    return len(errs) == 0, errs


def load_pred(out: Path):
    rows = []
    with open(out / "gloss_timeline.csv") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "idx": int(r["idx"]),
                    "start": float(r["start_sec"]),
                    "end":   float(r["end_sec"]),
                    "gloss": (r["gloss"] or "").strip(),
                    "non_manual": (r.get("non_manual") or "").strip().lower(),
                    "confidence": float(r["confidence"]) if r.get("confidence") else 0.0,
                })
            except (ValueError, KeyError):
                pass
    decoys = json.load(open(out / "decoy_rejections.json"))
    return rows, decoys


def iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def hungarian_match(preds, gts, iou_thr=0.3):
    """Return list of (gt_idx, pred_idx, iou) hits requiring strict gloss equality."""
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return None
    n_g, n_p = len(gts), len(preds)
    if n_g == 0 or n_p == 0:
        return []
    # cost = -IoU if gloss equal AND IoU > thr, else +1 (no hit)
    cost = np.full((n_g, n_p), 1.0, dtype=float)
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            if g["gloss"] != p["gloss"]:
                continue
            ov = iou(g["start"], g["end"], p["start"], p["end"])
            if ov > iou_thr:
                cost[i, j] = -ov
    row_ind, col_ind = linear_sum_assignment(cost)
    hits = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < 0:
            hits.append((int(r), int(c), float(-cost[r, c])))
    return hits


def gloss_match_decoy(preds_decoy, gts_decoy, iou_thr=0.3):
    """Hungarian match decoys: hit if IoU>thr AND decoy_type equal."""
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return None
    if not preds_decoy or not gts_decoy:
        return []
    n_g, n_p = len(gts_decoy), len(preds_decoy)
    cost = np.full((n_g, n_p), 1.0)
    for i, g in enumerate(gts_decoy):
        for j, p in enumerate(preds_decoy):
            if (p.get("decoy_type") or "").strip() != g["decoy_type"]:
                continue
            ov = iou(g["t_start"], g["t_end"], p.get("t_start", 0), p.get("t_end", 0))
            if ov > iou_thr:
                cost[i, j] = -ov
    row_ind, col_ind = linear_sum_assignment(cost)
    hits = [(int(r), int(c)) for r, c in zip(row_ind, col_ind) if cost[r, c] < 0]
    return hits


def split_sentences(md_text):
    # Extract '## English Translation' section
    m = re.search(r"^##\s*English\s*Translation\s*$(.*?)(?=^##\s|\Z)",
                  md_text, flags=re.M | re.S | re.I)
    if not m:
        return []
    body = m.group(1).strip()
    # Strip markdown bullets
    body = re.sub(r"^\s*[-*]\s+", "", body, flags=re.M)
    body = body.replace("\n", " ")
    sents = re.split(r"(?<=[.!?])\s+", body)
    return [s.strip() for s in sents if s.strip()]


def english_translation_text(md_text):
    return " ".join(split_sentences(md_text))


def evaluate(out: Path, gt: dict, fixtures: Path) -> dict:
    pred_rows, pred_decoys = load_pred(out)
    gt_signs = [{"gloss": s["gloss"], "start": s["start_sec"], "end": s["end_sec"]}
                for s in gt["signs"]]
    pred_signs = [{"gloss": p["gloss"], "start": p["start"], "end": p["end"]}
                  for p in pred_rows]
    n_gt = len(gt_signs)

    # ---- gloss_recall (Hungarian, IoU>0.2 + strict gloss) -------------------
    # GT timestamps are derived from linear interpolation across each
    # English sentence's audio span, so per-sign onsets can drift ±1s
    # relative to the actual sign motion. IoU threshold loosened from 0.3
    # to 0.2 to absorb that drift; the strict gloss match still gates
    # correctness.
    hits = hungarian_match(pred_signs, gt_signs, iou_thr=0.2)
    if hits is None:
        gloss_recall = {"ratio": 0.0, "hits": 0, "n_gt": n_gt, "ok": False,
                        "error": "scipy / numpy missing"}
        temporal_iou = {"ratio": 0.0, "ok": False, "error": "scipy / numpy missing"}
    else:
        ratio_r = len(hits) / max(1, n_gt)
        gloss_recall = {"hits": len(hits), "n_gt": n_gt,
                        "ratio": round(ratio_r, 3), "ok": ratio_r >= 0.55}
        if hits:
            mean_iou = sum(h[2] for h in hits) / len(hits)
        else:
            mean_iou = 0.0
        temporal_iou = {"mean_iou_over_hits": round(mean_iou, 3),
                        "n_hits": len(hits),
                        "ratio": round(mean_iou, 3),
                        # Threshold lowered 0.45 -> 0.30 in step with the
                        # IoU match relaxation; interpolated GT onsets
                        # can't realistically reach 0.45 mean IoU.
                        "ok": mean_iou >= 0.30 and len(hits) > 0}

    # ---- decoy_rejection ----------------------------------------------------
    gt_decoys = gt.get("decoys", [])
    d_hits = gloss_match_decoy(pred_decoys, gt_decoys, iou_thr=0.3)
    if d_hits is None:
        decoy_rejection = {"ratio": 0.0, "ok": False, "error": "scipy missing"}
    else:
        d_ratio = len(d_hits) / max(1, len(gt_decoys))
        decoy_rejection = {"hits": len(d_hits), "n_gt": len(gt_decoys),
                           "ratio": round(d_ratio, 3), "ok": d_ratio >= 0.75}

    # ---- translation_blue ---------------------------------------------------
    md_text = (out / "translation.md").read_text(encoding="utf-8", errors="ignore")
    agent_en = english_translation_text(md_text)
    gt_en = " ".join(gt["translation_gt"])
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu([agent_en], [[gt_en]]).score / 100.0
        translation_blue = {"bleu": round(bleu, 3), "ratio": round(bleu, 3),
                            "ok": bleu >= 0.30}
    except ImportError:
        translation_blue = {"bleu": 0.0, "ratio": 0.0, "ok": False,
                            "error": "sacrebleu missing"}

    # ---- cross_doc_consistency ---------------------------------------------
    cd_errs = []
    vocab_path = fixtures / "gloss_vocab.json"
    vocab = set(json.load(open(vocab_path))["glosses"]) if vocab_path.exists() else set()
    for r in pred_rows:
        if r["gloss"] not in vocab:
            cd_errs.append(f"OOV gloss: {r['gloss']}")
            if len(cd_errs) >= 30:
                break
    if not cd_errs:
        for i in range(1, len(pred_rows)):
            if pred_rows[i]["start"] < pred_rows[i-1]["start"]:
                cd_errs.append(f"non-monotonic at idx {i}: "
                               f"{pred_rows[i-1]['start']} -> {pred_rows[i]['start']}")
                break
    for r in pred_rows:
        if r["end"] <= r["start"]:
            cd_errs.append(f"non-positive duration at idx {r['idx']}")
            break
        if r["non_manual"] not in NM_ENUM and r["non_manual"] != "null":
            cd_errs.append(f"bad non_manual {r['non_manual']!r} at idx {r['idx']}")
            break
    # decoy intervals must not overlap any sign
    for d in pred_decoys:
        ds, de = d.get("t_start"), d.get("t_end")
        if ds is None or de is None or de <= ds:
            cd_errs.append(f"bad decoy interval: {d}")
            break
        for r in pred_rows:
            if iou(r["start"], r["end"], ds, de) > 0:
                cd_errs.append(f"decoy {ds}-{de} overlaps gloss row at {r['start']}")
                break
        if cd_errs and cd_errs[-1].startswith("decoy"):
            break
    # fluency_review numeric check
    fr_text = (out / "fluency_review.md").read_text(encoding="utf-8", errors="ignore")
    word_count = len(re.findall(r"\S+", fr_text))
    if word_count > 400:
        cd_errs.append(f"fluency_review too long: {word_count} words (>400)")
    nums = [int(x) for x in re.findall(r"\b\d+\b", fr_text)]
    if len(pred_rows) not in nums:
        cd_errs.append(f"fluency_review must report total signs = {len(pred_rows)}")
    # ≥8 sentences in EN translation
    sents = split_sentences(md_text)
    if len(sents) < 8:
        cd_errs.append(f"## English Translation has only {len(sents)} sentences (<8)")
    if "## Gloss Sequence" not in md_text:
        cd_errs.append("translation.md missing '## Gloss Sequence' section")
    cross_doc_consistency = {"errors": cd_errs[:8], "n_errors": len(cd_errs),
                             "ok": len(cd_errs) == 0}

    return {
        "gloss_recall": gloss_recall,
        "temporal_iou": temporal_iou,
        "decoy_rejection": decoy_rejection,
        "translation_blue": translation_blue,
        "cross_doc_consistency": cross_doc_consistency,
    }


TRANSLATION_AXES = [
    ("english_translation_fluent",
     "Does the `## English Translation` section in translation.md "
     "contain natural fluent English prose (>=8 sentences, articles "
     "and tenses present) — not raw token concatenation like "
     "'I learn ASL three year'?"),
    ("topic_asl_lecture",
     "Is the translation's content topic plausibly an ASL lecture / "
     "tutorial (e.g. learning ASL, deaf identity, classroom signs, "
     "common ASL phrases) — and not off-topic content like cooking "
     "or sports?"),
    ("gloss_sequence_present",
     "Does translation.md contain a `## Gloss Sequence` section with "
     "UPPERCASE ASL glosses (e.g. `IX-1 LEARN ASL THREE-YEAR`) whose "
     "length is in the order of magnitude of GROUND_DATA.n_pred_signs?"),
    ("no_placeholders",
     "Is translation.md free of placeholder / lorem-ipsum / `<...>` / "
     "`TODO` markers in either section?"),
    ("clean_markdown",
     "Does translation.md read as clean Markdown (correct section "
     "headers, no copy-paste of the rubric or instruction text into "
     "the body)?"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--fixtures", default="fixtures")
    args = ap.parse_args()
    out = Path(args.output); fix = Path(args.fixtures)
    ok, errs = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs},
                         indent=2)); return
    gt = json.load(open(args.gt))
    res = evaluate(out, gt, fix)

    # translation_quality_axes: 5-axis yes/no on translation.md
    md_text = (out / "translation.md").read_text(encoding="utf-8", errors="ignore")
    gr = res.get("gloss_recall", {})
    bleu = res.get("translation_blue", {})
    ground_for_translation = {
        "n_gt_signs": gr.get("n_gt"),
        "n_pred_signs_hits": gr.get("hits"),
        "n_pred_signs": len(list(csv.DictReader((out / "gloss_timeline.csv").open()))),
        "agent_bleu_score": bleu.get("bleu"),
        "lecture_topic": "ASL lecture / tutorial (How2Sign)",
    }
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground_for_translation, ensure_ascii=False)}\n\n"
        f"TRANSLATION.MD:\n{md_text[:6000]}\n"
    )
    tq = axes_yes_no_score(payload, TRANSLATION_AXES, threshold=0.70)
    res["translation_quality_axes"] = {
        "axes": tq.get("axes", []),
        "score": round(float(tq.get("score", 0.0)), 3),
        "ok": bool(tq.get("ok", False)),
        "threshold": 0.70,
    }

    # v3: deterministic dims dominate; only translation prose uses LLM.
    weights = {"gloss_recall": 0.23, "temporal_iou": 0.18,
               "decoy_rejection": 0.13, "translation_blue": 0.10,
               "cross_doc_consistency": 0.21, "translation_quality_axes": 0.15}
    score = 0.0
    for k, w in weights.items():
        if k == "cross_doc_consistency":
            score += w * (1.0 if res[k]["ok"] else 0.0)
        elif k == "translation_quality_axes":
            score += w * res[k]["score"]
        else:
            score += w * max(0.0, res[k].get("ratio", 0.0))
    score = round(score, 3)
    all_ok = all(v["ok"] for v in res.values())
    print(json.dumps({"pass": all_ok,
                      "checks": {k: v["ok"] for k, v in res.items()},
                      **res, "score": score, "weights": weights},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
