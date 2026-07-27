"""Grader for T77a Tropical Forest Bioacoustic Triage (v3).

v3 redesign:
  - Closed-vocab fields (species / decoy_type) graded by deterministic
    strict match. Closed vocab is the 25-name scientific-name list.
  - Free-form `field_report.md` graded by 5-axis yes/no
    (`field_report_quality_axes`), replacing single-shot VLM judge.
  - `tropical_species_vocab.json` and `decoy_categories.json` augmented
    with decoy entries (CLOSED_WITH_DECOYS).

6 dims (sum 1.0):
  event_recall                 >= 0.65 (0.20)  Hungarian IoU>0.3 + species exact
  species_macro_f1             >= 0.55 (0.15)  sklearn macro F1 over scorable 25
  chorus_overlap_iou           >= 0.60 (0.10)  Hungarian on 5 GT
  noise_rejection              >= 0.85 (0.10)  IoU>0.3 + decoy_type exact
  cross_doc_consistency        == 1.0  (0.25)  monotone+vocab+spectro+report
  field_report_quality_axes    >= 0.70 (0.20)  LLM yes/no axes
"""
import argparse, csv, json, os, re, sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))
from open_vocab_judge import axes_yes_no_score, mllm_image_axes
from grader_format import resolve_required, check_columns  # noqa: E402


# Multimodal checklist for `spectrogram.png` (very wide mel
# spectrogram of full recording with top-5 events as red rectangles +
# species labels). Anti-cheat content checklist — single-image yes/no
# across 4 axes.
SPECTROGRAM_AXES = [
    ("real_spectrogram_not_placeholder",
     "Is this image a real spectrogram visualization — NOT a single "
     "solid color, blank canvas, all-black image, or a single text "
     "caption?"),
    ("time_frequency_pattern_visible",
     "Does the image show a recognisable time-vs-frequency spectrogram "
     "pattern (horizontal time axis, vertical frequency axis, intensity "
     "bands) consistent with an audio mel spectrogram?"),
    ("has_red_annotation_boxes",
     "Are there visible red rectangles / bounding boxes overlaid on "
     "the spectrogram marking event regions?"),
    ("has_text_labels_for_species",
     "Are there visible text labels (e.g. species scientific names) "
     "drawn near the annotation boxes on the spectrogram?"),
]


FIELD_REPORT_AXES = [
    ("species_count_matches",
     "Does the report state a total unique species count matching "
     "GROUND_DATA.n_unique_species_pred?"),
    ("species_and_chorus_cited",
     "Does the report cite >=3 specific species by scientific name "
     "AND >=1 chorus hotspot time window (mm:ss) consistent with "
     "GROUND_DATA?"),
    ("most_active_window_named",
     "Does the report identify a single most-active 5-minute window "
     "with an event count plausible against GROUND_DATA event total?"),
    ("ecologist_tone",
     "Does the report read as coherent professional ecology prose "
     "with at least one conservation observation grounded in the data?"),
    ("no_fabrication_no_oov",
     "Is the report free of fabricated scientific names not in the "
     "GROUND_DATA species_vocab, and free of contradictions with the "
     "event/chorus counts?"),
]

REQUIRED = ["events.csv", "chorus_hotspots.json", "decoy_rejections.csv",
            "spectrogram.png", "field_report.md"]
EVENT_FIELDS = {"event_id", "start_sec", "end_sec", "species", "confidence", "intensity_db"}
DECOY_FIELDS = {"t_start", "t_end", "decoy_type"}
DCATS = {"wind", "rain", "aircraft", "human_speech",
         # decoys (CLOSED_WITH_DECOYS): accepted by vocab gate but
         # never present in GT, so flagging them harms precision.
         "boat_motor", "chainsaw", "vehicle_horn"}
REPORT_WORD_CAP = 500
EVT_LO, EVT_HI = 0.3, 30.0
CHORUS_MIN_DUR = 10.0
SPECTRO_MIN_W, SPECTRO_MIN_H = 3000, 800


def gating(out: Path):
    errs, resolved = resolve_required(out, REQUIRED)
    if errs: return False, errs, resolved
    miss = check_columns(resolved["events.csv"], EVENT_FIELDS)
    if miss: return False, [f"events.csv missing: {miss}"], resolved
    miss = check_columns(resolved["decoy_rejections.csv"], DECOY_FIELDS)
    if miss: return False, [f"decoy_rejections.csv missing: {miss}"], resolved
    try:
        ch = json.load(resolved["chorus_hotspots.json"].open())
        assert isinstance(ch, list)
    except (FileNotFoundError, json.JSONDecodeError, ValueError, AssertionError) as e:
        return False, [f"chorus_hotspots.json: {e}"], resolved
    return True, [], resolved


def hungarian(cost):
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    return linear_sum_assignment(np.asarray(cost))


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def event_recall_dim(events, gt_events):
    G = len(gt_events); P = len(events)
    if G == 0 or P == 0: return 0.0, 0, G
    cost = [[1.0] * G for _ in range(P)]
    for i, p in enumerate(events):
        for j, g in enumerate(gt_events):
            if p["species"] != g["species"]: continue
            io = iou((p["start_sec"], p["end_sec"]), (g["start_sec"], g["end_sec"]))
            if io > 0.3: cost[i][j] = 1.0 - io
    rows, cols = hungarian(cost)
    hits = sum(1 for r, c in zip(rows, cols) if cost[r][c] < 1.0)
    return hits / G, hits, G


def species_macro_f1(events, gt_events, vocab):
    pred = {sp: any(e["species"] == sp for e in events) for sp in vocab}
    gold = {sp: any(e["species"] == sp for e in gt_events) for sp in vocab}
    from sklearn.metrics import f1_score
    y_t = [int(gold[s]) for s in vocab]
    y_p = [int(pred[s]) for s in vocab]
    return float(f1_score(y_t, y_p, average="macro", zero_division=0))


def chorus_iou(pred, gt):
    G = len(gt); P = len(pred)
    if G == 0 or P == 0: return 0.0
    cost = [[1.0] * G for _ in range(P)]
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            cost[i][j] = 1.0 - iou((p["t_start"], p["t_end"]),
                                   (g["t_start"], g["t_end"]))
    rows, cols = hungarian(cost)
    matched = {c: 1.0 - cost[r][c] for r, c in zip(rows, cols)}
    return sum(matched.get(j, 0.0) for j in range(G)) / G


def noise_rej(pred, gt):
    if not gt: return 0.0, 0, 0
    used, hits = set(), 0
    for g in gt:
        for i, p in enumerate(pred):
            if i in used or p["decoy_type"] != g["decoy_type"]: continue
            if iou((p["t_start"], p["t_end"]), (g["t_start"], g["t_end"])) > 0.3:
                used.add(i); hits += 1; break
    return hits / len(gt), hits, len(gt)


def img_size(p):
    try:
        from PIL import Image
        with Image.open(p) as im: return im.size
    except (ImportError, FileNotFoundError, OSError):
        return None


def cross_doc(events, vocab, ch, decoys, spectro, report, spectro5):
    errs = []
    last = -1.0
    for e in events:
        if e["start_sec"] < last - 1e-3:
            errs.append(f"events.csv not monotone at {e['event_id']}"); break
        last = e["start_sec"]
    bad_sp = sorted({e["species"] for e in events if e["species"] not in vocab})
    if bad_sp: errs.append(f"out-of-vocab species: {bad_sp[:5]}")
    for e in events:
        if not (e["end_sec"] > e["start_sec"]):
            errs.append(f"event {e['event_id']} end<=start"); break
        d = e["end_sec"] - e["start_sec"]
        if d < EVT_LO or d > EVT_HI:
            errs.append(f"event {e['event_id']} length {d:.2f}s outside [{EVT_LO},{EVT_HI}]"); break
    for c in ch:
        miss = [k for k in ("hotspot_id", "t_start", "t_end", "n_species",
                            "species_list", "peak_intensity_db") if k not in c]
        if miss:
            errs.append(f"chorus row missing keys: {miss}"); continue
        if c["t_end"] - c["t_start"] < CHORUS_MIN_DUR:
            errs.append(f"chorus {c['hotspot_id']} duration < {CHORUS_MIN_DUR}s")
        sl = c.get("species_list") or []
        sl_set = set(sl)
        if c.get("n_species", 0) < 3 or len(sl_set) < 3:
            errs.append(f"chorus {c['hotspot_id']} distinct species < 3")
        bad = [s for s in sl if s not in vocab]
        if bad: errs.append(f"chorus {c['hotspot_id']} bad species: {bad[:3]}")
    bad = sorted({r["decoy_type"] for r in decoys if r["decoy_type"] not in DCATS})
    if bad: errs.append(f"decoy_rejections invalid types: {bad}")
    sz = img_size(spectro)
    if sz is None: errs.append("spectrogram.png unreadable")
    else:
        w, h = sz
        if w < SPECTRO_MIN_W or h < SPECTRO_MIN_H:
            errs.append(f"spectrogram.png {w}x{h} below {SPECTRO_MIN_W}x{SPECTRO_MIN_H}")
    if spectro5 is None:
        errs.append("spectrogram_annotations.json missing (5 boxes = events.csv top-5 by intensity_db)")
    elif len(spectro5) != 5:
        errs.append(f"spectrogram_annotations: need 5, got {len(spectro5)}")
    else:
        top5 = sorted(events, key=lambda e: -e["intensity_db"])[:5]
        T = {(round(e["start_sec"], 2), round(e["end_sec"], 2), e["species"]) for e in top5}
        try:
            A = {(round(float(a["start_sec"]), 2), round(float(a["end_sec"]), 2),
                  str(a["species"])) for a in spectro5}
        except (KeyError, ValueError, TypeError):
            errs.append("spectrogram_annotations entry malformed"); A = set()
        if A and T != A:
            errs.append("spectrogram_annotations != events.csv top-5 by intensity_db")
    rt = report.strip()
    n_words = len(re.findall(r"\S+", rt))
    if n_words > REPORT_WORD_CAP:
        errs.append(f"field_report.md {n_words} words > {REPORT_WORD_CAP}")
    n_unique = len({e["species"] for e in events})
    if not any(int(m.group(1)) == n_unique for m in re.finditer(r"\b(\d{1,3})\b", rt)):
        errs.append(f"field_report.md does not state total unique species count {n_unique}")
    return errs


def evaluate(out: Path, gt: dict, resolved: dict[str, Path]):
    with resolved["events.csv"].open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    events = []
    for r in rows:
        try:
            events.append({
                "event_id": r.get("event_id", ""),
                "start_sec": float(r["start_sec"]),
                "end_sec":   float(r["end_sec"]),
                "species":   (r.get("species") or "").strip(),
                "confidence": float(r.get("confidence") or 0.0),
                "intensity_db": float(r.get("intensity_db") or -90.0),
            })
        except (ValueError, KeyError, TypeError):
            pass
    with resolved["decoy_rejections.csv"].open(newline="") as fh:
        d_rows = list(csv.DictReader(fh))
    decoys = []
    for r in d_rows:
        try:
            decoys.append({"t_start": float(r["t_start"]),
                           "t_end":   float(r["t_end"]),
                           "decoy_type": (r.get("decoy_type") or "").strip()})
        except (ValueError, KeyError, TypeError):
            pass
    ch_pred = json.load(resolved["chorus_hotspots.json"].open())
    if not isinstance(ch_pred, list): ch_pred = []
    report = resolved["field_report.md"].read_text(encoding="utf-8", errors="replace")
    spectro5 = None
    p5 = out / "spectrogram_annotations.json"
    if p5.exists():
        try: spectro5 = json.load(open(p5))
        except (FileNotFoundError, json.JSONDecodeError): spectro5 = None

    vocab = list(gt["closed_vocab_species"])
    # Cross-doc vocab gate accepts the 25 scorable species PLUS optional
    # decoys (CLOSED_WITH_DECOYS) — listed in gt or fallback constant.
    decoy_species = list(gt.get("closed_vocab_species_decoys", [
        "Patagioenas plumbea", "Dacnis cayana",
        "Dendrocincla fuliginosa", "Pipra mentalis",
    ]))
    cross_vocab = set(vocab) | set(decoy_species)
    er, hits, gtot = event_recall_dim(events, gt["events"])
    macro = species_macro_f1(events, gt["events"], vocab)
    ic = chorus_iou(ch_pred, gt["chorus_hotspots"])
    nr, dh, dtot = noise_rej(decoys, gt["decoys"])
    cd = cross_doc(events, cross_vocab, ch_pred, decoys,
                   resolved["spectrogram.png"], report, spectro5)
    return {
        "event_recall":          {"ratio": round(er, 3),  "hits": hits, "gt": gtot,
                                  "ok": er >= 0.65},
        "species_macro_f1":      {"ratio": round(macro, 3), "n_classes": len(vocab),
                                  "ok": macro >= 0.55},
        "chorus_overlap_iou":    {"ratio": round(ic, 3),
                                  "n_pred": len(ch_pred), "n_gt": len(gt["chorus_hotspots"]),
                                  "ok": ic >= 0.60},
        "noise_rejection":       {"ratio": round(nr, 3), "hits": dh, "gt": dtot,
                                  "ok": nr >= 0.85},
        "cross_doc_consistency": {"errors": cd[:8], "n_errors": len(cd),
                                  "ok": len(cd) == 0},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--fixtures", default="fixtures")
    a = ap.parse_args()
    out = Path(a.output)
    ok, errs, resolved = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs}, indent=2))
        return
    gt = json.load(open(a.gt))
    res = evaluate(out, gt, resolved)
    ground_for_memo = {
        "n_events_pred": res["event_recall"].get("hits", 0) + 0,
        "n_unique_species": res["species_macro_f1"].get("ratio"),
        "n_chorus_pred": res["chorus_overlap_iou"].get("n_pred"),
        "n_chorus_gt": res["chorus_overlap_iou"].get("n_gt"),
        "n_decoys_pred": res["noise_rejection"].get("hits"),
        "n_decoys_gt": res["noise_rejection"].get("gt"),
        "vocab_size": res["species_macro_f1"].get("n_classes"),
    }
    # Recompute n_unique_species from events for memo grounding
    events_pred = []
    try:
        with resolved["events.csv"].open(newline="") as fh:
            for r in csv.DictReader(fh):
                events_pred.append((r.get("species") or "").strip())
    except Exception:
        pass
    ground_for_memo["n_unique_species_pred"] = len({s for s in events_pred if s})
    ground_for_memo["n_events_pred"] = len([s for s in events_pred if s])
    ground_for_memo["species_vocab"] = list(gt["closed_vocab_species"])
    memo_text = resolved["field_report.md"].read_text(encoding="utf-8", errors="replace")
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground_for_memo, ensure_ascii=False)}\n\n"
        f"FIELD_REPORT.MD:\n{memo_text[:6000]}\n"
    )
    fq = axes_yes_no_score(payload, FIELD_REPORT_AXES, threshold=0.70)
    fq_ok = bool(fq.get("ok")) or bool(fq.get("skipped"))
    fq_score = float(fq.get("score", 0.0))
    res["field_report_quality_axes"] = {"score": round(fq_score, 3),
                                        "axes": fq.get("axes", []),
                                        "skipped": bool(fq.get("skipped")),
                                        "ok": fq_ok,
                                        "threshold": 0.70}
    sa = mllm_image_axes(resolved["spectrogram.png"], SPECTROGRAM_AXES,
                          threshold=0.70)
    sa_ok = bool(sa.get("ok")) or bool(sa.get("skipped"))
    sa_score = float(sa.get("score", 0.0))
    res["spectrogram_image_axes"] = {"score": round(sa_score, 3),
                                     "axes": sa.get("axes", []),
                                     "skipped": bool(sa.get("skipped")),
                                     "ok": sa_ok,
                                     "threshold": 0.70}
    W = {"event_recall": 0.18, "species_macro_f1": 0.13,
         "chorus_overlap_iou": 0.09, "noise_rejection": 0.09,
         "cross_doc_consistency": 0.22, "field_report_quality_axes": 0.17,
         "spectrogram_image_axes": 0.12}
    score = 0.0
    for k, w in W.items():
        if k == "cross_doc_consistency":
            score += w * (1.0 if res[k]["ok"] else 0.0)
        elif k == "field_report_quality_axes":
            score += w * fq_score
        elif k == "spectrogram_image_axes":
            score += w * sa_score
        else:
            score += w * max(0.0, res[k].get("ratio", 0.0))
    score = round(max(0.0, min(score, 1.0)), 3)
    all_ok = all(v["ok"] for v in res.values())
    print(json.dumps({"pass": all_ok,
                      "checks": {k: v["ok"] for k, v in res.items()},
                      **res, "score": score, "weights": W},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
