"""Grader for T81a Underwater Coral Reef Bleaching & Reef-Fish Census (v3).

v3 redesign:
  - Closed-vocab fields (coral_type, fish species, bleaching_severity,
    decoy_type) remain strict normalised match — no LLM, no VLM. The
    coral vocab includes 4 substrate decoys (sand/rubble/algae_turf/
    coralline_algae) and the decoy_categories already act as PIP-overlay
    decoys vs genuine reef content.
  - conservation_memo.md is genuinely free-form prose; replaced single
    VLM judge_memo_grounded with 5-axis yes/no `conservation_quality_axes`.
  - Reweighted: deterministic dims dominate (~0.85); only the prose memo
    uses LLM axes (0.15).

6 dims (sum 1.0):
  coral_cover_mae               <= 8.0  (0.17)  per (quadrat x coral_type) cover_pct |delta|
  fish_species_macro_f1         >= 0.55 (0.17)  sklearn macro F1 over 30-species presence
  bleaching_severity_kappa      >= 0.55 (0.17)  cohen_kappa_score(weights='quadratic')
  decoy_rejection               >= 0.75 (0.13)  hits/8 with IoU>0.3 + decoy_type exact
  cross_doc_consistency         == 1.0  (0.21)  vocab + sums + memo<->csv reconciliation
  conservation_quality_axes     >= 0.70 (0.15)  5-axis yes/no on conservation_memo.md
"""
import argparse, csv, json, os, re, sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))

from open_vocab_judge import axes_yes_no_score, mllm_image_axes  # noqa: E402
from grader_format import resolve_required, check_columns  # noqa: E402


# Multimodal checklist for `benthic_map.png` (time-series strip plot with
# x=quadrat time, y=quadrat_id rows, colour=dominant coral_type, plus
# representative quadrat thumbnails inset). Anti-cheat: confirms the
# composite plot is actually rendered, not blank.
BENTHIC_MAP_AXES = [
    ("real_image_not_placeholder",
     "Is this image a real rendered figure with plots and graphics, "
     "NOT a single solid colour, blank canvas, all-black image, or "
     "single-text caption?"),
    ("strip_plot_or_heatmap_layout_visible",
     "Does the figure visibly contain a strip plot or heatmap layout "
     "(rows of horizontally-arranged time-series cells / coloured "
     "bands per quadrat)?"),
    ("multiple_color_categories_visible",
     "Are multiple distinct colour categories visible (encoding "
     "dominant coral types) rather than a single uniform colour?"),
    ("thumbnails_or_legend_inset_visible",
     "Does the figure include inset elements such as quadrat "
     "thumbnails, a colour legend, or labelled side-panels supporting "
     "the strip plot?"),
]


CONSERVATION_AXES = [
    ("bleached_fraction_cited",
     "Does conservation_memo.md state a bleached-cover fraction (% of "
     "total cover with severity >= 2) within +/-5 percentage points of "
     "GROUND_DATA.bleached_cover_pct?"),
    ("top_quadrats_named",
     "Does conservation_memo.md name at least 3 specific quadrat IDs "
     "(Q01-style) including all of GROUND_DATA.top3_worst_quadrats — "
     "no fabricated quadrat IDs not in the CSV?"),
    ("top_fish_named",
     "Does conservation_memo.md mention at least 3 of "
     "GROUND_DATA.top5_fish_species by name?"),
    ("conservation_priorities_present",
     "Does conservation_memo.md read as coherent reef-conservation "
     "English with at least one sentence on management priorities or "
     "follow-up monitoring (no lorem-ipsum, no bullet-only dump)?"),
    ("self_consistent_no_fabrication",
     "Is conservation_memo.md self-consistent with GROUND_DATA: "
     "bleaching severity phrasing matches the supplied distribution, "
     "decoy rejection rationale (divers / bubbles / particulate plumes "
     "/ snorkel flags) is grounded, and there are no fabricated species "
     "or quadrats?"),
]

REQUIRED = ["coral_cover_per_quadrat.csv", "fish_census.csv",
            "decoy_rejections.json", "benthic_map.png", "conservation_memo.md"]
CORAL_FIELDS = {"quadrat_id", "t_sec", "coral_type", "cover_pct", "bleaching_severity"}
FISH_FIELDS  = {"event_id", "t_sec", "species", "count", "confidence"}
DCATS = {"diver", "bubbles", "particulate_plume", "snorkel_flag"}
SUBSTRATES = {"sand", "rubble", "algae_turf", "coralline_algae"}
SUM_CAP = 102
MEMO_WORD_CAP = 500
IMG_MIN_W, IMG_MIN_H = 1280, 800


def gating(out: Path):
    errs, resolved = resolve_required(out, REQUIRED)
    if errs: return False, errs, resolved
    miss = check_columns(resolved["coral_cover_per_quadrat.csv"], CORAL_FIELDS)
    if miss: return False, [f"coral_cover_per_quadrat.csv missing cols: {miss}"], resolved
    miss = check_columns(resolved["fish_census.csv"], FISH_FIELDS)
    if miss: return False, [f"fish_census.csv missing cols: {miss}"], resolved
    try:
        d = json.load(resolved["decoy_rejections.json"].open())
        assert isinstance(d, list)
    except (FileNotFoundError, json.JSONDecodeError, ValueError, AssertionError) as e:
        return False, [f"decoy_rejections.json invalid: {e}"], resolved
    return True, [], resolved


def img_size(p: Path):
    try:
        from PIL import Image
        with Image.open(p) as im:
            return im.size
    except (ImportError, FileNotFoundError, OSError):
        return None


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def coral_mae_dim(coral_rows, gt):
    gt_index = {(q["quadrat_id"], r["coral_type"]): r for q in gt["quadrats"] for r in q["cover"]}
    pred_index = {(r["quadrat_id"], r["coral_type"]): r for r in coral_rows}
    diffs = []
    for k, gtr in gt_index.items():
        pr = pred_index.get(k)
        agent_pct = float(pr["cover_pct"]) if pr else 0.0
        diffs.append(abs(agent_pct - float(gtr["cover_pct"])))
    if not diffs:
        return 100.0, 0
    mae = sum(diffs) / len(diffs)
    return float(mae), len(diffs)


def fish_macro_f1(fish_rows, gt):
    from sklearn.metrics import f1_score
    vocab = list(gt["closed_vocab_fish"])
    pred = {sp: any(r["species"] == sp for r in fish_rows) for sp in vocab}
    gold = {sp: any(e["species"] == sp for e in gt["fish_events"]) for sp in vocab}
    y_t = [int(gold[s]) for s in vocab]
    y_p = [int(pred[s]) for s in vocab]
    return float(f1_score(y_t, y_p, average="macro", zero_division=0))


def bleach_kappa(coral_rows, gt):
    from sklearn.metrics import cohen_kappa_score
    pred_index = {(r["quadrat_id"], r["coral_type"]): int(r["bleaching_severity"]) for r in coral_rows}
    gt_index = {(q["quadrat_id"], r["coral_type"]): int(r["bleaching_severity"])
                for q in gt["quadrats"] for r in q["cover"]}
    keys = sorted(set(pred_index) & set(gt_index))
    if len(keys) < 5:
        return 0.0, 0
    y_t = [gt_index[k] for k in keys]
    y_p = [pred_index[k] for k in keys]
    if len(set(y_t)) < 2 and len(set(y_p)) < 2:
        return (1.0 if y_t == y_p else 0.0), len(keys)
    try:
        k = cohen_kappa_score(y_t, y_p, weights="quadratic", labels=[0, 1, 2, 3])
    except (ValueError, TypeError):
        return 0.0, len(keys)
    if k != k:                      # NaN guard
        return 0.0, len(keys)
    return float(k), len(keys)


def decoy_rej(decoys, gt):
    gtd = gt["decoys"]
    if not gtd: return 0.0, 0, 0
    used, hits = set(), 0
    for g in gtd:
        for i, d in enumerate(decoys):
            if i in used or d["decoy_type"] != g["decoy_type"]: continue
            if iou((d["t_start"], d["t_end"]), (g["t_start"], g["t_end"])) > gt["iou_thr_decoy"]:
                used.add(i); hits += 1; break
    return hits / len(gtd), hits, len(gtd)


def memo_top5_counts(fish_rows):
    sums = defaultdict(int)
    for r in fish_rows:
        try: sums[r["species"]] += int(r["count"])
        except (ValueError, KeyError, TypeError): pass
    return sorted(sums.items(), key=lambda x: (-x[1], x[0]))[:5]


def memo_bleached_pct(coral_rows):
    weighted = 0.0; total = 0.0
    for r in coral_rows:
        try:
            pct = float(r["cover_pct"]); sev = int(r["bleaching_severity"])
        except (ValueError, KeyError, TypeError):
            continue
        total += pct
        if sev >= 2: weighted += pct
    return (100.0 * weighted / total) if total > 0 else 0.0


def memo_top3_quadrats(coral_rows):
    burden = defaultdict(float)
    for r in coral_rows:
        try:
            pct = float(r["cover_pct"]); sev = int(r["bleaching_severity"])
        except (ValueError, KeyError, TypeError):
            continue
        burden[r["quadrat_id"]] += pct * sev
    return [q for q, _ in sorted(burden.items(), key=lambda x: (-x[1], x[0]))[:3]]


def cross_doc(coral_rows, fish_rows, decoys, vocab_c, vocab_f, memo, png):
    errs = []
    bad_c = sorted({r["coral_type"] for r in coral_rows if r["coral_type"] not in vocab_c})
    if bad_c: errs.append(f"out-of-vocab coral_type: {bad_c[:5]}")
    bad_f = sorted({r["species"] for r in fish_rows if r["species"] not in vocab_f})
    if bad_f: errs.append(f"out-of-vocab species: {bad_f[:5]}")
    bad_sub = [(r["quadrat_id"], r["coral_type"], r["bleaching_severity"]) for r in coral_rows
               if r["coral_type"] in SUBSTRATES and int(r.get("bleaching_severity", 0)) != 0]
    if bad_sub: errs.append(f"substrate row severity != 0: {bad_sub[:3]}")
    bad_sev = [r["bleaching_severity"] for r in coral_rows
               if int(r.get("bleaching_severity", -1)) not in (0, 1, 2, 3)]
    if bad_sev: errs.append(f"severity not in {{0,1,2,3}}: {bad_sev[:3]}")
    sums = defaultdict(float)
    for r in coral_rows:
        try: sums[r["quadrat_id"]] += float(r["cover_pct"])
        except (ValueError, KeyError, TypeError): pass
    over = [(q, round(s, 1)) for q, s in sums.items() if s > SUM_CAP]
    if over: errs.append(f"quadrat sum > {SUM_CAP}: {over[:3]}")
    qrows = defaultdict(int)
    for r in coral_rows: qrows[r["quadrat_id"]] += 1
    thin = [q for q, n in qrows.items() if n < 3]
    if thin: errs.append(f"quadrat with < 3 coral_type rows: {thin[:3]}")
    last_t = -1.0
    for r in fish_rows:
        try: t = float(r["t_sec"])
        except (ValueError, KeyError, TypeError): continue
        if t < last_t - 1e-3:
            errs.append("fish_census.csv not monotone in t_sec"); break
        last_t = t
    counts_bad = [r["count"] for r in fish_rows
                  if not (1 <= int(r.get("count", 0)) <= 50)]
    if counts_bad: errs.append(f"fish count outside [1,50]: {counts_bad[:3]}")
    if len(decoys) < 6: errs.append(f"decoy_rejections.json has {len(decoys)} rows, need >= 6")
    bad_d = sorted({d.get("decoy_type") for d in decoys if d.get("decoy_type") not in DCATS})
    if bad_d: errs.append(f"decoy_type not in vocab: {bad_d}")
    sz = img_size(png)
    if sz is None: errs.append("benthic_map.png unreadable")
    else:
        w, h = sz
        if w < IMG_MIN_W or h < IMG_MIN_H:
            errs.append(f"benthic_map.png {w}x{h} < {IMG_MIN_W}x{IMG_MIN_H}")
    n_words = len(re.findall(r"\S+", memo))
    if n_words > MEMO_WORD_CAP:
        errs.append(f"conservation_memo.md {n_words} words > {MEMO_WORD_CAP}")
    bp = memo_bleached_pct(coral_rows)
    nums = [float(m.group(1)) for m in re.finditer(r"(\d{1,3}(?:\.\d+)?)\s*%", memo)]
    if not any(abs(n - bp) <= max(5.0, 0.05 * bp + 1e-9) for n in nums):
        errs.append(f"memo bleached fraction (~{bp:.1f}%) not within +/-5% of any '%' figure in memo")
    top3_q = set(memo_top3_quadrats(coral_rows))
    found_q = set(re.findall(r"\bQ\d{2}\b", memo))
    if not top3_q.issubset(found_q):
        errs.append(f"memo missing top-3 quadrat IDs: need {sorted(top3_q)}")
    top5 = memo_top5_counts(fish_rows)
    miss_sp = [sp for sp, _ in top5 if sp.lower() not in memo.lower()]
    if miss_sp:
        errs.append(f"memo missing top-5 fish species mentions: {miss_sp[:5]}")
    return errs


def evaluate(out: Path, gt: dict, resolved: dict[str, Path]):
    coral_rows = []
    with resolved["coral_cover_per_quadrat.csv"].open(newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                coral_rows.append({
                    "quadrat_id": (r.get("quadrat_id") or "").strip(),
                    "t_sec":      float(r.get("t_sec") or 0),
                    "coral_type": (r.get("coral_type") or "").strip(),
                    "cover_pct":  float(r.get("cover_pct") or 0),
                    "bleaching_severity": int(float(r.get("bleaching_severity") or 0)),
                })
            except (ValueError, KeyError, TypeError): pass
    fish_rows = []
    with resolved["fish_census.csv"].open(newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                fish_rows.append({
                    "event_id": (r.get("event_id") or "").strip(),
                    "t_sec":    float(r.get("t_sec") or 0),
                    "species":  (r.get("species") or "").strip(),
                    "count":    int(float(r.get("count") or 0)),
                    "confidence": float(r.get("confidence") or 0),
                })
            except (ValueError, KeyError, TypeError): pass
    decoys = []
    raw_d = json.load(resolved["decoy_rejections.json"].open())
    if isinstance(raw_d, list):
        for d in raw_d:
            try:
                decoys.append({
                    "t_start":    float(d["t_start"]),
                    "t_end":      float(d["t_end"]),
                    "decoy_type": (d.get("decoy_type") or "").strip(),
                })
            except (ValueError, KeyError, TypeError): pass
    memo = resolved["conservation_memo.md"].read_text(encoding="utf-8", errors="replace")

    mae, mae_n = coral_mae_dim(coral_rows, gt)
    f1 = fish_macro_f1(fish_rows, gt)
    kappa, kappa_n = bleach_kappa(coral_rows, gt)
    rej, dh, dt = decoy_rej(decoys, gt)
    cd_errs = cross_doc(coral_rows, fish_rows, decoys,
                        set(gt["closed_vocab_coral"]), set(gt["closed_vocab_fish"]),
                        memo, resolved["benthic_map.png"])
    return {
        "coral_cover_mae":        {"mae": round(mae, 3), "n_pairs": mae_n,
                                    "ok": mae <= 8.0,
                                    "ratio": max(0.0, 1.0 - mae / 8.0)},
        "fish_species_macro_f1":  {"ratio": round(f1, 3), "n_classes": 30,
                                    "ok": f1 >= 0.55},
        "bleaching_severity_kappa": {"kappa": round(kappa, 3), "n_overlap": kappa_n,
                                    "ok": kappa >= 0.55, "ratio": max(0.0, kappa)},
        "decoy_rejection":        {"ratio": round(rej, 3), "hits": dh, "gt": dt,
                                    "ok": rej >= 0.75},
        "cross_doc_consistency":  {"errors": cd_errs[:8], "n_errors": len(cd_errs),
                                    "ok": len(cd_errs) == 0},
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
    # Compute summary ground for memo
    coral_rows_for_summary = []
    with resolved["coral_cover_per_quadrat.csv"].open(newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                coral_rows_for_summary.append({
                    "quadrat_id": (r.get("quadrat_id") or "").strip(),
                    "cover_pct": float(r.get("cover_pct") or 0),
                    "bleaching_severity": int(float(r.get("bleaching_severity") or 0)),
                })
            except (ValueError, KeyError, TypeError): pass
    fish_rows_for_summary = []
    with resolved["fish_census.csv"].open(newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                fish_rows_for_summary.append({
                    "species": (r.get("species") or "").strip(),
                    "count": int(float(r.get("count") or 0)),
                })
            except (ValueError, KeyError, TypeError): pass
    bp = memo_bleached_pct(coral_rows_for_summary)
    top3_q = memo_top3_quadrats(coral_rows_for_summary)
    top5 = memo_top5_counts(fish_rows_for_summary)
    ground_for_memo = {
        "bleached_cover_pct": round(bp, 2),
        "coral_cover_mae": res["coral_cover_mae"].get("mae"),
        "bleaching_kappa": res["bleaching_severity_kappa"].get("kappa"),
        "fish_macro_f1": res["fish_species_macro_f1"].get("ratio"),
        "top3_worst_quadrats": top3_q,
        "top5_fish_species": [sp for sp, _ in top5],
        "n_decoys_hit": res["decoy_rejection"].get("hits"),
        "n_decoys_gt": res["decoy_rejection"].get("gt"),
    }
    memo_text = resolved["conservation_memo.md"].read_text(encoding="utf-8", errors="ignore")
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground_for_memo, ensure_ascii=False)}\n\n"
        f"CONSERVATION_MEMO.MD:\n{memo_text[:6000]}\n"
    )
    cq = axes_yes_no_score(payload, CONSERVATION_AXES, threshold=0.70)
    cq_score = float(cq.get("score", 0.0))
    cq_ok = bool(cq.get("ok", False))
    res["conservation_quality_axes"] = {
        "axes": cq.get("axes", []),
        "score": round(cq_score, 3),
        "ok": cq_ok,
        "threshold": 0.70,
    }
    # MLLM image content checklist on rendered benthic_map.
    ba = mllm_image_axes(resolved["benthic_map.png"], BENTHIC_MAP_AXES,
                         threshold=0.70)
    ba_score = float(ba.get("score", 0.0))
    res["benthic_map_image_axes"] = {
        "axes": ba.get("axes", []),
        "score": round(ba_score, 3),
        "ok": bool(ba.get("ok", False)),
        "threshold": 0.70,
    }
    W = {"coral_cover_mae": 0.17, "fish_species_macro_f1": 0.17,
         "bleaching_severity_kappa": 0.17, "decoy_rejection": 0.13,
         "cross_doc_consistency": 0.11, "conservation_quality_axes": 0.15,
         "benthic_map_image_axes": 0.10}
    score = 0.0
    for k, w in W.items():
        if k == "cross_doc_consistency":
            score += w * (1.0 if res[k]["ok"] else 0.0)
        elif k == "conservation_quality_axes":
            score += w * cq_score
        elif k == "benthic_map_image_axes":
            score += w * ba_score
        else:
            score += w * max(0.0, res[k].get("ratio", 0.0))
    score = round(max(0.0, min(score, 1.0)), 3)
    all_ok = all(v["ok"] for v in res.values())
    print(json.dumps({"pass": all_ok,
                      "checks": {k: v["ok"] for k, v in res.items()},
                      **res, "score": score, "weights": W}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
