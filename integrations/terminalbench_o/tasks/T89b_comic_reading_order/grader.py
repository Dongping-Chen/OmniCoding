"""Grader for T89a Vintage Newspaper Comic Strip Reading Order + Speaker Audit.
6 dims (sum 1.0):
  panel_order_kendall_tau    >= 0.65 (0.18) per-page Kendall's tau on matched panels
  ocr_cer                    <= 0.20 (0.13) jiwer CER over 91 GT bubbles
  speaker_attribution_acc    >= 0.55 (0.18) exact speaker_id on matched dialogue+thought
  decoy_rejection            >= 0.60 (0.13) hits/5 with page_id + IoU>0.3 + type
  cross_doc_consistency      == 1.0  (0.28) vocab + permutation + memo cross-refs
  memo_quality               >= 0.55 (0.10) VLM judge on narrative.md
"""
import argparse, csv, json, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))
from open_vocab_judge import axes_yes_no_score, mllm_image_axes  # noqa: E402
from grader_format import resolve_required, check_columns  # noqa: E402

# Multimodal checklist for `reading_order_overlay.png` (4-panel composite of
# comic pages with reading_order_rank numerals overlaid on detected panel
# centroids — mix of Krazy Kat + Little Nemo). Anti-cheat single-image yes/no
# across 4 axes.
OVERLAY_AXES = [
    ("real_image_not_placeholder",
     "Is this image a real visual rendering — NOT a single solid color, "
     "blank canvas, all-black image, or single-text caption?"),
    ("four_comic_pages_visible_in_composite",
     "Does this image visibly contain FOUR comic-strip pages arranged as "
     "a composite / 2x2 (or similar) layout?"),
    ("numeric_rank_overlays_visible_on_panels",
     "Are numeric rank labels (e.g. 1, 2, 3, ...) overlaid on top of the "
     "comic panels at their centroids, indicating reading order?"),
    ("comic_page_aesthetic_not_blank_canvas",
     "Do the visible elements look like vintage newspaper-comic artwork "
     "(line-art panels, speech bubbles, hand-drawn characters) rather "
     "than blank canvases or modern UI screenshots?"),
]


NARRATIVE_AXES = [
    ("five_arcs_with_refs",
     "Does the narrative present 5 story arcs, each citing at least "
     "one page_id (e.g. page_05) and at least one character_id from "
     "GROUND_DATA's character_vocab?"),
    ("dialogue_grounded_in_bubbles",
     "Are dialogue lines quoted in the prose grounded in GROUND_DATA's "
     "bubbles ocr_text — i.e. not invented and not attributed to a "
     "speaker absent from character_vocab?"),
    ("reading_order_causal_flow",
     "Does each arc's narration follow the agent's panel reading-order "
     "as panel-to-panel beats (no random scrambling)?"),
    ("sound_effects_unattributed",
     "Are vintage-strip onomatopoeia (POW! BIFF! BAM!) referred to as "
     "sound effects and NEVER given a speaker name (e.g. NOT 'Ignatz "
     "shouts POW!')?"),
    ("counts_cited",
     "Does the prose cite GROUND_DATA's `n_panels_total` and "
     "`n_bubbles_total` figures (or values within ±1) somewhere?"),
]


def memo_score(narrative_path: Path, ground: dict) -> dict:
    summary = narrative_path.read_text()
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground, ensure_ascii=False)}\n\n"
        f"NARRATIVE.MD:\n{summary[:6000]}\n"
    )
    return axes_yes_no_score(payload, NARRATIVE_AXES, threshold=0.70)


REQUIRED = ["panel_order.csv", "bubbles.csv", "decoy_rejections.json",
            "narrative.md", "reading_order_overlay.png"]
PANEL_FIELDS = {"page_id", "panel_idx", "bbox", "reading_order_rank"}
BUBBLE_FIELDS = {"bubble_id", "page_id", "panel_idx", "bbox",
                 "bubble_type", "ocr_text", "speaker_id_or_null"}
BUBBLE_TYPES = {"dialogue", "thought", "narration_box", "sound_effect"}
DCATS = {"fake_modern_text_overlay", "wrong_century_logo"}
NULL_SPEAKER_TYPES = {"sound_effect", "narration_box"}
DIALOG_TYPES = {"dialogue", "thought"}
MEMO_WORD_CAP = 600
IMG_MIN_W, IMG_MIN_H = 1280, 800
PANEL_IOU_THR = 0.5
BUBBLE_IOU_THR = 0.5
DECOY_IOU_THR = 0.3
MIN_PANELS = 120
MIN_BUBBLES = 80


def img_size(p: Path):
    try:
        from PIL import Image
        with Image.open(p) as im: return im.size
    except Exception: return None


def gating(out: Path):
    errs, resolved = resolve_required(out, REQUIRED)
    if errs: return False, errs, resolved
    miss = check_columns(resolved["panel_order.csv"], PANEL_FIELDS)
    if miss: return False, [f"panel_order.csv missing cols: {miss}"], resolved
    miss = check_columns(resolved["bubbles.csv"], BUBBLE_FIELDS)
    if miss: return False, [f"bubbles.csv missing cols: {miss}"], resolved
    try:
        d = json.load(resolved["decoy_rejections.json"].open())
        assert isinstance(d, list)
    except Exception as e:
        return False, [f"decoy_rejections.json invalid: {e}"], resolved
    return True, [], resolved


def _parse_bbox(raw):
    if isinstance(raw, list): val = raw
    else:
        s = str(raw).strip()
        try: val = json.loads(s)
        except Exception:
            try: val = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)[:4]]
            except Exception: return None
    if not isinstance(val, list) or len(val) != 4: return None
    try:
        x1, y1, x2, y2 = [float(v) for v in val]
    except Exception: return None
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def iou_2d(a, b):
    if a is None or b is None: return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    u = aa + bb - inter
    return (inter / u) if u > 0 else 0.0


def parse_outputs(out: Path, resolved: dict[str, Path]):
    panels, bubbles, decoys = [], [], []
    with resolved["panel_order.csv"].open(newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                panels.append({
                    "page_id": (r.get("page_id") or "").strip(),
                    "panel_idx": int(float(r.get("panel_idx") or 0)),
                    "bbox": _parse_bbox(r.get("bbox")),
                    "reading_order_rank": int(float(r.get("reading_order_rank") or 0)),
                })
            except Exception: pass
    with resolved["bubbles.csv"].open(newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                spk = (r.get("speaker_id_or_null") or "").strip()
                bubbles.append({
                    "bubble_id": (r.get("bubble_id") or "").strip(),
                    "page_id": (r.get("page_id") or "").strip(),
                    "panel_idx": int(float(r.get("panel_idx") or 0)),
                    "bbox": _parse_bbox(r.get("bbox")),
                    "bubble_type": (r.get("bubble_type") or "").strip(),
                    "ocr_text": (r.get("ocr_text") or ""),
                    "speaker_id": None if spk.lower() in ("", "null", "none") else spk,
                })
            except Exception: pass
    raw = json.load(resolved["decoy_rejections.json"].open())
    if isinstance(raw, list):
        for d in raw:
            try:
                decoys.append({
                    "page_id": (d.get("page_id") or "").strip(),
                    "bbox": _parse_bbox(d.get("bbox")),
                    "decoy_type": (d.get("decoy_type") or "").strip(),
                })
            except Exception: pass
    memo = resolved["narrative.md"].read_text(encoding="utf-8", errors="replace")
    return panels, bubbles, decoys, memo


def panel_kendall(panels, gt):
    try:
        from scipy.stats import kendalltau
    except Exception:
        return 0.0, 0
    by_page = defaultdict(list)
    for p in panels: by_page[p["page_id"]].append(p)
    tau_sum, n_pages_used = 0.0, 0
    total_pages = len(gt["pages"])
    for gp in gt["pages"]:
        pid = gp["page_id"]
        gt_panels = gp["panels"]
        agent_panels = by_page.get(pid, [])
        if not gt_panels or not agent_panels: continue
        gt_ranks, ag_ranks = [], []
        used = set()
        for gpan in gt_panels:
            best_j, best_iou = -1, 0.0
            for j, ap in enumerate(agent_panels):
                if j in used or ap["bbox"] is None: continue
                v = iou_2d(ap["bbox"], gpan["bbox"])
                if v >= PANEL_IOU_THR and v > best_iou:
                    best_j, best_iou = j, v
            if best_j >= 0:
                used.add(best_j)
                gt_ranks.append(gpan["reading_order_rank"])
                ag_ranks.append(agent_panels[best_j]["reading_order_rank"])
        if len(gt_ranks) >= 2:
            tau, _ = kendalltau(gt_ranks, ag_ranks)
            if tau != tau: tau = 0.0  # NaN guard
            tau_sum += float(tau); n_pages_used += 1
    if total_pages == 0: return 0.0, 0
    return (tau_sum / total_pages), n_pages_used


def collect_gt_bubbles(gt):
    out = []
    for gp in gt["pages"]:
        for gpan in gp["panels"]:
            for gb in gpan.get("bubbles", []):
                out.append({"page_id": gp["page_id"], **gb})
    return out


def match_bubbles(agent_bubbles, gt_bubbles):
    by_page_agent = defaultdict(list)
    for i, b in enumerate(agent_bubbles): by_page_agent[b["page_id"]].append((i, b))
    pairs = []  # list of (gt_index, agent_index)
    used_a = set()
    for gi, gb in enumerate(gt_bubbles):
        cand = by_page_agent.get(gb["page_id"], [])
        best_j, best_iou = -1, 0.0
        for j, ab in cand:
            if j in used_a or ab["bbox"] is None: continue
            v = iou_2d(ab["bbox"], gb["bbox"])
            if v >= BUBBLE_IOU_THR and v > best_iou:
                best_j, best_iou = j, v
        if best_j >= 0:
            used_a.add(best_j); pairs.append((gi, best_j))
    return pairs


def ocr_cer_dim(agent_bubbles, gt_bubbles, pairs):
    try:
        import jiwer
    except Exception:
        return 1.0, 0, 0
    total, n_gt = 0.0, len(gt_bubbles)
    if n_gt == 0: return 1.0, 0, 0
    matched_idx = {gi for gi, _ in pairs}
    matched_n = 0
    for gi, gb in enumerate(gt_bubbles):
        if gi in matched_idx:
            ai = next(j for g, j in pairs if g == gi)
            agent_txt = (agent_bubbles[ai]["ocr_text"] or "").strip()
            gt_txt = (gb.get("ocr_text") or "").strip()
            if not gt_txt:
                cer = 0.0 if not agent_txt else 1.0
            else:
                try: cer = float(jiwer.cer(gt_txt, agent_txt))
                except Exception: cer = 1.0
            cer = min(1.0, max(0.0, cer))
            total += cer; matched_n += 1
        else:
            total += 1.0
    return (total / n_gt), matched_n, n_gt


def speaker_acc_dim(agent_bubbles, gt_bubbles, pairs):
    if not pairs: return 0.0, 0, 0
    n_eligible, n_correct = 0, 0
    for gi, ai in pairs:
        gb = gt_bubbles[gi]
        ab = agent_bubbles[ai]
        gt_type = gb.get("bubble_type")
        if gt_type in DIALOG_TYPES:
            n_eligible += 1
            gt_spk = gb.get("speaker_id")
            ag_spk = ab.get("speaker_id")
            if gt_spk == ag_spk and gt_spk is not None:
                n_correct += 1
        elif gt_type in NULL_SPEAKER_TYPES:
            n_eligible += 1
            if ab.get("speaker_id") is None:
                n_correct += 1
    if n_eligible == 0: return 0.0, 0, 0
    return (n_correct / n_eligible), n_correct, n_eligible


def decoy_rej(decoys, gt_decoys):
    used, hits = set(), 0
    for g in gt_decoys:
        for i, d in enumerate(decoys):
            if i in used: continue
            if d["page_id"] != g["page_id"]: continue
            if d["decoy_type"] != g["decoy_type"]: continue
            if iou_2d(d["bbox"], g["bbox"]) > DECOY_IOU_THR:
                used.add(i); hits += 1; break
    n = len(gt_decoys)
    return (hits / n if n else 0.0), hits, n


def cross_doc(panels, bubbles, decoys, memo, png, gt):
    errs = []
    char_vocab = set(gt["closed_vocab_character"])
    btype_vocab = set(gt["closed_vocab_bubble"])
    dtype_vocab = set(gt["closed_vocab_decoy"])
    if len(panels) < MIN_PANELS:
        errs.append(f"panel_order.csv has {len(panels)} rows (< {MIN_PANELS})")
    if len(bubbles) < MIN_BUBBLES:
        errs.append(f"bubbles.csv has {len(bubbles)} rows (< {MIN_BUBBLES})")
    bad_bt = sorted({b["bubble_type"] for b in bubbles if b["bubble_type"] not in btype_vocab})
    if bad_bt: errs.append(f"bubble_type out of vocab: {bad_bt[:5]}")
    bad_sp = sorted({b["speaker_id"] for b in bubbles
                     if b["speaker_id"] is not None and b["speaker_id"] not in char_vocab})
    if bad_sp: errs.append(f"speaker_id out of vocab: {bad_sp[:5]}")
    bad_null = [(b["bubble_id"], b["bubble_type"]) for b in bubbles
                if b["bubble_type"] in NULL_SPEAKER_TYPES and b["speaker_id"] is not None]
    if bad_null: errs.append(f"sound_effect/narration_box has non-null speaker: {bad_null[:3]}")
    bad_dt = sorted({d.get("decoy_type") for d in decoys if d.get("decoy_type") not in dtype_vocab})
    if bad_dt: errs.append(f"decoy_type out of vocab: {bad_dt[:5]}")
    by_page = defaultdict(list)
    for p in panels: by_page[p["page_id"]].append(p)
    bad_perm = []
    for pid, ps in by_page.items():
        ranks = sorted(p["reading_order_rank"] for p in ps)
        if ranks != list(range(len(ps))):
            bad_perm.append(pid)
    if bad_perm: errs.append(f"reading_order_rank not 0..n-1 permutation: {bad_perm[:3]}")
    dup_bid = [bid for bid, c in Counter(b["bubble_id"] for b in bubbles).items() if c > 1]
    if dup_bid: errs.append(f"duplicate bubble_id: {dup_bid[:3]}")
    bid_pat = re.compile(r"^B\d{3,}$")
    bad_bid = [b["bubble_id"] for b in bubbles if not bid_pat.match(b["bubble_id"] or "")]
    if bad_bid: errs.append(f"bubble_id format violation: {bad_bid[:3]}")
    n_words = len(re.findall(r"\S+", memo))
    if n_words > MEMO_WORD_CAP:
        errs.append(f"narrative.md {n_words} words (> {MEMO_WORD_CAP})")
    bubble_chars_present = {b["speaker_id"] for b in bubbles
                             if b["speaker_id"] in char_vocab}
    tokens = set(re.findall(r"\b[a-z][a-z_]+\b", memo.lower()))
    cited_chars = tokens & char_vocab & bubble_chars_present
    if len(cited_chars) < 3:
        errs.append(f"narrative cites {len(cited_chars)} vocab character_id present in bubbles.csv (need >= 3)")
    bad_cite_char = (tokens & char_vocab) - bubble_chars_present
    if bad_cite_char:
        errs.append(f"narrative cites character_id absent from bubbles.csv: {sorted(bad_cite_char)[:5]}")
    cited_pages = set(re.findall(r"\bpage_\d{2}\b", memo))
    if len(cited_pages) < 3:
        errs.append(f"narrative cites {len(cited_pages)} page_id (need >= 3)")
    sz = img_size(png)
    if sz is None: errs.append("reading_order_overlay.png unreadable")
    elif sz[0] < IMG_MIN_W or sz[1] < IMG_MIN_H:
        errs.append(f"reading_order_overlay.png {sz[0]}x{sz[1]} < {IMG_MIN_W}x{IMG_MIN_H}")
    return errs


def evaluate(out: Path, gt: dict, resolved: dict[str, Path]):
    panels, bubbles, decoys, memo = parse_outputs(out, resolved)
    gt_bubbles = collect_gt_bubbles(gt)
    pairs = match_bubbles(bubbles, gt_bubbles)
    tau, n_pages = panel_kendall(panels, gt)
    cer, matched_n, n_gt_b = ocr_cer_dim(bubbles, gt_bubbles, pairs)
    spk, n_corr, n_elig = speaker_acc_dim(bubbles, gt_bubbles, pairs)
    rej, dh, dt = decoy_rej(decoys, gt["decoys"])
    cd = cross_doc(panels, bubbles, decoys, memo,
                   resolved["reading_order_overlay.png"], gt)
    return {
        "panel_order_kendall_tau": {"tau": round(float(tau), 3),
                                     "n_pages_matched": n_pages,
                                     "ratio": max(0.0, float(tau)),
                                     "ok": tau >= 0.65},
        "ocr_cer": {"cer": round(float(cer), 3), "matched": matched_n, "n_gt": n_gt_b,
                    "ratio": max(0.0, 1.0 - cer / 0.20),
                    "ok": cer <= 0.20 and matched_n > 0},
        "speaker_attribution_acc": {"ratio": round(float(spk), 3),
                                     "correct": n_corr, "eligible": n_elig,
                                     "ok": spk >= 0.55},
        "decoy_rejection": {"ratio": round(rej, 3), "hits": dh, "gt": dt,
                            "ok": rej >= 0.60},
        "cross_doc_consistency": {"errors": cd[:8], "n_errors": len(cd),
                                   "ok": len(cd) == 0},
    }, panels, bubbles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--fixtures", default="fixtures")
    a = ap.parse_args()
    out = Path(a.output)
    ok, errs, resolved = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs}, indent=2)); return
    gt = json.load(open(a.gt))
    res, panels, bubbles = evaluate(out, gt, resolved)
    char_counts = Counter(b["speaker_id"] for b in bubbles
                          if b["speaker_id"] in set(gt["closed_vocab_character"]))
    page_panels = Counter(p["page_id"] for p in panels)
    ground = {
        "n_panels_total": len(panels),
        "n_bubbles_total": len(bubbles),
        "top_characters_by_lines": [c for c, _ in char_counts.most_common(5)],
        "top_pages_by_panels": [p for p, _ in page_panels.most_common(5)],
        "n_decoys_hit": res["decoy_rejection"].get("hits"),
        "n_decoys_gt":  res["decoy_rejection"].get("gt"),
        "panel_kendall_tau": res["panel_order_kendall_tau"].get("tau"),
        "ocr_cer": res["ocr_cer"].get("cer"),
        "speaker_acc": res["speaker_attribution_acc"].get("ratio"),
    }
    mq = memo_score(resolved["narrative.md"], ground)
    mq_score = float(mq.get("score", 0.0))
    res["memo_quality"] = {"score": round(mq_score, 3),
                            "reasons": mq.get("reasons", "")[:200],
                            "ok": bool(mq.get("ok"))}
    # MLLM image content checklist on rendered reading_order_overlay.
    oa = mllm_image_axes(resolved["reading_order_overlay.png"], OVERLAY_AXES,
                         threshold=0.70)
    res["overlay_image_axes"] = {
        "axes": oa.get("axes", []),
        "score": round(float(oa.get("score", 0.0)), 3),
        "ok": bool(oa.get("ok", False)),
        "threshold": 0.70,
    }
    W = {"panel_order_kendall_tau": 0.16, "ocr_cer": 0.12,
         "speaker_attribution_acc": 0.16, "decoy_rejection": 0.12,
         "cross_doc_consistency": 0.24, "memo_quality": 0.10,
         "overlay_image_axes": 0.10}
    score = 0.0
    for k, w in W.items():
        if k == "cross_doc_consistency":
            score += w * (1.0 if res[k]["ok"] else 0.0)
        elif k == "memo_quality":
            score += w * mq_score
        elif k == "overlay_image_axes":
            score += w * res[k]["score"]
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
