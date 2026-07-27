"""Grader for T108a Political Speech Rhetoric Analysis (v3).

v3 pattern:
  - argparse CLI (--output, --gt) matching the rest of tasks_v3, single
    JSON result on stdout — no positional args, no human-readable text.
  - Closed-vocab dims (rhetoric_device ∈ rhetoric_vocab.json, theme ∈
    theme_vocab.json) are hard-gated.
  - Deterministic recall/IoU dims drive the bulk of the score; the
    analysis memo is graded by `axes_yes_no_score` over 4 yes/no axes
    at low weight.
  - Aligns to the actual GT schema in reference/gt.json (`phrase`,
    `start_time`, `rhetoric_device_inventory`) — the previous grader
    expected fields that GT does not produce.

7 dimensions (weights sum to 1.0):
  key_phrase_recall          >= 0.40   (weight 0.20)
  segment_iou_macro          >= 0.45   (weight 0.15)
  segment_theme_acc          >= 0.50   (weight 0.15)
  rhetoric_inventory_recall  >= 0.45   (weight 0.20)
  closed_vocab               == 1.0    (weight 0.15)
  cross_doc_consistency      == 1.0    (weight 0.10)
  memo_axes                  >= 0.55   (weight 0.05)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))

try:
    from open_vocab_judge import axes_yes_no_score  # noqa: E402
except Exception:
    axes_yes_no_score = None

REQUIRED_FILES = [
    "key_phrases.json",
    "segments.json",
    "rhetoric_inventory.json",
    "analysis.md",
]
PHRASE_TIME_TOL = 5.0
INVENTORY_TIME_TOL = 3.0
SEGMENT_BOUNDARY_TOL = 5.0
TEXT_SIM_MIN = 0.40
WORD_MIN = 300
WORD_MAX = 500

MEMO_AXES = [
    ("rhetorical_strategy_named",
     "Does ANALYSIS.MD discuss the speech's overall rhetorical strategy "
     "(naming concrete devices such as anaphora, antithesis, chiasmus, "
     "metaphor, parallelism — not just generic 'rhetoric')?"),
    ("audience_addressed",
     "Does ANALYSIS.MD discuss audience considerations (who the speech "
     "is addressed to, intended effect on listeners)?"),
    ("structure_or_pattern",
     "Does ANALYSIS.MD describe structure / organisation / progression "
     "of the speech (segments, build-up, turning points)?"),
    ("effectiveness_grounded",
     "Does ANALYSIS.MD assess effectiveness with concrete textual "
     "evidence — quoting or paraphrasing real lines from the speech, "
     "not lorem ipsum or generic praise?"),
]


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _toks(s: str) -> set[str]:
    return set(_norm(s).split())


def _text_sim(a: str, b: str) -> float:
    A, B = _toks(a), _toks(b)
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))


def _phrase_time(p: dict) -> float:
    for k in ("timestamp", "start_time", "t_start", "time"):
        v = p.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return -1.0


def _gating(out: Path) -> tuple[bool, list[str]]:
    errs: list[str] = []
    for fn in REQUIRED_FILES:
        p = out / fn
        if not p.exists():
            errs.append(f"missing: {fn}")
        elif p.stat().st_size == 0:
            errs.append(f"empty: {fn}")
    return (not errs), errs


def grade_key_phrases(sub: dict, gt: dict) -> dict:
    gt_ph = gt.get("key_phrases", []) or []
    sub_ph = sub.get("phrases", []) or []
    matched = set()
    details = []
    for sp in sub_ph:
        sp_text = sp.get("text") or sp.get("phrase") or ""
        sp_time = _phrase_time(sp)
        sp_dev = _norm(sp.get("rhetoric_device", ""))
        best = (-1, 0.0, None)
        for i, gp in enumerate(gt_ph):
            if i in matched:
                continue
            gt_text = gp.get("phrase") or gp.get("text") or ""
            gt_time = float(gp.get("start_time", gp.get("timestamp", -1)) or -1)
            gt_dev = _norm(gp.get("rhetoric_device", ""))
            sim = _text_sim(sp_text, gt_text)
            if sim < TEXT_SIM_MIN:
                continue
            time_ok = (sp_time >= 0 and gt_time >= 0
                        and abs(sp_time - gt_time) <= PHRASE_TIME_TOL)
            score = sim
            if time_ok:
                score += 0.20
            if sp_dev and sp_dev == gt_dev:
                score += 0.10
            if score > best[1]:
                best = (i, score, gp)
        if best[0] >= 0 and best[1] >= TEXT_SIM_MIN:
            matched.add(best[0])
            details.append({"sub_text": sp_text[:60], "gt_idx": best[0],
                            "score": round(best[1], 3)})
    n_gt = len(gt_ph)
    recall = len(matched) / max(1, n_gt)
    return {"matched": len(matched), "n_gt": n_gt,
            "ratio": round(recall, 3), "details": details[:10],
            "ok": recall >= 0.40}


def grade_segments(sub: dict, gt: dict) -> tuple[dict, dict]:
    gt_seg = gt.get("rhetoric_segments", []) or []
    sub_seg = sub.get("segments", []) or []
    if not gt_seg or not sub_seg:
        return ({"ratio": 0.0, "ok": False, "n_gt": len(gt_seg),
                  "n_sub": len(sub_seg), "details": []},
                {"matched": 0, "n_gt": len(gt_seg), "ratio": 0.0, "ok": False})
    iou_per = []
    theme_match = 0
    used_sub = set()
    for gi, gs in enumerate(gt_seg):
        gs_s = float(gs.get("start_time", -1))
        gs_e = float(gs.get("end_time", -1))
        gs_theme = _norm(gs.get("theme", ""))
        best = (-1, 0.0)
        for si, ss in enumerate(sub_seg):
            if si in used_sub:
                continue
            ss_s = float(ss.get("start_time", -1))
            ss_e = float(ss.get("end_time", -1))
            if ss_e <= ss_s or gs_e <= gs_s:
                continue
            inter = max(0.0, min(gs_e, ss_e) - max(gs_s, ss_s))
            union = max(gs_e, ss_e) - min(gs_s, ss_s)
            iou = inter / max(1e-6, union)
            if iou > best[1]:
                best = (si, iou)
        if best[0] >= 0:
            used_sub.add(best[0])
            iou_per.append(best[1])
            ss_theme = _norm(sub_seg[best[0]].get("theme", ""))
            if ss_theme and gs_theme and ss_theme == gs_theme:
                theme_match += 1
        else:
            iou_per.append(0.0)
    iou_macro = sum(iou_per) / max(1, len(iou_per))
    theme_ratio = theme_match / max(1, len(gt_seg))
    iou_dim = {"iou_per_gt": [round(x, 3) for x in iou_per],
                "macro_iou": round(iou_macro, 3),
                "n_gt": len(gt_seg), "n_sub": len(sub_seg),
                "ok": iou_macro >= 0.45}
    theme_dim = {"matched": theme_match, "n_gt": len(gt_seg),
                  "ratio": round(theme_ratio, 3),
                  "ok": theme_ratio >= 0.50}
    return iou_dim, theme_dim


def grade_inventory(sub: dict, gt: dict) -> dict:
    gt_inv: dict = gt.get("rhetoric_device_inventory", {}) or {}
    sub_devices: dict = (sub.get("devices") or {}) if isinstance(
        sub.get("devices"), dict) else {}
    n_gt_total = sum(len(v) for v in gt_inv.values())
    if n_gt_total == 0:
        return {"matched": 0, "n_gt": 0, "ratio": 0.0, "ok": False}
    sub_flat = []
    for dev, instances in sub_devices.items():
        if not isinstance(instances, list):
            continue
        for inst in instances:
            if not isinstance(inst, dict):
                continue
            sub_flat.append({"device": _norm(dev),
                              "time": _phrase_time(inst),
                              "text": inst.get("text", "")})
    matched = 0
    matched_pairs: set[tuple[str, int]] = set()
    for sp in sub_flat:
        for dev, instances in gt_inv.items():
            for j, inst in enumerate(instances):
                key = (dev, j)
                if key in matched_pairs:
                    continue
                if sp["device"] != _norm(dev):
                    continue
                gt_t = float(inst.get("time", -1))
                if (sp["time"] >= 0 and gt_t >= 0
                        and abs(sp["time"] - gt_t) <= INVENTORY_TIME_TOL):
                    sim = _text_sim(sp["text"], inst.get("text", ""))
                    if sim >= 0.30:
                        matched += 1
                        matched_pairs.add(key)
                        break
    ratio = matched / max(1, n_gt_total)
    return {"matched": matched, "n_gt": n_gt_total,
            "ratio": round(ratio, 3), "ok": ratio >= 0.45}


def grade_closed_vocab(sub_phrases: dict, sub_segments: dict,
                          sub_inventory: dict, fixtures: Path) -> dict:
    rh_path = fixtures / "rhetoric_vocab.json"
    th_path = fixtures / "theme_vocab.json"
    rhet = set()
    themes = set()
    if rh_path.exists():
        try:
            data = json.loads(rh_path.read_text())
            rhet = set(data.get("devices") or data.get("rhetoric_devices") or [])
        except Exception:
            pass
    if th_path.exists():
        try:
            data = json.loads(th_path.read_text())
            themes = set(data.get("themes") or data.get("theme_categories") or [])
        except Exception:
            pass
    errs = []
    for p in (sub_phrases.get("phrases") or []):
        d = (p.get("rhetoric_device") or "").strip()
        if d and rhet and d not in rhet:
            errs.append(f"phrase device not in vocab: {d}")
    for s in (sub_segments.get("segments") or []):
        t = (s.get("theme") or "").strip()
        if t and themes and t not in themes:
            errs.append(f"segment theme not in vocab: {t}")
        d = (s.get("primary_device") or "").strip()
        if d and rhet and d not in rhet:
            errs.append(f"segment primary_device not in vocab: {d}")
    for dev in (sub_inventory.get("devices") or {}):
        if rhet and dev not in rhet:
            errs.append(f"inventory device not in vocab: {dev}")
    return {"errors": errs[:8], "n_errors": len(errs),
            "ok": len(errs) == 0}


def grade_cross_doc(sub_phrases: dict, sub_segments: dict,
                     sub_inventory: dict, analysis_text: str,
                     duration: float) -> dict:
    errs: list[str] = []
    phrases = sub_phrases.get("phrases") or []
    if not (4 <= len(phrases) <= 6):
        errs.append(f"key_phrases count {len(phrases)} not in [4,6]")
    segments = sub_segments.get("segments") or []
    if not (4 <= len(segments) <= 6):
        errs.append(f"segments count {len(segments)} not in [4,6]")
    if segments:
        ordered = sorted(segments, key=lambda s: s.get("start_time", 0))
        prev_end = 0.0
        for s in ordered:
            ss = float(s.get("start_time", -1))
            se = float(s.get("end_time", -1))
            if ss < 0 or se <= ss:
                errs.append(f"segment time invalid: {ss}-{se}")
                continue
            if abs(ss - prev_end) > SEGMENT_BOUNDARY_TOL:
                errs.append(
                    f"gap or overlap at boundary {prev_end:.1f}->{ss:.1f}")
            prev_end = se
        if duration > 0 and abs(prev_end - duration) > SEGMENT_BOUNDARY_TOL:
            errs.append(f"segments end at {prev_end:.1f} vs duration {duration:.1f}")
        if ordered[0].get("start_time", 0) > SEGMENT_BOUNDARY_TOL:
            errs.append("first segment does not start near 0")
    devs = sub_inventory.get("devices") or {}
    if not isinstance(devs, dict):
        errs.append("inventory.devices must be a dict")
        devs = {}
    n_dev_types = sum(1 for v in devs.values() if isinstance(v, list) and v)
    n_inst = sum(len(v) for v in devs.values() if isinstance(v, list))
    if n_dev_types < 3:
        errs.append(f"inventory has {n_dev_types} device types (need ≥3)")
    if n_inst < 6:
        errs.append(f"inventory has {n_inst} instances (need ≥6)")
    wc = len((analysis_text or "").split())
    if not (WORD_MIN <= wc <= WORD_MAX):
        errs.append(f"analysis word count {wc} not in [{WORD_MIN},{WORD_MAX}]")
    return {"errors": errs[:8], "word_count": wc,
            "n_dev_types": n_dev_types, "n_inst": n_inst,
            "ok": len(errs) == 0}


def grade_memo_axes(analysis_text: str) -> dict:
    if axes_yes_no_score is None:
        return {"score": 0.0, "axes": [], "ok": False,
                "note": "open_vocab_judge unavailable"}
    try:
        res = axes_yes_no_score(analysis_text or "",
                                  MEMO_AXES, threshold=0.55)
        score = float(res.get("score", 0.0))
        return {"score": round(score, 3),
                "axes": res.get("axes", []),
                "ok": score >= 0.55}
    except Exception as e:
        return {"score": 0.0, "axes": [], "ok": False,
                "note": f"judge error: {type(e).__name__}: {str(e)[:120]}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True,
                      help="agent output directory")
    ap.add_argument("--gt", required=True,
                      help="path to reference/gt.json")
    ap.add_argument("--fixtures", default=None,
                      help="path to fixtures/ (for closed-vocab files); "
                           "defaults to <task_dir>/fixtures")
    args = ap.parse_args()

    out = Path(args.output)
    fixtures_dir = (Path(args.fixtures) if args.fixtures
                     else Path(__file__).resolve().parent / "fixtures")

    ok_gate, errs = _gating(out)
    if not ok_gate:
        print(json.dumps({"pass": False, "stage": "gating",
                            "errors": errs}, indent=2))
        return

    sub_phrases = json.loads((out / "key_phrases.json").read_text())
    sub_segments = json.loads((out / "segments.json").read_text())
    sub_inventory = json.loads((out / "rhetoric_inventory.json").read_text())
    analysis_text = (out / "analysis.md").read_text()
    gt = json.loads(Path(args.gt).read_text())
    duration = float(gt.get("metadata", {}).get("duration_seconds", 0))

    kp = grade_key_phrases(sub_phrases, gt)
    seg_iou, seg_theme = grade_segments(sub_segments, gt)
    inv = grade_inventory(sub_inventory, gt)
    cv = grade_closed_vocab(sub_phrases, sub_segments, sub_inventory,
                              fixtures_dir)
    cdc = grade_cross_doc(sub_phrases, sub_segments, sub_inventory,
                            analysis_text, duration)
    memo = grade_memo_axes(analysis_text)

    weights = {
        "key_phrase_recall": 0.20,
        "segment_iou_macro": 0.15,
        "segment_theme_acc": 0.15,
        "rhetoric_inventory_recall": 0.20,
        "closed_vocab": 0.15,
        "cross_doc_consistency": 0.10,
        "memo_axes": 0.05,
    }
    score = (
        weights["key_phrase_recall"] * kp["ratio"]
        + weights["segment_iou_macro"] * seg_iou["macro_iou"]
        + weights["segment_theme_acc"] * seg_theme["ratio"]
        + weights["rhetoric_inventory_recall"] * inv["ratio"]
        + weights["closed_vocab"] * (1.0 if cv["ok"] else 0.0)
        + weights["cross_doc_consistency"] * (1.0 if cdc["ok"] else 0.0)
        + weights["memo_axes"] * memo.get("score", 0.0)
    )
    all_ok = all([
        kp["ok"], seg_iou["ok"], seg_theme["ok"], inv["ok"],
        cv["ok"], cdc["ok"], memo.get("ok", False),
    ])
    result = {
        "pass": bool(all_ok),
        "checks": {
            "key_phrase_recall_ok": kp["ok"],
            "segment_iou_macro_ok": seg_iou["ok"],
            "segment_theme_acc_ok": seg_theme["ok"],
            "rhetoric_inventory_recall_ok": inv["ok"],
            "closed_vocab_ok": cv["ok"],
            "cross_doc_consistency_ok": cdc["ok"],
            "memo_axes_ok": memo.get("ok", False),
        },
        "weights": weights,
        "key_phrase_recall": kp,
        "segment_iou_macro": seg_iou,
        "segment_theme_acc": seg_theme,
        "rhetoric_inventory_recall": inv,
        "closed_vocab": cv,
        "cross_doc_consistency": cdc,
        "memo_axes": memo,
        "score_so_far": round(score, 3),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
