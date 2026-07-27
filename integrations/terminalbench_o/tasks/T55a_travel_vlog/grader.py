"""Grader for T55a Travel Vlog Itinerary Builder (v3).

v3 pattern:
  - Closed-vocab dims (place_recall, decoy_rejection, time_iou,
    activity_class_acc) are deterministic — GT-driven, no LLM.
  - recap.md is genuinely free-form prose; replaced single
    `judge_memo_grounded` continuous-score with `axes_yes_no_score` over 5
    independent axes.
  - Reweighted: deterministic dims ~0.90, axes-LLM dim 0.10.

6 dimensions (weights sum 1.0):
  place_recall            >= 0.70  (weight 0.23)
  decoy_rejection         <= 0.20  (weight 0.13)  lower-is-better
  time_iou                >= 0.45  (weight 0.18)
  activity_class_acc      >= 0.65  (weight 0.13)
  cross_doc_consistency   == 1.0   (weight 0.23)
  memo_quality_axes       >= 0.60  (weight 0.10)  axes yes/no on recap.md
"""
import argparse, json, os, re, sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))

from open_vocab_judge import axes_yes_no_score  # noqa: E402

MEMO_AXES = [
    ("city_named_early",
     "Does RECAP.MD name the destination city in the first one or two "
     "sentences, matching GROUND_DATA.city?"),
    ("specific_places_highlighted",
     "Does RECAP.MD highlight 2-3 specific place names that appear in "
     "GROUND_DATA.itinerary_places (not generic 'a museum' or 'a park')?"),
    ("transport_sentence",
     "Does RECAP.MD include at least one concrete sentence on transport "
     "(e.g., 'mostly on foot with one bus transfer') — not a generic mention?"),
    ("cohesive_prose",
     "Does RECAP.MD read as cohesive English travel prose — not a bullet "
     "checklist of CSV fields, not lorem ipsum, not a count dump?"),
    ("no_fabrication",
     "Is RECAP.MD free of fabricated place names (no decoy place names "
     "mentioned that the agent did not visit, per GROUND_DATA.itinerary_places)?"),
]

VOCAB_ACTIVITY = {"sightseeing", "dining", "transit", "shopping", "lodging"}
VOCAB_TRANSPORT = {"walk", "bus", "train", "taxi", "bicycle", "none"}
REQUIRED_FILES = ["itinerary.json", "pins.geojson", "recap.md"]
TIME_TOLERANCE = 5.0
COORD_TOL = 1e-4
MAX_RECAP_WORDS = 320
MAX_SEGMENTS = 30


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
    try:
        json.load(open(out / "itinerary.json"))
    except Exception as e:
        return False, [f"itinerary.json not parseable: {e}"]
    try:
        json.load(open(out / "pins.geojson"))
    except Exception as e:
        return False, [f"pins.geojson not parseable: {e}"]
    return True, []


def load_vocab(fixtures: Path, gt: dict | None = None):
    pv = json.load(open(fixtures / "places_vocab.json"))
    if "places" in pv:
        all_places = {p["name"]: p for p in pv["places"]}
        if gt is not None:
            gt_names = {v["name"] for v in gt.get("visited_places", [])}
            targets = {n: p for n, p in all_places.items() if n in gt_names}
            decoys = {n: p for n, p in all_places.items() if n not in gt_names}
        else:
            targets = dict(all_places)
            decoys = {}
    else:
        # legacy schema
        targets = {p["name"]: p for p in pv["target_places"]}
        decoys = {p["name"]: p for p in pv["decoy_places"]}
        all_places = {**targets, **decoys}
    return targets, decoys, all_places


def cross_doc_check(itinerary, pins, recap, all_places, duration):
    errs = []
    if not isinstance(itinerary, list):
        errs.append("itinerary.json is not a list")
        return errs
    if not (1 <= len(itinerary) <= MAX_SEGMENTS):
        errs.append(f"itinerary length {len(itinerary)} not in [1, {MAX_SEGMENTS}]")
    last_end = -1.0
    for i, seg in enumerate(itinerary):
        if not isinstance(seg, dict):
            errs.append(f"segment[{i}] not an object"); continue
        place = seg.get("place")
        if place not in all_places:
            errs.append(f"segment[{i}] place '{place}' not in vocab")
        try:
            t0 = float(seg["t_start"]); t1 = float(seg["t_end"])
        except (KeyError, TypeError, ValueError):
            errs.append(f"segment[{i}] missing/invalid t_start/t_end"); continue
        if not (0 <= t0 < t1 <= duration + 0.5):
            errs.append(f"segment[{i}] times {t0:.2f}-{t1:.2f} out of [0, {duration:.1f}]")
        if t0 < last_end - 1e-3:
            errs.append(f"segment[{i}] overlap: t_start={t0:.2f} < prev t_end={last_end:.2f}")
        last_end = max(last_end, t1)
        ac = seg.get("activity_class")
        if ac not in VOCAB_ACTIVITY:
            errs.append(f"segment[{i}] activity_class '{ac}' not in vocab")
        tm = seg.get("transport_mode_in")
        if tm not in VOCAB_TRANSPORT:
            errs.append(f"segment[{i}] transport_mode_in '{tm}' not in vocab")

    # pins.geojson
    if not isinstance(pins, dict) or pins.get("type") != "FeatureCollection":
        errs.append("pins.geojson missing type=FeatureCollection")
        return errs
    feats = pins.get("features")
    if not isinstance(feats, list):
        errs.append("pins.geojson features not a list"); return errs
    unique_places = []
    seen = set()
    for seg in itinerary:
        p = seg.get("place")
        if p in all_places and p not in seen:
            seen.add(p); unique_places.append(p)
    if len(feats) != len(unique_places):
        errs.append(f"pins.geojson has {len(feats)} features, expected {len(unique_places)} unique places")
    seen_pin_names = set()
    for j, f in enumerate(feats):
        if not isinstance(f, dict): errs.append(f"feature[{j}] not an object"); continue
        if f.get("type") != "Feature":
            errs.append(f"feature[{j}] type != 'Feature'")
        geom = f.get("geometry") or {}
        if geom.get("type") != "Point":
            errs.append(f"feature[{j}] geometry.type != 'Point'")
        coords = geom.get("coordinates")
        if not (isinstance(coords, list) and len(coords) == 2):
            errs.append(f"feature[{j}] coordinates not a length-2 list"); continue
        try:
            lon, lat = float(coords[0]), float(coords[1])
        except (TypeError, ValueError):
            errs.append(f"feature[{j}] coordinates not numeric"); continue
        props = f.get("properties") or {}
        name = props.get("name")
        if name not in all_places:
            errs.append(f"feature[{j}] name '{name}' not in vocab"); continue
        if name in seen_pin_names:
            errs.append(f"feature[{j}] duplicate pin for '{name}'")
        seen_pin_names.add(name)
        ref = all_places[name]
        if abs(lon - ref["lon"]) > COORD_TOL or abs(lat - ref["lat"]) > COORD_TOL:
            errs.append(f"feature[{j}] '{name}' coords ({lon},{lat}) != vocab ({ref['lon']},{ref['lat']})")
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            errs.append(f"feature[{j}] coords out of geo range — likely [lat,lon] swap")

    # recap word count
    n_words = len(re.findall(r"\S+", recap))
    if n_words > MAX_RECAP_WORDS:
        errs.append(f"recap.md word count {n_words} > {MAX_RECAP_WORDS}")
    if n_words < 30:
        errs.append(f"recap.md word count {n_words} < 30 (too short)")
    return errs


def interval_iou(a0, a1, b0, b1, tol=0.0):
    a0, a1 = a0 - tol, a1 + tol
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def evaluate(out: Path, gt: dict, fixtures: Path):
    itinerary = json.load(open(out / "itinerary.json"))
    pins = json.load(open(out / "pins.geojson"))
    recap = (out / "recap.md").read_text()

    targets, decoys, all_places = load_vocab(fixtures, gt)
    duration = float(gt["duration_sec"])
    cd_errs = cross_doc_check(itinerary, pins, recap, all_places, duration)

    # Filter to valid segments (place in vocab, valid times) for metric computation
    valid_segs = []
    for seg in itinerary:
        if not isinstance(seg, dict): continue
        p = seg.get("place")
        if p not in all_places: continue
        try:
            t0 = float(seg["t_start"]); t1 = float(seg["t_end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= t0 < t1 <= duration + 0.5): continue
        valid_segs.append({**seg, "t_start": t0, "t_end": t1})

    pred_target_places = {s["place"] for s in valid_segs if s["place"] in targets}
    pred_decoy_places = {s["place"] for s in valid_segs if s["place"] in decoys}

    gt_targets = {v["name"]: v for v in gt["visited_places"]}

    # place_recall
    n_targets = len(gt_targets)
    n_recalled = sum(1 for n in gt_targets if n in pred_target_places)
    place_recall = n_recalled / max(1, n_targets)

    # decoy_rejection (lower is better → fraction of decoys flagged)
    n_decoys = max(1, len(decoys))
    decoy_rate = len(pred_decoy_places) / n_decoys

    # time_iou: per recalled GT place, take the union of agent's segments for
    # that place and IoU vs GT [t_start, t_end].  Average across recalled.
    ious = []
    for name, gt_p in gt_targets.items():
        if name not in pred_target_places: continue
        agent_segs = [(s["t_start"], s["t_end"]) for s in valid_segs if s["place"] == name]
        gt0, gt1 = float(gt_p["t_start"]), float(gt_p["t_end"])
        # Compute IoU using the union of agent intervals vs the GT interval
        agent_segs.sort()
        merged = []
        for a0, a1 in agent_segs:
            if merged and a0 <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], a1))
            else:
                merged.append((a0, a1))
        inter = sum(max(0.0, min(a1, gt1) - max(a0, gt0)) for a0, a1 in merged)
        agent_union = sum(a1 - a0 for a0, a1 in merged)
        gt_len = gt1 - gt0
        union_len = agent_union + gt_len - inter
        if union_len <= 0:
            ious.append(0.0)
        else:
            # Apply ±tolerance by expanding GT then re-clipping intersection
            inter_tol = sum(max(0.0, min(a1, gt1 + TIME_TOLERANCE) - max(a0, gt0 - TIME_TOLERANCE))
                            for a0, a1 in merged)
            inter_tol = min(inter_tol, agent_union, gt_len + 2 * TIME_TOLERANCE)
            iou_val = inter_tol / union_len
            ious.append(min(1.0, max(0.0, iou_val)))
    time_iou = sum(ious) / len(ious) if ious else 0.0

    # activity_class_acc: among rows whose place is a GT target, fraction whose
    # activity matches GT activity.
    n_act_total, n_act_correct = 0, 0
    for s in valid_segs:
        if s["place"] not in gt_targets: continue
        n_act_total += 1
        gt_act = gt_targets[s["place"]].get("activity_class")
        if s.get("activity_class") == gt_act:
            n_act_correct += 1
    act_acc = n_act_correct / n_act_total if n_act_total else 0.0

    itinerary_places = sorted({s["place"] for s in valid_segs})
    ground_for_memo = {
        "city": gt.get("city"),
        "itinerary_places": itinerary_places,
        "n_segments": len(valid_segs),
    }

    return {
        "place_recall": {"ratio": round(place_recall, 3), "tp": n_recalled, "n": n_targets,
                         "ok": place_recall >= 0.70},
        "decoy_rejection": {"ratio": round(decoy_rate, 3),
                            "n_decoys_in_itinerary": len(pred_decoy_places),
                            "n_decoys_total": len(decoys),
                            "ok": decoy_rate <= 0.20},
        "time_iou": {"ratio": round(time_iou, 3), "n_pairs": len(ious),
                     "ok": time_iou >= 0.45},
        "activity_class_acc": {"ratio": round(act_acc, 3),
                               "tp": n_act_correct, "n": n_act_total,
                               "ok": act_acc >= 0.65},
        "cross_doc_consistency": {"errors": cd_errs[:8], "n_errors": len(cd_errs),
                                  "ok": len(cd_errs) == 0},
        "_meta": {"ground_for_memo": ground_for_memo},
    }


def memo_quality_axes(out: Path, ground: dict) -> dict:
    """v3: Replace single-LLM continuous judge with 5 yes/no axes."""
    recap = (out / "recap.md").read_text()
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground, ensure_ascii=False)}\n\n"
        f"RECAP.MD:\n{recap[:6000]}\n"
    )
    return axes_yes_no_score(payload, MEMO_AXES, threshold=0.60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--fixtures", default="fixtures")
    args = ap.parse_args()
    out, fix = Path(args.output), Path(args.fixtures)
    ok, errs = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs}, indent=2))
        return
    gt = json.load(open(args.gt))
    res = evaluate(out, gt, fix)
    meta = res.pop("_meta", {})
    ground_for_memo = meta.get("ground_for_memo", {})

    # v3: replace single-LLM continuous judge with axes-yes/no.
    mq = memo_quality_axes(out, ground_for_memo)
    res["memo_quality_axes"] = {
        "axes": mq.get("axes", []),
        "score": mq.get("score", 0.0),
        "ok": bool(mq.get("ok", False)),
        "threshold": 0.60,
    }

    all_ok = all(v["ok"] for v in res.values())
    # v3 weights (sum = 1.000): deterministic dims dominate (~0.90).
    weights = {"place_recall": 0.23, "decoy_rejection": 0.13, "time_iou": 0.18,
               "activity_class_acc": 0.13, "cross_doc_consistency": 0.23,
               "memo_quality_axes": 0.10}
    score = 0.0
    for k, w in weights.items():
        if k == "cross_doc_consistency":
            score += w * (1.0 if res[k]["ok"] else 0.0)
        elif k == "decoy_rejection":
            score += w * max(0.0, 1.0 - res[k]["ratio"])
        elif k == "memo_quality_axes":
            score += w * float(res[k].get("score", 0.0))
        else:
            score += w * max(0.0, res[k]["ratio"])
    print(json.dumps({"pass": all_ok, "checks": {k: v["ok"] for k, v in res.items()},
                      **res, "score": round(score, 3), "weights": weights},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
