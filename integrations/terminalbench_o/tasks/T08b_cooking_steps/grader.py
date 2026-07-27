"""Grader for T08b Cooking Steps (v2 — semantic match + LLM yes/no axes).

v2 redesign:
  - `temporal_coverage` replaces `mean_iou`. For each GT segment, take the
    max tIoU against any predicted clip; coverage = mean over GT. This is
    independent of the predicted clip count, so agents that merge GT
    substeps still get partial credit.
  - `ingredients_semantic_jaccard` replaces plain Jaccard. Ingredient
    tokens are compared via batched LLM semantic equivalence (fish↔salmon,
    chilies↔chili, steak↔meat all match).
  - `title_semantic_alignment` replaces token Jaccard. Hungarian-match
    pred to GT by tIoU, then ask the LLM (single batch) whether each
    matched (title, sentence) pair is semantically related.
  - `recipe_quality_axes` replaces single VLM judge: 5 yes/no axes.

Capability dims (~0.55): temporal_coverage 0.30
                         + ingredients_semantic_jaccard 0.15
                         + title_semantic_alignment 0.10
Quality dims    (~0.30): recipe_quality_axes 0.30
Format / sanity (~0.15): gating + partition_and_consistency

Usage:
    python grader.py --output /workspace/output \
                     --steps_gt reference/steps_gt.json \
                     --ingr_gt  reference/ingredients_gt.json
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))

from open_vocab_judge import (  # noqa: E402
    llm_semantic_pairs,
    axes_yes_no_score,
    mllm_video_axes,
)


# Multimodal checklist for one cooking step clip (steps/step_01.mp4 —
# the first clip the timeline iterates over). Sample 4 evenly-spaced
# frames; majority vote per axis across frames.
STEP_CONTENT_AXES = [
    ("real_cooking_footage_not_placeholder",
     "Is this frame natural cooking-video footage — NOT a black screen, "
     "color bars, test pattern, or single-color placeholder?"),
    ("food_or_kitchen_action_visible",
     "Is at least one of food, ingredients, hands cooking, kitchen "
     "utensils, or a kitchen scene visible in this frame?"),
    ("continuous_video_not_static_card",
     "Does this frame look like it was sampled from continuous cooking "
     "video footage rather than a static title card / recipe text card "
     "/ single-color slide?"),
    ("framing_focuses_on_cooking_step",
     "Does the framing of this frame focus on the cooking step (close "
     "or medium shot of the food / cooking action) rather than an "
     "irrelevant or off-topic subject?"),
]


# ---------------------------------------------------------------------- #
# Gating
# ---------------------------------------------------------------------- #
def gating(out: Path, timeline: list, n_clips: int) -> tuple[bool, list[str]]:
    errs = []
    for f in ["recipe.md", "timeline.json"]:
        if not (out / f).exists():
            errs.append(f"missing {f}")
    if not (out / "steps").is_dir():
        errs.append("missing steps/")
    if not (4 <= n_clips <= 25):
        errs.append(f"n_clips={n_clips} not in [4,25]")
    last_end = -1.0
    for i, st in enumerate(timeline):
        for k in ("index", "title", "start_sec", "end_sec", "clip"):
            if k not in st:
                errs.append(f"timeline[{i}] missing {k}")
                continue
        s, e = st.get("start_sec", 0), st.get("end_sec", 0)
        if e <= s:
            errs.append(f"timeline[{i}] end <= start")
        if s < last_end - 1e-3:
            errs.append(f"timeline[{i}] overlaps previous")
        last_end = e
        clip = out / st.get("clip", "")
        if not clip.exists():
            errs.append(f"clip not found: {st.get('clip')}")
            continue
        dur = ffprobe_duration(clip)
        if dur is None:
            errs.append(f"unreadable: {clip.name}")
        else:
            if not (3.0 <= dur <= 240.0):
                errs.append(f"{clip.name} duration {dur:.1f}s out of [3,240]")
            if abs(dur - (e - s)) > 0.5:
                errs.append(f"{clip.name} duration {dur:.2f} != end-start {e-s:.2f}")
    return len(errs) == 0, errs


def ffprobe_duration(p: Path) -> float | None:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(p)],
            stderr=subprocess.STDOUT,
        )
        return float(out.strip())
    except Exception:
        return None


# ---------------------------------------------------------------------- #
# temporal_coverage — mean over GT of best-match tIoU vs any predicted clip
# ---------------------------------------------------------------------- #
def _iou(p_s: float, p_e: float, g_s: float, g_e: float) -> float:
    inter = max(0.0, min(p_e, g_e) - max(p_s, g_s))
    union = max(p_e, g_e) - min(p_s, g_s)
    return inter / union if union > 0 else 0.0


def temporal_coverage(pred: list, gt: list) -> dict:
    """For each GT segment, find the pred with max IoU; coverage is the
    mean over GT. Independent of the predicted clip count."""
    if not gt:
        return {"coverage": 0.0, "per_gt_iou": [], "n_pred": len(pred), "n_gt": 0}
    if not pred:
        return {"coverage": 0.0,
                "per_gt_iou": [0.0] * len(gt),
                "n_pred": 0, "n_gt": len(gt)}
    per = []
    for g in gt:
        gs, ge = g["start_sec"], g["end_sec"]
        best = 0.0
        for p in pred:
            ps, pe = p["start_sec"], p["end_sec"]
            iou = _iou(ps, pe, gs, ge)
            if iou > best:
                best = iou
        per.append(best)
    cov = sum(per) / len(per)
    return {
        "coverage": round(float(cov), 3),
        "per_gt_iou": [round(x, 3) for x in per],
        "n_pred": len(pred),
        "n_gt": len(gt),
    }


# ---------------------------------------------------------------------- #
# ingredients_semantic_jaccard — Jaccard with LLM-judged equivalence
# ---------------------------------------------------------------------- #
def parse_recipe_ingredients(md: str) -> set[str]:
    m = re.search(r"##\s*Ingredients(.+?)(?=\n##|\Z)", md, re.S | re.I)
    if not m:
        return set()
    out = set()
    for line in m.group(1).splitlines():
        line = line.strip().lstrip("-*").strip()
        if not line:
            continue
        for token in re.findall(r"[a-zA-Z]+", line.lower()):
            if len(token) > 2:
                out.add(token)
    return out


# v3: hard-coded synonym map for the closed 13-item GT vocabulary.
# This replaces the LLM judge — the GT list is small + stable, and an
# explicit dict is more deterministic + faster + cheaper.
_INGR_SYNONYMS: dict[str, set[str]] = {
    "fish":    {"fish", "salmon", "tuna", "cod", "trout", "haddock",
                "fillet", "seafood"},
    "meat":    {"meat", "steak", "beef", "pork", "lamb", "veal",
                "mutton", "sausage"},
    "chicken": {"chicken", "poultry", "hen"},
    "chili":   {"chili", "chilli", "chillies", "chilies", "pepper_hot",
                "jalapeno", "habanero", "thai_chili"},
    "garlic":  {"garlic"},
    "ginger":  {"ginger"},
    "lemon":   {"lemon", "citrus"},
    "onion":   {"onion", "shallot", "scallion", "leek"},
    "pasta":   {"pasta", "spaghetti", "penne", "macaroni",
                "angel_hair", "linguine"},
    "pepper":  {"pepper", "peppercorn", "black_pepper", "white_pepper"},
    "rice":    {"rice", "basmati", "jasmine_rice", "short_grain"},
    "bread":   {"bread", "toast", "loaf", "baguette"},
    "cream":   {"cream", "heavy_cream", "double_cream", "creme"},
}


def _norm_ingr(t: str) -> str:
    return str(t or "").strip().lower().replace("-", "_").replace(" ", "_")


def ingredients_jaccard_strict(pred_tokens: set[str],
                                gt_tokens: set[str]) -> dict:
    """Recall-only ingredient match (was Jaccard).

    GT is built by regex over the YouTube chapter titles, so it is a
    *subset* of the actual ingredients used in the recipe — agents that
    correctly enumerate every visible ingredient would be punished by
    plain Jaccard for the extras the GT failed to capture.

    The metric is now `recall = |pred ∩ gt| / |gt|`, returned in the
    `jaccard` field for backwards compatibility with downstream weights.
    Tokens predicted but not in GT (`pred_only`) are reported for
    inspection but no longer reduce the score.
    """
    pred_norm = {_norm_ingr(t) for t in pred_tokens}
    gt_norm = {_norm_ingr(t) for t in gt_tokens}
    if not pred_norm and not gt_norm:
        return {"jaccard": 0.0, "matched_pred": [], "matched_gt": [],
                "pred_only": [], "gt_only": [], "examples": []}
    matched_pred: set[str] = set()
    matched_gt: set[str] = set()
    sample_matches: list[tuple[str, str]] = []
    for g in gt_norm:
        syns = _INGR_SYNONYMS.get(g, {g})
        for p in pred_norm:
            if p in syns:
                matched_pred.add(p)
                matched_gt.add(g)
                if len(sample_matches) < 12:
                    sample_matches.append((p, g))
    recall = len(matched_gt) / len(gt_norm) if gt_norm else 0.0
    return {
        "jaccard": round(float(recall), 3),  # actually recall; field name kept
        "metric": "recall_over_gt",
        "matched_pred": sorted(matched_pred),
        "matched_gt": sorted(matched_gt),
        "pred_only": sorted(pred_norm - matched_pred),
        "gt_only": sorted(gt_norm - matched_gt),
        "examples": [list(t) for t in sample_matches],
        "n_pred": len(pred_norm),
        "n_gt": len(gt_norm),
    }


# ---------------------------------------------------------------------- #
# title_semantic_alignment — Hungarian by tIoU + LLM yes/no per pair
# ---------------------------------------------------------------------- #
_TITLE_EXAMPLES = [
    ("Peel garlic", "Peeling Garlic", True),
    ("Cook rice", "How To Cook the Perfect Rice Basmati", True),
    ("Chop the onion", "How To Chop an Onion", True),
    ("Brown the meat", "Browning Meat or Fish", True),
    ("Sharpen knife", "Veg Peeler", False),
    ("Marinade chicken", "How To Zest the Lemon", False),
]


def title_semantic_alignment(timeline: list, gt: list) -> dict:
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
    except ImportError:
        raise SystemExit("scipy + numpy required: pip install scipy numpy")
    if not timeline or not gt:
        return {"passing_ratio": 0.0, "passing_pairs": 0,
                "total_pairs": 0, "details": []}
    P, G = len(timeline), len(gt)
    cost = np.ones((P, G))
    for i, p in enumerate(timeline):
        for j, g in enumerate(gt):
            iou = _iou(p["start_sec"], p["end_sec"],
                      g["start_sec"], g["end_sec"])
            cost[i, j] = 1.0 - iou
    ri, ci = linear_sum_assignment(cost)
    pairs = []
    matched_meta = []
    for i, j in zip(ri, ci):
        iou = 1.0 - cost[i, j]
        if iou < 0.1:
            continue
        pred_t = (timeline[i].get("title") or "").strip()
        gt_t = (gt[j].get("sentence") or gt[j].get("title") or "").strip()
        if not pred_t or not gt_t:
            continue
        pairs.append((pred_t, gt_t))
        matched_meta.append({"iou": round(float(iou), 2),
                             "pred": pred_t[:60],
                             "gt": gt_t[:60]})
    if not pairs:
        return {"passing_ratio": 0.0, "passing_pairs": 0,
                "total_pairs": 0, "details": []}
    sem = llm_semantic_pairs(pairs, kind="cooking step description",
                             examples=_TITLE_EXAMPLES)
    passing = 0
    details = []
    for meta, eq in zip(matched_meta, sem):
        meta["semantic_match"] = bool(eq)
        if eq:
            passing += 1
        details.append(meta)
    total = len(pairs)
    return {
        "passing_pairs": passing,
        "total_pairs": total,
        "passing_ratio": round(passing / total, 3) if total else 0.0,
        "details": details,
    }


# ---------------------------------------------------------------------- #
# recipe_quality_axes — 5 yes/no axes
# ---------------------------------------------------------------------- #
RECIPE_AXES = [
    ("dish_title_specific",
     "Does the recipe.md begin with a top-level '# <Dish Name>' header that "
     "names a real dish or cooking technique (NOT a generic placeholder like "
     "'Recipe', 'Untitled', 'Cooking Video Recipe', or just 'Steps')?"),
    ("ingredients_concrete",
     "Does the '## Ingredients' section list concrete, real-world ingredient "
     "items (e.g. 'garlic', 'rice', 'chicken') and NOT placeholder text such "
     "as 'ingredient 1', 'item A', or empty bullets?"),
    ("steps_imperative",
     "Does each numbered step have an imperative-titled action describing "
     "the actual cooking technique (e.g. 'Sauté the onions', 'Chop the "
     "garlic'), and NOT generic placeholder titles like 'Step 1: do step 1' "
     "or 'Step N'?"),
    ("clip_links_present",
     "Does every step in the '## Steps' section link to its corresponding "
     "'steps/step_NN.mp4' clip file (e.g. via '→ steps/step_01.mp4' or a "
     "Markdown link), so each step can be played back?"),
    ("no_lorem_ipsum",
     "Does the recipe.md as a whole read as fluent, coherent recipe prose, "
     "with NO lorem-ipsum, NO repeated boilerplate, and NO obvious filler "
     "placeholders?"),
]


def recipe_quality_axes(out: Path, ingr_gt: list, timeline: list) -> dict:
    md = (out / "recipe.md").read_text(encoding="utf-8") if (out / "recipe.md").exists() else ""
    timeline_summary = [
        {"index": st.get("index"),
         "title": str(st.get("title", ""))[:80],
         "clip": st.get("clip")}
        for st in timeline
    ]
    payload = (
        "GROUND_DATA (candidate vocabulary, for grounding only):\n"
        f"{json.dumps({'candidate_ingredients_gt': list(ingr_gt)[:30], 'n_timeline_steps': len(timeline), 'timeline': timeline_summary[:25]}, ensure_ascii=False)}\n\n"
        f"RECIPE.MD:\n{md[:6000]}\n"
    )
    return axes_yes_no_score(payload, RECIPE_AXES, threshold=0.70)


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps_gt", required=True)
    ap.add_argument("--ingr_gt", required=True)
    args = ap.parse_args()

    out = Path(args.output)
    try:
        timeline = json.load((out / "timeline.json").open())
    except Exception as e:
        print(json.dumps({"pass": False, "stage": "gating",
                          "errors": [f"timeline.json: {e}"]}, indent=2))
        return

    ok, errs = gating(out, timeline, len(timeline))
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating",
                          "errors": errs}, indent=2))
        return

    gt_steps = json.load(open(args.steps_gt))
    cov = temporal_coverage(timeline, gt_steps)

    md = (out / "recipe.md").read_text(encoding="utf-8")
    pred_ing = parse_recipe_ingredients(md)
    gt_ing = set(json.load(open(args.ingr_gt)))
    ing = ingredients_jaccard_strict(pred_ing, gt_ing)

    ta = title_semantic_alignment(timeline, gt_steps)

    rq = recipe_quality_axes(out, sorted(gt_ing), timeline)
    rq_ok = bool(rq.get("ok", False))

    first_clip_rel = (timeline[0].get("clip") if timeline else None) or "steps/step_01.mp4"
    sv = mllm_video_axes(out / first_clip_rel, STEP_CONTENT_AXES,
                          n_frames=4, threshold=0.70)
    sv_ok = bool(sv.get("ok", False))

    cov_ok = cov["coverage"] >= 0.55
    ing_ok = ing["jaccard"] >= 0.70  # field is now recall over GT (not Jaccard)
    ta_ok = ta["passing_ratio"] >= 0.50
    format_ok = True  # gating already passed; this is the format dim

    passed = cov_ok and ing_ok and ta_ok and rq_ok and format_ok and sv_ok

    # Reweight (sum = 1.000):
    #   capability ~0.55: temporal_coverage 0.30
    #                     + ingredients_semantic_jaccard 0.15
    #                     + title_semantic_alignment 0.10
    #   quality    ~0.30: recipe_quality_axes 0.20
    #                     + step_video_axes 0.10
    #   format     ~0.15: gating 0.15
    score = float(
        0.30 * cov["coverage"]
        + 0.15 * ing["jaccard"]
        + 0.10 * ta["passing_ratio"]
        + 0.20 * float(rq.get("score", 0.0))
        + 0.10 * float(sv.get("score", 0.0))
        + 0.15 * (1.0 if format_ok else 0.0)
    )

    result = {
        "pass": passed,
        "checks": {
            "temporal_coverage": cov_ok,
            "ingredients_semantic_jaccard": ing_ok,
            "title_semantic_alignment": ta_ok,
            "recipe_quality_axes": rq_ok,
            "step_video_axes": sv_ok,
            "format_gating": format_ok,
        },
        "temporal_coverage": {**cov, "ok": cov_ok, "threshold": 0.55},
        "ingredients_semantic_jaccard": {**ing, "ok": ing_ok, "threshold": 0.70},
        "title_semantic_alignment": {**ta, "ok": ta_ok, "threshold": 0.50},
        "recipe_quality_axes": {
            "axes": rq.get("axes", []),
            "score": rq.get("score", 0.0),
            "ok": rq_ok,
            "threshold": 0.70,
        },
        "step_video_axes": {
            "axes": sv.get("axes", []),
            "score": sv.get("score", 0.0),
            "ok": sv_ok,
            "threshold": 0.70,
            "sampled_clip": first_clip_rel,
        },
        "format_gating": {"ok": format_ok},
        "score": round(score, 3),
        "weights": {
            "temporal_coverage": 0.30,
            "ingredients_semantic_jaccard": 0.15,
            "title_semantic_alignment": 0.10,
            "recipe_quality_axes": 0.20,
            "step_video_axes": 0.10,
            "format_gating": 0.15,
        },
        "thresholds": {
            "temporal_coverage": 0.55,
            "ingredients_semantic_jaccard": 0.70,
            "title_semantic_alignment": 0.50,
            "recipe_quality_axes": 0.70,
            "step_video_axes": 0.70,
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
