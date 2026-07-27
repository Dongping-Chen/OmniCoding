"""Grader for T54a Old-Time Radio Drama Cast Sheet (v3).

v3 pattern:
  - Closed-vocab dims (character_role_acc, sfx_recall, script_alignment_iou,
    diarization_der) are deterministic — GT-driven, no LLM.
  - production_notes.md is genuinely free-form prose; replaced single
    `judge_memo_grounded` continuous-score with `axes_yes_no_score` over 5
    independent axes.
  - Reweighted: deterministic dims ~0.90, axes-LLM dim 0.10.

6 dims (weights sum 1.0): diarization_der le 0.30 (0.23), character_role_acc
ge 0.65 (0.18), script_alignment_iou ge 0.55 (0.18), sfx_recall ge 0.55 (0.13),
cross_doc_consistency == 1.0 (0.18), memo_quality_axes ge 0.60 (0.10).
"""
import argparse, csv, json, os, re, sys
from pathlib import Path

try:
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    HAVE_PYANNOTE = True
except ModuleNotFoundError:
    Annotation = None  # type: ignore[assignment]
    Segment = None  # type: ignore[assignment]
    DiarizationErrorRate = None  # type: ignore[assignment]
    HAVE_PYANNOTE = False

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))

from open_vocab_judge import axes_yes_no_score  # noqa: E402
from grader_format import _norm, resolve_required, check_columns, load_csv_norm as load_csv  # noqa: E402

MEMO_AXES = [
    ("sponsor_segment_time",
     "Does PRODUCTION_NOTES.MD mention the time bracket (start/end seconds "
     "or mm:ss) of the sponsor or PSA segment, consistent with GROUND_DATA?"),
    ("narrator_voice_discussed",
     "Does PRODUCTION_NOTES.MD state whether the narrator voice is in-character "
     "or omniscient, with at least one concrete reason (voice cue, credit "
     "context, or script evidence)?"),
    ("lead_actors_named",
     "Does PRODUCTION_NOTES.MD name at least two lead actors (voice_actor_guess "
     "from cast_sheet) and give a concrete reason for each (e.g., voice cue, "
     "line count, first appearance)?"),
    ("coherent_prose",
     "Does PRODUCTION_NOTES.MD read as coherent English production prose — not "
     "a CSV-style row dump, not a bullet checklist of counts, not lorem ipsum?"),
    ("no_fabrication",
     "Is PRODUCTION_NOTES.MD free of fabricated character names or contradictions "
     "with GROUND_DATA (e.g., claiming 8 characters when GROUND_DATA shows 5)?"),
]

REQUIRED = ["cast_sheet.csv", "diarization.csv", "script_alignment.csv",
            "sfx_log.csv", "production_notes.md"]
CAST_F  = {"character_name", "voice_actor_guess", "line_count",
           "total_speak_sec", "first_appearance_sec"}
DIAR_F  = {"segment_no", "t_start", "t_end", "character", "line_text"}
ALIGN_F = {"script_line_no", "diarization_segment_no", "confidence"}
SFX_F   = {"t_at", "sfx_type", "source_clue"}
SFX_TOL = 3.0
DER_NRM = 0.6
NOTES_CAP = 250

def gating(out: Path):
    errs, resolved = resolve_required(out, REQUIRED)
    if errs:
        return False, errs, resolved
    for fn, req in [("cast_sheet.csv", CAST_F), ("diarization.csv", DIAR_F),
                    ("script_alignment.csv", ALIGN_F), ("sfx_log.csv", SFX_F)]:
        miss = check_columns(resolved[fn], req)
        if miss:
            errs.append(f"{fn} missing fields: {miss}")
    return len(errs) == 0, errs, resolved


def f(v, d=None):
    return d if v is None or v == "" else float(v)


def i(v, d=None):
    return d if v is None or v == "" else int(float(v))


def to_ann(rows, key="character"):
    a = Annotation()
    for k, r in enumerate(rows):
        s, e = f(r.get("t_start")), f(r.get("t_end"))
        c = (r.get(key) or "").strip()
        if s is None or e is None or e <= s or not c:
            continue
        a[Segment(s, e), k] = c
    return a


def simple_diarization_error_rate(ref_rows, hyp_rows, duration: float) -> float:
    """Fallback DER when pyannote is unavailable.

    The task data is single-label, non-overlapped drama dialogue.  Sampling at
    50 ms and ignoring a 250 ms collar around reference boundaries gives a
    deterministic approximation of pyannote's collar=0.25, skip_overlap=True
    behavior without adding a heavyweight runtime dependency.
    """

    collar = 0.25
    step = 0.05

    def rows_to_spans(rows):
        spans = []
        for r in rows:
            s, e = f(r.get("t_start")), f(r.get("t_end"))
            c = (r.get("character") or "").strip()
            if s is None or e is None or e <= s or not c:
                continue
            spans.append((float(s), float(e), c))
        return spans

    ref = rows_to_spans(ref_rows)
    hyp = rows_to_spans(hyp_rows)
    ref_boundaries = [x for s, e, _ in ref for x in (s, e)]

    def in_ref_collar(t):
        return any(abs(t - b) <= collar for b in ref_boundaries)

    def label_at(spans, t):
        for s, e, c in spans:
            if s <= t < e:
                return c
        return None

    ref_time = 0.0
    err_time = 0.0
    t = 0.0
    while t < duration:
        mid = min(duration, t + step / 2.0)
        if not in_ref_collar(mid):
            r = label_at(ref, mid)
            h = label_at(hyp, mid)
            if r is not None:
                ref_time += step
            if r != h and (r is not None or h is not None):
                err_time += step
        t += step
    return err_time / max(ref_time, step)


def evaluate(out: Path, gt: dict, fix: Path, resolved: dict[str, Path]):
    duration = float(gt["duration_sec"])
    diar = load_csv(resolved["diarization.csv"])
    cast = load_csv(resolved["cast_sheet.csv"])
    align = load_csv(resolved["script_alignment.csv"])
    sfx = load_csv(resolved["sfx_log.csv"])
    notes = resolved["production_notes.md"].read_text()

    # --- 1. DER ---
    if HAVE_PYANNOTE:
        hyp = to_ann(diar)
        ref = to_ann(gt["diarization_gt"])
        if len(hyp) == 0:
            der = 1.0
        else:
            der = float(DiarizationErrorRate(collar=0.25, skip_overlap=True)
                        (ref, hyp, uem=Segment(0.0, duration)))
    else:
        der = simple_diarization_error_rate(gt["diarization_gt"], diar, duration)
    der_score = max(0.0, 1.0 - der / DER_NRM)

    # --- 2. character_role_acc ---
    # Match on normalized character name AND normalized actor name, so
    # casing / whitespace / hyphenation don't masquerade as content errors.
    pred = {}
    for r in cast:
        c = (r.get("character_name") or "").strip()
        if c:
            pred[_norm(c)] = _norm(r.get("voice_actor_guess") or "")
    gt_roles = gt["actor_role_gt"]
    n_match = sum(1 for r, a in gt_roles.items()
                  if pred.get(_norm(r), "") == _norm(a))
    role_acc = n_match / max(1, len(gt_roles))

    # --- 3. script_alignment_iou ---
    diar_by = {}
    for r in diar:
        sn = i(r.get("segment_no"))
        s, e = f(r.get("t_start")), f(r.get("t_end"))
        if sn is not None and s is not None and e is not None and e > s:
            diar_by[sn] = (s, e)
    gt_diar_by = {d["segment_no"]: (d["t_start"], d["t_end"])
                   for d in gt["diarization_gt"]}
    gt_span = {a["script_line_no"]: gt_diar_by[a["segment_no"]]
                for a in gt["alignment_gt"]}
    pred_align = {}
    for r in align:
        sl = i(r.get("script_line_no"))
        sn = i(r.get("diarization_segment_no"))
        if sl is not None and sn is not None:
            pred_align[sl] = sn
    ious = []
    for sl, (gs, ge) in gt_span.items():
        psn = pred_align.get(sl)
        if psn is None or psn not in diar_by:
            ious.append(0.0)
            continue
        ps, pe = diar_by[psn]
        inter = max(0.0, min(ge, pe) - max(gs, ps))
        union = max(ge, pe) - min(gs, ps)
        ious.append(inter / union if union > 0 else 0.0)
    align_iou = sum(ious) / max(1, len(ious))

    # --- 4. sfx_recall ---
    pred_sfx = []
    for r in sfx:
        t, st = f(r.get("t_at")), (r.get("sfx_type") or "").strip()
        if t is not None and st:
            pred_sfx.append((t, _norm(st)))
    used, matched = set(), 0
    for g in gt["sfx_gt"]:
        gt_t, gt_st = float(g["t_at"]), _norm(g["sfx_type"])
        best = None
        for k, (pt, pst) in enumerate(pred_sfx):
            if k in used or pst != gt_st:
                continue
            if abs(pt - gt_t) <= SFX_TOL and (
                    best is None
                    or abs(pt - gt_t) < abs(pred_sfx[best][0] - gt_t)):
                best = k
        if best is not None:
            used.add(best)
            matched += 1
    sfx_rec = matched / max(1, len(gt["sfx_gt"]))

    # --- 5. cross_doc_consistency ---
    # Compare on normalized identifiers so "OPERATOR 1" / "Operator #1" /
    # "operator-1" are treated as the same entity. The original (display)
    # forms are still surfaced in error messages for human review.
    cd = []
    sfx_vocab = {_norm(c) for c in json.load(
        (fix / "sfx_taxonomy.json").open())["sfx_classes"]}
    known_actors = {_norm(a) for a in json.load(
        (fix / "show_metadata.json").open())["known_actors"]}
    valid_chars = {_norm(d["character"]) for d in gt["script_lines"]} \
        | {_norm("SPONSOR")}

    for r in sfx:
        st = (r.get("sfx_type") or "").strip()
        if st and _norm(st) not in sfx_vocab:
            cd.append(f"sfx_type out of vocab: {st}")
        t = f(r.get("t_at"))
        if t is not None and not (0.0 <= t <= duration):
            cd.append(f"sfx t_at out of [0,{duration}]: {t}")

    diar_chars, diar_segs = set(), set()
    for r in diar:
        s, e = f(r.get("t_start")), f(r.get("t_end"))
        c = (r.get("character") or "").strip()
        sn = i(r.get("segment_no"))
        if s is None or e is None:
            cd.append(f"diar row {sn}: missing time")
            continue
        if e <= s:
            cd.append(f"diar seg {sn}: t_end<=t_start")
        if not (0.0 <= s <= duration and 0.0 <= e <= duration):
            cd.append(f"diar seg {sn}: time out of [0,{duration}]")
        if c and _norm(c) not in valid_chars:
            cd.append(f"diar character not in script: {c}")
        if c:
            diar_chars.add(_norm(c))
        if sn is not None:
            diar_segs.add(sn)

    cast_chars = set()
    for r in cast:
        c = (r.get("character_name") or "").strip()
        a = (r.get("voice_actor_guess") or "").strip()
        if c:
            cast_chars.add(_norm(c))
            if _norm(c) not in valid_chars:
                cd.append(f"cast character not in script: {c}")
        if a and _norm(a) not in known_actors:
            cd.append(f"voice_actor not in known_actors: {a}")

    miss = cast_chars - diar_chars
    if miss:
        cd.append(
            f"cast characters absent from diarization: {sorted(miss)[:3]}")

    align_segs = {i(r.get("diarization_segment_no")) for r in align
                  if i(r.get("diarization_segment_no")) is not None}
    bad = align_segs - diar_segs
    if bad:
        cd.append(f"alignment refs unknown segment_no: {sorted(bad)[:3]}")

    nw = len(re.findall(r"\S+", notes))
    if nw > NOTES_CAP:
        cd.append(f"production_notes {nw} > {NOTES_CAP} words")

    ground_for_memo = {
        "n_characters": len(cast_chars),
        "n_diar_segments": len(diar),
        "n_sfx": len(pred_sfx),
        "duration_sec": duration,
        "cast_characters": sorted(cast_chars),
    }

    return {
        "diarization_der":      {"der": round(der, 3),
                                  "score": round(der_score, 3),
                                  "ok": der <= 0.30},
        "character_role_acc":   {"matched": n_match,
                                  "total": len(gt_roles),
                                  "value": round(role_acc, 3),
                                  "ok": role_acc >= 0.65},
        "script_alignment_iou": {"n_lines": len(ious),
                                  "value": round(align_iou, 3),
                                  "ok": align_iou >= 0.55},
        "sfx_recall":           {"matched": matched,
                                  "n_gt": len(gt["sfx_gt"]),
                                  "value": round(sfx_rec, 3),
                                  "ok": sfx_rec >= 0.55},
        "cross_doc_consistency":{"errors": cd[:8],
                                  "n_errors": len(cd),
                                  "ok": len(cd) == 0},
        "_meta": {"ground_for_memo": ground_for_memo},
    }


def memo_quality_axes(notes_path: Path, ground: dict) -> dict:
    """v3: Replace single-LLM continuous judge with 5 yes/no axes."""
    notes = notes_path.read_text()
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground, ensure_ascii=False)}\n\n"
        f"PRODUCTION_NOTES.MD:\n{notes[:6000]}\n"
    )
    return axes_yes_no_score(payload, MEMO_AXES, threshold=0.60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--fixtures", default="fixtures")
    args = ap.parse_args()
    out, fix = Path(args.output), Path(args.fixtures)
    ok, errs, resolved = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating",
                          "errors": errs}, indent=2))
        return
    gt = json.load(open(args.gt))
    res = evaluate(out, gt, fix, resolved)
    meta = res.pop("_meta", {})
    ground_for_memo = meta.get("ground_for_memo", {})

    # v3: replace single-LLM continuous judge with axes-yes/no.
    mq = memo_quality_axes(resolved["production_notes.md"], ground_for_memo)
    res["memo_quality_axes"] = {
        "axes": mq.get("axes", []),
        "score": mq.get("score", 0.0),
        "ok": bool(mq.get("ok", False)),
        "threshold": 0.60,
    }

    all_ok = all(v["ok"] for v in res.values())
    # v3 weights (sum = 1.000): deterministic dims dominate (~0.90).
    w = {"diarization_der": 0.23, "character_role_acc": 0.18,
         "script_alignment_iou": 0.18, "sfx_recall": 0.13,
         "cross_doc_consistency": 0.18, "memo_quality_axes": 0.10}

    def dv(k, d):
        if k == "diarization_der":      return d["score"]
        if k == "cross_doc_consistency": return 1.0 if d["ok"] else 0.0
        if k == "memo_quality_axes":     return float(d.get("score", 0.0))
        return d["value"]

    score = round(sum(w[k] * dv(k, res[k]) for k in w), 3)
    print(json.dumps({"pass": all_ok,
                      "checks": {k: v["ok"] for k, v in res.items()},
                      "score": score, "weights": w, **res},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
