"""Grader for T03a Multi-Cam Auto Edit (v3 deterministic).

v3 redesign:
  - Removed expensive VLM checks: `edit_visual` (frame-by-frame VLM) and
    `cam_visual_consistency` (VLM per shot).
  - Replaced single-LLM `summary_quality` with 4-axis yes/no (`summary_axes`).
  - Kept lightweight checks: ffprobe duration, deterministic speaker alignment.
  - Reweighted: deterministic dims ~0.85, axes ~0.15.
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))
from open_vocab_judge import axes_yes_no_score


def gating(out: Path):
    errs = []
    if not (out / "final.mp4").exists():
        errs.append("missing final.mp4")
    if not (out / "shot_log.json").exists():
        errs.append("missing shot_log.json")
    if not (out / "switch_report.csv").exists():
        errs.append("missing switch_report.csv")
    if not (out / "summary.md").exists():
        errs.append("missing summary.md")
    if not (out / "speaker_map.json").exists():
        errs.append("missing speaker_map.json")
    return len(errs) == 0, errs


def probe_duration(path: Path) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of",
             "default=noprint_wrappers=1:nokey=1", str(path)],
            text=True)
        return float(out.strip())
    except Exception:
        return 0.0


def evaluate(out: Path, gt, src: Path):
    shot_log = json.load((out / "shot_log.json").open())
    shots = shot_log.get("shots", [])
    rows = list(csv.DictReader((out / "switch_report.csv").open()))
    summary = (out / "summary.md").read_text()
    try:
        agent_map = json.load((out / "speaker_map.json").open())
    except Exception:
        agent_map = {}

    src_dur = probe_duration(src)
    out_dur = probe_duration(out / "final.mp4")
    duration_ok = abs(src_dur - out_dur) <= 0.2
    gt_turns = gt["turns"]

    # ---- speaker_map_correct (agent's S1<->camHk mapping vs GT) ----
    # The 2-speaker / 2-camera permutation is binary: blind-guess hits
    # 50%. Two reinforcing requirements raise the bar:
    #   (a) Mapping must be specified for EVERY GT speaker (S1 and S2,
    #       and any future Sk), not just one. Missing speakers fail.
    #   (b) The agent's mapping must be SELF-CONSISTENT with its own
    #       shot_log — for each speaker `Sk` claimed to map to
    #       `camHj`, that camera must dominate during Sk's GT turns
    #       in the agent's OWN shot_log (≥ 60% of dwell time, with
    #       at least 3 distinct GT turns of that speaker observed).
    # Random guessing would have to align both halves of (a)+(b),
    # which is well below 50%.
    gt_map = gt.get("cam_mapping", {"S1": "camH1", "S2": "camH2"})
    gt_speakers = sorted(gt_map.keys())
    map_pairs_match = all(
        agent_map.get(spk) == gt_map.get(spk) for spk in gt_speakers
    )
    map_pairs_match = map_pairs_match and len(agent_map) >= len(gt_speakers)

    # Self-consistency: aggregate per-speaker camera dwell from
    # agent's shot_log against GT turns.
    SELF_CONS_MIN_DWELL = 0.60
    SELF_CONS_MIN_TURNS = 3
    spk_dwell = {spk: {} for spk in gt_speakers}
    spk_turn_count = {spk: 0 for spk in gt_speakers}
    for tr in gt_turns:
        spk = tr["speaker"]
        if spk not in spk_dwell:
            continue
        spk_turn_count[spk] += 1
        # Sample at 0.1 s grid inside this turn.
        tt = float(tr["start"])
        while tt < float(tr["end"]):
            for s in shots:
                if s.get("start", -1) <= tt <= s.get("end", -1):
                    cam = s.get("camera")
                    spk_dwell[spk][cam] = spk_dwell[spk].get(cam, 0.0) + 0.1
                    break
            tt += 0.1

    self_consistent = True
    self_cons_detail = {}
    for spk in gt_speakers:
        claimed = agent_map.get(spk)
        dwell = spk_dwell[spk]
        total = sum(dwell.values())
        share = (dwell.get(claimed, 0.0) / total) if total > 0 else 0.0
        self_cons_detail[spk] = {
            "claimed_cam": claimed,
            "dwell_share_in_claimed": round(share, 3),
            "n_gt_turns": spk_turn_count[spk],
        }
        if (spk_turn_count[spk] < SELF_CONS_MIN_TURNS
                or share < SELF_CONS_MIN_DWELL
                or claimed not in {"camH1", "camH2"}):
            self_consistent = False
    # Also require the two close-up cameras be distinct across S1/S2
    claimed_set = {agent_map.get(spk) for spk in gt_speakers}
    if len(claimed_set) < len(gt_speakers):
        self_consistent = False

    map_match = map_pairs_match and self_consistent

    # ---- speaker_alignment (active cam matches GT speaker) ----
    sa_total = 0
    sa_ok = 0
    # Sample at 0.5 s grid; check active speaker (GT) vs active camera
    # in shots.
    def cam_at(t):
        for s in shots:
            if s.get("start", -1) <= t <= s.get("end", -1):
                return s.get("camera")
        return None

    def expected_cam(spk):
        # Use GT cam_mapping so the alignment test is grounded in the
        # *correct* speaker -> close-up assignment, regardless of
        # whether the agent picked the right mapping.
        if spk in ("BOTH", "SILENCE"):
            return "camA"
        return gt_map.get(spk)

    t = 0.0
    while t < src_dur:
        spk = None
        for tr in gt_turns:
            if tr["start"] <= t < tr["end"]:
                spk = tr["speaker"]
                break
        if spk is None:
            t += 0.5
            continue
        sa_total += 1
        cam = cam_at(t)
        if cam == expected_cam(spk):
            sa_ok += 1
        t += 0.5
    sa_ratio = sa_ok / max(1, sa_total)

    # ---- shot_length_min (every shot ≥ 1.0 s) ----
    sl_ok = all(
        (s.get("end", 0) - s.get("start", 0)) >= 1.0 - 1e-3
        for s in shots) and len(shots) > 0

    # ---- rule_compliance for camA ----
    rc_total = 2
    rc_pass = 0
    long_camA = sum(
        1 for s in shots
        if s.get("camera") == "camA"
        and (s.get("end", 0) - s.get("start", 0)) > 5.0)
    if long_camA <= 1:
        rc_pass += 1
    # camA covers all BOTH+SILENCE > 1.5 s GT turns at midpoint
    boths = [tr for tr in gt_turns
             if tr["speaker"] in ("BOTH", "SILENCE")
             and (tr["end"] - tr["start"]) > 1.5]
    if boths:
        hits = sum(1 for tr in boths
                    if cam_at((tr["start"] + tr["end"]) / 2) == "camA")
        if hits / len(boths) >= 0.7:
            rc_pass += 1
    else:
        rc_pass += 1
    rc_ratio = rc_pass / rc_total

    # ---- cross_doc_consistency ----
    cdc_ok = True; cdc_err = []
    # shot_log switches count == csv row count
    if shot_log.get("switches") is not None:
        if len(rows) != shot_log.get("switches"):
            cdc_ok = False
            cdc_err.append(
                f"switches {shot_log.get('switches')} vs csv {len(rows)}")
    # csv first row's prev_camera + last new_camera makes sense
    if rows:
        cams_in_csv = {r.get("new_camera") for r in rows} | \
                      {r.get("prev_camera") for r in rows}
        if not cams_in_csv.issubset({"camA", "camH1", "camH2"}):
            cdc_ok = False
            cdc_err.append(f"csv cams {cams_in_csv}")
    # summary mentions cadence
    if "switch" not in summary.lower() and "cadence" not in summary.lower():
        cdc_ok = False; cdc_err.append("summary missing cadence")
    if "wide" not in summary.lower() and "cama" not in summary.lower():
        cdc_ok = False; cdc_err.append("summary missing wide usage")

    return {
        "duration_match": {
            "src": round(src_dur, 2),
            "out": round(out_dur, 2),
            "ok": duration_ok,
        },
        "speaker_map_correct": {
            "agent_map": agent_map,
            "gt_map": gt_map,
            "pairs_match_gt": map_pairs_match,
            "self_consistent_with_shot_log": self_consistent,
            "self_consistency_detail": self_cons_detail,
            "self_cons_thresholds": {
                "min_dwell_share": SELF_CONS_MIN_DWELL,
                "min_turns_per_speaker": SELF_CONS_MIN_TURNS,
            },
            "ok": map_match,
        },
        "speaker_alignment": {
            "matched": sa_ok, "total": sa_total,
            "ratio": round(sa_ratio, 3),
            "ok": sa_ratio >= 0.85,
        },
        "shot_length_min": {
            "ok": sl_ok,
        },
        "rule_compliance": {
            "ratio": round(rc_ratio, 3),
            "ok": rc_ratio >= 1.0,
        },
        "cross_doc_consistency": {
            "errors": cdc_err[:5],
            "ok": cdc_ok,
        },
    }


SUMMARY_AXES = [
    ("switching_cadence_quantified",
     "Does the memo include a `## Switching cadence` section that quantifies "
     "cadence (e.g. total switches, dominant camera, reasoning) — not just a "
     "heading, but actual numbers from GROUND_DATA?"),
    ("wide_usage_explained",
     "Does the memo include a `## Wide usage` section explaining when/why "
     "camA was used (BOTH speaking, silence, B-roll), grounded in the "
     "GROUND_DATA shot distribution rather than placeholder text?"),
    ("fluent_prose",
     "Does the memo read as fluent prose ≤ 250 words, not lorem ipsum / "
     "placeholder / pure heading dump?"),
    ("self_consistent",
     "Are the cadence numbers / camera names in the memo consistent with the "
     "agent's own shot_log (per GROUND_DATA)?"),
]


def summary_axes_score(out: Path) -> dict:
    try:
        shot_log = json.load((out / "shot_log.json").open())
        shots = shot_log.get("shots", [])
    except Exception:
        shots = []
    cam_counts = {}
    for s in shots:
        cam_counts[s.get("camera", "?")] = cam_counts.get(s.get("camera", "?"), 0) + 1
    ground = {
        "n_shots": len(shots),
        "cam_counts": cam_counts,
        "switches": shot_log.get("switches") if shots else None,
    }
    summary = (out / "summary.md").read_text()
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground, ensure_ascii=False)}\n\n"
        f"SUMMARY.MD:\n{summary[:6000]}\n"
    )
    return axes_yes_no_score(payload, SUMMARY_AXES, threshold=0.70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--src", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    ok, errs = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs},
                         indent=2))
        return
    gt = json.load(open(args.gt))
    res = evaluate(out, gt, Path(args.src))

    # NOTE (v3): Removed expensive VLM checks:
    # - edit_visual (frame-by-frame VLM on final.mp4)
    # - cam_visual_consistency (VLM per shot)
    # These are redundant with deterministic speaker_alignment and add
    # cross-model bias. Duration check via ffprobe is sufficient for
    # basic sanity.

    sq = summary_axes_score(out)
    res["summary_axes"] = {
        "axes": sq.get("axes", []),
        "score": sq.get("score", 0.0),
        "ok": bool(sq.get("ok", False)),
        "threshold": 0.70,
    }

    all_ok = all(v["ok"] for v in res.values())
    # v3: deterministic dims dominate (~0.85), only summary uses LLM (~0.15).
    score = (
        0.10 * (1 if res["duration_match"]["ok"] else 0)
        + 0.15 * (1 if res["speaker_map_correct"]["ok"] else 0)
        + 0.25 * res["speaker_alignment"]["ratio"]
        + 0.10 * (1 if res["shot_length_min"]["ok"] else 0)
        + 0.15 * res["rule_compliance"]["ratio"]
        + 0.10 * (1 if res["cross_doc_consistency"]["ok"] else 0)
        + 0.15 * res["summary_axes"]["score"]
    )
    print(json.dumps({
        "pass": all_ok,
        "checks": {k: v["ok"] for k, v in res.items()},
        **res,
        "score_so_far": round(score, 3),
        "weights": {
            "duration_match": 0.10,
            "speaker_map_correct": 0.15,
            "speaker_alignment": 0.25,
            "shot_length_min": 0.10,
            "rule_compliance": 0.15,
            "cross_doc_consistency": 0.10,
            "summary_axes": 0.15,
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
