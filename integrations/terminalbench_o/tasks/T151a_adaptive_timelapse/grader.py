#!/usr/bin/env python3
"""T151a — deterministic grader for adaptive time-lapse.

Inputs (default --output-dir = /workspace/output):
  lapse/clip_NNN.mp4       30 s ± 2 s, 640×360 @ 25 fps yuv420p
  speed_curve.csv          per (clip_id, out_second, src_t_start,
                              src_t_end, src_speed_factor); monotonic

Reference:
  reference/events.json    per-clip GT event windows

Fixtures:
  fixtures/clips.json      per-clip metadata

Score dimensions (matching task.yaml):
  event_preservation       ≥ 0.75  weight 0.35
  total_duration           ≥ 0.99  weight 0.10
  monotonicity             ≥ 0.99  weight 0.10
  compression_ratio        ≥ 4.0   weight 0.10
  quiet_compression_ratio  ≥ 5.5   weight 0.10
  speed_smoothness         ≤ 1.5   weight 0.10
  output_continuity        ≥ 0.95  weight 0.05
  format_compliance        ≥ 0.99  weight 0.10
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2  # type: ignore
import numpy as np

TASK_DIR = Path(__file__).resolve().parent

DIMS = [
    ("event_preservation", 0.35, 0.75, "ge"),
    ("total_duration", 0.10, 0.99, "ge"),
    ("monotonicity", 0.10, 0.99, "ge"),
    ("compression_ratio", 0.10, 4.0, "ge"),
    ("quiet_compression_ratio", 0.10, 5.5, "ge"),
    ("speed_smoothness", 0.10, 1.5, "le"),
    ("output_continuity", 0.05, 0.95, "ge"),
    ("format_compliance", 0.10, 0.99, "ge"),
]

EXPECTED_FPS = 25
EXPECTED_W = 640
EXPECTED_H = 360
TARGET_LAPSE_S = 30.0
LAPSE_TOL_S = 2.0
EVENT_COVERAGE_THRESHOLD_S = 1.0


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def parse_speed_curve(p: Path) -> Dict[str, List[Tuple[int, float, float, float]]]:
    """Returns clip_id -> [(out_second, src_t_start, src_t_end, src_speed_factor)]."""
    res: Dict[str, List[Tuple[int, float, float, float]]] = {}
    with p.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            cid = row.get("clip_id") or row.get("id")
            if not cid:
                continue
            try:
                osec = int(float(row["out_second"]))
                ts = float(row["src_t_start"])
                te = float(row["src_t_end"])
                sf = float(row["src_speed_factor"])
            except (KeyError, ValueError):
                continue
            res.setdefault(cid, []).append((osec, ts, te, sf))
    for cid in res:
        res[cid].sort(key=lambda x: x[0])
    return res


def ffprobe_video(p: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries",
           "stream=width,height,r_frame_rate,nb_frames,duration",
           "-of", "json", str(p)]
    try:
        out = subprocess.check_output(cmd, timeout=30)
        info = json.loads(out)["streams"][0]
    except Exception as e:
        return {"error": str(e)}
    w = int(info.get("width", 0))
    h = int(info.get("height", 0))
    num, _, den = info.get("r_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den) if float(den) else 0.0
    nb_frames = int(info.get("nb_frames", 0)) or 0
    duration = float(info.get("duration", 0.0))
    if nb_frames == 0 and fps > 0 and duration > 0:
        nb_frames = int(round(fps * duration))
    return {"width": w, "height": h, "fps": fps,
            "nb_frames": nb_frames, "duration": duration}


def consecutive_frame_stats(video_path: Path,
                            max_frames: int = 200) -> dict:
    from skimage.metrics import structural_similarity as ssim  # type: ignore

    cap = cv2.VideoCapture(str(video_path))
    prev = None
    ssims: List[float] = []
    var_vals: List[float] = []
    while len(ssims) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        var_vals.append(float(np.var(gray)))
        if prev is not None:
            try:
                v = ssim(prev, gray, data_range=255)
            except Exception:
                v = 0.0
            ssims.append(float(v))
        prev = gray
    cap.release()
    if not ssims:
        return {"median_ssim": 0.0, "frac_static": 1.0, "mean_var": 0.0}
    arr = np.array(ssims)
    return {
        "median_ssim": float(np.median(arr)),
        "frac_static": float(np.mean(arr > 0.999)),
        "mean_var": float(np.mean(var_vals)) if var_vals else 0.0,
    }


def gating(out_dir: Path, expected_ids: List[str]) -> List[str]:
    errs: List[str] = []
    if not (out_dir / "lapse").is_dir():
        errs.append("output/lapse/ missing")
    if not (out_dir / "speed_curve.csv").exists():
        errs.append("output/speed_curve.csv missing")
    missing = [c for c in expected_ids
               if not (out_dir / "lapse" / f"{c}.mp4").exists()]
    if missing:
        errs.append(
            f"{len(missing)} mp4(s) missing under output/lapse/ "
            f"(first: {missing[:3]})"
        )
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path,
                    default=Path("/workspace/output"))
    ap.add_argument("--reference-dir", type=Path,
                    default=TASK_DIR / "reference")
    ap.add_argument("--fixtures-dir", type=Path,
                    default=TASK_DIR / "fixtures")
    ap.add_argument("--report", type=Path,
                    default=TASK_DIR / "grader_report.json")
    args = ap.parse_args()

    out = args.output_dir
    ref = args.reference_dir
    fix = args.fixtures_dir

    clips = _read_json(fix / "clips.json")["clips"]
    events_data = _read_json(ref / "events.json")["clips"]
    events_by_id = {c["id"]: c["events"] for c in events_data}

    expected_ids = [c["id"] for c in clips]
    gating_errs = gating(out, expected_ids)
    if gating_errs:
        report = {"pass": False, "score": 0.0,
                  "gating_errors": gating_errs, "dims": {}}
        args.report.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 1

    speed_curves = parse_speed_curve(out / "speed_curve.csv")

    n_events_total = 0
    n_events_preserved = 0
    duration_pass: List[bool] = []
    monotonic_fracs: List[float] = []
    compression_vals: List[float] = []
    quiet_compress_vals: List[float] = []
    smoothness_vals: List[float] = []
    continuity_pass: List[bool] = []
    fmt_pass = 0
    fmt_total = len(clips)

    per_clip: List[dict] = []

    for clip in clips:
        cid = clip["id"]
        rec = {"id": cid}
        out_mp4 = out / "lapse" / f"{cid}.mp4"
        info = ffprobe_video(out_mp4)
        rec["video_info"] = info

        out_dur = float(info.get("duration", 0.0))
        fmt_ok = (
            info.get("width", 0) == EXPECTED_W
            and info.get("height", 0) == EXPECTED_H
            and abs(info.get("fps", 0.0) - EXPECTED_FPS) < 0.5
            and abs(out_dur - TARGET_LAPSE_S) <= LAPSE_TOL_S
        )
        if fmt_ok:
            fmt_pass += 1
        rec["format_ok"] = fmt_ok
        duration_pass.append(abs(out_dur - TARGET_LAPSE_S) <= LAPSE_TOL_S)

        rows = speed_curves.get(cid, [])
        rec["n_speed_rows"] = len(rows)

        # event_preservation: per-event coverage
        events = events_by_id.get(cid, [])
        rec["n_events"] = len(events)
        ev_records: List[dict] = []
        for ev in events:
            es = float(ev["t_start"])
            ee = float(ev["t_end"])
            covered = 0.0
            for (_, ts, te, _) in rows:
                ov_lo = max(es, ts)
                ov_hi = min(ee, te)
                if ov_hi > ov_lo:
                    # The agent maps 1 s of output → [ts, te] of source.
                    # Output-side coverage of this event = the fraction
                    # of source overlap inside the row, projected back
                    # onto the 1-s output slot.
                    src_span = max(te - ts, 1e-9)
                    coverage_out_seconds = (ov_hi - ov_lo) / src_span
                    covered += coverage_out_seconds
            preserved = covered >= EVENT_COVERAGE_THRESHOLD_S
            ev_records.append({
                "t_start": es, "t_end": ee,
                "out_coverage_s": round(covered, 3),
                "preserved": preserved,
            })
            n_events_total += 1
            if preserved:
                n_events_preserved += 1
        rec["events"] = ev_records

        # monotonicity
        if len(rows) >= 2:
            mono_ok = sum(
                1 for i in range(len(rows) - 1)
                if rows[i + 1][1] >= rows[i][2] - 1e-6
            )
            mono_frac = mono_ok / (len(rows) - 1)
        else:
            mono_frac = 1.0
        monotonic_fracs.append(mono_frac)
        rec["monotonicity"] = round(mono_frac, 4)

        # compression_ratio
        in_dur = float(clip["duration_seconds"])
        if out_dur > 0:
            cr = in_dur / out_dur
        else:
            cr = 0.0
        compression_vals.append(cr)
        rec["compression_ratio"] = round(cr, 3)

        # quiet_compression_ratio: rows whose [ts, te] is fully outside any event
        def in_any_event(ts: float, te: float) -> bool:
            for ev in events:
                es = float(ev["t_start"])
                ee = float(ev["t_end"])
                if min(te, ee) > max(ts, es):
                    return True
            return False

        quiet_factors: List[float] = []
        for (_, ts, te, sf) in rows:
            if not in_any_event(ts, te):
                quiet_factors.append(sf)
        if quiet_factors:
            qcr = float(np.mean(quiet_factors))
        else:
            qcr = 0.0  # all rows hit events — penalise
        quiet_compress_vals.append(qcr)
        rec["quiet_compression_ratio"] = round(qcr, 3)

        # speed_smoothness: per-clip CoV of speed factors
        if len(rows) >= 2:
            sfs = np.array([r[3] for r in rows], dtype=np.float64)
            mu = float(np.mean(sfs))
            sd = float(np.std(sfs))
            cov = sd / max(mu, 1e-6)
        else:
            cov = 100.0
        smoothness_vals.append(cov)
        rec["speed_cv"] = round(cov, 4)

        # output continuity
        if fmt_ok:
            cs = consecutive_frame_stats(out_mp4)
        else:
            cs = {"median_ssim": 0.0, "frac_static": 1.0, "mean_var": 0.0}
        rec["consecutive_stats"] = {
            "median_ssim": round(cs["median_ssim"], 4),
            "frac_static": round(cs["frac_static"], 4),
            "mean_var": round(cs["mean_var"], 2),
        }
        continuity_pass.append(
            cs["median_ssim"] > 0.05  # very low threshold: time-lapse skips fast
            and cs["frac_static"] < 0.50
            and cs["mean_var"] > 25.0
        )

        per_clip.append(rec)

    def _avg(xs: List[float], default: float = 0.0) -> float:
        return float(np.mean(xs)) if xs else default

    agg = {
        "event_preservation": (n_events_preserved / max(n_events_total, 1))
                              if n_events_total else 1.0,
        "total_duration": sum(duration_pass) / max(len(duration_pass), 1)
                          if duration_pass else 0.0,
        "monotonicity": _avg(monotonic_fracs, 1.0),
        "compression_ratio": _avg(compression_vals, 0.0),
        "quiet_compression_ratio": _avg(quiet_compress_vals, 0.0),
        "speed_smoothness": _avg(smoothness_vals, 100.0),
        "output_continuity": (sum(continuity_pass) / max(len(continuity_pass), 1))
                             if continuity_pass else 0.0,
        "format_compliance": fmt_pass / max(fmt_total, 1),
    }

    dims_report: Dict[str, dict] = {}
    score = 0.0
    for name, weight, thr, direction in DIMS:
        v = float(agg[name])
        ok = v >= thr if direction == "ge" else v <= thr
        if direction == "ge":
            cred = max(0.0, min(1.0, v / max(thr, 1e-9)))
        else:
            cred = max(0.0, min(1.0, thr / max(abs(v), 1e-9)))
        score += weight * (1.0 if ok else cred)
        dims_report[name] = {
            "value": round(v, 4),
            "threshold": thr,
            "direction": direction,
            "pass": bool(ok),
            "weight": weight,
        }

    overall_pass = all(d["pass"] for d in dims_report.values())
    report = {
        "pass": overall_pass,
        "score": round(score, 4),
        "dims": dims_report,
        "per_clip": per_clip,
        "n_events_total": n_events_total,
        "n_events_preserved": n_events_preserved,
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({"pass": overall_pass, "score": report["score"],
                      "dims": dims_report,
                      "n_events_total": n_events_total,
                      "n_events_preserved": n_events_preserved},
                     indent=2))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
