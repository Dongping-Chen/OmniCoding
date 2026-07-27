#!/usr/bin/env python3
"""Deterministic grader for T155a spatial soundfield event rescue."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf  # type: ignore

TASK_DIR = Path(__file__).resolve().parent
SR = 48000
VOCAB = {"vocals", "drums", "bass", "other"}

DIMS = {
    "event_temporal_f1": (0.20, 0.65, "ge"),
    "doa_angular_error_deg": (0.16, 20.0, "le"),
    "overlap_event_recall": (0.12, 0.55, "ge"),
    "beamformed_snr_db": (0.16, 5.0, "ge"),
    "stereo_remix_quality": (0.10, 0.90, "ge"),
    "doa_track_validity": (0.10, 0.90, "ge"),
    "cross_file_consistency": (0.08, 1.0, "eq"),
    "format_compliance": (0.08, 0.99, "ge"),
}


def read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def read_wav(p: Path):
    x, sr = sf.read(p)
    return x.astype(np.float32), sr


def read_events(p: Path) -> list[dict]:
    rows = []
    if not p.exists():
        return rows
    with p.open() as f:
        for row in csv.DictReader(f):
            try:
                if row["event_class"] not in VOCAB:
                    continue
                rows.append({
                    "event_id": row["event_id"],
                    "scene_id": row["scene_id"],
                    "t_start": float(row["t_start"]),
                    "t_end": float(row["t_end"]),
                    "event_class": row["event_class"],
                    "confidence": float(row.get("confidence", 1.0)),
                })
            except Exception:
                continue
    return rows


def read_gt(p: Path) -> list[dict]:
    rows = []
    with p.open() as f:
        for row in csv.DictReader(f):
            row["t_start"] = float(row["t_start"])
            row["t_end"] = float(row["t_end"])
            row["azimuth_deg"] = float(row["azimuth_deg"])
            row["elevation_deg"] = float(row["elevation_deg"])
            rows.append(row)
    return rows


def tiou(a: dict, b: dict) -> float:
    inter = max(0.0, min(a["t_end"], b["t_end"]) - max(a["t_start"], b["t_start"]))
    union = max(a["t_end"], b["t_end"]) - min(a["t_start"], b["t_start"])
    return inter / union if union > 0 else 0.0


def match_events(pred: list[dict], gt: list[dict]) -> list[tuple[dict, dict]]:
    pairs = []
    used = set()
    for p in sorted(pred, key=lambda r: -r.get("confidence", 1.0)):
        best = None
        best_iou = 0.0
        for i, g in enumerate(gt):
            if i in used or p["scene_id"] != g["scene_id"] or p["event_class"] != g["event_class"]:
                continue
            v = tiou(p, g)
            if v > best_iou:
                best_iou = v
                best = i
        if best is not None and best_iou >= 0.3:
            used.add(best)
            pairs.append((p, gt[best]))
    return pairs


def doa_mean(p: Path) -> Dict[str, tuple[float, float, float]]:
    res = {}
    if not p.exists():
        return res
    vals: Dict[str, list[tuple[float, float, float]]] = {}
    with p.open() as f:
        for row in csv.DictReader(f):
            try:
                eid = row["event_id"]
                az = float(row["azimuth_deg"])
                el = float(row["elevation_deg"])
                conf = float(row.get("confidence", 1.0))
                vals.setdefault(eid, []).append((az, el, conf))
            except Exception:
                continue
    for eid, rows in vals.items():
        w = np.array([max(0.0, r[2]) for r in rows])
        if w.sum() <= 0:
            w = np.ones(len(rows))
        az = float(np.average([r[0] for r in rows], weights=w))
        el = float(np.average([r[1] for r in rows], weights=w))
        conf = float(np.mean([r[2] for r in rows]))
        res[eid] = (az, el, conf)
    return res


def angular_error(az1, el1, az2, el2) -> float:
    a1, e1, a2, e2 = map(math.radians, [az1, el1, az2, el2])
    v1 = np.array([math.cos(e1) * math.cos(a1), math.cos(e1) * math.sin(a1), math.sin(e1)])
    v2 = np.array([math.cos(e2) * math.cos(a2), math.cos(e2) * math.sin(a2), math.sin(e2)])
    c = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.degrees(math.acos(c))


def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    if est.ndim > 1:
        est = est.mean(axis=1)
    if ref.ndim > 1:
        ref = ref.mean(axis=1)
    n = min(len(ref), len(est))
    if n <= 0:
        return -30.0
    ref = ref[:n]
    est = est[:n]
    return float(10 * np.log10((np.mean(ref ** 2) + 1e-9) / (np.mean((ref - est) ** 2) + 1e-9)))


def overlap_gt(gt: list[dict]) -> set[str]:
    out = set()
    for i, a in enumerate(gt):
        for j, b in enumerate(gt):
            if i >= j or a["scene_id"] != b["scene_id"]:
                continue
            if min(a["t_end"], b["t_end"]) > max(a["t_start"], b["t_start"]):
                out.add(a["event_id"])
                out.add(b["event_id"])
    return out


def remix_quality(out: Path, scenes: list[dict]) -> float:
    vals = []
    for s in scenes:
        try:
            x, sr = read_wav(out / "stereo_remix" / f"{s['id']}.wav")
            dur = len(x) / sr
            checks = [
                sr == SR,
                x.ndim == 2 and x.shape[1] == 2,
                abs(dur - float(s["duration_seconds"])) <= 0.05,
                float(np.max(np.abs(x))) <= 0.99,
                float(np.sqrt(np.mean(x ** 2))) > 0.002,
            ]
            vals.append(sum(1.0 for c in checks if c) / len(checks))
        except Exception:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


def doa_validity(p: Path, pred: list[dict]) -> float:
    means = doa_mean(p)
    vals = []
    for ev in pred:
        item = means.get(ev["event_id"])
        if item is None:
            vals.append(0.0)
        else:
            az, el, conf = item
            vals.append(1.0 if -180 <= az <= 180 and -90 <= el <= 90 and 0 <= conf <= 1 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def cross_file(out: Path, pred: list[dict], scenes: list[dict]) -> float:
    checks = [(out / "events.csv").exists(), (out / "doa_tracks.csv").exists(),
              (out / "spatial_summary.json").exists(), (out / "processing_manifest.json").exists()]
    for ev in pred:
        checks.append((out / "beamformed_events" / f"{ev['event_id']}.wav").exists())
    for s in scenes:
        checks.append((out / "stereo_remix" / f"{s['id']}.wav").exists())
    try:
        mf = read_json(out / "processing_manifest.json")
        checks.append(int(mf.get("inputs_processed", -1)) == len(scenes))
    except Exception:
        checks.append(False)
    return sum(1.0 for x in checks if x) / len(checks) if checks else 0.0


def format_score(out: Path, pred: list[dict], scenes: list[dict]) -> float:
    vals = []
    for ev in pred:
        try:
            x, sr = read_wav(out / "beamformed_events" / f"{ev['event_id']}.wav")
            vals.append(1.0 if sr == SR and (x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1)) else 0.0)
        except Exception:
            vals.append(0.0)
    for s in scenes:
        try:
            x, sr = read_wav(out / "stereo_remix" / f"{s['id']}.wav")
            vals.append(1.0 if sr == SR and x.ndim == 2 and x.shape[1] == 2 else 0.0)
        except Exception:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


def metric_ok(name: str, value: float) -> bool:
    _, thr, direction = DIMS[name]
    if direction == "eq":
        return abs(value - thr) <= 1e-9
    return value >= thr if direction == "ge" else value <= thr


def metric_credit(name: str, value: float) -> float:
    _, thr, direction = DIMS[name]
    if direction == "eq":
        return 1.0 if metric_ok(name, value) else value
    return max(0.0, min(1.0, value / max(thr, 1e-9))) if direction == "ge" else max(0.0, min(1.0, thr / max(value, 1e-9)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("/workspace/output"))
    ap.add_argument("--fixtures-dir", type=Path, default=TASK_DIR / "fixtures")
    ap.add_argument("--reference-dir", type=Path, default=TASK_DIR / "reference")
    ap.add_argument("--report", type=Path, default=TASK_DIR / "grader_report.json")
    args = ap.parse_args()

    scenes = read_json(args.fixtures_dir / "scenes.json")["scenes"]
    gt = read_gt(args.reference_dir / "events_gt.csv")
    pred = read_events(args.output_dir / "events.csv")
    pairs = match_events(pred, gt)
    precision = len(pairs) / len(pred) if pred else 0.0
    recall = len(pairs) / len(gt) if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    doa = doa_mean(args.output_dir / "doa_tracks.csv")
    angs = []
    snrs = []
    for p, g in pairs:
        if p["event_id"] in doa:
            az, el, _ = doa[p["event_id"]]
            angs.append(angular_error(az, el, float(g["azimuth_deg"]), float(g["elevation_deg"])))
        else:
            angs.append(180.0)
        try:
            ref_a, _ = read_wav(args.reference_dir / "event_audio" / f"{g['event_id']}.wav")
            est_a, _ = read_wav(args.output_dir / "beamformed_events" / f"{p['event_id']}.wav")
            snrs.append(snr_db(ref_a, est_a))
        except Exception:
            snrs.append(-30.0)
    ov = overlap_gt(gt)
    hit_overlap = {g["event_id"] for _, g in pairs if g["event_id"] in ov}
    overlap_recall = len(hit_overlap) / len(ov) if ov else 1.0
    values = {
        "event_temporal_f1": f1,
        "doa_angular_error_deg": float(np.mean(angs)) if angs else 180.0,
        "overlap_event_recall": overlap_recall,
        "beamformed_snr_db": float(np.mean(snrs)) if snrs else -30.0,
        "stereo_remix_quality": remix_quality(args.output_dir, scenes),
        "doa_track_validity": doa_validity(args.output_dir / "doa_tracks.csv", pred),
        "cross_file_consistency": cross_file(args.output_dir, pred, scenes),
        "format_compliance": format_score(args.output_dir, pred, scenes),
    }
    dims = {}
    score = 0.0
    all_ok = True
    for name, value in values.items():
        weight, threshold, direction = DIMS[name]
        ok = metric_ok(name, value)
        all_ok = all_ok and ok
        score += weight * metric_credit(name, value)
        dims[name] = {"value": round(value, 4), "threshold": threshold, "direction": direction, "ok": ok, "weight": weight}
    report = {"pass": bool(all_ok), "score": round(score, 4), "dims": dims}
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
