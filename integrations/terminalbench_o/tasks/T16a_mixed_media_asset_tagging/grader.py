"""Grader for T16a Mixed Media Asset Tagging (v2 — open vocabulary).

v2 redesign:
  - tag_vocabulary is OPEN: no closed VALID_TAGS check.
  - tag_macro_f1 (strict string match) replaced by tag_semantic_macro_f1
    that uses an LLM judge to fold synonyms into the GT tags.
  - tag_grounding: VLM verifies on 6 random sampled (file, top-tag)
    image pairs that the tag is visually present.
  - flag_grounding: VLM verifies on 4 random sampled flagged image
    files that the flag reason is visible.
  - summary_quality replaced by summary_quality_axes (5 yes/no axes).
  - vocab_and_partition keeps partition + per-file 1–4 tag count +
    manifest dedup; drops the closed-vocab tag check.

Capability dims  (~0.55): type_detection_acc + ext_mismatch_recall +
                          tag_semantic_macro_f1 + flag_recall
Grounding dims   (~0.20): tag_grounding + flag_grounding
Quality dims     (~0.20): summary_quality_axes
Format / sanity  (~0.05): vocab_and_partition
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))

from open_vocab_judge import (  # noqa: E402
    llm_semantic_pairs, vlm_grounding, axes_yes_no_score,
)


SUMMARY_AXES = [
    ("counts_present",
     "Does SUMMARY.MD report type counts (image/video/audio/unknown), "
     "top tags, and flag counts that match GROUND_DATA within +/-1?"),
    ("top_tag_rationale",
     "Does SUMMARY.MD's `## Top tags` section name the top tags AND "
     "give a one-line rationale or descriptor (not just a bare list)?"),
    ("flag_overview",
     "Does SUMMARY.MD's `## Flag overview` section break down flag "
     "types with counts consistent with GROUND_DATA's flag_counts?"),
    ("recommendations_present",
     "Does SUMMARY.MD include at least one concrete remediation "
     "recommendation (e.g. dedup by sha1, fix wrong-extension files, "
     "re-shoot off-topic uploads)?"),
    ("no_fabrication",
     "Is SUMMARY.MD free of fabricated filenames or wildly invented "
     "tags / flag types not present in GROUND_DATA?"),
]


def gating(out: Path):
    errs = []
    for f in ["manifest.csv", "tag_index.json",
              "flagged.csv", "summary.md"]:
        if not (out / f).exists():
            errs.append(f"missing {f}")
    return len(errs) == 0, errs


def _parse_tags(value):
    """Accept tags as JSON list, semicolon-list, or comma-list."""
    if isinstance(value, list):
        return [str(t).strip() for t in value if str(t).strip()]
    s = (value or "").strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(t).strip() for t in arr]
        except Exception:
            pass
    if ";" in s:
        return [t.strip() for t in s.split(";") if t.strip()]
    return [t.strip() for t in s.split(",") if t.strip()]


def _load_tag_index(p: Path) -> dict[str, list]:
    """Return {filename: [tags]} preserving order. Tolerates:
       1) {"items": [{"filename":..., "tags":[...]}, ...]}
       2) [{"filename":..., "tags":[...]}, ...]
       3) {tag: [filename, ...], ...}  (inverted index — convert)
    """
    try:
        raw = json.load(p.open())
    except Exception:
        return {}
    if isinstance(raw, dict) and "items" in raw and isinstance(raw["items"], list):
        records = raw["items"]
        return {r.get("filename"): list(r.get("tags") or [])
                for r in records if isinstance(r, dict)}
    if isinstance(raw, list):
        return {r.get("filename"): list(r.get("tags") or [])
                for r in raw if isinstance(r, dict)}
    if isinstance(raw, dict):
        out = {}
        for tag, files in raw.items():
            if not isinstance(files, list):
                continue
            for fn in files:
                out.setdefault(fn, []).append(tag)
        return out
    return {}


def _is_mismatch(m: dict) -> bool:
    if "ext_match" in m and m.get("ext_match") not in (None, ""):
        v = str(m["ext_match"]).strip().lower()
        return v in ("0", "false", "no")
    if "ext_mismatch" in m and m.get("ext_mismatch") not in (None, ""):
        v = str(m["ext_mismatch"]).strip().lower()
        return v in ("1", "true", "yes")
    return False


def _norm_tag(t: str) -> str:
    return str(t or "").strip().lower().replace("-", "_").replace(" ", "_")


def _strict_tag_f1(pred_by_file: dict[str, list],
                   gt_by_file: dict[str, list]) -> tuple[float, dict]:
    """Per-GT-tag macro-F1 with strict normalised string match.

    v3: closed-vocab tagging is fully GT-defined; no LLM needed. Tags
    are case- and separator-normalised (`B-Roll` == `b_roll`)."""
    gt_tags = set()
    for tags in gt_by_file.values():
        for t in (tags or []):
            t2 = _norm_tag(t)
            if t2:
                gt_tags.add(t2)
    f1s = []
    per_tag = []
    all_files = set(list(gt_by_file.keys()) + list(pred_by_file.keys()))
    for g in sorted(gt_tags):
        tp = fp = fn = 0
        for fname in all_files:
            gset = {_norm_tag(t) for t in (gt_by_file.get(fname) or [])}
            pset = {_norm_tag(t) for t in (pred_by_file.get(fname) or [])}
            in_gt = g in gset
            in_pred = g in pset
            if in_gt and in_pred:
                tp += 1
            elif in_pred and not in_gt:
                fp += 1
            elif in_gt and not in_pred:
                fn += 1
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-6, prec + rec)
        f1s.append(f1)
        per_tag.append({"tag": g, "tp": tp, "fp": fp, "fn": fn,
                         "f1": round(f1, 3)})
    macro = sum(f1s) / max(1, len(f1s))
    return macro, {
        "per_tag": per_tag,
        "gt_tag_vocab_size": len(gt_tags),
    }


def evaluate(out: Path, gt):
    manifest = list(csv.DictReader((out / "manifest.csv").open()))
    tags_by_name = _load_tag_index(out / "tag_index.json")
    flagged = list(csv.DictReader((out / "flagged.csv").open()))

    uploads_gt = gt["uploads"]
    n = len(uploads_gt)

    def _fname(m):
        return (m.get("filename") or m.get("file_name")
                or m.get("file") or m.get("name"))
    by_name = {_fname(m): m for m in manifest}
    if not tags_by_name:
        tags_by_name = {fn: _parse_tags(m.get("tags"))
                        for fn, m in by_name.items()}

    # ---- type_detection_acc ----
    correct = 0
    for fname, gv in uploads_gt.items():
        m = by_name.get(fname, {})
        if (m.get("detected_type") or "").strip() == gv["actual_type"]:
            correct += 1
    type_acc = correct / max(1, n)

    # ---- ext_mismatch_recall ----
    gt_mismatches = {fn for fn, gv in uploads_gt.items() if gv["ext_match"] == 0}
    pred_mismatches = {_fname(m) for m in manifest if _is_mismatch(m)}
    em_match = len(gt_mismatches & pred_mismatches)
    em_recall = em_match / max(1, len(gt_mismatches))

    # ---- tag_macro_f1 (strict GT match, v3) ----
    gt_tag_by_file = {fn: list(gv.get("tags") or [])
                       for fn, gv in uploads_gt.items()}
    macro_f1, sem_dbg = _strict_tag_f1(tags_by_name, gt_tag_by_file)

    # ---- flag_recall ----
    gt_flagged = {fn for fn, gv in uploads_gt.items() if gv["flag"]}
    pred_flagged = {r.get("filename") for r in flagged}
    fl_match = len(gt_flagged & pred_flagged)
    fl_recall = fl_match / max(1, len(gt_flagged))

    # ---- vocab_and_partition (loosened — no closed VALID_TAGS) ----
    vp_ok = True; vp_err = []
    for fn, tags_list in tags_by_name.items():
        if fn is None:
            continue
        n_tags = len(tags_list)
        if not (1 <= n_tags <= 4):
            vp_ok = False; vp_err.append(f"{fn} has {n_tags} tags")
    seen = [_fname(m) for m in manifest]
    if len(seen) != len(set(seen)):
        vp_ok = False; vp_err.append("dup filenames in manifest")
    if set(seen) != set(uploads_gt.keys()):
        vp_ok = False; vp_err.append(
            f"manifest covers {len(seen)}, want {len(uploads_gt)}")
    media = {_fname(m) for m in manifest
              if (m.get("detected_type") or "") in ("image", "video", "audio")}
    tag_names = set(tags_by_name.keys()) - {None}
    extras = tag_names - media
    if extras:
        vp_ok = False
        vp_err.append(f"tag_index has non-media files: {sorted(extras)[:3]}")
    for r in flagged:
        flag = (r.get("flag_type") or r.get("flag_reason")
                or r.get("reason") or "").strip()
        if not flag:
            vp_ok = False; vp_err.append(f"empty flag for {r.get('filename')}")

    return {
        "type_detection_acc": {
            "correct": correct, "n": n,
            "ratio": round(type_acc, 3),
            "ok": type_acc >= 0.95,
        },
        "ext_mismatch_recall": {
            "matched": em_match, "n_gt": len(gt_mismatches),
            "ratio": round(em_recall, 3),
            "ok": em_recall >= 0.85,
        },
        "tag_macro_f1": {
            "macro_f1": round(macro_f1, 3),
            "per_tag": sem_dbg["per_tag"],
            "gt_tag_vocab_size": sem_dbg["gt_tag_vocab_size"],
            "ok": macro_f1 >= 0.55,
        },
        "flag_recall": {
            "matched": fl_match, "n_gt": len(gt_flagged),
            "ratio": round(fl_recall, 3),
            "ok": fl_recall >= 0.80,
        },
        "vocab_and_partition": {
            "errors": vp_err[:5],
            "ok": vp_ok,
        },
        "_meta": {
            "by_name": by_name,
            "tags_by_name": tags_by_name,
            "flagged_rows": flagged,
        },
    }


def _is_image_file(fname: str, manifest_row: dict | None,
                   gt_entry: dict | None) -> bool:
    """Used to gate VLM grounding to only image uploads (the VLM judge
    handles still images; videos / audio aren't ingestible here)."""
    if manifest_row:
        dt = (manifest_row.get("detected_type") or "").strip().lower()
        if dt == "image":
            return True
        if dt in ("video", "audio", "unknown"):
            return False
    if gt_entry and gt_entry.get("actual_type") == "image":
        return True
    ext = Path(fname).suffix.lower()
    return ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--uploads",
                    default=str(Path(__file__).resolve().parent
                                / "fixtures" / "uploads"),
                    help="Directory with the original upload files for "
                         "VLM grounding.")
    args = ap.parse_args()
    out = Path(args.output)
    ok, errs = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs},
                         indent=2))
        return
    gt = json.load(open(args.gt))
    res = evaluate(out, gt)
    meta = res.pop("_meta", {})
    by_name = meta.get("by_name", {})
    tags_by_name = meta.get("tags_by_name", {})
    flagged_rows = meta.get("flagged_rows", [])
    uploads_dir = Path(args.uploads)

    # NOTE (v3): tag_grounding + flag_grounding removed. The tag and
    # flag vocabularies are closed; GT match is authoritative. VLM
    # verification was redundant + introduced cross-model bias and
    # noise. summary_quality_axes is the only judge-LLM dim retained
    # because summary.md is genuinely free-form prose.

    # ---- summary_quality_axes ----
    try:
        type_counts = Counter((m.get("detected_type") or "").strip()
                                for m in by_name.values())
        tag_freq = Counter()
        for tags in tags_by_name.values():
            for t in (tags or []):
                tag_freq[t] += 1
        top_tags = tag_freq.most_common(5)
        flag_counts = Counter(
            (r.get("flag_type") or r.get("flag_reason")
             or r.get("reason") or "").strip()
            for r in flagged_rows)
        ground_for_memo = {
            "type_counts": dict(type_counts),
            "top_tags": top_tags,
            "flag_counts": dict(flag_counts),
            "n_files": len(by_name),
            "n_tagged": sum(1 for v in tags_by_name.values() if v),
            "n_flagged": len(flagged_rows),
        }
    except Exception:
        ground_for_memo = {}
    summary_text = (out / "summary.md").read_text()
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground_for_memo, ensure_ascii=False)}\n\n"
        f"SUMMARY.MD:\n{summary_text[:6000]}\n"
    )
    sq = axes_yes_no_score(payload, SUMMARY_AXES, threshold=0.70)
    res["summary_quality_axes"] = {
        "axes": sq.get("axes", []),
        "score": sq.get("score", 0.0),
        "ok": bool(sq.get("ok", False)),
        "threshold": 0.70,
    }

    all_ok = all(v["ok"] for v in res.values())
    # v3: deterministic dims dominate. Only summary uses LLM judge.
    score = (
        0.20 * res["type_detection_acc"]["ratio"]
        + 0.15 * res["ext_mismatch_recall"]["ratio"]
        + 0.25 * res["tag_macro_f1"]["macro_f1"]
        + 0.15 * res["flag_recall"]["ratio"]
        + 0.05 * (1 if res["vocab_and_partition"]["ok"] else 0)
        + 0.20 * res["summary_quality_axes"]["score"]
    )
    print(json.dumps({
        "pass": all_ok,
        "checks": {k: v["ok"] for k, v in res.items()},
        **res,
        "score_so_far": round(score, 3),
        "weights": {
            "type_detection_acc": 0.20,
            "ext_mismatch_recall": 0.15,
            "tag_macro_f1": 0.25,
            "flag_recall": 0.15,
            "vocab_and_partition": 0.05,
            "summary_quality_axes": 0.20,
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
