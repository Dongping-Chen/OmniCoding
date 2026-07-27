"""Grader for T56a Press Conference Q-A Linker (v3).

v3 pattern:
  - Closed-vocab dims (der, qa_pair_recall, topic_acc, closure) are
    deterministic — GT-driven, no LLM.
  - summary.md is genuinely free-form prose; replaced single
    `judge_memo_grounded` continuous-score with `axes_yes_no_score` over 5
    independent axes.
  - Reweighted: deterministic dims ~0.90, axes-LLM dim 0.10.

6 dimensions (weights sum 1.0):
  der                     <= 0.25  (weight 0.18)
  qa_pair_recall          >= 0.70  (weight 0.23)
  topic_acc               >= 0.65  (weight 0.18)
  closure                 == 1.0   (weight 0.13)
  cross_doc_consistency   == 1.0   (weight 0.18)
  memo_quality_axes       >= 0.60  (weight 0.10)  axes yes/no on summary.md
"""
import argparse, csv, json, os, re, sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))

from open_vocab_judge import axes_yes_no_score  # noqa: E402
from grader_format import resolve_required, check_columns  # noqa: E402

MEMO_AXES = [
    ("speaker_counts_present",
     "Does SUMMARY.MD state the number of principals and the number of "
     "distinct reporters, both matching GROUND_DATA.n_principals / "
     "n_reporters (within +/-0)?"),
    ("topic_distribution_discussed",
     "Does SUMMARY.MD report topic distribution counts across the Q-A pairs, "
     "consistent with GROUND_DATA.topic_counts (within +/-1 per topic)?"),
    ("specific_qa_highlighted",
     "Does SUMMARY.MD highlight 1-2 specific Q-A exchanges with concrete "
     "content (not just 'a question was asked' or generic placeholders)?"),
    ("cohesive_prose",
     "Does SUMMARY.MD read as cohesive English legal-press prose — not a "
     "CSV-like row dump, not lorem ipsum, not a placeholder bullet list?"),
    ("no_fabrication",
     "Is SUMMARY.MD free of fabricated speaker IDs or contradictions with "
     "GROUND_DATA (e.g., claiming 5 principals when GROUND_DATA shows 2)?"),
]

REQUIRED = ["diarization.csv", "qa_pairs.json", "summary.md"]
DIAR_FIELDS = {"utterance_no", "speaker_role", "speaker_id"}
ROLE_VOCAB = {"principal", "reporter"}
SPK_RE = re.compile(r"^(principal|reporter)_(\d{2})$")
SUMMARY_CAP = 320  # 300 + 20 buffer
GAP_MIN, GAP_MAX = 1, 5
IOU_T = 0.5


def gating(out: Path):
    errs, resolved = resolve_required(out, REQUIRED)
    if errs: return False, errs, resolved
    miss = check_columns(resolved["diarization.csv"], DIAR_FIELDS)
    if miss: return False, [f"diarization.csv missing fields: {miss}"], resolved
    return True, [], resolved


def compute_der(pred, gt):
    """DER over utterance_no via best agent->GT label mapping (Hungarian)."""
    from scipy.optimize import linear_sum_assignment
    nos = sorted(set(pred) & set(gt))
    if not nos: return 1.0
    pl = sorted({pred[n] for n in nos}); gl = sorted({gt[n] for n in nos})
    pi = {l: i for i, l in enumerate(pl)}; gi = {l: i for i, l in enumerate(gl)}
    N = max(len(pl), len(gl))
    cost = [[0] * N for _ in range(N)]
    for n in nos:
        cost[pi[pred[n]]][gi[gt[n]]] -= 1
    r, c = linear_sum_assignment(cost)
    p2g = {pl[ri]: gl[ci] for ri, ci in zip(r.tolist(), c.tolist())
           if ri < len(pl) and ci < len(gl)}
    correct = sum(1 for n in nos if p2g.get(pred[n]) == gt[n])
    return 1.0 - correct / len(nos)


def iou_1d(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union > 0 else 0.0


def match_qa(pred, gt):
    cand = []
    for ip, p in enumerate(pred):
        for ig, g in enumerate(gt):
            qi = iou_1d(p["q_utt_range"], g["q_utt_range"])
            ai = iou_1d(p["a_utt_range"], g["a_utt_range"])
            if qi >= IOU_T and ai >= IOU_T:
                cand.append((qi + ai, ip, ig))
    cand.sort(key=lambda t: -t[0])
    up, ug, m = set(), set(), []
    for _, ip, ig in cand:
        if ip in up or ig in ug: continue
        up.add(ip); ug.add(ig); m.append((ip, ig))
    return m


def parse_qa(qa):
    out, errs = [], []
    for j, p in enumerate(qa or []):
        try:
            qr = [int(p["q_utt_range"][0]), int(p["q_utt_range"][1])]
            ar = [int(p["a_utt_range"][0]), int(p["a_utt_range"][1])]
            out.append({
                "id": p.get("id", f"qa_{j+1:03d}"),
                "q_utt_range": qr, "a_utt_range": ar,
                "q_speaker_id": (p.get("q_speaker_id") or "").strip(),
                "a_speaker_id": (p.get("a_speaker_id") or "").strip(),
                "topic": (p.get("topic") or "").strip(),
            })
        except (KeyError, ValueError, TypeError, IndexError):
            errs.append(f"malformed qa entry #{j}")
    return out, errs


def evaluate(out: Path, gt: dict, fix: Path, resolved: dict[str, Path]):
    diar = list(csv.DictReader(resolved["diarization.csv"].open(newline="")))
    qa_raw = json.load(resolved["qa_pairs.json"].open())
    summary = resolved["summary.md"].read_text(errors="ignore")
    transcript = list(csv.DictReader((fix / "transcript.csv").open(newline="")))
    vocab = set(json.load((fix / "topics_vocab.json").open())["topics"])

    pred_by_no, pred_role, diar_errs = {}, {}, []
    for r in diar:
        try: n = int(r["utterance_no"])
        except (KeyError, ValueError, TypeError):
            diar_errs.append(f"bad utterance_no: {r}"); continue
        pred_by_no[n] = (r.get("speaker_id") or "").strip()
        pred_role[n]  = (r.get("speaker_role") or "").strip()

    gt_by_no = {d["utterance_no"]: d["speaker_id"] for d in gt["diarization_gt"]}
    der_value = compute_der(pred_by_no, gt_by_no)

    pred_pairs, pp_errs = parse_qa(qa_raw)
    gt_pairs = gt["qa_pairs_gt"]
    matches = match_qa(pred_pairs, gt_pairs)
    qa_recall = len(matches) / max(1, len(gt_pairs))
    topic_acc = (sum(1 for ip, ig in matches
                     if pred_pairs[ip]["topic"] == gt_pairs[ig]["topic"])
                 / len(matches)) if matches else 0.0

    # ---- closure ----
    cl_errs = []
    transcript_nos = {int(r["utterance_no"]) for r in transcript}
    for p in pred_pairs:
        qr, ar = p["q_utt_range"], p["a_utt_range"]
        if not (qr[0] <= qr[1] and ar[0] <= ar[1]):
            cl_errs.append(f"{p['id']}: bad range order"); continue
        if any(x not in transcript_nos for x in (qr[0], qr[1], ar[0], ar[1])):
            cl_errs.append(f"{p['id']}: utterance_no out of range"); continue
        gap = ar[0] - qr[1]
        if not (GAP_MIN <= gap <= GAP_MAX):
            cl_errs.append(f"{p['id']}: gap {gap} not in [{GAP_MIN},{GAP_MAX}]")
        q_sids = {pred_by_no.get(n, "") for n in range(qr[0], qr[1] + 1)}
        a_sids = {pred_by_no.get(n, "") for n in range(ar[0], ar[1] + 1)}
        if len(q_sids) != 1 or "" in q_sids:
            cl_errs.append(f"{p['id']}: q range spans multiple/missing speaker_ids")
        elif next(iter(q_sids)) != p["q_speaker_id"]:
            cl_errs.append(f"{p['id']}: q_speaker_id mismatch with diarization")
        if len(a_sids) != 1 or "" in a_sids:
            cl_errs.append(f"{p['id']}: a range spans multiple/missing speaker_ids")
        elif next(iter(a_sids)) != p["a_speaker_id"]:
            cl_errs.append(f"{p['id']}: a_speaker_id mismatch with diarization")

    reporter_counts = Counter(sid for n, sid in pred_by_no.items()
                              if pred_role.get(n) == "reporter" and sid)
    reporter_ids = {sid for sid, c in reporter_counts.items() if c >= 2}
    q_used = {p["q_speaker_id"] for p in pred_pairs if p["q_speaker_id"]}
    miss_rep = reporter_ids - q_used
    if miss_rep:
        cl_errs.append(f"{len(miss_rep)} reporter(s) (>=2 utts) without any Q "
                       f"(e.g. {sorted(miss_rep)[:3]})")
    if not pred_pairs:
        cl_errs.append("qa_pairs.json is empty")
    closure_ok = len(cl_errs) == 0

    # ---- cross_doc_consistency ----
    cd_errs = list(diar_errs) + list(pp_errs)
    if len(diar) != len(transcript):
        cd_errs.append(f"diarization rows {len(diar)} != transcript {len(transcript)}")
    miss = transcript_nos - set(pred_by_no)
    extra = set(pred_by_no) - transcript_nos
    if miss:  cd_errs.append(f"diarization missing utterance_no: {sorted(miss)[:5]}")
    if extra: cd_errs.append(f"diarization extra utterance_no: {sorted(extra)[:5]}")

    used_sids = set()
    for n in sorted(pred_by_no):
        sid, rol = pred_by_no[n], pred_role.get(n, "")
        if rol not in ROLE_VOCAB:
            cd_errs.append(f"utt {n}: role {rol!r} out of vocab"); continue
        m = SPK_RE.match(sid)
        if not m:
            cd_errs.append(f"utt {n}: speaker_id format: {sid!r}"); continue
        if m.group(1) != rol:
            cd_errs.append(f"utt {n}: role/id prefix mismatch ({rol} vs {sid})")
        used_sids.add(sid)
    for p in pred_pairs:
        for k in ("q_speaker_id", "a_speaker_id"):
            sid = p[k]
            if sid and sid not in used_sids:
                cd_errs.append(f"{p['id']}: {k} {sid!r} not in diarization.csv")
        if p["topic"] and p["topic"] not in vocab:
            cd_errs.append(f"{p['id']}: topic {p['topic']!r} out of vocab")
    n_words = len(re.findall(r"\S+", summary))
    if n_words == 0: cd_errs.append("summary.md is empty")
    elif n_words > SUMMARY_CAP:
        cd_errs.append(f"summary.md too long: {n_words} words (cap 300+20)")
    consistency_ok = len(cd_errs) == 0

    n_principals = len({sid for n, sid in pred_by_no.items()
                        if pred_role.get(n) == "principal" and sid})
    n_reporters = len({sid for n, sid in pred_by_no.items()
                       if pred_role.get(n) == "reporter" and sid})
    topic_counts = Counter(p["topic"] for p in pred_pairs if p["topic"])
    ground_for_memo = {
        "n_principals": n_principals,
        "n_reporters": n_reporters,
        "n_qa_pairs": len(pred_pairs),
        "topic_counts": dict(topic_counts),
    }

    return {
        "der":            {"value": round(der_value, 3),
                           "n_utts": len(pred_by_no),
                           "ok": der_value <= 0.25},
        "qa_pair_recall": {"matched": len(matches),
                           "n_gt": len(gt_pairs), "n_pred": len(pred_pairs),
                           "ratio": round(qa_recall, 3),
                           "ok": qa_recall >= 0.70},
        "topic_acc":      {"matched_pairs": len(matches),
                           "topic_hits": sum(1 for ip, ig in matches
                                              if pred_pairs[ip]["topic"]
                                              == gt_pairs[ig]["topic"]),
                           "ratio": round(topic_acc, 3),
                           "ok": topic_acc >= 0.65},
        "closure":        {"errors": cl_errs[:8], "n_errors": len(cl_errs),
                           "ok": closure_ok},
        "cross_doc_consistency": {"errors": cd_errs[:8],
                                  "n_errors": len(cd_errs),
                                  "ok": consistency_ok},
        "_meta": {"ground_for_memo": ground_for_memo},
    }


def memo_quality_axes(summary_path: Path, ground: dict) -> dict:
    """v3: Replace single-LLM continuous judge with 5 yes/no axes."""
    summary = summary_path.read_text(errors="ignore")
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground, ensure_ascii=False)}\n\n"
        f"SUMMARY.MD:\n{summary[:6000]}\n"
    )
    return axes_yes_no_score(payload, MEMO_AXES, threshold=0.60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--fixtures", default="fixtures")
    a = ap.parse_args()
    out, fix = Path(a.output), Path(a.fixtures)
    ok, errs, resolved = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs}, indent=2))
        return
    res = evaluate(out, json.load(open(a.gt)), fix, resolved)
    meta = res.pop("_meta", {})
    ground_for_memo = meta.get("ground_for_memo", {})

    # v3: replace single-LLM continuous judge with axes-yes/no.
    mq = memo_quality_axes(resolved["summary.md"], ground_for_memo)
    res["memo_quality_axes"] = {
        "axes": mq.get("axes", []),
        "score": mq.get("score", 0.0),
        "ok": bool(mq.get("ok", False)),
        "threshold": 0.60,
    }

    all_ok = all(v["ok"] for v in res.values())
    # v3 weights (sum = 1.000): deterministic dims dominate (~0.90).
    w = {"der": 0.18, "qa_pair_recall": 0.23, "topic_acc": 0.18,
         "closure": 0.13, "cross_doc_consistency": 0.18, "memo_quality_axes": 0.10}
    score = round(
        w["der"] * max(0.0, min(1.0, 1.0 - res["der"]["value"] / 0.25))
        + w["qa_pair_recall"] * res["qa_pair_recall"]["ratio"]
        + w["topic_acc"] * res["topic_acc"]["ratio"]
        + w["closure"] * (1.0 if res["closure"]["ok"] else 0.0)
        + w["cross_doc_consistency"] * (1.0 if res["cross_doc_consistency"]["ok"] else 0.0)
        + w["memo_quality_axes"] * float(res["memo_quality_axes"].get("score", 0.0)),
        3)
    print(json.dumps({"pass": all_ok,
                      "checks": {k: v["ok"] for k, v in res.items()},
                      **res, "score": score, "weights": w},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
