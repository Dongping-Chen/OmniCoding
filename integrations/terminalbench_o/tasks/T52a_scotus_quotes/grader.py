"""Grader for T52a SCOTUS Oral Argument Quote Pack (v3).

v3 pattern:
  - Closed-vocab justice / advocate / cited_case fields are CLOSED_WITH_DECOYS.
    The cited_cases_all vocab already has 8 decoy cases that are NEVER
    cited in the Counterman GT pairs (only Elonis + Watts are real);
    cross_doc_consistency rejects out-of-vocab cases.
  - quote_audio_grounding (ASR vs agent quote) stays — it's a
    deterministic per-clip WER metric, not a single-LLM judge.
  - summary.md is genuinely free-form prose; replaced single
    `judge_memo_grounded` continuous-score with `axes_yes_no_score`
    over 5 yes/no axes.
  - Reweighted: deterministic dims ~0.85, axes-LLM 0.15.
"""
import argparse, csv, json, os, random, re, shutil, subprocess, sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))

from open_vocab_judge import axes_yes_no_score  # noqa: E402

REQUIRED_FILES = ["quote_pack.json", "summary.md"]
ID_RE = re.compile(r"^qa_\d{3}$")
QA_KEYS = ["id", "justice", "question_quote", "q_utt_no",
           "advocate", "answer_quote", "a_utt_no",
           "cited_cases", "q_clip", "a_clip"]


def gating(out: Path):
    errs = []
    for f in REQUIRED_FILES:
        p = out / f
        if not p.exists():
            errs.append(f"missing: {f}")
        elif p.stat().st_size == 0:
            errs.append(f"empty: {f}")
    if errs:
        return False, errs, None
    try:
        pack = json.loads((out / "quote_pack.json").read_text())
    except Exception as e:
        return False, [f"quote_pack.json not parseable JSON: {e}"], None
    if not isinstance(pack, list):
        return False, ["quote_pack.json is not a JSON array"], None
    if len(pack) < 12:
        return False, [f"quote_pack has only {len(pack)} entries (need >=12)"], None
    for i, p in enumerate(pack):
        if not isinstance(p, dict):
            return False, [f"entry {i} is not an object"], None
        miss = [k for k in QA_KEYS if k not in p]
        if miss:
            return False, [f"entry {i} missing keys: {miss}"], None
    return True, [], pack


def ffprobe_duration(path: Path):
    if not path.exists():
        return None
    if shutil.which("ffprobe") is None:
        return None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip())
    except Exception:
        return None


def load_transcript(fixtures: Path):
    rows = list(csv.DictReader((fixtures / "transcript.csv").open(newline="")))
    by_no = {}
    for r in rows:
        try:
            n = int(r["utterance_no"])
            by_no[n] = {
                "utterance_no": n,
                "t_start": float(r["t_start"]),
                "t_end":   float(r["t_end"]),
                "speaker_role": r["speaker_role"],
                "speaker_name": r["speaker_name"],
                "text": r["text"],
            }
        except (KeyError, ValueError):
            continue
    return by_no


def cross_doc_check(pack, gt, fixtures, out: Path, transcript):
    errs = []
    voc_just = set(gt["vocab_justices"])
    voc_adv  = set(gt["vocab_advocates"])
    voc_cite = set(gt["cited_cases_all"])
    n_utt = gt["n_utterances"]

    seen_ids = set()
    for i, p in enumerate(pack):
        pid = (p.get("id") or "").strip()
        if not ID_RE.match(pid):
            errs.append(f"entry {i} id '{pid}' not matching qa_\\d{{3}}")
        elif pid in seen_ids:
            errs.append(f"duplicate id {pid}")
        seen_ids.add(pid)

        j = (p.get("justice") or "").strip()
        if j not in voc_just:
            errs.append(f"{pid} justice '{j}' not in vocab")
        a = (p.get("advocate") or "").strip()
        if a not in voc_adv:
            errs.append(f"{pid} advocate '{a}' not in vocab")

        for key in ("q_utt_no", "a_utt_no"):
            try:
                v = int(p.get(key))
            except (TypeError, ValueError):
                errs.append(f"{pid} {key} not an integer")
                continue
            if not (0 <= v < n_utt):
                errs.append(f"{pid} {key}={v} out of [0,{n_utt})")
            elif v not in transcript:
                errs.append(f"{pid} {key}={v} missing from transcript.csv")

        cc = p.get("cited_cases")
        if not isinstance(cc, list):
            errs.append(f"{pid} cited_cases not a list")
        else:
            for c in cc:
                if c not in voc_cite:
                    errs.append(f"{pid} cited_case '{c}' not in vocab")

        for ck in ("q_clip", "a_clip"):
            cp = (p.get(ck) or "").strip()
            if not cp:
                errs.append(f"{pid} {ck} empty")
                continue
            if not (out / cp).exists():
                errs.append(f"{pid} {ck} file missing: {cp}")

        # role must match: q_utt should be justice, a_utt advocate
        try:
            qn = int(p.get("q_utt_no"))
            an = int(p.get("a_utt_no"))
            if qn in transcript and transcript[qn]["speaker_role"] != "scotus_justice":
                errs.append(f"{pid} q_utt_no={qn} speaker_role is not scotus_justice")
            if an in transcript and transcript[an]["speaker_role"] != "advocate":
                errs.append(f"{pid} a_utt_no={an} speaker_role is not advocate")
        except (TypeError, ValueError):
            pass

    summary = (out / "summary.md").read_text(errors="ignore")
    n_words = len(re.findall(r"\S+", summary))
    if n_words > 300:
        errs.append(f"summary.md has {n_words} words (> 300)")
    return errs


def evaluate(out: Path, gt, fixtures: Path, pack):
    transcript = load_transcript(fixtures)

    # 1) justice_q_recall
    gt_pairs = gt["gt_pairs"]
    gt_qnos = {int(p["q_utt_no"]) for p in gt_pairs}
    agent_qnos = set()
    for p in pack:
        try: agent_qnos.add(int(p["q_utt_no"]))
        except (TypeError, ValueError): pass
    hit_q = gt_qnos & agent_qnos
    q_recall = len(hit_q) / max(1, len(gt_qnos))

    # 2) qa_pairing_acc — over agent pairs whose q_utt_no matches a GT q
    gt_by_q = {int(p["q_utt_no"]): int(p["a_utt_no"]) for p in gt_pairs}
    matched = 0
    correct = 0
    for p in pack:
        try:
            qn = int(p["q_utt_no"])
            an = int(p["a_utt_no"])
        except (TypeError, ValueError):
            continue
        if qn in gt_by_q:
            matched += 1
            if abs(an - gt_by_q[qn]) <= 2:
                correct += 1
    pairing_acc = correct / max(1, matched) if matched > 0 else 0.0

    # 3) cited_case_f1 — agent's cited_cases UNION (deduped) vs cases
    # actually mentioned in the GT pairs. The full vocab has distractors,
    # so emitting every closed-vocab case should be penalized by precision.
    agent_cites = set()
    for p in pack:
        cc = p.get("cited_cases")
        if isinstance(cc, list):
            for c in cc:
                if isinstance(c, str):
                    agent_cites.add(c)
    gt_cites = set()
    for p in gt_pairs:
        for c in p.get("cited_cases") or []:
            gt_cites.add(c)
    cite_hits = len(agent_cites & gt_cites)
    cite_precision = cite_hits / len(agent_cites) if agent_cites else (1.0 if not gt_cites else 0.0)
    cite_recall = cite_hits / len(gt_cites) if gt_cites else 1.0
    cite_f1 = (2 * cite_precision * cite_recall / (cite_precision + cite_recall)
               if (cite_precision + cite_recall) else 0.0)

    # 4) clip_duration_match — per-clip
    n_clips, n_ok = 0, 0
    bad = []
    for p in pack:
        for ck, utt_key in (("q_clip", "q_utt_no"), ("a_clip", "a_utt_no")):
            n_clips += 1
            cp = (p.get(ck) or "").strip()
            if not cp:
                bad.append(f"{p.get('id')}/{ck}: empty")
                continue
            full = out / cp
            dur = ffprobe_duration(full)
            if dur is None:
                bad.append(f"{p.get('id')}/{ck}: ffprobe failed or file missing")
                continue
            try:
                un = int(p.get(utt_key))
            except (TypeError, ValueError):
                bad.append(f"{p.get('id')}/{ck}: bad utt_no")
                continue
            if un not in transcript:
                bad.append(f"{p.get('id')}/{ck}: utt {un} not in transcript")
                continue
            utt_dur = transcript[un]["t_end"] - transcript[un]["t_start"]
            if dur > 30.0 + 0.05:
                bad.append(f"{p.get('id')}/{ck}: dur={dur:.2f}s > 30s")
                continue
            if abs(dur - utt_dur) > 5.0 + 0.05:
                bad.append(f"{p.get('id')}/{ck}: dur={dur:.2f} vs utt_dur={utt_dur:.2f}")
                continue
            n_ok += 1
    clip_ratio = n_ok / max(1, n_clips)

    # 5) cross_doc_consistency
    cd_errs = cross_doc_check(pack, gt, fixtures, out, transcript)

    return {
        "justice_q_recall":       {"hit": len(hit_q), "n_gt": len(gt_qnos), "ratio": round(q_recall, 3),
                                   "ok": q_recall >= 0.75},
        "qa_pairing_acc":         {"correct": correct, "n_matched": matched, "ratio": round(pairing_acc, 3),
                                   "ok": pairing_acc >= 0.75},
        "cited_case_f1":          {"hit": cite_hits, "n_gt": len(gt_cites),
                                   "n_agent": len(agent_cites),
                                   "precision": round(cite_precision, 3),
                                   "recall": round(cite_recall, 3),
                                   "ratio": round(cite_f1, 3),
                                   "ok": cite_f1 >= 0.50},
        "clip_duration_match":    {"ok_clips": n_ok, "n_clips": n_clips, "ratio": round(clip_ratio, 3),
                                   "errors": bad[:6], "ok": clip_ratio >= 1.0 - 1e-6},
        "cross_doc_consistency":  {"errors": cd_errs[:8], "n_errors": len(cd_errs),
                                   "ok": len(cd_errs) == 0},
    }


# --------------------------------------------------------------------------- #
# Quote audio grounding: ASR per-pair Q/A wav clips and compare to the
# agent's question_quote / answer_quote text in quote_pack.json.
# Anti-cheat: silence / wrong-content wavs at high CER will fail.
# --------------------------------------------------------------------------- #
def quote_audio_grounding(out: Path, pack: list, n_sample: int = 5) -> dict:
    if os.environ.get("CLAW_BENCH_SKIP_ASR") == "1":
        return {"skipped": True, "ratio": 0.0, "ok": False,
                "reasons": "CLAW_BENCH_SKIP_ASR=1"}
    try:
        from asr_judge import asr_score  # type: ignore
    except Exception as e:
        return {"skipped": True, "ratio": 0.0, "ok": False,
                "reasons": f"asr_judge import failed: {e}"}
    if not pack:
        return {"ratio": 0.0, "ok": False, "reasons": "empty pack"}

    rng = random.Random(0x5C0775)
    sample = list(pack)
    rng.shuffle(sample)
    sample = sample[:n_sample]

    PASS_WER = 0.45
    n_clips = 0
    n_pass = 0
    werr_total = 0.0
    per_clip = []
    for p in sample:
        for clip_key, text_key in (("q_clip", "question_quote"),
                                    ("a_clip", "answer_quote")):
            cp = (p.get(clip_key) or "").strip()
            ref = (p.get(text_key) or "").strip()
            if not cp or not ref:
                continue
            wav = out / cp
            if not wav.exists():
                per_clip.append({"id": p.get("id"), "side": clip_key,
                                  "missing": True})
                continue
            n_clips += 1
            try:
                r = asr_score(wav, ref, language="en", metric="wer")
            except Exception as e:
                per_clip.append({"id": p.get("id"), "side": clip_key,
                                  "asr_err": str(e)[:120]})
                continue
            werr_total += r["wer"]
            passed = r["wer"] <= PASS_WER
            if passed:
                n_pass += 1
            per_clip.append({
                "id": p.get("id"),
                "side": clip_key,
                "wer": round(r["wer"], 3),
                "passed": passed,
                "asr_preview": (r.get("transcript") or "")[:120],
                "ref_preview": ref[:120],
            })
    ratio = n_pass / max(1, n_clips)
    mean_wer = werr_total / max(1, n_clips)
    return {
        "n_clips": n_clips,
        "n_pass": n_pass,
        "ratio": round(ratio, 3),
        "mean_wer": round(mean_wer, 3),
        "per_clip": per_clip,
        "ok": n_clips >= 4 and ratio >= 0.60,
        "threshold_wer": PASS_WER,
        "threshold_ratio": 0.60,
    }


# --------------------------------------------------------------------------- #
# Summary axes-yes/no judge (replaces single-LLM continuous score).
# --------------------------------------------------------------------------- #
SUMMARY_AXES = [
    ("central_question_stated",
     "Does SUMMARY.MD state the central legal question of the case "
     "(true threats / mens rea standard on the facts of Counterman v. "
     "Colorado, or comparable framing)?"),
    ("justice_dividing_lines",
     "Does SUMMARY.MD name the dividing lines among the justices "
     "(objective vs subjective intent standard, recklessness, etc.)?"),
    ("party_positions_named",
     "Does SUMMARY.MD name petitioner / respondent positions or "
     "arguments (e.g. petitioner's free-speech concern; respondent's "
     "victim-protection rationale)?"),
    ("legal_prose_concrete",
     "Does SUMMARY.MD read as fluent legal-press English (~ <= 300 words) "
     "referencing specific Counterman-related concepts, not placeholder "
     "or lorem-ipsum?"),
    ("self_consistent_with_pack",
     "Is SUMMARY.MD self-consistent with the agent's quote_pack — names "
     "of justices/advocates mentioned in the summary appear in "
     "GROUND_DATA's justices_seen / advocates_seen?"),
]


def summary_quality(out: Path, pack: list, gt: dict) -> dict:
    ground = {
        "case": "Counterman v. Colorado (22-138, 2023 OT)",
        "n_pairs": len(pack),
        "justices_seen": sorted({(p.get("justice") or "")
                                  for p in pack if p.get("justice")})[:9],
        "advocates_seen": sorted({(p.get("advocate") or "")
                                   for p in pack if p.get("advocate")})[:5],
    }
    summary = (out / "summary.md").read_text()
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground, ensure_ascii=False)}\n\n"
        f"SUMMARY.MD:\n{summary[:6000]}\n"
    )
    return axes_yes_no_score(payload, SUMMARY_AXES, threshold=0.60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--fixtures", default="fixtures")
    args = ap.parse_args()
    out, fix = Path(args.output), Path(args.fixtures)

    ok, errs, pack = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs}, indent=2))
        return
    gt = json.load(open(args.gt))
    res = evaluate(out, gt, fix, pack)

    qag = quote_audio_grounding(out, pack)
    sq = summary_quality(out, pack, gt)
    res["quote_audio_grounding"] = {**qag, "ok": qag.get("ok", False)}
    res["summary_quality_axes"] = {
        "axes": sq.get("axes", []),
        "score": sq.get("score", 0.0),
        "ok": bool(sq.get("ok", False)),
        "threshold": 0.60,
    }

    all_ok = all(v["ok"] for v in res.values())
    # v3: deterministic dims dominate (~0.85). Axes-LLM 0.15.
    weights = {
        "justice_q_recall":       0.15,
        "qa_pairing_acc":         0.20,
        "cited_case_f1":          0.15,
        "clip_duration_match":    0.10,
        "cross_doc_consistency":  0.15,
        "quote_audio_grounding":  0.10,
        "summary_quality_axes":   0.15,
    }
    def _component(k):
        if k in ("clip_duration_match", "cross_doc_consistency"):
            return 1.0 if res[k]["ok"] else 0.0
        if k == "summary_quality_axes":
            return res[k].get("score", 0.0)
        return max(0.0, res[k].get("ratio", 0.0))
    score = round(sum(weights[k] * _component(k) for k in weights), 3)
    print(json.dumps({"pass": all_ok, "checks": {k: v["ok"] for k, v in res.items()},
                      **res, "score": score, "weights": weights},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
