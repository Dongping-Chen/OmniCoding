#!/usr/bin/env python3
"""Evaluate predictions on VideoZeroBench.

This script implements the benchmark protocol described in
`2604.01569v1.pdf` in the same directory:

- Level-1: answer accuracy with temporal + spatial evidence provided
- Level-2: answer accuracy with temporal evidence provided
- Level-3: answer accuracy without evidence hints
- Level-4: Level-3 correct answer AND temporal grounding tIoU > 0.3
- Level-5: Level-3 correct answer AND temporal grounding tIoU > 0.3
  AND spatial grounding vIoU > 0.3

The paper text available locally specifies the Level-4/5 decision rule and the
0.3 thresholds. It does not spell out the low-level tIoU/vIoU implementation in
the extracted text, so this script makes the following explicit choices:

- Answer correctness uses normalized exact match.
- Temporal grounding uses IoU over the union of predicted and ground-truth
  intervals after merging overlaps.
- Spatial grounding groups boxes by timestamp, computes IoU between the union
  of predicted boxes and the union of ground-truth boxes at each timestamp, and
  averages across timestamps.
- If a matched sample lacks the corresponding ground-truth grounding
  annotations, the grounding score is 0.

Prediction file format
======================

The prediction file can be JSON or JSONL. Each record must contain `question_id`
and may contain these fields:

- `level1_answer`
- `level2_answer`
- `answer` or `level3_answer`
- `temporal` or `level4_temporal`
- `spatial` or `level5_spatial`

It also accepts the JSON output produced by `run_codex_cli_videozerobench.py`.
In that case, the evaluator automatically falls back to nested fields such as:

- `runs.level1.parsed_answer`
- `runs.level2.parsed_answer`
- `runs.level3.parsed_answer`
- `runs.level4.parsed_temporal`
- `runs.level5.parsed_spatial`

Accepted temporal formats:
- string: "From <12.3 seconds> to <45.6 seconds>. From <50> to <60>."
- list of dicts: [{"start": 12.3, "end": 45.6}, ...]
- list of pairs: [[12.3, 45.6], [50, 60]]

Accepted spatial formats:
- JSON string or Python-literal string with objects like:
  [{"time": 12.3, "bbox_2d": [[100, 200, 300, 400]]}, ...]
- list of dicts with keys such as `time`, `bbox_2d`, `boxes`, `box`, `bbox`

Predicted boxes may use either normalized [0, 1] coordinates or [0, 1000]
coordinates as in some paper prompts. The script normalizes the latter.
"""

from __future__ import annotations

import ast
import json
import math
import re
import statistics
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


INTERVAL_RE = re.compile(
    r"from\s*<?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:seconds?|s)?\s*>?\s*"
    r"to\s*<?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:seconds?|s)?\s*>?",
    flags=re.IGNORECASE,
)


def load_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path} is empty")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    parsed = json.loads(text)
    if isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
        return parsed["questions"]
    if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
        return parsed["results"]
    return parsed


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def numeric_normalize(text: str) -> float | None:
    cleaned = text.strip().replace(",", "")
    if re.fullmatch(r"[-+]?[0-9]+(?:\.[0-9]+)?", cleaned):
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    numeric = numeric_normalize(text)
    if numeric is not None:
        if math.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.6f}".rstrip("0").rstrip(".")
    chars = []
    for ch in text.lower():
        category = unicodedata.category(ch)
        if category.startswith("P") or category.startswith("S"):
            chars.append(" ")
        else:
            chars.append(ch)
    normalized = re.sub(r"\s+", " ", "".join(chars)).strip()
    return normalized


def answer_exact_match(pred: Any, gold: Any) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    cleaned = sorted((s, e) for s, e in intervals if s is not None and e is not None and s < e)
    merged: list[tuple[float, float]] = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def intervals_length(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def interval_intersection_length(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    i = 0
    j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            total += end - start
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def temporal_iou(
    gold: list[tuple[float, float]], pred: list[tuple[float, float]]
) -> float:
    gold_m = merge_intervals(gold)
    pred_m = merge_intervals(pred)
    if not gold_m or not pred_m:
        return 0.0
    intersection = interval_intersection_length(gold_m, pred_m)
    union = intervals_length(gold_m) + intervals_length(pred_m) - intersection
    return intersection / union if union > 0 else 0.0


def parse_intervals(value: Any) -> list[tuple[float, float]]:
    if value is None:
        return []
    if isinstance(value, str):
        intervals = []
        for start, end in INTERVAL_RE.findall(value):
            s = as_float(start)
            e = as_float(end)
            if s is not None and e is not None and s < e:
                intervals.append((s, e))
        return intervals
    if isinstance(value, list):
        intervals = []
        for item in value:
            if isinstance(item, dict):
                s = as_float(item.get("start"))
                e = as_float(item.get("end"))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                s = as_float(item[0])
                e = as_float(item[1])
            else:
                continue
            if s is not None and e is not None and s < e:
                intervals.append((s, e))
        return intervals
    return []


def normalize_box(box: Any) -> list[float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    coords = [as_float(x) for x in box]
    if any(v is None for v in coords):
        return None
    x1, y1, x2, y2 = [float(v) for v in coords]  # type: ignore[arg-type]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
        x1 /= 1000.0
        y1 /= 1000.0
        x2 /= 1000.0
        y2 /= 1000.0
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def parse_spatial(value: Any) -> dict[float, list[list[float]]]:
    if value is None:
        return {}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                return {}
        return parse_spatial(parsed)
    if not isinstance(value, list):
        return {}

    grouped: dict[float, list[list[float]]] = defaultdict(list)
    for item in value:
        if not isinstance(item, dict):
            continue
        time_value = as_float(item.get("time", item.get("timestamp")))
        if time_value is None:
            continue

        raw_boxes = (
            item.get("bbox_2d")
            or item.get("boxes")
            or item.get("bbox")
            or item.get("box")
            or []
        )
        if isinstance(raw_boxes, (list, tuple)) and len(raw_boxes) == 4 and not isinstance(
            raw_boxes[0], (list, tuple)
        ):
            raw_boxes = [raw_boxes]

        for raw_box in raw_boxes:
            box = normalize_box(raw_box)
            if box is not None:
                grouped[float(time_value)].append(box)
    return dict(grouped)


def group_gold_boxes(evidence_boxes: list[dict[str, Any]]) -> dict[float, list[list[float]]]:
    grouped: dict[float, list[list[float]]] = defaultdict(list)
    for item in evidence_boxes:
        time_value = as_float(item.get("time"))
        box = normalize_box(item.get("box"))
        if time_value is None or box is None:
            continue
        grouped[float(time_value)].append(box)
    return dict(grouped)


def rectangle_union_area(boxes: list[list[float]]) -> float:
    if not boxes:
        return 0.0
    xs = sorted({box[0] for box in boxes} | {box[2] for box in boxes})
    if len(xs) < 2:
        return 0.0
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        ys: list[tuple[float, float]] = []
        for x1, y1, x2, y2 in boxes:
            if x1 < right and x2 > left:
                ys.append((y1, y2))
        if not ys:
            continue
        ys.sort()
        merged: list[tuple[float, float]] = []
        for start, end in ys:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        y_total = sum(end - start for start, end in merged)
        area += (right - left) * y_total
    return area


def union_iou_boxes(gold_boxes: list[list[float]], pred_boxes: list[list[float]]) -> float:
    if not gold_boxes or not pred_boxes:
        return 0.0
    gold_area = rectangle_union_area(gold_boxes)
    pred_area = rectangle_union_area(pred_boxes)
    combined_area = rectangle_union_area(gold_boxes + pred_boxes)
    intersection = gold_area + pred_area - combined_area
    union = combined_area
    return max(0.0, intersection) / union if union > 0 else 0.0


def align_time_key(pred: dict[float, list[list[float]]], target: float, tolerance: float) -> float | None:
    best_key = None
    best_delta = None
    for key in pred:
        delta = abs(key - target)
        if delta <= tolerance and (best_delta is None or delta < best_delta):
            best_key = key
            best_delta = delta
    return best_key


def visual_iou(
    gold: dict[float, list[list[float]]],
    pred: dict[float, list[list[float]]],
    time_tolerance: float,
) -> float:
    if not gold or not pred:
        return 0.0
    per_frame = []
    for time_value, gold_boxes in gold.items():
        matched_key = align_time_key(pred, time_value, time_tolerance)
        pred_boxes = pred.get(matched_key, []) if matched_key is not None else []
        per_frame.append(union_iou_boxes(gold_boxes, pred_boxes))
    return statistics.fmean(per_frame) if per_frame else 0.0


def get_prediction_field(record: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in record:
            return record[name]
    runs = record.get("runs")
    if isinstance(runs, dict):
        nested_aliases = {
            "level1_answer": ("level1", "parsed_answer"),
            "level2_answer": ("level2", "parsed_answer"),
            "level3_answer": ("level3", "parsed_answer"),
            "answer": ("level3", "parsed_answer"),
            "prediction": ("level3", "parsed_answer"),
            "level4_temporal": ("level4", "parsed_temporal"),
            "temporal": ("level4", "parsed_temporal"),
            "temporal_prediction": ("level4", "parsed_temporal"),
            "level4_prediction": ("level4", "parsed_temporal"),
            "level5_spatial": ("level5", "parsed_spatial"),
            "spatial": ("level5", "parsed_spatial"),
            "spatial_prediction": ("level5", "parsed_spatial"),
            "level5_prediction": ("level5", "parsed_spatial"),
        }
        for name in names:
            nested = nested_aliases.get(name)
            if not nested:
                continue
            level_key, field_key = nested
            level_record = runs.get(level_key)
            if isinstance(level_record, dict) and field_key in level_record:
                return level_record[field_key]
    return None


def has_prediction_field(record: dict[str, Any], names: list[str]) -> bool:
    for name in names:
        if name in record:
            return True
    runs = record.get("runs")
    if isinstance(runs, dict):
        nested_aliases = {
            "level1_answer": ("level1", "parsed_answer"),
            "level2_answer": ("level2", "parsed_answer"),
            "level3_answer": ("level3", "parsed_answer"),
            "answer": ("level3", "parsed_answer"),
            "prediction": ("level3", "parsed_answer"),
            "level4_temporal": ("level4", "parsed_temporal"),
            "temporal": ("level4", "parsed_temporal"),
            "temporal_prediction": ("level4", "parsed_temporal"),
            "level4_prediction": ("level4", "parsed_temporal"),
            "level5_spatial": ("level5", "parsed_spatial"),
            "spatial": ("level5", "parsed_spatial"),
            "spatial_prediction": ("level5", "parsed_spatial"),
            "level5_prediction": ("level5", "parsed_spatial"),
        }
        for name in names:
            nested = nested_aliases.get(name)
            if not nested:
                continue
            level_key, field_key = nested
            level_record = runs.get(level_key)
            if isinstance(level_record, dict) and field_key in level_record:
                return True
    return False


def was_level_requested(record: dict[str, Any], level: int) -> bool | None:
    levels = record.get("levels_requested")
    if not isinstance(levels, list):
        return None
    normalized = set()
    for value in levels:
        try:
            normalized.add(int(value))
        except (TypeError, ValueError):
            continue
    return level in normalized


def build_prediction_index(records: Any) -> dict[str, dict[str, Any]]:
    if isinstance(records, dict):
        if all(isinstance(v, dict) for v in records.values()):
            index = {}
            for key, value in records.items():
                merged = dict(value)
                merged.setdefault("question_id", key)
                index[str(merged["question_id"])] = merged
            return index
        raise ValueError("JSON object predictions must map question_id -> record")
    if not isinstance(records, list):
        raise ValueError("predictions must be a list, JSONL file, or question_id mapping")
    index = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("every prediction record must be a JSON object")
        question_id = record.get("question_id")
        if question_id is None:
            raise ValueError("every prediction record must include question_id")
        index[str(question_id)] = record
    return index


def records_to_list(records: Any) -> list[dict[str, Any]]:
    if isinstance(records, dict):
        if all(isinstance(v, dict) for v in records.values()):
            rows: list[dict[str, Any]] = []
            for key, value in records.items():
                merged = dict(value)
                merged.setdefault("question_id", key)
                rows.append(merged)
            return rows
        raise ValueError("JSON object predictions must map question_id -> record")
    if not isinstance(records, list):
        raise ValueError("predictions must be a list, JSONL file, or question_id mapping")
    rows = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("every prediction record must be a JSON object")
        rows.append(dict(record))
    return rows


def metric_mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def percent(value: float) -> float:
    return round(value * 100.0, 2)


def evaluate(
    annotations: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    time_tolerance: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    annotation_index = {str(sample["question_id"]): sample for sample in annotations}
    matched_question_ids = [qid for qid in predictions if qid in annotation_index]
    matched_annotations = [annotation_index[qid] for qid in matched_question_ids]

    if not matched_annotations:
        raise ValueError("No prediction question_id values matched the annotation file.")

    rows = []
    for sample in matched_annotations:
        qid = str(sample["question_id"])
        pred = predictions.get(qid, {})

        answer_l1 = get_prediction_field(pred, ["level1_answer"])
        answer_l2 = get_prediction_field(pred, ["level2_answer"])
        answer_l3 = get_prediction_field(pred, ["level3_answer", "answer", "prediction"])
        temporal_raw = get_prediction_field(
            pred,
            ["level4_temporal", "temporal", "temporal_prediction", "level4_prediction"],
        )
        spatial_raw = get_prediction_field(
            pred,
            ["level5_spatial", "spatial", "spatial_prediction", "level5_prediction"],
        )

        gold_answer = sample["answer"]
        l1_correct = answer_exact_match(answer_l1, gold_answer) if answer_l1 is not None else None
        l2_correct = answer_exact_match(answer_l2, gold_answer) if answer_l2 is not None else None
        l3_correct = answer_exact_match(answer_l3, gold_answer)

        gold_temporal = [
            (float(item["start"]), float(item["end"]))
            for item in sample.get("evidence_windows", [])
            if "start" in item and "end" in item
        ]
        pred_temporal = parse_intervals(temporal_raw)
        tiou = temporal_iou(gold_temporal, pred_temporal)

        gold_spatial = group_gold_boxes(sample.get("evidence_boxes", []))
        pred_spatial = parse_spatial(spatial_raw)
        viou = visual_iou(gold_spatial, pred_spatial, time_tolerance=time_tolerance)

        l1_requested = was_level_requested(pred, 1)
        l2_requested = was_level_requested(pred, 2)
        l3_requested = was_level_requested(pred, 3)
        l4_requested = was_level_requested(pred, 4)
        l5_requested = was_level_requested(pred, 5)

        row = {
            "question_id": qid,
            "category": sample.get("category", ""),
            "language": sample.get("language", ""),
            "evidence_span": sample.get("evidence_span", ""),
            "capabilities": sample.get("annotation_capabilities", []),
            "reference_answer": gold_answer,
            "reference_evidence_windows": sample.get("evidence_windows", []),
            "reference_evidence_boxes": sample.get("evidence_boxes", []),
            "has_temporal_gt": bool(gold_temporal),
            "has_spatial_gt": bool(gold_spatial),
            "l1_attempted": l1_requested if l1_requested is not None else has_prediction_field(pred, ["level1_answer"]),
            "l2_attempted": l2_requested if l2_requested is not None else has_prediction_field(pred, ["level2_answer"]),
            "l3_attempted": l3_requested
            if l3_requested is not None
            else has_prediction_field(pred, ["level3_answer", "answer", "prediction"]),
            "l4_attempted": l4_requested
            if l4_requested is not None
            else has_prediction_field(
                pred,
                ["level4_temporal", "temporal", "temporal_prediction", "level4_prediction"],
            ),
            "l5_attempted": l5_requested
            if l5_requested is not None
            else has_prediction_field(
                pred,
                ["level5_spatial", "spatial", "spatial_prediction", "level5_prediction"],
            ),
            "l1_correct": l1_correct,
            "l2_correct": l2_correct,
            "l3_correct": l3_correct,
            "tiou": tiou,
            "viou": viou,
            "l4_correct": l3_correct and tiou > 0.3,
            "l5_correct": l3_correct and tiou > 0.3 and viou > 0.3,
        }
        rows.append(row)

    report = summarize_rows(rows)
    report["num_predictions_loaded"] = len(predictions)
    report["num_predictions_matched"] = len(matched_question_ids)
    report["num_annotations_evaluated"] = len(matched_annotations)
    report["num_predictions_unmatched"] = len(predictions) - len(matched_question_ids)
    report["num_predictions"] = len(matched_question_ids)
    report["num_annotations"] = len(matched_annotations)
    report["time_tolerance_seconds"] = time_tolerance
    return report, {row["question_id"]: row for row in rows}


def build_questions_eval(
    prediction_records: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    eval_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    annotation_index = {str(sample["question_id"]): sample for sample in annotations}
    questions_eval: list[dict[str, Any]] = []
    for record in prediction_records:
        enriched = dict(record)
        question_id = str(enriched.get("question_id"))
        sample = annotation_index.get(question_id)
        eval_row = eval_rows.get(question_id)

        enriched["eval_matched_annotation"] = sample is not None
        if sample is None:
            enriched["eval_error"] = "question_id not found in annotations"
            questions_eval.append(enriched)
            continue

        enriched["reference_answer"] = sample.get("answer")
        enriched["reference_evidence_windows"] = sample.get("evidence_windows", [])
        enriched["reference_evidence_boxes"] = sample.get("evidence_boxes", [])

        if eval_row is not None:
            enriched["eval_l1_correct"] = eval_row.get("l1_correct")
            enriched["eval_l2_correct"] = eval_row.get("l2_correct")
            enriched["eval_l3_correct"] = eval_row.get("l3_correct")
            enriched["eval_tiou"] = round(float(eval_row.get("tiou", 0.0)), 6)
            enriched["eval_viou"] = round(float(eval_row.get("viou", 0.0)), 6)
            enriched["eval_l4_correct"] = eval_row.get("l4_correct")
            enriched["eval_l5_correct"] = eval_row.get("l5_correct")
            enriched["eval_has_temporal_gt"] = eval_row.get("has_temporal_gt")
            enriched["eval_has_spatial_gt"] = eval_row.get("has_spatial_gt")

        questions_eval.append(enriched)
    return questions_eval


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    l1_attempted_rows = [r for r in rows if r["l1_attempted"]]
    l2_attempted_rows = [r for r in rows if r["l2_attempted"]]
    l3_attempted_rows = [r for r in rows if r["l3_attempted"]]
    l4_attempted_rows = [r for r in rows if r["l4_attempted"]]
    l5_attempted_rows = [r for r in rows if r["l5_attempted"]]

    overall = {
        "level1_accuracy": percent(metric_mean([r["l1_correct"] for r in rows if r["l1_correct"] is not None])),
        "level2_accuracy": percent(metric_mean([r["l2_correct"] for r in rows if r["l2_correct"] is not None])),
        "level3_accuracy": percent(metric_mean([float(r["l3_correct"]) for r in rows])),
        "mean_tiou_all": percent(metric_mean([r["tiou"] for r in rows])),
        "mean_tiou_annotated": percent(
            metric_mean([r["tiou"] for r in rows if r["has_temporal_gt"]])
        ),
        "level4_accuracy": percent(metric_mean([float(r["l4_correct"]) for r in rows])),
        "mean_viou_all": percent(metric_mean([r["viou"] for r in rows])),
        "mean_viou_annotated": percent(
            metric_mean([r["viou"] for r in rows if r["has_spatial_gt"]])
        ),
        "level5_accuracy": percent(metric_mean([float(r["l5_correct"]) for r in rows])),
        "temporal_annotation_coverage": percent(
            metric_mean([float(r["has_temporal_gt"]) for r in rows])
        ),
        "spatial_annotation_coverage": percent(
            metric_mean([float(r["has_spatial_gt"]) for r in rows])
        ),
        "spatio_temporal_annotation_coverage": percent(
            metric_mean([float(r["has_temporal_gt"] and r["has_spatial_gt"]) for r in rows])
        ),
        "level1_attempted_count": len(l1_attempted_rows),
        "level1_correct_count": sum(1 for r in l1_attempted_rows if r["l1_correct"]),
        "level2_attempted_count": len(l2_attempted_rows),
        "level2_correct_count": sum(1 for r in l2_attempted_rows if r["l2_correct"]),
        "level3_attempted_count": len(l3_attempted_rows),
        "level3_correct_count": sum(1 for r in l3_attempted_rows if r["l3_correct"]),
        "level4_attempted_count": len(l4_attempted_rows),
        "level4_correct_count": sum(1 for r in l4_attempted_rows if r["l4_correct"]),
        "level5_attempted_count": len(l5_attempted_rows),
        "level5_correct_count": sum(1 for r in l5_attempted_rows if r["l5_correct"]),
    }

    def grouped(key: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[key])].append(row)
        return {
            group: {
                "count": len(group_rows),
                "level3_accuracy": percent(metric_mean([float(r["l3_correct"]) for r in group_rows])),
                "level4_accuracy": percent(metric_mean([float(r["l4_correct"]) for r in group_rows])),
                "level5_accuracy": percent(metric_mean([float(r["l5_correct"]) for r in group_rows])),
                "mean_tiou_all": percent(metric_mean([r["tiou"] for r in group_rows])),
                "mean_viou_all": percent(metric_mean([r["viou"] for r in group_rows])),
            }
            for group, group_rows in sorted(groups.items(), key=lambda item: item[0])
        }

    capabilities: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for capability in row["capabilities"]:
            capabilities[str(capability)].append(row)

    by_capability = {
        capability: {
            "count": len(group_rows),
            "level3_accuracy": percent(metric_mean([float(r["l3_correct"]) for r in group_rows])),
            "level4_accuracy": percent(metric_mean([float(r["l4_correct"]) for r in group_rows])),
            "level5_accuracy": percent(metric_mean([float(r["l5_correct"]) for r in group_rows])),
            "mean_tiou_all": percent(metric_mean([r["tiou"] for r in group_rows])),
            "mean_viou_all": percent(metric_mean([r["viou"] for r in group_rows])),
        }
        for capability, group_rows in sorted(capabilities.items(), key=lambda item: item[0])
    }

    return {
        "overall": overall,
        "by_category": grouped("category"),
        "by_language": grouped("language"),
        "by_evidence_span": grouped("evidence_span"),
        "by_capability": by_capability,
    }


def print_report(*args, **kwargs):
    from videozerobench_evaluator_cli import print_report as _impl
    return _impl(*args, **kwargs)



def parse_args(*args, **kwargs):
    from videozerobench_evaluator_cli import parse_args as _impl
    return _impl(*args, **kwargs)



def main(*args, **kwargs):
    from videozerobench_evaluator_cli import main as _impl
    return _impl(*args, **kwargs)



if __name__ == "__main__":
    sys.exit(main())
