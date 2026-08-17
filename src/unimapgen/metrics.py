"""UniMapGen v6 mIoU, mask AP, and Chamfer AP evaluation."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.distance import cdist

from .codec import decode_lanes


CLASSES = ("Curb", "Laneline", "Virtualline")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}


def load_result_records(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record["_pred_lanes"] = decode_lanes(record["hq_pred"])
                record["_gt_lanes"] = decode_lanes(record["hq_gt"])
                records.append(record)
            except Exception as error:
                errors.append({"line": str(line_number), "error": str(error)})
    return records, errors


def rasterize_by_class(
    lanes: Iterable[dict], width: int = 896, height: int = 896, lane_width: int = 6
) -> dict[str, np.ndarray]:
    masks: dict[str, Image.Image] = {}
    for lane in lanes:
        category = lane.get("category")
        points = lane.get("sample_points", [])
        if category not in CLASS_TO_INDEX or len(points) < 2:
            continue
        if category not in masks:
            masks[category] = Image.new("L", (width, height), 0)
        flat_points = [int(coord) for point in points for coord in point]
        ImageDraw.Draw(masks[category]).line(flat_points, fill=255, width=lane_width)
    return {category: np.asarray(mask) > 0 for category, mask in masks.items()}


def semantic_mask(
    lanes: Iterable[dict], width: int = 896, height: int = 896, lane_width: int = 6
) -> np.ndarray:
    combined = np.full((height, width), 255, dtype=np.uint8)
    for category, mask in rasterize_by_class(lanes, width, height, lane_width).items():
        combined[mask] = CLASS_TO_INDEX[category]
    return combined


def compute_miou(
    records: Iterable[dict],
    *,
    width: int = 896,
    height: int = 896,
    lane_width: int = 6,
) -> dict[str, Any]:
    """Reproduce the original foreground-only semantic mIoU protocol."""

    intersections = np.zeros(len(CLASSES), dtype=np.float64)
    unions = np.zeros(len(CLASSES), dtype=np.float64)
    gt_areas = np.zeros(len(CLASSES), dtype=np.float64)
    sample_count = 0
    for record in records:
        pred = semantic_mask(record["_pred_lanes"], width, height, lane_width)
        gt = semantic_mask(record["_gt_lanes"], width, height, lane_width)
        valid = gt != 255
        pred = pred[valid]
        gt = gt[valid]
        for class_index in range(len(CLASSES)):
            pred_class = pred == class_index
            gt_class = gt == class_index
            intersections[class_index] += np.logical_and(pred_class, gt_class).sum()
            unions[class_index] += np.logical_or(pred_class, gt_class).sum()
            gt_areas[class_index] += gt_class.sum()
        sample_count += 1
    per_class = np.divide(
        intersections,
        unions,
        out=np.full_like(intersections, np.nan),
        where=unions > 0,
    )
    return {
        "mIoU": float(np.nanmean(per_class) * 100),
        "per_class": {
            category: None if np.isnan(value) else float(value * 100)
            for category, value in zip(CLASSES, per_class)
        },
        "samples": sample_count,
    }


_LEGACY_PROBABILITY = re.compile(
    r"Log Probability:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), decoded_token:\s*(.*)$"
)


def probability_tokens(record: dict[str, Any]) -> list[tuple[str, float]]:
    structured = record.get("token_logprobs")
    if isinstance(structured, list) and structured:
        return [
            (str(item.get("token", "")), float(item.get("probability", 0.0)))
            for item in structured
        ]
    legacy = record.get("pred_probs", [])
    if isinstance(legacy, str):
        try:
            legacy = json.loads(legacy)
        except json.JSONDecodeError:
            legacy = []
    output = []
    for item in legacy:
        match = _LEGACY_PROBABILITY.search(str(item))
        if match:
            output.append((match.group(2), float(match.group(1))))
    return output


def category_token_scores(record: dict[str, Any], count: int) -> list[float]:
    scores = [
        probability
        for token, probability in probability_tokens(record)
        if any(category in token for category in CLASSES)
    ]
    return (scores + [1.0] * count)[:count]


def geometric_line_scores(record: dict[str, Any], count: int) -> list[float]:
    scores: list[float] = []
    current: list[float] = []
    capturing = False
    for token, probability in probability_tokens(record):
        if "sample_points" in token:
            current = []
            capturing = True
            continue
        if capturing and ("c_" in token or any(category in token for category in CLASSES)):
            current.append(max(probability, np.finfo(np.float64).tiny))
        if capturing and any(category in token for category in CLASSES):
            scores.append(float(math.exp(np.mean(np.log(current)))))
            current = []
            capturing = False
    if not scores:
        scores = category_token_scores(record, count)
    return (scores + [1.0] * count)[:count]


def compute_mask_ap(
    records: Iterable[dict],
    *,
    width: int = 896,
    height: int = 896,
    lane_width: int = 6,
) -> dict[str, Any]:
    try:
        import torch
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError as error:
        raise RuntimeError(
            "mask AP requires torchmetrics and pycocotools from requirements.txt"
        ) from error

    metric = MeanAveragePrecision(
        iou_type="segm", class_metrics=True, extended_summary=False
    )
    sample_count = 0
    for record in records:
        predictions = record["_pred_lanes"]
        ground_truth = record["_gt_lanes"]
        scores = category_token_scores(record, len(predictions))

        def convert(lanes: list[dict], *, predicted: bool) -> dict[str, Any]:
            masks = []
            labels = []
            kept_scores = []
            for index, lane in enumerate(lanes):
                category = lane.get("category")
                points = lane.get("sample_points", [])
                if category not in CLASS_TO_INDEX or len(points) < 2:
                    continue
                mask = Image.new("L", (width, height), 0)
                flat = [int(coord) for point in points for coord in point]
                ImageDraw.Draw(mask).line(flat, fill=255, width=lane_width)
                masks.append(torch.from_numpy((np.asarray(mask) > 0).copy()))
                labels.append(CLASS_TO_INDEX[category])
                if predicted:
                    kept_scores.append(scores[index])
            data: dict[str, Any] = {
                "masks": torch.stack(masks)
                if masks
                else torch.zeros((0, height, width), dtype=torch.bool),
                "labels": torch.tensor(labels, dtype=torch.int64),
            }
            if predicted:
                data["scores"] = torch.tensor(kept_scores, dtype=torch.float32)
            return data

        metric.update(
            [convert(predictions, predicted=True)],
            [convert(ground_truth, predicted=False)],
        )
        sample_count += 1
    result = metric.compute()
    classes = [int(item) for item in result.get("classes", [])]
    per_class_values = result.get("map_per_class", [])
    per_class = {
        CLASSES[class_index]: float(per_class_values[position])
        for position, class_index in enumerate(classes)
    }
    return {
        "mAP": float(result["map"]),
        "mAP@50": float(result["map_50"]),
        "mAP@75": float(result["map_75"]),
        "per_class": per_class,
        "samples": sample_count,
    }


def resample_points(points: Iterable[Iterable[float]], count: int = 50) -> np.ndarray:
    points_array = np.asarray(list(points), dtype=np.float64)
    if len(points_array) == 0:
        return np.zeros((count, 2), dtype=np.float64)
    if len(points_array) == 1:
        return np.repeat(points_array, count, axis=0)
    segment_lengths = np.linalg.norm(np.diff(points_array, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] <= 0:
        return np.repeat(points_array[:1], count, axis=0)
    positions = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack(
        [np.interp(positions, cumulative, points_array[:, axis]) for axis in range(2)]
    )


def chamfer_distance(first: np.ndarray, second: np.ndarray) -> float:
    distances = cdist(first, second, metric="euclidean")
    return float((distances.min(axis=1).mean() + distances.min(axis=0).mean()) / 2)


def average_precision(true_positive: np.ndarray, false_positive: np.ndarray, gt_count: int) -> float:
    if gt_count <= 0:
        return float("nan")
    tp = np.cumsum(true_positive)
    fp = np.cumsum(false_positive)
    recall = tp / gt_count
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def compute_chamfer_ap(
    records: Iterable[dict],
    *,
    thresholds: tuple[float, ...] = (12.0, 16.0, 26.0, 36.0),
    sample_points: int = 50,
) -> dict[str, Any]:
    records = list(records)
    class_results: dict[str, list[float]] = {}
    for category in CLASSES:
        gt_count = sum(
            1
            for record in records
            for lane in record["_gt_lanes"]
            if lane.get("category") == category and len(lane.get("sample_points", [])) >= 2
        )
        detections: list[tuple[float, np.ndarray, np.ndarray]] = []
        for record in records:
            predictions = [
                lane
                for lane in record["_pred_lanes"]
                if lane.get("category") == category and len(lane.get("sample_points", [])) >= 2
            ]
            ground_truth = [
                lane
                for lane in record["_gt_lanes"]
                if lane.get("category") == category and len(lane.get("sample_points", [])) >= 2
            ]
            all_scores = geometric_line_scores(record, len(record["_pred_lanes"]))
            scores = [
                all_scores[index]
                for index, lane in enumerate(record["_pred_lanes"])
                if lane.get("category") == category and len(lane.get("sample_points", [])) >= 2
            ]
            pred_arrays = [resample_points(lane["sample_points"], sample_points) for lane in predictions]
            gt_arrays = [resample_points(lane["sample_points"], sample_points) for lane in ground_truth]
            if pred_arrays and gt_arrays:
                distance_matrix = np.asarray(
                    [
                        [chamfer_distance(prediction, target) for target in gt_arrays]
                        for prediction in pred_arrays
                    ]
                )
            else:
                distance_matrix = np.empty((len(pred_arrays), len(gt_arrays)))
            detections.append((0.0, np.asarray(scores), distance_matrix))

        aps = []
        for threshold in thresholds:
            ranked: list[tuple[float, float, float]] = []
            for _, scores, distance_matrix in detections:
                gt_used = np.zeros(distance_matrix.shape[1], dtype=bool)
                for pred_index in np.argsort(-scores):
                    score = float(scores[pred_index])
                    if distance_matrix.shape[1] == 0:
                        ranked.append((score, 0.0, 1.0))
                        continue
                    gt_index = int(np.argmin(distance_matrix[pred_index]))
                    if distance_matrix[pred_index, gt_index] <= threshold and not gt_used[gt_index]:
                        gt_used[gt_index] = True
                        ranked.append((score, 1.0, 0.0))
                    else:
                        ranked.append((score, 0.0, 1.0))
            ranked.sort(key=lambda item: item[0], reverse=True)
            tp = np.asarray([item[1] for item in ranked], dtype=np.float64)
            fp = np.asarray([item[2] for item in ranked], dtype=np.float64)
            aps.append(average_precision(tp, fp, gt_count))
        class_results[category] = aps

    mean_ap = []
    for index in range(len(thresholds)):
        values = np.asarray([class_results[name][index] for name in CLASSES])
        mean_ap.append(float(np.nanmean(values)))
    def finite_or_none(value: float) -> float | None:
        return None if math.isnan(value) else value

    return {
        "thresholds": list(thresholds),
        "mAP": {
            str(int(threshold)): finite_or_none(value)
            for threshold, value in zip(thresholds, mean_ap)
        },
        "per_class": {
            category: {
                str(int(threshold)): finite_or_none(value)
                for threshold, value in zip(thresholds, class_results[category])
            }
            for category in CLASSES
        },
        "samples": len(records),
    }


def evaluate_file(
    input_file: Path,
    metrics: Iterable[str],
    *,
    width: int = 896,
    height: int = 896,
    lane_width: int = 6,
) -> dict[str, Any]:
    records, errors = load_result_records(input_file)
    output: dict[str, Any] = {
        "input": str(input_file),
        "parsed_samples": len(records),
        "parse_errors": errors,
    }
    for metric in metrics:
        if metric == "miou":
            output[metric] = compute_miou(
                records, width=width, height=height, lane_width=lane_width
            )
        elif metric == "mask_ap":
            output[metric] = compute_mask_ap(
                records, width=width, height=height, lane_width=lane_width
            )
        elif metric == "chamfer_ap":
            output[metric] = compute_chamfer_ap(records)
        else:
            raise ValueError(f"unknown metric: {metric}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        default="miou,mask_ap,chamfer_ap",
        help="Comma-separated subset of miou, mask_ap, chamfer_ap.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--height", type=int, default=896)
    parser.add_argument("--lane-width", type=int, default=6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    result = evaluate_file(
        args.input_file,
        metrics,
        width=args.width,
        height=args.height,
        lane_width=args.lane_width,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if result["parse_errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
