"""Render input, ground truth, and prediction comparison panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .codec import decode_lanes, decode_structure
from .inference import resolve_image


COLORS = {
    "Curb": (60, 60, 255),
    "Laneline": (80, 230, 80),
    "Virtualline": (255, 190, 60),
}


def draw_lanes(image: np.ndarray, lanes: list[dict], thickness: int = 3) -> np.ndarray:
    output = image.copy()
    for lane in lanes:
        points = lane.get("sample_points", [])
        if len(points) < 2:
            continue
        color = COLORS.get(lane.get("category"), (255, 255, 255))
        polyline = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(output, [polyline], False, color, thickness, cv2.LINE_AA)
        for point in points:
            cv2.circle(output, tuple(map(int, point)), 3, color, -1, cv2.LINE_AA)
    return output


def _decode_ref_list(fragment: str) -> list[dict]:
    try:
        decoded = decode_structure(fragment)
        return decoded if isinstance(decoded, list) else []
    except Exception:
        return []


def draw_prompt_refs(image: np.ndarray, prompt: str) -> np.ndarray:
    output = image.copy()
    start_marker = "线开始提示点为："
    end_marker = "线结束提示点为："
    if start_marker not in prompt or end_marker not in prompt:
        return output
    start_fragment, end_fragment = prompt.split(start_marker, 1)[1].split(end_marker, 1)
    for item in _decode_ref_list(start_fragment):
        point = item.get("ref_point")
        if isinstance(point, list) and len(point) == 2:
            cv2.circle(output, tuple(map(int, point)), 12, (255, 0, 255), -1)
    for item in _decode_ref_list(end_fragment):
        point = item.get("ref_point")
        if isinstance(point, list) and len(point) == 2:
            cv2.circle(output, tuple(map(int, point)), 12, (0, 255, 255), -1)
    return output


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    banner = np.zeros((52, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        banner,
        title,
        (16, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([banner, image])


def visualize_record(record: dict, image_root: Path, output_dir: Path) -> Path:
    image_path = resolve_image(record["image"], image_root)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read {image_path}")
    ground_truth = decode_lanes(record["hq_gt"])
    prediction = decode_lanes(record["hq_pred"])
    pred_panel = draw_prompt_refs(draw_lanes(image, prediction), record.get("prompt", ""))
    panels = [
        add_title(image, "Satellite image"),
        add_title(draw_lanes(image, ground_truth), "Ground truth"),
        add_title(pred_panel, "Prediction"),
    ]
    combined = cv2.hconcat(panels)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(record['image']).stem}.jpg"
    if not cv2.imwrite(str(output_path), combined):
        raise RuntimeError(f"failed to write {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    written = 0
    errors = 0
    with args.input_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if args.limit is not None and written >= args.limit:
                break
            try:
                output = visualize_record(
                    json.loads(line), args.image_root, args.output_dir
                )
                print(output)
                written += 1
            except Exception as error:
                errors += 1
                print(f"visualization error: {error}")
    print(json.dumps({"written": written, "errors": errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
