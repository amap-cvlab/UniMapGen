"""Convert full satellite maps and polyline annotations into v6 tiles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, box


CATEGORY_ALIASES = {
    "lane line": "Laneline",
    "laneline": "Laneline",
    "virtual line": "Virtualline",
    "virtualline": "Virtualline",
    "curb": "Curb",
}


def normalize_category(value: str) -> str:
    normalized = CATEGORY_ALIASES.get(value.replace("_", " ").strip().lower())
    if normalized is None:
        raise ValueError(f"unsupported category: {value!r}")
    return normalized


def sliding_positions(length: int, window: int, step: int, cover_edge: bool) -> list[int]:
    if length < window:
        raise ValueError(f"image dimension {length} is smaller than window {window}")
    positions = list(range(0, length - window + 1, step))
    edge = length - window
    if cover_edge and positions[-1] != edge:
        positions.append(edge)
    return positions


def sample_polyline(points: Iterable[Iterable[float]], spacing: float = 40.0) -> list[list[int]]:
    line = LineString(points)
    if line.length <= 0:
        return []
    distances = list(range(0, int(math.floor(line.length)), max(1, int(spacing))))
    distances.append(line.length)
    sampled: list[list[int]] = []
    for distance in distances:
        point = line.interpolate(distance)
        xy = [int(round(point.x)), int(round(point.y))]
        if not sampled or sampled[-1] != xy:
            sampled.append(xy)
    return sampled


def _line_components(geometry) -> Iterator[LineString]:
    if isinstance(geometry, LineString):
        if geometry.length > 0:
            yield geometry
    elif isinstance(geometry, (MultiLineString, GeometryCollection)):
        for part in geometry.geoms:
            yield from _line_components(part)


def _orient_like_original(component: LineString, original: LineString) -> list[tuple[float, float]]:
    coords = list(component.coords)
    if original.project(component.boundary.geoms[0]) > original.project(component.boundary.geoms[-1]):
        coords.reverse()
    return coords


def clip_lane_to_tile(
    lane: dict,
    *,
    x0: int,
    y0: int,
    tile_size: int = 896,
    sample_distance: float = 40.0,
) -> list[dict]:
    """Clip one global-coordinate lane and return local-coordinate v6 lanes."""

    raw_points = lane.get("points", lane.get("sample_points"))
    if not isinstance(raw_points, list) or len(raw_points) < 2:
        return []
    original = LineString(raw_points)
    if original.length <= 0:
        return []

    # v6 coordinates are constrained to [0, 895] for an 896-pixel patch.
    tile = box(x0, y0, x0 + tile_size - 1, y0 + tile_size - 1)
    clipped = original.intersection(tile)
    output: list[dict] = []
    for component in _line_components(clipped):
        global_coords = _orient_like_original(component, original)
        local_coords = [(x - x0, y - y0) for x, y in global_coords]
        sampled = sample_polyline(local_coords, spacing=sample_distance)
        sampled = [
            [min(tile_size - 1, max(0, x)), min(tile_size - 1, max(0, y))]
            for x, y in sampled
        ]
        if len(sampled) < 2:
            continue
        # Determine endpoint types after orienting the clipped component.  Shapely
        # does not promise that an intersection keeps the input line direction.
        starts_at_original = original.project(Point(global_coords[0])) <= 1e-6
        ends_at_original = (
            original.length - original.project(Point(global_coords[-1])) <= 1e-6
        )
        output.append(
            {
                "start_point": sampled[0],
                "start_type": "<start_point>" if starts_at_original else "<cut_point>",
                "end_point": sampled[-1],
                "end_type": "<end_point>" if ends_at_original else "</cut_point>",
                "sample_points": sampled,
                "category": normalize_category(str(lane.get("category", ""))),
            }
        )
    return output


def order_lanes(lanes: list[dict]) -> list[dict]:
    """Apply the v4/v6 near-to-far ordering used to reduce decoding ambiguity."""

    return sorted(
        lanes,
        key=lambda lane: (
            lane["start_point"][0] ** 2 + lane["start_point"][1] ** 2,
            lane["category"],
            lane["end_point"][0],
            lane["end_point"][1],
        ),
    )


def load_annotations(path: Path) -> dict[str, list[dict]]:
    """Load a JSON mapping, JSONL records, or a directory of per-image JSON files."""

    if path.is_dir():
        return {
            item.stem: json.loads(item.read_text(encoding="utf-8"))
            for item in sorted(path.glob("*.json"))
        }
    if path.suffix.lower() == ".jsonl":
        result: dict[str, list[dict]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                image = record.get("image") or record.get("image_name")
                lanes = record.get("lanes") or record.get("annotations")
                result[Path(image).stem] = lanes
        return result

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {Path(key).stem: value for key, value in payload.items()}
    if isinstance(payload, list):
        result = {}
        for record in payload:
            image = record.get("image") or record.get("image_name")
            lanes = record.get("lanes") or record.get("annotations")
            result[Path(image).stem] = lanes
        return result
    raise ValueError(f"unsupported annotation container in {path}")


def preprocess_dataset(
    images_dir: Path,
    annotations_path: Path,
    output_dir: Path,
    *,
    tile_size: int = 896,
    step: int = 896,
    sample_distance: float = 40.0,
    cover_edge: bool = False,
    keep_empty: bool = False,
) -> tuple[Path, int]:
    annotations = load_annotations(annotations_path)
    tiles_dir = output_dir / "images"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "samples.jsonl"
    count = 0

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    )
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for image_path in image_paths:
            lanes = annotations.get(image_path.stem)
            if lanes is None:
                raise KeyError(f"missing annotations for {image_path.name}")
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                xs = sliding_positions(image.width, tile_size, step, cover_edge)
                ys = sliding_positions(image.height, tile_size, step, cover_edge)
                for y0 in ys:
                    for x0 in xs:
                        tile_lanes = []
                        for lane in lanes:
                            tile_lanes.extend(
                                clip_lane_to_tile(
                                    lane,
                                    x0=x0,
                                    y0=y0,
                                    tile_size=tile_size,
                                    sample_distance=sample_distance,
                                )
                            )
                        tile_lanes = order_lanes(tile_lanes)
                        if not tile_lanes and not keep_empty:
                            continue
                        tile_name = (
                            f"{image_path.stem}_{x0}_{y0}_{x0 + tile_size}_{y0 + tile_size}.png"
                        )
                        image.crop((x0, y0, x0 + tile_size, y0 + tile_size)).save(
                            tiles_dir / tile_name
                        )
                        record = {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": json.dumps(tile_lanes, ensure_ascii=False),
                                },
                                {"role": "assistant", "content": "balabala"},
                            ],
                            "images": [tile_name],
                            "videos": [],
                        }
                        manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1
    return manifest_path, count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=896)
    parser.add_argument("--step", type=int, default=896)
    parser.add_argument("--sample-distance", type=float, default=40.0)
    parser.add_argument(
        "--cover-edge",
        action="store_true",
        help="Append a final overlapping tile so the full image is covered.",
    )
    parser.add_argument("--keep-empty", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest, count = preprocess_dataset(
        args.images_dir,
        args.annotations,
        args.output_dir,
        tile_size=args.tile_size,
        step=args.step,
        sample_distance=args.sample_distance,
        cover_edge=args.cover_edge,
        keep_empty=args.keep_empty,
    )
    print(f"Wrote {count} tiles and {manifest}")


if __name__ == "__main__":
    main()
