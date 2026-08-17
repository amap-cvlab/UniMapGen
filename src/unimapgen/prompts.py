"""Global-stitching prompt construction for the UniMapGen v6 model."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .codec import encode_text


BASE_PROMPT = "<image>请预测图内所有车道线及其类别"
REF_PROMPT = (
    "<image>请参考提示点预测图内所有车道线及其类别。"
    "线开始提示点为：{starts}线结束提示点为：{ends}"
)

_TILE_PATTERN = re.compile(
    r"^(?P<prefix>.+)_(?P<x0>-?\d+)_(?P<y0>-?\d+)_(?P<x1>-?\d+)_(?P<y1>-?\d+)$"
)


@dataclass(frozen=True)
class TileId:
    prefix: str
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def key(self) -> tuple[str, int, int]:
        return self.prefix, self.x0, self.y0


def parse_tile_id(image_name: str | Path) -> TileId:
    stem = Path(image_name).stem
    match = _TILE_PATTERN.match(stem)
    if not match:
        raise ValueError(
            f"tile name must end in _x0_y0_x1_y1, received: {image_name}"
        )
    groups = match.groupdict()
    return TileId(
        prefix=groups["prefix"],
        x0=int(groups["x0"]),
        y0=int(groups["y0"]),
        x1=int(groups["x1"]),
        y1=int(groups["y1"]),
    )


def sort_key(image_name: str | Path) -> tuple[str, int, int]:
    tile = parse_tile_id(image_name)
    return tile.prefix, tile.y0, tile.x0


def _is_right(point: list[int], tile_size: int) -> bool:
    return point[0] + point[1] >= tile_size and point[0] > point[1]


def _is_down(point: list[int], tile_size: int) -> bool:
    return point[0] + point[1] >= tile_size and point[0] <= point[1]


def _neighbor_refs(
    lanes: Iterable[dict] | None, *, axis: str, tile_size: int
) -> tuple[list[dict], list[dict]]:
    starts: list[dict] = []
    ends: list[dict] = []
    for lane in lanes or []:
        category = lane.get("category")
        for point_key, type_key, expected_type in (
            ("start_point", "start_type", "<cut_point>"),
            ("end_point", "end_type", "</cut_point>"),
        ):
            point = lane.get(point_key)
            if lane.get(type_key) != expected_type or not isinstance(point, list):
                continue
            visible = _is_right(point, tile_size) if axis == "left" else _is_down(point, tile_size)
            if not visible:
                continue
            translated = [int(point[0]), int(point[1])]
            if axis == "left":
                translated[0] = max(0, translated[0] - tile_size)
            else:
                translated[1] = max(0, translated[1] - tile_size)

            if expected_type == "<cut_point>":
                ends.append(
                    {
                        "end_type": "</cut_point>",
                        "ref_point": translated,
                        "category": category,
                    }
                )
            else:
                starts.append(
                    {
                        "start_type": "<cut_point>",
                        "ref_point": translated,
                        "category": category,
                    }
                )
    return starts, ends


def build_stitching_prompt(
    left_lanes: Iterable[dict] | None,
    up_lanes: Iterable[dict] | None,
    *,
    tile_size: int = 896,
) -> str:
    """Build the exact Chinese prompt layout used by v6 global inference."""

    if not left_lanes and not up_lanes:
        return BASE_PROMPT
    left_starts, left_ends = _neighbor_refs(left_lanes, axis="left", tile_size=tile_size)
    up_starts, up_ends = _neighbor_refs(up_lanes, axis="up", tile_size=tile_size)
    starts = left_starts + up_starts
    ends = left_ends + up_ends
    prompt = REF_PROMPT.format(
        starts=json.dumps(starts, ensure_ascii=False, separators=(",", ":")),
        ends=json.dumps(ends, ensure_ascii=False, separators=(",", ":")),
    )
    return encode_text(prompt).replace('"', "")
