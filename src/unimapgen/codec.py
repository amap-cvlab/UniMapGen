"""Codec for the structured tokens used by the UniMapGen v6 checkpoint."""

from __future__ import annotations

import json
import re
from typing import Any


KEY_TOKENS = {
    "start_point": "|start_point|",
    "start_type": "|start_type|",
    "end_point": "|end_point|",
    "end_type": "|end_type|",
    "sample_points": "|sample_points|",
    "ref_point": "|ref_point|",
    "category": "|category|",
    "type": "|type|",
    "boundary": "|boundary|",
    "occlusion": "|occlusion|",
}

VALUE_TOKENS = {
    "<start_point>",
    "<split_point>",
    "<cut_point>",
    "<link_point>",
    "<end_point>",
    "</split_point>",
    "</cut_point>",
    "</link_point>",
    "Laneline",
    "Virtualline",
    "Curb",
    "None",
    "Others",
    "Shortdashed",
    "Dashed",
    "Thicksolid",
    "Solid",
    "False",
    "True",
    "Major",
    "Minor",
    "No",
}

_TOKEN_VALUE_PATTERN = re.compile(
    r'"(' + "|".join(re.escape(value) for value in sorted(VALUE_TOKENS, key=len, reverse=True)) + r')"'
)
_COORD_PATTERN = re.compile(r"(?<!<)\b\d+\b(?!>)")


def encode_text(text: str) -> str:
    """Apply v6 tokens to a JSON fragment or prompt containing JSON fragments."""

    text = _COORD_PATTERN.sub(lambda match: f"<c_{match.group(0)}>", text)
    for key, token in KEY_TOKENS.items():
        text = text.replace(f'"{key}"', token)

    def encode_value(match: re.Match[str]) -> str:
        value_text = match.group(1)
        return value_text if value_text.startswith("<") else f"<{value_text}>"

    text = _TOKEN_VALUE_PATTERN.sub(encode_value, text)
    return text.replace(">,<", "><").replace(" ", "")


def encode_structure(value: Any) -> str:
    """Encode JSON-compatible data with the v6 key/value/coordinate tokens."""

    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return encode_text(text)


def encode_lanes(lanes: list[dict[str, Any]]) -> str:
    return encode_structure(lanes)


def decode_structure(text: str) -> Any:
    """Decode UniMapGen tokens into JSON-compatible Python values."""

    if not isinstance(text, str):
        raise TypeError("encoded UniMapGen output must be a string")

    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end >= start:
        text = text[start : end + 1]

    text = text.replace("><", ">,<")
    for key, token in KEY_TOKENS.items():
        text = text.replace(token, f'"{key}"')
    text = re.sub(r"<c_(\d+)>", r"\1", text)

    for value in sorted(VALUE_TOKENS, key=len, reverse=True):
        token = value if value.startswith("<") else f"<{value}>"
        text = text.replace(token, json.dumps(value, ensure_ascii=False))
    text = text.replace("<out_pad>", '"<out_pad>"')
    return json.loads(text)


def decode_lanes(text: str) -> list[dict[str, Any]]:
    """Decode, validate, and clean a list of predicted lane polylines."""

    payload = decode_structure(text)
    if not isinstance(payload, list):
        raise ValueError("prediction is not a JSON list")

    lanes: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        points = []
        for point in item.get("sample_points", []):
            if (
                isinstance(point, list)
                and len(point) == 2
                and all(isinstance(coord, (int, float)) for coord in point)
            ):
                points.append([int(round(point[0])), int(round(point[1]))])
        cleaned = dict(item)
        cleaned["sample_points"] = points
        if points:
            cleaned.setdefault("start_point", points[0])
            cleaned.setdefault("end_point", points[-1])
        lanes.append(cleaned)
    return lanes
