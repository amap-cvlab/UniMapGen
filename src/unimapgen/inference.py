"""Sequential vLLM inference with UniMapGen v6 global stitching prompts."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from .codec import decode_lanes, encode_lanes
from .prompts import build_stitching_prompt, parse_tile_id, sort_key


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_image(image: str, image_root: Path) -> Path:
    path = Path(image)
    if not path.is_absolute():
        path = image_root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _extract_ground_truth(record: dict[str, Any]) -> list[dict]:
    messages = record.get("messages", [])
    if not messages:
        return []
    content = messages[0].get("content", "[]")
    if isinstance(content, list):
        content = next(
            (item.get("text", "[]") for item in content if item.get("type") == "text"),
            "[]",
        )
    payload = json.loads(content)
    if not isinstance(payload, list):
        raise ValueError("the first message content must encode a lane list")
    return payload


def _token_logprob_records(generation) -> tuple[list[dict[str, Any]], list[str]]:
    structured: list[dict[str, Any]] = []
    legacy: list[str] = []
    for token_id, alternatives in zip(generation.token_ids, generation.logprobs or []):
        chosen = alternatives.get(token_id) if hasattr(alternatives, "get") else None
        if chosen is None:
            chosen = next(iter(alternatives.values()))
        probability = math.exp(float(chosen.logprob))
        decoded = chosen.decoded_token
        structured.append(
            {"token_id": int(token_id), "probability": probability, "token": decoded}
        )
        legacy.append(
            f"Token ID: {token_id}, Log Probability: {probability}, decoded_token: {decoded}"
        )
    return structured, legacy


def normalize_vllm_config(config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Normalize the Qwen2-VL mRoPE marker expected by vLLM 0.6.1.

    Some Transformers versions serialize Qwen2-VL's ``mrope_section`` with
    ``rope_type=default``.  vLLM 0.6.1 then treats it as conventional RoPE and
    incorrectly requires a ``factor``.  The model architecture is mRoPE, so the
    compatibility-safe marker is ``mrope``.  No numeric model value is changed.
    """

    normalized = json.loads(json.dumps(config))
    rope_scaling = normalized.get("rope_scaling")
    if not isinstance(rope_scaling, dict) or "mrope_section" not in rope_scaling:
        return normalized, False
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if rope_type != "default":
        return normalized, False
    rope_scaling["rope_type"] = "mrope"
    rope_scaling["type"] = "mrope"
    return normalized, True


def stage_vllm_checkpoint(model_path: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Create an ephemeral symlink view when the mRoPE marker needs repair."""

    config_path = model_path / "config.json"
    if not config_path.is_file():
        return model_path, None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    normalized, changed = normalize_vllm_config(config)
    if not changed:
        return model_path, None

    temporary = tempfile.TemporaryDirectory(prefix="unimapgen-vllm-")
    staged_path = Path(temporary.name)
    for source in model_path.iterdir():
        if source.name == "config.json":
            continue
        os.symlink(source.resolve(), staged_path / source.name, target_is_directory=source.is_dir())
    (staged_path / "config.json").write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    warnings.warn(
        "staged a temporary checkpoint view with rope_type=mrope for vLLM 0.6.1",
        RuntimeWarning,
        stacklevel=2,
    )
    return staged_path, temporary


class VLLMEngine:
    """Thin compatibility wrapper around the original vLLM 0.6.1 runtime."""

    def __init__(
        self,
        model_path: Path,
        *,
        max_model_len: int = 12000,
        max_tokens: int = 16000,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        legacy_image_only_prompt: bool = True,
    ) -> None:
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        staged_model_path, self._temporary_model_dir = stage_vllm_checkpoint(model_path)
        self.processor = AutoProcessor.from_pretrained(str(staged_model_path))
        self.model = LLM(
            model=str(staged_model_path),
            max_model_len=max_model_len,
            max_num_seqs=1,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_tokens,
            stop_token_ids=[],
            skip_special_tokens=False,
            logprobs=1,
        )
        self.legacy_image_only_prompt = legacy_image_only_prompt

    def generate(
        self, image: Path, prompt: str, auxiliary_images: Iterable[Path]
    ) -> tuple[str, list[dict[str, Any]], list[str]]:
        from qwen_vl_utils import process_vision_info

        # The checkpoint stores a tokenizer-level, string-content chat
        # template.  AutoProcessor.apply_chat_template() does not inherit it
        # in the validated Transformers build, so render through the tokenizer
        # and provide Qwen2-VL's canonical image placeholder explicitly.
        template_content = "<|vision_start|><|image_pad|><|vision_end|>"
        if not self.legacy_image_only_prompt:
            template_content += prompt.removeprefix("<image>")
        template_messages = [{"role": "user", "content": template_content}]
        text = self.processor.tokenizer.apply_chat_template(
            template_messages, tokenize=False, add_generation_prompt=True
        )

        # The released v6 checkpoint was evaluated with one satellite image and
        # two black auxiliary slots. Preserve that tensor layout by default.
        vision_content = [{"type": "image", "image": str(image)}]
        vision_content.extend(
            {"type": "image", "image": str(path)} for path in auxiliary_images
        )
        image_inputs, _ = process_vision_info(
            [{"role": "user", "content": vision_content}]
        )
        inputs = {
            "prompt": text,
            "multi_modal_data": {"image": [image_inputs]},
        }
        outputs = self.model.generate(inputs, self.sampling_params, use_tqdm=False)
        generation = outputs[0].outputs[0]
        structured, legacy = _token_logprob_records(generation)
        return generation.text, structured, legacy


def run_inference(
    input_file: Path,
    image_root: Path,
    output_file: Path,
    *,
    engine: VLLMEngine | None,
    blank_image: Path,
    tile_size: int = 896,
    limit: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    records = read_jsonl(input_file)
    records.sort(key=lambda record: sort_key(record["images"][0]))
    if limit is not None:
        records = records[:limit]

    predictions: dict[tuple[str, int, int], list[dict]] = {}
    completed_images: set[str] = set()
    if resume and output_file.exists():
        for previous in read_jsonl(output_file):
            image_name = previous["image"]
            completed_images.add(image_name)
            try:
                predictions[parse_tile_id(image_name).key] = decode_lanes(
                    previous["hq_pred"]
                )
            except Exception:
                predictions[parse_tile_id(image_name).key] = []
    elif output_file.exists():
        output_file.unlink()

    total_seconds = 0.0
    succeeded = 0
    failed = 0
    for record in tqdm(records, desc="UniMapGen inference"):
        image_name = record["images"][0]
        if image_name in completed_images:
            continue
        tile = parse_tile_id(image_name)
        left = predictions.get((tile.prefix, tile.x0 - tile.width, tile.y0))
        up = predictions.get((tile.prefix, tile.x0, tile.y0 - tile.height))
        prompt = build_stitching_prompt(left, up, tile_size=tile_size)
        image_path = resolve_image(image_name, image_root)
        ground_truth = _extract_ground_truth(record)

        start = time.perf_counter()
        try:
            if engine is None:
                prediction = encode_lanes(ground_truth)
                token_logprobs: list[dict[str, Any]] = []
                legacy_probs: list[str] = []
            else:
                prediction, token_logprobs, legacy_probs = engine.generate(
                    image_path, prompt, [blank_image, blank_image]
                )
            elapsed = time.perf_counter() - start
            total_seconds += elapsed
            try:
                parsed_prediction = decode_lanes(prediction)
                parse_error = None
            except Exception as error:
                parsed_prediction = []
                parse_error = str(error)
            predictions[tile.key] = parsed_prediction
            append_jsonl(
                output_file,
                {
                    "prompt": prompt,
                    "hq_gt": encode_lanes(ground_truth),
                    "image": image_name,
                    "hq_pred": prediction,
                    "token_logprobs": token_logprobs,
                    "pred_probs": json.dumps(legacy_probs, ensure_ascii=False),
                    "latency_seconds": elapsed,
                    "parse_error": parse_error,
                },
            )
            succeeded += 1
        except Exception as error:
            failed += 1
            append_jsonl(
                output_file,
                {
                    "prompt": prompt,
                    "hq_gt": encode_lanes(ground_truth),
                    "image": image_name,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
    return {
        "requested": len(records),
        "succeeded": succeeded,
        "failed": failed,
        "total_seconds": total_seconds,
        "average_seconds": total_seconds / succeeded if succeeded else None,
        "output_file": str(output_file),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument(
        "--blank-image",
        type=Path,
        default=Path("data/example/empty_img_896.jpg"),
    )
    parser.add_argument("--max-model-len", type=int, default=12000)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tile-size", type=int, default=896)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--include-text-prompt",
        action="store_true",
        help="Include the Chinese stitching text in the chat template. The v6 legacy run used image-only template text.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip model loading and copy GT to predictions to test the pipeline.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.dry_run and args.model_path is None:
        raise SystemExit("--model-path is required unless --dry-run is set")
    if not args.blank_image.is_file():
        raise SystemExit(f"blank image not found: {args.blank_image}")

    engine = None
    if not args.dry_run:
        engine = VLLMEngine(
            args.model_path,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            legacy_image_only_prompt=not args.include_text_prompt,
        )
    summary = run_inference(
        args.input_file,
        args.image_root,
        args.output_file,
        engine=engine,
        blank_image=args.blank_image.resolve(),
        tile_size=args.tile_size,
        limit=args.limit,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
