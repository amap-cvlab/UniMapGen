# Example inference data

This directory contains six `896×896` tiles from one validation map. The
records are ordered left-to-right and then top-to-bottom so they exercise the
v6 neighbor-prompt path as well as single-tile inference.

- `samples.jsonl`: Qwen/LLaMA-Factory-style records with lane GT.
- `images/`: the six referenced satellite tiles.
- `empty_img_896.jpg`: the black auxiliary image used by the legacy v6 input
  layout.
- `predictions_v6_checkpoint26000.jsonl`: real predictions from the designated final v6
  checkpoint, including token probabilities and timing.
- `metrics_v6_checkpoint26000.json`: mIoU and Chamfer AP for those predictions.

The sample is intentionally small. It is useful for smoke tests, not for
reporting benchmark accuracy.
