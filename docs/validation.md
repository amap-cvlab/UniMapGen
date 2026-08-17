# v6 checkpoint-26000 validation

Validation was run on 2026-08-17 with the designated final v6 checkpoint from
experiment `0502_v6_opensatmap20_896` at training step `26000`, one NVIDIA A100
80GB GPU, Python 3.10, PyTorch 2.4.0, vLLM 0.6.1, and the six example tiles
included in this repository.

## Result

| Check | Result |
| --- | ---: |
| Unit tests | 8 passed |
| Dry-run inference | 6/6 succeeded |
| Real checkpoint inference | 6/6 succeeded |
| Structurally parsed predictions | 6/6 |
| Generation time (model already loaded) | 23.191 s |
| Average generation time | 3.865 s/tile |
| mIoU | 55.7083% |
| Chamfer AP @ 12 px | 0.416550 |
| Chamfer AP @ 16 px | 0.416550 |
| Chamfer AP @ 26 px | 0.465073 |
| Chamfer AP @ 36 px | 0.465073 |
| Visualizations | 6/6 written |

Per-class mIoU was 52.4021% for `Curb`, 83.2745% for `Laneline`, and 31.4482%
for `Virtualline`.

The validation environment did not contain the optional `torchmetrics` and
`pycocotools` packages, so mask AP was not executed there. Both packages are
pinned in `requirements.txt`; the mask AP implementation remains covered by
the same input parsing and rasterization exercised by the other metrics.

Artifacts:

- [predictions](../data/example/predictions_v6_checkpoint26000.jsonl)
- [metric JSON](../data/example/metrics_v6_checkpoint26000.json)
- [stitched-tile visualization](validation_checkpoint26000.jpg)

This six-tile sample is a pipeline check, not a benchmark result. Reproduce any
reported benchmark on the complete authorized validation split.
