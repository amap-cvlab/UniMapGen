<div align="center">
<a id="readme-top"></a>
<h1>UniMapGen: A Generative Framework<br>for Large-Scale Map Construction<br>from Multi-modal Data</h1>
<h3><strong>🎉🎉 AAAI 2026 Oral 🎉🎉</strong></h3>

<a href="https://arxiv.org/pdf/2509.22262"><img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white" alt="arXiv paper"></a>
<a href="https://amap-cvlab.github.io/UniMapGen/"><img src="https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white" alt="Project page"></a>

[Yujian Yuan](https://scholar.google.com/citations?user=FYwn1-YAAAAJ&hl=zh-CN)<sup>1,2,\*</sup>,
Changjie Wu<sup>1,\*</sup>,
[Xinyuan Chang](https://scholar.google.com.hk/citations?user=5OnPBVYAAAAJ&hl=zh-CN)<sup>1,\*</sup>,
Sijin Wang<sup>1,\*</sup>,
Hang Zhang<sup>1</sup>,
Shiyi Liang<sup>1,3</sup>,
Shuang Zeng<sup>1,3</sup>,
Mu Xu<sup>1,†</sup>

<sup>1</sup>Amap, Alibaba Group,
<sup>2</sup>The Hong Kong University of Science and Technology,<br>
<sup>3</sup>Xi'an Jiaotong University

<sup>\*Equal contribution; †Corresponding author</sup>

**UniMapGen** is a generative unified framework that autoregressively generates
**smooth and topologically consistent** vectorized maps from multi-modal inputs,
enabling scalable, occlusion-robust city-scale mapping without costly on-site
data collection.
</div>

[Chinese](README-ZH.md)

This release provides a standalone satellite-image lane-map pipeline. It encodes
each `896×896` tile as directed polylines in the `Curb`, `Laneline`, and
`Virtualline` classes and stitches large maps tile by tile using connection-point
hints from the left and upper neighbors.

## Release status

- [x] Inference code
- [x] Evaluation code
- [x] Visualization code
- [x] Data processing code
- [x] UniMapGen package code
- [ ] UniMapGen checkpoint download

## Repository layout

```text
.
├── checkpoints/                 # Checkpoint placement; weights are excluded by .gitignore
├── data/example/                # Example tiles, real predictions, and validation metrics
├── docs/                        # Validation report and example visualization
├── scripts/
│   ├── preprocess.py            # 4096×4096 images and polyline annotations → tile JSONL
│   ├── infer.py                 # vLLM inference and global stitching
│   ├── evaluate.py              # mIoU, mask AP, and Chamfer AP
│   ├── visualize.py             # Satellite/GT/prediction comparison panels
│   └── run_tests.py             # Lightweight test runner when pytest is unavailable
├── src/unimapgen/               # Reusable Python package
└── tests/                       # Codec, clipping, stitching, and metric tests
```

## 1. Install the environment

```bash
conda create -n unimapgen python=3.10 -y
conda activate unimapgen

# Install the CUDA 12.4 build of PyTorch first so later extensions can find torch.
pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
pip install flash-attn==2.6.1 --no-build-isolation
pip install -e . --no-deps
```

If you only need preprocessing, mIoU, Chamfer AP, and visualization, you may
omit `vllm`, `flash-attn`, `torchmetrics`, and `pycocotools`.

## 2. Prepare the checkpoint

Download link: coming soon.

```text
checkpoints/unimapgen-v6/
├── config.json
├── generation_config.json
├── preprocessor_config.json
├── tokenizer.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
└── model.safetensors.index.json
```

## 3. Preprocess the data

### Input format

Each source image is typically `4096×4096`. Annotations may be provided as a
JSON mapping, a JSONL file, or a directory containing one JSON file per image.
Each polyline must contain at least:

```json
{
  "category": "Lane line",
  "points": [[4095, 1483.2], [4076.8, 1418.8], [4054.4, 1350.3]]
}
```

Accepted category names are `Lane line`, `Virtual line`, and `Curb`. They are
normalized to `Laneline`, `Virtualline`, and `Curb` in the output.

Example JSON mapping:

```json
{
  "city_tile_001.png": [
    {"category": "Lane line", "points": [[0, 20], [200, 30], [900, 50]]}
  ]
}
```

### Run preprocessing

```bash
python scripts/preprocess.py \
  --images-dir /path/to/full_images \
  --annotations /path/to/annotations.json \
  --output-dir data/processed \
  --tile-size 896 \
  --step 896 \
  --sample-distance 40
```

The script clips polylines to each tile, marks `<cut_point>` and
`</cut_point>` endpoints, samples points uniformly every 40 pixels, and orders
lanes from near to far according to the distance between their starting point
and the upper-left corner. It writes:

```text
data/processed/
├── images/*.png
└── samples.jsonl
```

By default, the script does not cover the remainder when `4096` is not divisible
by `896`. Add `--cover-edge` to cover the full image with an overlapping final
window.

## 4. Run inference

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \
  --model-path checkpoints/unimapgen-v6 \
  --input-file data/example/samples.jsonl \
  --image-root data/example/images \
  --blank-image data/example/empty_img_896.jpg \
  --output-file outputs/example_v6.jsonl \
  --max-model-len 12000 \
  --max-tokens 16000
```

Important behavior:

- Tiles are sorted from left to right and then top to bottom within the same
  source map.
- Each tile reads predictions already produced for its left and upper neighbors
  and adds their boundary connection points to the prompt.
- The checkpoint stores its chat template in the tokenizer configuration. The
  script renders the Qwen2-VL image placeholder through the tokenizer, which is
  compatible with environments where the processor does not inherit that
  template.
- `--include-text-prompt` adds the Chinese neighbor-stitching text to the chat
  template. This is useful for ablation studies but is not part of the original
  v6 validation setting.
- Add `--resume` after an interruption to restore neighbor predictions from an
  existing JSONL file.
- Each output record contains `hq_pred`, machine-readable token probabilities,
  latency, and a parse-error field.
- If a Qwen2-VL configuration serializes `mrope_section` with
  `rope_type=default`, the script creates a temporary symlink-only checkpoint
  view and changes the marker to the `mrope` value expected by vLLM 0.6.1. The
  original configuration and weights are not modified.

## 5. Evaluate predictions

Compute all three metrics listed in the project handoff:

```bash
python scripts/evaluate.py \
  --input-file outputs/example_v6.jsonl \
  --metrics miou,mask_ap,chamfer_ap \
  --output outputs/example_v6_metrics.json
```

The metric definitions match the original experiment:

- `mIoU`: rasterizes polylines with a six-pixel line width and evaluates the
  three foreground classes. To reproduce the experiment, ground-truth
  background pixels are excluded.
- `mask AP`: rasterizes each polyline as an instance mask, uses its category
  token probability as confidence, and reports COCO-style mAP, AP50, and AP75.
- `Chamfer AP`: resamples each polyline to 50 points, uses bidirectional Chamfer
  distance, and evaluates thresholds of `12/16/26/36` pixels.

For lightweight evaluation only:

```bash
python scripts/evaluate.py \
  --input-file outputs/example_v6.jsonl \
  --metrics miou,chamfer_ap
```

## 6. Visualize predictions

```bash
python scripts/visualize.py \
  --input-file outputs/example_v6.jsonl \
  --image-root data/example/images \
  --output-dir visualizations/example_v6
```

Each output is a three-panel comparison containing the original satellite
image, ground truth, and prediction. Colors are fixed: red for `Curb`, green for
`Laneline`, and orange-blue for `Virtualline`. Purple and yellow markers indicate
start and end connection hints received from neighboring tiles.

## Citation

If UniMapGen is useful in your research or applications, please consider giving
the project a star and citing it with the following BibTeX entry:

```bibtex
@article{yuan2025unimapgen,
  title={UniMapGen: A Generative Framework for Large-Scale Map Construction from Multi-modal Data},
  author={Yuan, Yujian and Wu, Changjie and Chang, Xinyuan and Wang, Sijin and Zhang, Hang and Liang, Shiyi and Zeng, Shuang and Xu, Mu},
  journal={arXiv preprint arXiv:2509.22262},
  year={2025}
}
```

<p align="right"><a href="#readme-top"><img src="https://img.shields.io/badge/back%20to%20top-red?style=flat" alt="Back to top"></a></p>

## Acknowledgements

Our work is primarily based on
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and the dataset comes
from [OpenSatMap](https://github.com/bjzhb666/OpenSatMap-offical). We sincerely
thank the authors for their work.

The code is released under the [Apache License 2.0](LICENSE). The dependencies
and model weights remain subject to their respective licenses. Including example
data does not alter the licensing terms of the original dataset.

<p align="right"><a href="#readme-top"><img src="https://img.shields.io/badge/back%20to%20top-red?style=flat" alt="Back to top"></a></p>
