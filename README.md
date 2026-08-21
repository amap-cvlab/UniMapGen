# UniMapGen

UniMapGen 从卫星图像生成车道级矢量地图。模型把每个 `896×896` 图块编码为一组有向折线，支持 `Curb`、`Laneline` 和 `Virtualline` 三类，并通过左侧/上侧图块的连接点提示完成大图逐块拼接。

本仓库是面向开源发布重新整理的独立工程，只包含数据预处理、推理、评测、可视化和小型示例数据
## 目录

```text
.
├── checkpoints/                 # 权重放置说明，权重本身被 .gitignore 排除
├── data/example/                # 示例图块、真实预测和验证指标
├── docs/                        # 验证报告和示例可视化
├── scripts/
│   ├── preprocess.py            # 4096 大图与折线标注 → 图块 JSONL
│   ├── infer.py                 # vLLM 推理与全局拼接
│   ├── evaluate.py              # mIoU、mask AP、Chamfer AP
│   ├── visualize.py             # 卫星图/GT/预测三联图
│   ├── run_tests.py             # 无 pytest 时的轻量测试入口
├── src/unimapgen/               # 可复用 Python 包
└── tests/                       # 格式、裁剪、拼接和指标测试
```

## 1. 安装环境


```bash
conda create -n unimapgen python=3.10 -y
conda activate unimapgen

# 先安装 CUDA 12.4 版 PyTorch，确保后续扩展能找到 torch。
pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
pip install flash-attn==2.6.1 --no-build-isolation
pip install -e . --no-deps
```

若只运行预处理、mIoU、Chamfer AP 和可视化，可不安装 `vllm`、`flash-attn`、`torchmetrics` 与 `pycocotools`。

## 2. 准备 checkpoint

下载地址 comming soon

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

## 3. 数据预处理

### 输入格式

每张原图通常为 `4096×4096`。标注可以是一个 JSON 映射、JSONL 文件，或每图一个 JSON 的目录。单条折线至少包含：

```json
{
  "category": "Lane line",
  "points": [[4095, 1483.2], [4076.8, 1418.8], [4054.4, 1350.3]]
}
```

类别名接受 `Lane line`、`Virtual line`、`Curb`，输出统一为 `Laneline`、`Virtualline`、`Curb`。

JSON 映射示例：

```json
{
  "city_tile_001.png": [
    {"category": "Lane line", "points": [[0, 20], [200, 30], [900, 50]]}
  ]
}
```

### 运行

```bash
python scripts/preprocess.py \
  --images-dir /path/to/full_images \
  --annotations /path/to/annotations.json \
  --output-dir data/processed \
  --tile-size 896 \
  --step 896 \
  --sample-distance 40
```

脚本会裁剪折线、标记 `<cut_point>`/`</cut_point>`、每 40 像素均匀采样、按起点距左上角由近到远排序，并生成：

```text
data/processed/
├── images/*.png
└── samples.jsonl
```

默认不补齐 `4096` 无法被 `896` 整除后剩余的边缘。若希望覆盖完整图像，增加 `--cover-edge`；这会产生一个重叠的末端窗口。


## 5. 推理

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

重要行为：

- 图块会按“同一大图内从左到右、从上到下”排序。
- 当前图块会读取已经生成的左侧和上侧预测，把边界连接点写入提示。
- checkpoint 的 chat template 保存在 tokenizer 配置中；脚本通过 tokenizer 渲染 Qwen2-VL 图像占位符，兼容该环境下 processor 不继承模板的情况。
- `--include-text-prompt` 会把中文连接提示也写进 chat template，适合做消融，但不属于原始 v6 验证设置。
- 中断后可加 `--resume`，脚本会从已有 JSONL 恢复邻居预测。
- 每条输出保留 `hq_pred`、可解析的 token 概率、耗时和解析错误字段。
- 若 Qwen2-VL 配置把 `mrope_section` 写成 `rope_type=default`，脚本会在系统临时目录创建只含软链接的 checkpoint 视图，并把标记改为 vLLM 0.6.1 需要的 `mrope`；原权重和原配置不会被修改。

## 6. 评测

一次计算交接文档列出的三套指标：

```bash
python scripts/evaluate.py \
  --input-file outputs/example_v6.jsonl \
  --metrics miou,mask_ap,chamfer_ap \
  --output outputs/example_v6_metrics.json
```

指标定义与原实验保持一致：

- `mIoU`：将折线按 6 像素线宽栅格化，统计三个前景类别；为复现实验，GT 背景像素不参与统计。
- `mask AP`：把每条折线栅格化为实例 mask，以类别 token 概率作为置信度，报告 COCO 风格 mAP、AP50 和 AP75。
- `Chamfer AP`：每条折线重采样 50 点，使用双向 Chamfer 距离，阈值为 `12/16/26/36` 像素。

若只需轻量指标，可运行：

```bash
python scripts/evaluate.py \
  --input-file outputs/example_v6.jsonl \
  --metrics miou,chamfer_ap
```

## 7. 可视化

```bash
python scripts/visualize.py \
  --input-file outputs/example_v6.jsonl \
  --image-root data/example/images \
  --output-dir visualizations/example_v6
```

每个输出为三联图：原卫星图、GT、预测。颜色固定为：红色 `Curb`、绿色 `Laneline`、橙蓝色 `Virtualline`；紫色和黄色圆点分别表示来自相邻图块的开始/结束连接提示。


## 许可证与致谢

代码按 [Apache License 2.0](LICENSE) 发布。模型基于 Qwen2-VL，并沿用了 LLaMA-Factory 与 vLLM 的训练/推理生态；这些依赖和模型权重分别受其自身许可证约束。示例数据不改变原数据集的许可要求。
