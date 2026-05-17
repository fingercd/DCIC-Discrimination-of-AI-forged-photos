# DCIC 图像伪造分析 - 训练与预测全流程

## 一、全流程概览

```
① prepare_data.py      → data/dataset.json
② prepare_vlm_data.py   → data/vlm_train.json
③ train_unet.py        → checkpoints/unet_best.pt
④ train_vlm.py         → checkpoints/qwen3.5-9b-lora/lora_adapter
⑤ inference.py         → data/inference_results.json + inference_metrics.json
⑥ generate_csv.py      → data/submission.csv
```

---

## 二、执行顺序（PyCharm 右键 Run）

| 步骤 | 脚本 | 依赖 | 输出 |
|------|------|------|------|
| 1 | `scripts/prepare_data.py` | `ForgeryAnalysis_Stage_1_Train/` | `data/dataset.json` |
| 2 | `scripts/prepare_vlm_data.py` | ① | `data/vlm_train.json` |
| 3 | `scripts/train_unet.py` | ① | `checkpoints/unet_best.pt` |
| 4 | `scripts/train_vlm.py` | ② + `models/qwen3.5-9b` | `checkpoints/qwen3.5-9b-lora/lora_adapter` |
| 5 | `scripts/inference.py` | ③ ④ | `data/inference_results.json` |
| 6 | `scripts/generate_csv.py` | ⑤ | `data/submission.csv` |

**说明**：③ 和 ④ 可并行训练；⑤ 需指定 `--vlm_model models/qwen3.5-9b`，LoRA 通过 `--lora_path checkpoints/qwen3.5-9b-lora/lora_adapter` 指定。Plan B 分割模型默认为 **DeepLabV3Plus + ResNet50**，checkpoint 内会保存 `architecture`、`encoder_name`、`size`、`threshold` 供推理使用。

---

## 三、参数调优指南

### 1. 分割模型训练 (`scripts/train_unet.py`)

当前默认：**DeepLabV3Plus + ResNet50**。可在 4090 24GB 下以 `--batch 8` 训练 512×512。

| 参数 | 默认 | 调参建议 | 说明 |
|------|------|----------|------|
| `--architecture` | deeplabv3plus | unet / deeplabv3plus | 分割架构 |
| `--encoder` | resnet50 | resnet34 / resnet50 / efficientnet-b4 等 | 编码器 |
| `--epochs` | 30 | 20~50 | 过拟合可减，欠拟合可增 |
| `--batch` | 8 | 8~16（24GB 下 DeepLabV3+ 建议 8） | 显存不足时减小 |
| `--lr` | 1e-4 | 5e-5~2e-4 | 损失震荡可降低 |
| `--size` | 512 | 384/512/640 | 大图更准但更慢 |
| `--val_ratio` | 0.1 | 0.1~0.2 | 验证集比例 |

### 2. VLM 微调 (`scripts/train_vlm.py` 顶部 CONFIG)

| 参数 | 默认 | 调参建议 | 说明 |
|------|------|----------|------|
| `quantization_bit` | 4 | 4/8/0 | 4=QLoRA 省显存，0=全精度 |
| `lora_rank` | 64 | 32~128 | 越大表达能力越强，显存越高 |
| `lora_alpha` | 128 | 64~256 | 通常为 rank 的 2 倍 |
| `per_device_train_batch_size` | 1 | 1~2 | 显存决定 |
| `gradient_accumulation_steps` | 8 | 4~16 | 有效 batch = batch × 此值 |
| `num_train_epochs` | 3 | 2~5 | 数据少可多训几轮 |
| `learning_rate` | 1e-4 | 5e-5~2e-4 | 过大易不稳定 |
| `max_length` | 8192 | 1024~8192 | 越长显存越高（当前为截断上限） |

### 3. 推理 (`scripts/inference.py`)

| 参数 | 默认 | 调参建议 | 说明 |
|------|------|----------|------|
| `--plan` | A | A/B | A=VLM+SAM2，B=VLM+分割模型（默认 DeepLabV3+） |
| `--vlm_model` | None | 必填 | 如 `models/qwen3.5-9b` |
| `--unet_ckpt` | checkpoints/unet_best.pt | 自定义 | Plan B 分割模型权重 |
| `--unet_arch` | 从 ckpt 读 | unet / deeplabv3plus | 覆盖 checkpoint 内架构（兼容旧 ckpt） |
| `--unet_encoder` | 从 ckpt 读 | resnet34 / resnet50 等 | 覆盖 checkpoint 内编码器 |
| `--unet_size` | 从 ckpt 读或 512 | 384/512 | 推理输入尺寸 |
| `--test_dir` | Test/Image | 自定义 | 测试图片目录 |

---

## 四、显存与参数参考

| 显存 | 分割模型 (DeepLabV3+ ResNet50) batch | VLM (4bit) | 建议 |
|------|--------------------------------------|------------|------|
| 12GB | 4~6 | 可训 | batch=1, grad_accum=8 |
| 24GB | 8~12 | 可训 | 默认 batch=8，显存有余可试 12 |
| 40GB+ | 16~24 | 可训 | batch=2, rank=128 |

---

## 五、推理命令示例

**Plan A（VLM + SAM2）：**
```bash
python scripts/inference.py --vlm_model models/qwen3.5-9b --lora_path checkpoints/qwen3.5-9b-lora/lora_adapter --plan A
python scripts/generate_csv.py
```

**Plan B（VLM + 分割模型，默认从 checkpoint 读架构/编码器/尺寸）：**
```bash
python scripts/train_unet.py
python scripts/inference.py --vlm_model models/qwen3.5-9b --lora_path checkpoints/qwen3.5-9b-lora/lora_adapter --plan B --unet_ckpt checkpoints/unet_best.pt
python scripts/generate_csv.py
```

若使用旧版 UNet+ResNet34 的 checkpoint，推理时需显式指定：`--unet_arch unet --unet_encoder resnet34`。
