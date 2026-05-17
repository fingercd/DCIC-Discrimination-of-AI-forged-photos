"""
Fine-tune Qwen3.5-VL with LoRA (4-bit QLoRA).
Uses data/vlm_train.json. Saves LoRA adapter to checkpoints/qwen3.5-9b-lora/lora_adapter.
Run prepare_vlm_data.py first.
Requires: pip install bitsandbytes
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ============ 可在此修改参数 ============
# 不压缩时与 inference 的 NO_COMPRESS_MAX_PIXELS 一致，避免训练时缩小原图
NO_COMPRESS_MAX_PIXELS = 8192 * 8192
# 轻压缩：总像素上限 640×640，与 prepare_vlm_data --light-compress 一致
LIGHT_MAX_PIXELS = 960 * 960

CONFIG = {
    "model_path": str(ROOT / "models" / "qwen3.5-9b"),
    "data_path": str(ROOT / "data" / "vlm_train.json"),
    "output_dir": str(ROOT / "checkpoints" / "qwen3.5-9b-lora"),
    "compress": "light",  # 轻压缩 640²，与 prepare_vlm_data --light-compress 一致
    "quantization_bit": 4,
    "lora_rank": 64,
    "lora_alpha": 128,
    "per_device_train_batch_size": 3,
    "gradient_accumulation_steps": 1,
    "num_train_epochs": 3,
    "learning_rate": 1e-4,
    "warmup_ratio": 0.1,
    # 仅作截断上限（当前用动态长度时不强制 pad）；若改回固定长度，此处为 pad/truncate 目标
    "max_length":  8192,
    "logging_steps": 1,
    "save_steps": 100,
    "save_total_limit": 3,
    "dataloader_num_workers": 16,
}
# =======================================


class VLMDataset(Dataset):
    """Dataset from vlm_train.json (sharegpt format)."""

    def __init__(self, data_path: str, processor):
        self.processor = processor
        with open(data_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        r = self.records[idx]
        convs = r["conversations"]
        images = r.get("images", [])
        if not images:
            for c in convs:
                if c.get("from") == "human" and c.get("image"):
                    images.append(c["image"])
                    break
        image_path = images[0] if images else None

        messages = []
        for c in convs:
            role = "user" if c["from"] == "human" else "assistant"
            content = c["value"]
            if c.get("image"):
                text_part = content.replace("<image>\n", "").strip()
                messages.append({
                    "role": role,
                    "content": [{"type": "image", "image": c["image"]}, {"type": "text", "text": text_part}],
                })
            else:
                messages.append({"role": role, "content": content})

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        image_list = [Image.open(image_path).convert("RGB")] if image_path and Path(image_path).exists() else None
        # 不强制 pad 到固定长度：按实际 token 长度，在 collate 时再按 batch 内最大长度 pad
        inputs = self.processor(
            text=[text],
            images=image_list,
            return_tensors="pt",
            padding=False,
            truncation=False,
            return_attention_mask=True,
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        inputs["labels"][inputs["labels"] == pad_id] = -100
        # 只对 assistant 回复算 loss：用户消息+图像+assistant 前缀（<think>\n\n</think>\n\n）全部 mask
        prefix_messages = [m for m in messages if m["role"] == "user"]
        if prefix_messages:
            text_prefix = self.processor.apply_chat_template(
                prefix_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prefix_inputs = self.processor(
                text=[text_prefix],
                images=image_list,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )
            prefix_len = prefix_inputs["input_ids"].size(1)
            if prefix_len < inputs["input_ids"].size(0):
                inputs["labels"][:prefix_len] = -100
        inputs["pad_token_id"] = pad_id
        return inputs


def collate_fn(examples):
    pad_id = examples[0]["pad_token_id"]
    max_len = max(e["input_ids"].size(0) for e in examples)

    def pad_to_max(tensor, fill, length):
        if tensor.size(0) == length:
            return tensor
        out = torch.full((length,), fill, dtype=tensor.dtype, device=tensor.device)
        out[: tensor.size(0)] = tensor
        return out

    batch = {
        "input_ids": torch.stack([pad_to_max(e["input_ids"], pad_id, max_len) for e in examples]),
        "attention_mask": torch.stack([pad_to_max(e["attention_mask"], 0, max_len) for e in examples]),
        "labels": torch.stack([pad_to_max(e["labels"], -100, max_len) for e in examples]),
    }
    if "pixel_values" in examples[0]:
        # Qwen3.5-VL processor returns variable-length vision tokens per image as (N, D).
        # The model expects them concatenated across the batch, with `image_grid_thw` providing
        # per-image grid sizes to split internally.
        batch["pixel_values"] = torch.cat([e["pixel_values"] for e in examples], dim=0)
    if "image_grid_thw" in examples[0]:
        # Keep per-image grid metadata (B, 3). In __getitem__ we squeeze(0),
        # so each example typically has shape (3,). Stack to (B, 3).
        grids = [e["image_grid_thw"] for e in examples]
        batch["image_grid_thw"] = torch.stack(grids, dim=0) if grids[0].dim() == 1 else torch.cat(grids, dim=0)
    return batch


def main() -> None:
    delay_seconds = 0 * 3600
    print(f"[Delay] 等待 {delay_seconds / 3600:.1f} 小时后开始训练…", flush=True)
    time.sleep(delay_seconds)
    print("[Delay] 等待结束，开始训练。", flush=True)

    from transformers import AutoProcessor, TrainingArguments, Trainer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType

    try:
        from transformers import Qwen3_5ForConditionalGeneration as ModelClass
    except ImportError:
        from transformers import AutoModelForVision2Seq as ModelClass

    model_path = CONFIG["model_path"]
    data_path = CONFIG["data_path"]
    output_dir = CONFIG["output_dir"]

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Run prepare_vlm_data.py first. Missing: {data_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    quantization_config = None
    if CONFIG["quantization_bit"] == 4:
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except ImportError:
            print("Warning: bitsandbytes not installed. Install with: pip install bitsandbytes")

    # 是否压缩：False=不压缩，True=强压缩(1280×28²)，"light"=轻压缩(640²)，与 prepare_vlm_data 一致
    compress = CONFIG.get("compress", False)
    if compress is True:
        min_pixels, max_pixels = 256 * 28 * 28, 1280 * 28 * 28
        print("图像: 强压缩(1280×28²)", flush=True)
    elif compress == "light":
        min_pixels, max_pixels = 256 * 28 * 28, LIGHT_MAX_PIXELS
        print("图像: 轻压缩(640²)", flush=True)
    else:
        min_pixels, max_pixels = 28 * 28, NO_COMPRESS_MAX_PIXELS
        print("图像: 不压缩", flush=True)
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    model = ModelClass.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    if quantization_config is None:
        model = model.to(device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = VLMDataset(data_path, processor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        num_train_epochs=CONFIG["num_train_epochs"],
        learning_rate=CONFIG["learning_rate"],
        warmup_ratio=CONFIG["warmup_ratio"],
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        save_total_limit=CONFIG["save_total_limit"],
        remove_unused_columns=False,
        dataloader_num_workers=CONFIG.get("dataloader_num_workers", 0),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    trainer.train()
    lora_save_dir = Path(output_dir) / "lora_adapter"
    lora_save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(lora_save_dir)
    processor.save_pretrained(lora_save_dir)
    print(f"LoRA adapter saved to {lora_save_dir}")


if __name__ == "__main__":
    main()
