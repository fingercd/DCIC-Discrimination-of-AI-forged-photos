"""
仅对「问题图片」重跑 VLM + location，并将结果精准覆盖回原 inference_results.json。
重跑范围 = 你维护的列表文件（--images）中的 image_name ∪ 自动识别为「解释全是坐标乱码」的条目。
其他类型错误请加入列表文件。
运行后先等待 4 小时再执行推理与写回。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from src.utils import load_image, mask_to_rle, parse_vlm_output
import inference as inference_module

# 重跑专用：自然语言分析即可，不必 [Detection]/[Explanation] 格式，label 由 parse_vlm_output 从正文推断
RERUN_USER_PROMPT = (
    "请分析这张场景文本图像是否存在伪造。"
    "用自然语言、有逻辑地说明是否存在异常、哪些位置有问题以及原因。"
    "只需给出你的分析结论和解释即可，不必使用 [Detection] 或 [Explanation] 等格式标记。"
)
RERUN_MAX_NEW_TOKENS = 1024


def run_rerun_vlm_inference(
    image_paths: list[Path],
    model_path: str,
    device: str,
    lora_path: str | None = None,
    batch_size: int = 1,
    callback=None,
    no_compress: bool = True,
    light_compress: bool = False,
) -> list[dict]:
    """重跑专用 VLM 推理：使用 RERUN_USER_PROMPT 与 RERUN_MAX_NEW_TOKENS，不修改 inference.py。"""
    try:
        from transformers import AutoProcessor
        from peft import PeftModel
        try:
            from transformers import Qwen3_5ForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as ModelClass
    except ImportError:
        raise ImportError("For inline VLM run: pip install transformers peft")

    lora_dir = Path(lora_path) if lora_path else (ROOT / "checkpoints" / "qwen3.5-9b-lora" / "lora_adapter")
    MIN_P = getattr(inference_module, "MIN_PIXELS", 256 * 28 * 28)
    MAX_P = getattr(inference_module, "MAX_PIXELS", 1280 * 28 * 28)
    LIGHT_MAX = getattr(inference_module, "LIGHT_MAX_PIXELS", 1280 * 1280)
    NO_COMPRESS_MAX = getattr(inference_module, "NO_COMPRESS_MAX_PIXELS", 8192 * 8192)
    _min = 28 * 28 if no_compress else MIN_P
    _max = NO_COMPRESS_MAX if no_compress else (LIGHT_MAX if light_compress else MAX_P)
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, min_pixels=_min, max_pixels=_max
    )
    img_proc = getattr(processor, "image_processor", None)
    if img_proc is not None:
        if no_compress:
            setattr(img_proc, "min_pixels", 28 * 28)
            setattr(img_proc, "max_pixels", NO_COMPRESS_MAX)
        elif light_compress:
            setattr(img_proc, "min_pixels", MIN_P)
            setattr(img_proc, "max_pixels", LIGHT_MAX)
        else:
            setattr(img_proc, "min_pixels", MIN_P)
            setattr(img_proc, "max_pixels", MAX_P)
    if batch_size > 1:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    model = ModelClass.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    if lora_dir.exists():
        model = PeftModel.from_pretrained(model, str(lora_dir))
    model.eval()

    results = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        t0_batch = time.perf_counter()
        images = [load_image(p) for p in batch_paths]
        messages_one = [
            {"role": "user", "content": [{"type": "image", "image": images[0]}, {"type": "text", "text": RERUN_USER_PROMPT}]}
        ]
        text_one = processor.apply_chat_template(
            messages_one, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        texts = [text_one] * len(images)
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=RERUN_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        padded_input_len = inputs["input_ids"].shape[1]
        batch_results = []
        for i in range(out.size(0)):
            gen_ids = out[i, padded_input_len:].cpu()
            response = processor.decode(gen_ids, skip_special_tokens=True)
            parsed = parse_vlm_output(response)
            batch_results.append(parsed)
        elapsed_batch = time.perf_counter() - t0_batch
        elapsed_per = elapsed_batch / len(batch_paths)
        for p, parsed in zip(batch_paths, batch_results):
            if callback:
                callback(p, parsed, elapsed_per)
            results.append(parsed)
        del inputs, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


# 与 inference 一致
CONFIG = {
    "input": str(ROOT / "data" / "inference_results.json"),
    "output": None,  # 默认与 input 相同
    "test_dir": str(ROOT / "ForgeryAnalysis_Stage_1_Test" / "Image"),
    "plan": "B",
    "vlm_model": str(ROOT / "models" / "qwen3.5-9b"),
    "lora_path": str(ROOT / "checkpoints" / "qwen3.5-9b-lora" / "lora_adapter"),
    "unet_ckpt": str(ROOT / "checkpoints" / "unet_best.pt"),
    "unet_arch": None,
    "unet_encoder": None,
    "unet_size": None,
    "sam2_model": "facebook/sam2.1-hiera-large",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "vlm_no_compress": False,
    "vlm_light_compress": True,
}


def load_rerun_list(images_path: str | Path | None) -> set[str]:
    """从 txt（每行一个 image_name）或 JSON 数组文件读取要重跑的 image_name 集合。空行、# 开头行忽略。"""
    if not images_path:
        return set()
    path = Path(images_path)
    if not path.exists():
        return set()
    names = set()
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for x in data:
                if isinstance(x, str) and x.strip():
                    names.add(x.strip())
        return names
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.add(line)
    return names


def is_explanation_gibberish(record: dict) -> bool:
    """
    判定 explanation 是否「全是坐标乱码」：以 [Detection]/[Bboxes] 为主、无正常说明文字。
    满足其一即视为乱码：
    - explanation 以 [Detection] 或 [Bboxes] 开头且整段中没有 [Explanation]；
    - 或含有 [Explanation] 但其后内容极短或绝大部分为数字/逗号/方括号。
    """
    expl = (record.get("explanation") or "").strip()
    if not expl:
        return False
    expl_lower = expl.lower()
    # 以 [Detection] 或 [Bboxes] 开头且没有 [Explanation] -> 整段被当 fallback
    if expl_lower.startswith("[detection]") or expl_lower.startswith("[bboxes]"):
        if "[explanation]" not in expl_lower:
            return True
    # 有 [Explanation] 时，取其后内容，若极短或绝大部分为数字/逗号/方括号则视为乱码
    if "[explanation]" in expl_lower:
        idx = expl_lower.index("[explanation]")
        after_start = idx + len("[explanation]")
        after = expl[after_start:].strip()
        if len(after) <= 20:
            return True
        digit_bracket = sum(1 for c in after if c in "0123456789[], \n\t")
        if len(after) > 0 and (digit_bracket / len(after)) >= 0.8:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="对问题图片重跑 VLM+location 并精准覆盖回 JSON")
    parser.add_argument("--input", type=str, default=CONFIG["input"], help="现有推理结果 JSON")
    parser.add_argument("--output", type=str, default=None, help="写回路径，默认与 --input 相同")
    parser.add_argument("--test_dir", type=str, default=CONFIG["test_dir"], help="测试图片目录")
    parser.add_argument("--images", type=str, default=None, help="问题图片列表：txt 每行一个 image_name，或 JSON 数组；与自动识别结果取并集")
    parser.add_argument("--plan", type=str, choices=["A", "B"], default=CONFIG["plan"])
    parser.add_argument("--vlm_model", type=str, default=CONFIG["vlm_model"])
    parser.add_argument("--lora_path", type=str, default=CONFIG["lora_path"])
    parser.add_argument("--unet_ckpt", type=str, default=CONFIG["unet_ckpt"])
    parser.add_argument("--unet_arch", type=str, default=CONFIG["unet_arch"])
    parser.add_argument("--unet_encoder", type=str, default=CONFIG["unet_encoder"])
    parser.add_argument("--unet_size", type=int, default=CONFIG["unet_size"])
    parser.add_argument("--sam2_model", type=str, default=CONFIG["sam2_model"])
    parser.add_argument("--device", type=str, default=CONFIG["device"])
    parser.add_argument("--vlm_no_compress", action="store_true", default=CONFIG["vlm_no_compress"])
    parser.add_argument("--vlm_compress", action="store_false", dest="vlm_no_compress")
    parser.add_argument("--vlm_light_compress", action="store_true", default=CONFIG["vlm_light_compress"])
    parser.add_argument("--explanation_print_len", type=int, default=80)
    args = parser.parse_args()

    delay_seconds = 4 * 3600
    print(f"[Delay] 等待 {delay_seconds / 3600:.1f} 小时后开始重跑推理…", flush=True)
    time.sleep(delay_seconds)
    print("[Delay] 等待结束，开始执行。", flush=True)

    output_path = args.output if args.output is not None else args.input
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"test_dir not found: {test_dir}")

    with open(args.input, "r", encoding="utf-8") as f:
        full_list = json.load(f)
    if not isinstance(full_list, list):
        raise ValueError("JSON root must be a list")

    # 要重跑的 image_name = 列表文件 ∪ 自动识别（解释全是坐标乱码）
    from_list = load_rerun_list(args.images)
    auto_detect = {r["image_name"] for r in full_list
                   if isinstance(r, dict) and "image_name" in r and "explanation" in r
                   and is_explanation_gibberish(r)}
    rerun_names = from_list | auto_detect
    print(f"[Rerun] 列表文件: {len(from_list)} 条，自动识别(解释乱码): {len(auto_detect)} 条，并集: {len(rerun_names)} 条", flush=True)
    if not rerun_names:
        print("无待重跑图片，退出。")
        return

    # 在 test_dir 下找到对应路径（与 inference 一致：.jpg/.jpeg/.png）
    all_files = {p.name: p for p in test_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    rerun_paths = []
    for name in sorted(rerun_names):
        if name in all_files:
            rerun_paths.append(all_files[name])
        else:
            print(f"  [Skip] 列表中但 test_dir 无此文件: {name}", flush=True)
    if not rerun_paths:
        print("无有效图片路径，退出。")
        return
    print(f"待重跑 {len(rerun_paths)} 张: {[p.name for p in rerun_paths[:5]]}{' ...' if len(rerun_paths) > 5 else ''}", flush=True)

    # 与 inference 一致的 VLM 推理
    def print_cb(path: Path, result: dict, elapsed: float) -> None:
        label = int(result.get("label", 0))
        expl = (result.get("explanation") or "").strip().replace("\n", " ")[: args.explanation_print_len]
        if len((result.get("explanation") or "")) > args.explanation_print_len:
            expl += "…"
        print(f"  [{path.name}] label={label} 用时={elapsed:.1f}s  expl={expl!r}", flush=True)

    vlm_batch = 1 if args.vlm_no_compress else 1
    print("  [VLM] 加载模型并推理（重跑专用 prompt + max_new_tokens=%d）…" % RERUN_MAX_NEW_TOKENS, flush=True)
    vlm_results = run_rerun_vlm_inference(
        rerun_paths,
        args.vlm_model,
        args.device,
        args.lora_path,
        batch_size=vlm_batch,
        callback=print_cb,
        no_compress=args.vlm_no_compress,
        light_compress=args.vlm_light_compress and not args.vlm_no_compress,
    )
    vlm_by_name = {rerun_paths[i].name: vlm_results[i] for i in range(len(rerun_paths))}

    # 按 Plan A 或 B 生成 location，组装为与 inference 相同的单条结构
    run_vlm_result = lambda name: vlm_by_name.get(name, {"label": 0, "bboxes": [], "explanation": ""})
    if args.plan == "A":
        for p in rerun_paths:
            r = run_vlm_result(p.name)
            label = int(r.get("label", 0))
            bboxes = r.get("bboxes") or []
            explanation = r.get("explanation") or ""
            if label == 0 or not bboxes:
                location = ""
            else:
                img = load_image(p)
                mask = inference_module.sam2_bboxes_to_mask(img, bboxes, args.device, args.sam2_model)
                location = mask_to_rle(mask)
            vlm_by_name[p.name] = {"image_name": p.name, "label": label, "location": location, "explanation": explanation}
    else:
        masks = inference_module.unet_predict_masks(
            rerun_paths,
            args.unet_ckpt,
            args.device,
            architecture=args.unet_arch,
            encoder_name=args.unet_encoder,
            size=args.unet_size,
        )
        for i, p in enumerate(rerun_paths):
            r = run_vlm_result(p.name)
            label = int(r.get("label", 0))
            explanation = r.get("explanation") or ""
            mask = masks[i]
            if label == 0:
                mask = np.zeros_like(mask)
            location = "" if label == 0 else mask_to_rle(mask)
            vlm_by_name[p.name] = {"image_name": p.name, "label": label, "location": location, "explanation": explanation}

    # 按 image_name 精准覆盖回 full_list
    index_by_name = {}
    for idx, r in enumerate(full_list):
        if isinstance(r, dict) and "image_name" in r:
            index_by_name[r["image_name"]] = idx
    for name, rec in vlm_by_name.items():
        if name in index_by_name:
            full_list[index_by_name[name]] = rec
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_list, f, ensure_ascii=False, indent=2)
    print(f"已写回 {output_path}，共覆盖 {len(vlm_by_name)} 条。")


if __name__ == "__main__":
    main()
