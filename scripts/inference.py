"""
Unified inference: Plan A (VLM + SAM2) or Plan B (VLM + 分割模型).
使用已训好的 VLM LoRA + 分割 checkpoint 对测试集推理，产出 generate_csv 所需的 per-image 结果。
Supports --vlm_results_json (precomputed) or --vlm_model (run VLM inline).
Saves metrics (counts, timing, label distribution) to JSON.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_image, mask_to_rle, parse_vlm_output

# ============ 可在此修改默认参数（PyCharm 直接运行或命令行覆盖） ============
# 与 prepare_vlm_data / train_vlm 一致（压缩模式）
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
LIGHT_MAX_PIXELS = 1280 * 1280
NO_COMPRESS_MAX_PIXELS = 8192 * 8192
USER_PROMPT = (
    "请分析这张场景文本图像是否存在伪造。\n"
    "请按以下格式输出：\n"
    "第一行：[Detection] 0 或 1（0=真实，1=伪造）\n"
    "第二行起：[Explanation] 用自然语言、有逻辑地说明是否存在异常、哪些位置有问题以及原因。"
    "若需指出具体区域，可在说明中按需写出坐标或位置描述，不必单独列出 [Bboxes] 行。严禁整段只堆砌坐标，必须写成连贯的说明文字。"
)

# 每处理多少张图片写入一次 JSON（断点保存，防止中断丢失）
WRITE_EVERY_N_IMAGES = 24

CONFIG = {
    "test_dir": str(ROOT / "ForgeryAnalysis_Stage_1_Test" / "Image"),  # 若图片直接在 Test 下无 Image 子目录，改为 ROOT / "ForgeryAnalysis_Stage_1_Test"
    "plan": "B",
    "vlm_model": str(ROOT / "models" / "qwen3.5-9b"),
    "lora_path": str(ROOT / "checkpoints" / "qwen3.5-9b-lora" / "checkpoint-300"),
    "unet_ckpt": str(ROOT / "checkpoints" / "unet_best.pt"),
    "unet_arch": None,
    "unet_encoder": None,
    "unet_size": None,
    "output": str(ROOT / "data" / "inference_results.json"),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "resume": True,
    "vlm_batch_size": 8,
    "explanation_print_len": 80,
    "vlm_no_compress": False,  # VLM 用轻量压缩（与 train_vlm 格式一致）
    "vlm_light_compress": True,
}
# =============================================================================


def _write_results_json(
    output_path: str,
    image_paths: list,
    existing_by_name: dict,
    completed_todo: dict,
    placeholder_pending: bool = True,
) -> None:
    """按 image_paths 顺序组装结果并写入 JSON。每满 10 张或结束时调用。explanation 为完整内容，不截断。"""
    full = []
    for p in image_paths:
        if p.name in existing_by_name:
            full.append(existing_by_name[p.name])
        elif p.name in completed_todo:
            full.append(completed_todo[p.name])
        elif placeholder_pending:
            full.append({"image_name": p.name, "label": -1, "location": "", "explanation": "pending"})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)


def _truncate_expl(s: str, max_len: int) -> str:
    """仅用于终端显示截断；写入 JSON 时始终使用完整 explanation，不经过此函数。"""
    s = (s or "").strip().replace("\n", " ")
    return (s[:max_len] + "…") if len(s) > max_len else s


# --- VLM inference (Qwen3.5-VL + LoRA，与 train_vlm / prepare_vlm_data 一致) ---
def run_vlm_inference(
    image_paths: list[Path],
    model_path: str,
    device: str,
    lora_path: str | None = None,
    batch_size: int = 1,
    callback=None,
    no_compress: bool = True,
    light_compress: bool = False,
    reuse_model=None,
    reuse_processor=None,
) -> tuple[list[dict], object, object]:
    """Run VLM on each image (or batch); return (list of {label, bboxes, explanation}, model, processor).
    reuse_model/reuse_processor: 若提供则复用，避免重复加载。返回 (results, model, processor) 供下次复用。"""
    try:
        from transformers import AutoProcessor
        from peft import PeftModel
        try:
            from transformers import Qwen3_5ForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as ModelClass
    except ImportError:
        raise ImportError("For inline VLM run: pip install transformers peft")

    if reuse_model is not None and reuse_processor is not None:
        model, processor = reuse_model, reuse_processor
    else:
        lora_dir = Path(lora_path) if lora_path else (ROOT / "checkpoints" / "qwen3.5-9b-lora" / "checkpoint-300")
        _min = 28 * 28 if no_compress else MIN_PIXELS
        _max = NO_COMPRESS_MAX_PIXELS if no_compress else (LIGHT_MAX_PIXELS if light_compress else MAX_PIXELS)
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, min_pixels=_min, max_pixels=_max
        )
        img_proc = getattr(processor, "image_processor", None)
        if img_proc is not None:
            if no_compress:
                setattr(img_proc, "min_pixels", 28 * 28)
                setattr(img_proc, "max_pixels", NO_COMPRESS_MAX_PIXELS)
            elif light_compress:
                setattr(img_proc, "min_pixels", MIN_PIXELS)
                setattr(img_proc, "max_pixels", LIGHT_MAX_PIXELS)
            else:
                setattr(img_proc, "min_pixels", MIN_PIXELS)
                setattr(img_proc, "max_pixels", MAX_PIXELS)
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
        # 加载本批所有图像
        images = [load_image(p) for p in batch_paths]
        # 每条样本同一 prompt，生成一条模板再复制
        messages_one = [
            {"role": "user", "content": [{"type": "image", "image": images[0]}, {"type": "text", "text": USER_PROMPT}]}
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
            out = model.generate(**inputs, max_new_tokens=1024, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id)
        # 整批 pad 后长度一致，生成部分从 padded_input_len 开始（不能用 per-sample 长度，否则会解进输入段导致 label 错）
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
        # 每批结束后释放本批显存，避免长时间推理越跑越慢（碎片化/缓存堆积）
        del inputs, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return (results, model, processor)


# --- SAM2 or bbox fallback: bboxes -> merged mask ---
def _bboxes_to_rect_mask(h: int, w: int, bboxes: list[list[int]]) -> np.ndarray:
    """Fallback: fill rectangles in mask (no SAM2)."""
    import cv2
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in bboxes:
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def sam2_bboxes_to_mask(image: np.ndarray, bboxes: list[list[int]], device: str, model_name: str = "facebook/sam2.1-hiera-large") -> np.ndarray:
    """Return binary mask (H,W). Uses SAM2 if available, else rectangle mask."""
    h, w = image.shape[:2]
    if not bboxes:
        return np.zeros((h, w), dtype=np.uint8)
    try:
        from transformers import Sam2Processor, Sam2Model
    except ImportError:
        return _bboxes_to_rect_mask(h, w, bboxes)
    try:
        processor = Sam2Processor.from_pretrained(model_name)
        model = Sam2Model.from_pretrained(model_name).to(device)
        model.eval()
        # Input boxes: (batch, num_boxes, 4) in xyxy format; processor may expect normalized
        input_boxes = torch.tensor([bboxes], dtype=torch.float32, device=device)
        inputs = processor(images=[image], input_boxes=input_boxes, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        pred_masks = getattr(outputs, "pred_masks", outputs.logits)
        if pred_masks.dim() == 4:
            pred_masks = pred_masks[0]
        pred_masks = (pred_masks.sigmoid() > 0.5).cpu().numpy()
        merged = np.zeros((h, w), dtype=np.uint8)
        import cv2
        for m in pred_masks:
            if m.shape[0] != h or m.shape[1] != w:
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            merged = np.maximum(merged, m)
        return merged
    except Exception:
        return _bboxes_to_rect_mask(h, w, bboxes)


# --- Segmentation model inference (Plan B: from checkpoint with optional metadata) ---
def unet_predict_masks(
    image_paths: list[Path],
    ckpt_path: str,
    device: str,
    architecture: str | None = None,
    encoder_name: str | None = None,
    size: int | None = None,
    threshold: float = 0.5,
) -> list[np.ndarray]:
    """Return list of binary masks (H,W) per image. Builds model from checkpoint metadata; CLI args override."""
    import cv2
    from src.unet_model import build_unet, load_unet_checkpoint

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    # Prefer CLI overrides, then checkpoint metadata; legacy checkpoints lack architecture/encoder_name -> assume unet+resnet34
    arch = architecture if architecture is not None else ckpt.get("architecture")
    if arch is None:
        arch = "unet"
    enc = encoder_name if encoder_name is not None else ckpt.get("encoder_name")
    if enc is None:
        enc = "resnet34"
    infer_size = size if size is not None else ckpt.get("size", 512)
    thresh = ckpt.get("threshold", threshold)

    model = build_unet(encoder_name=enc, encoder_weights="imagenet", in_channels=3, architecture=arch)
    if isinstance(state, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    masks = []
    for p in image_paths:
        img = load_image(p)
        h, w = img.shape[:2]
        img_s = cv2.resize(img, (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img_s).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            logits = model(x)
        m = (logits.sigmoid().squeeze().cpu().numpy() > thresh).astype(np.uint8)
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        masks.append(m)
    return masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default=CONFIG["test_dir"], help="测试图片目录（默认 ForgeryAnalysis_Stage_1_Test/Image）")
    parser.add_argument("--plan", type=str, choices=["A", "B"], default=CONFIG["plan"], help="A: VLM+SAM2 用 bbox 生成 mask；B: VLM+分割模型在原图尺寸上出 mask")
    parser.add_argument("--vlm_results_json", type=str, default=None, help="预计算 VLM 结果 JSON，若提供则不再跑 VLM")
    parser.add_argument("--vlm_model", type=str, default=CONFIG["vlm_model"], help="VLM 基座路径（如 models/qwen3.5-9b）")
    parser.add_argument("--lora_path", type=str, default=CONFIG["lora_path"], help="LoRA 适配器目录（如 checkpoints/qwen3.5-9b-lora/checkpoint-300）")
    parser.add_argument("--unet_ckpt", type=str, default=CONFIG["unet_ckpt"], help="分割模型 checkpoint（Plan B）")
    parser.add_argument("--unet_arch", type=str, default=CONFIG["unet_arch"], help="覆盖 checkpoint：unet | deeplabv3plus")
    parser.add_argument("--unet_encoder", type=str, default=CONFIG["unet_encoder"], help="覆盖 encoder（如 resnet50）")
    parser.add_argument("--unet_size", type=int, default=CONFIG["unet_size"], help="分割推理输入尺寸（默认从 ckpt 或 512）")
    parser.add_argument("--sam2_model", type=str, default="facebook/sam2.1-hiera-large")
    parser.add_argument("--output", type=str, default=CONFIG["output"], help="推理结果 JSON，供 generate_csv 读取")
    parser.add_argument("--metrics_output", type=str, default=None, help="指标 JSON 路径（默认：output 同目录 inference_metrics.json）")
    parser.add_argument("--device", type=str, default=CONFIG["device"])
    parser.add_argument("--resume", action="store_true", default=CONFIG.get("resume", True), help="断点续跑：已存在于 output 的图片不再推理")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="禁用断点续跑，全部重跑")
    parser.add_argument("--vlm_batch_size", type=int, default=CONFIG.get("vlm_batch_size", 1), help="VLM 每批同时推理的图片数（真 batch，显存允许可试 2、4、8）")
    parser.add_argument("--explanation_print_len", type=int, default=CONFIG.get("explanation_print_len", 80), help="每张图打印时 explanation 截断长度")
    parser.add_argument("--vlm_no_compress", action="store_true", default=CONFIG.get("vlm_no_compress", True), help="VLM 推理不压缩原图")
    parser.add_argument("--vlm_compress", action="store_false", dest="vlm_no_compress", help="VLM 推理时强压缩(1280×28²)")
    parser.add_argument("--vlm_light_compress", action="store_true", default=CONFIG.get("vlm_light_compress", False), help="VLM 推理时轻压缩 640²（与 train_vlm compress=light 一致，需与 --vlm_compress 二选一）")
    parser.add_argument("--write_every_n", type=int, default=WRITE_EVERY_N_IMAGES, help="每处理 N 张图片写入一次 JSON（默认 10）")
    args = parser.parse_args()

    t_start = time.perf_counter()
    test_dir = Path(args.test_dir)
    image_paths = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not image_paths:
        raise FileNotFoundError(f"No images in {test_dir}")

    # 断点续跑：已存在结果不再跑，只跑新图
    existing_by_name = {}
    if args.resume and Path(args.output).exists():
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                existing_list = json.load(f)
            # 只把已预测的（label 0 或 1）视为已有；label=-1（pending）的需重新推理
            existing_by_name = {
                r["image_name"]: r for r in existing_list
                if isinstance(r, dict) and "image_name" in r and r.get("label", -1) in (0, 1)
            }
            if existing_by_name:
                print(f"[Resume] 已有 {len(existing_by_name)} 条有效结果（label=0/1），未预测（label=-1）的将重新推理。", flush=True)
                print("若之前推理结果为全 label=0，请删除 output JSON 或使用 --no-resume 重新推理。", flush=True)
        except Exception:
            existing_by_name = {}
    todo_paths = [p for p in image_paths if p.name not in existing_by_name]
    print(f"测试图总数: {len(image_paths)}，待推理: {len(todo_paths)}（跳过已存在 {len(existing_by_name)} 张）", flush=True)
    if not todo_paths and existing_by_name:
        print("无新图需推理，直接合并并写回结果。", flush=True)

    def print_cb(path: Path, result: dict, elapsed: float) -> None:
        # 终端仅显示截断后的 explanation；JSON 中保存完整 explanation（见下方 completed_todo 赋值）
        label = int(result.get("label", 0))
        expl = _truncate_expl(result.get("explanation") or "", args.explanation_print_len)
        print(f"  [{path.name}] label={label} 用时={elapsed:.1f}s  expl={expl!r}", flush=True)

    # VLM results: 预计算 JSON 或跑模型（仅对 todo 跑，带打印回调）
    vlm_source = ""
    vlm_results_todo: list[dict] = []
    if args.vlm_results_json:
        with open(args.vlm_results_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            vlm_results_full = data
        else:
            vlm_results_full = [data.get(p.name, {"label": 0, "bboxes": [], "explanation": ""}) for p in image_paths]
        vlm_source = f"vlm_results_json:{args.vlm_results_json}"
        vlm_results_by_name = {image_paths[i].name: vlm_results_full[i] for i in range(len(image_paths)) if i < len(vlm_results_full)}
    elif args.vlm_model:
        vlm_source = f"vlm_model:{args.vlm_model}"
        vlm_results_by_name = {}
    else:
        raise ValueError("Provide --vlm_results_json or --vlm_model")

    # 合并：按 image_paths 顺序，已有用 existing，新跑用 vlm_results_by_name
    def get_vlm_result(name: str) -> dict:
        if name in existing_by_name:
            r = existing_by_name[name]
            return {"label": int(r.get("label", 0)), "bboxes": r.get("bboxes", []), "explanation": r.get("explanation", "")}
        return vlm_results_by_name.get(name, {"label": 0, "bboxes": [], "explanation": ""})

    vlm_batch = args.vlm_batch_size

    # 按批处理 todo，每 write_every_n 张执行 VLM+分割+写入，确保每 10 张保存一次
    completed_todo = {}
    n = args.write_every_n
    vlm_model, vlm_processor = None, None

    def flush_results(placeholder: bool = True) -> None:
        _write_results_json(args.output, image_paths, existing_by_name, completed_todo, placeholder_pending=placeholder)

    if args.plan == "A":
        for start in range(0, len(todo_paths), n):
            chunk = todo_paths[start : start + n]
            if args.vlm_results_json:
                vlm_chunk = [get_vlm_result(p.name) for p in chunk]
            else:
                vlm_chunk_raw, vlm_model, vlm_processor = run_vlm_inference(
                    chunk, args.vlm_model, args.device, args.lora_path,
                    batch_size=vlm_batch, callback=print_cb,
                    no_compress=args.vlm_no_compress,
                    light_compress=args.vlm_light_compress and not args.vlm_no_compress,
                    reuse_model=vlm_model, reuse_processor=vlm_processor,
                )
                vlm_chunk = vlm_chunk_raw
            for i, p in enumerate(chunk):
                r = vlm_chunk[i]
                label = int(r.get("label", 0))
                bboxes = r.get("bboxes") or []
                explanation = r.get("explanation") or ""
                if label == 0 or not bboxes:
                    location = ""
                else:
                    img = load_image(p)
                    mask = sam2_bboxes_to_mask(img, bboxes, args.device, args.sam2_model)
                    location = mask_to_rle(mask)
                completed_todo[p.name] = {"image_name": p.name, "label": label, "location": location, "explanation": explanation}
            flush_results(placeholder=True)
            print(f"  [Write] 已处理 {len(completed_todo)} 张，写入 {args.output}", flush=True)
    else:
        # Plan B: 每 chunk 执行 VLM+Unet+写入，确保每 10 张保存一次
        for start in range(0, len(todo_paths), n):
            chunk = todo_paths[start : start + n]
            if args.vlm_results_json:
                vlm_chunk = [get_vlm_result(p.name) for p in chunk]
            else:
                vlm_chunk_raw, vlm_model, vlm_processor = run_vlm_inference(
                    chunk, args.vlm_model, args.device, args.lora_path,
                    batch_size=vlm_batch, callback=print_cb,
                    no_compress=args.vlm_no_compress,
                    light_compress=args.vlm_light_compress and not args.vlm_no_compress,
                    reuse_model=vlm_model, reuse_processor=vlm_processor,
                )
                vlm_chunk = vlm_chunk_raw
            masks_chunk = unet_predict_masks(
                chunk,
                args.unet_ckpt,
                args.device,
                architecture=args.unet_arch,
                encoder_name=args.unet_encoder,
                size=args.unet_size,
            )
            for i, p in enumerate(chunk):
                r = vlm_chunk[i]
                label = int(r.get("label", 0))
                explanation = r.get("explanation") or ""  # 完整保存到 JSON，不截断
                mask = masks_chunk[i]
                if label == 0:
                    mask = np.zeros_like(mask)
                location = "" if label == 0 else mask_to_rle(mask)
                completed_todo[p.name] = {"image_name": p.name, "label": label, "location": location, "explanation": explanation}
            flush_results(placeholder=True)
            print(f"  [Write] 已处理 {len(completed_todo)} 张，写入 {args.output}", flush=True)

    flush_results(placeholder=False)
    _pending = {"label": -1, "location": "", "explanation": "pending"}
    results = [
        existing_by_name.get(p.name) or completed_todo.get(p.name) or {"image_name": p.name, **_pending}
        for p in image_paths
    ]
    print(f"Wrote {len(results)} results to {args.output}")

    # Compute and save metrics
    t_end = time.perf_counter()
    total_time_s = t_end - t_start
    num_images = len(results)
    num_predicted_forged = sum(1 for r in results if int(r.get("label", 0)) == 1)
    num_predicted_real = num_images - num_predicted_forged
    metrics = {
        "num_images": num_images,
        "num_predicted_forged": num_predicted_forged,
        "num_predicted_real": num_predicted_real,
        "label_distribution": {"forged": num_predicted_forged, "real": num_predicted_real},
        "total_time_seconds": round(total_time_s, 2),
        "time_per_image_seconds": round(total_time_s / max(num_images, 1), 2),
        "plan": args.plan,
        "vlm_source": vlm_source,
        "output_path": args.output,
    }
    metrics_path = args.metrics_output or str(Path(args.output).parent / "inference_metrics.json")
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
