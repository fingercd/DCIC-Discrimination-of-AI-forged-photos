"""
Convert dataset.json to LLaMA-Factory sharegpt multi-modal format for VLM fine-tuning.
可选是否压缩：不压缩时 bbox 保持原图坐标；压缩时与 train_vlm 的 min_pixels/max_pixels 一致。
Output: data/vlm_train.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import BBOX_PATTERN

DATA_DIR = ROOT / "data"
DATASET_JSON = DATA_DIR / "dataset.json"
VLM_TRAIN_JSON = DATA_DIR / "vlm_train.json"

# 与 train_vlm 压缩模式下的 processor 一致
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
PATCH_SIZE = 28
# 轻压缩：总像素上限 640×640，与 train_vlm compress="light" 一致，bbox 需同步缩放
LIGHT_MAX_PIXELS = 960 * 960

# 默认轻压缩(640²)；--no-compress=不压缩，--compress=强压缩(1280×28²)
USER_PROMPT = (
    "请分析这张场景文本图像是否存在伪造，并严格按以下格式输出：\n"
    "[Detection] 0或1\n"
    "[Bboxes] [[x1,y1,x2,y2], ...]（若无伪造则为[]）\n"
    "[Explanation] 你的分析说明"
)


def get_resized_size(
    w_orig: int,
    h_orig: int,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> tuple[int, int]:
    """
    与 Qwen2VL 的 smart_resize 规则一致：保持宽高比，总像素落在 [min_pixels, max_pixels]，
    宽高为 PATCH_SIZE 的倍数。与 train_vlm 的 processor 行为对齐。
    """
    min_p = min_pixels if min_pixels is not None else MIN_PIXELS
    max_p = max_pixels if max_pixels is not None else MAX_PIXELS
    total = w_orig * h_orig
    if total <= 0:
        return (PATCH_SIZE, PATCH_SIZE)
    if total > max_p:
        scale = math.sqrt(max_p / total)
    elif total < min_p:
        scale = math.sqrt(min_p / total)
    else:
        scale = 1.0
    w_new = max(PATCH_SIZE, round(w_orig * scale / PATCH_SIZE) * PATCH_SIZE)
    h_new = max(PATCH_SIZE, round(h_orig * scale / PATCH_SIZE) * PATCH_SIZE)
    if w_new * h_new > max_p:
        scale2 = math.sqrt(max_p / (w_new * h_new))
        w_new = max(PATCH_SIZE, round(w_new * scale2 / PATCH_SIZE) * PATCH_SIZE)
        h_new = max(PATCH_SIZE, round(h_new * scale2 / PATCH_SIZE) * PATCH_SIZE)
    return (int(w_new), int(h_new))


def scale_bboxes(
    bboxes: list[list[int]], w_orig: int, h_orig: int, w_new: int, h_new: int
) -> list[list[int]]:
    """将原图坐标的 bbox 按缩放比例换算到 (w_new, h_new) 空间。"""
    if w_orig <= 0 or h_orig <= 0:
        return bboxes
    scale_x = w_new / w_orig
    scale_y = h_new / h_orig
    return [
        [
            round(x1 * scale_x),
            round(y1 * scale_y),
            round(x2 * scale_x),
            round(y2 * scale_y),
        ]
        for x1, y1, x2, y2 in bboxes
    ]


def replace_bboxes_in_caption(caption: str, scaled_bboxes: list[list[int]]) -> str:
    """
    将 caption 中按顺序出现的 bbox 字符串替换为 scaled_bboxes 中对应坐标。
    与 prepare_data 使用相同的 BBOX_PATTERN，保证一一对应。
    """
    if not scaled_bboxes:
        return caption
    result = []
    idx = 0
    last_end = 0
    for m in BBOX_PATTERN.finditer(caption):
        if idx >= len(scaled_bboxes):
            warnings.warn(
                f"Caption has more bbox matches than scaled_bboxes (idx={idx}), skipping remainder."
            )
            break
        result.append(caption[last_end : m.start()])
        result.append(f"[{scaled_bboxes[idx][0]}, {scaled_bboxes[idx][1]}, {scaled_bboxes[idx][2]}, {scaled_bboxes[idx][3]}]")
        last_end = m.end()
        idx += 1
    result.append(caption[last_end:])
    if idx < len(scaled_bboxes):
        warnings.warn(
            f"Caption has fewer bbox matches ({idx}) than scaled_bboxes ({len(scaled_bboxes)})."
        )
    return "".join(result)


def build_assistant_content(
    record: dict,
    *,
    bboxes: list[list[int]] | None = None,
    caption: str | None = None,
) -> str:
    label = record["label"]
    bboxes = bboxes if bboxes is not None else record.get("bboxes") or []
    caption = caption if caption is not None else record.get("caption") or ""
    bbox_str = json.dumps(bboxes, ensure_ascii=False) if bboxes else "[]"
    return (
        f"[Detection] {label}\n"
        f"[Bboxes] {bbox_str}\n"
        f"[Explanation] {caption}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 VLM 训练用 vlm_train.json")
    parser.add_argument("--compress", action="store_true", help="强压缩：bbox 按 1280×28² 缩放（与 train_vlm compress=True 一致）")
    parser.add_argument("--light-compress", action="store_true", help="轻压缩：bbox 按 640×640 上限缩放（与 train_vlm compress=\"light\" 一致）")
    parser.add_argument("--no-compress", action="store_true", help="不压缩：bbox 保持原图坐标（默认不传参时为轻压缩）")
    args = parser.parse_args()
    # 优先级：--no-compress > --light-compress > --compress；不传任何参数时默认轻压缩(640²)
    if args.no_compress:
        compress, light_compress = False, False
    elif args.light_compress or not args.compress:
        compress, light_compress = False, True
    else:
        compress, light_compress = True, False

    if not DATASET_JSON.exists():
        raise FileNotFoundError(f"Run prepare_data.py first. Missing: {DATASET_JSON}")

    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)

    mode = "轻压缩(640²)" if light_compress else ("强压缩(1280×28²)" if compress else "不压缩")
    print(f"生成模式: {mode}（bbox 与 train_vlm 的 compress 选项需一致）", flush=True)
    out_list = []
    for r in records:
        image_path = ROOT / r["image_path"]
        if not image_path.exists():
            continue

        # 原图尺寸
        try:
            from PIL import Image
            with Image.open(image_path) as im:
                w_orig, h_orig = im.size
        except Exception as e:
            warnings.warn(f"Skip {image_path}: cannot read image size ({e})")
            continue

        bboxes_orig = r.get("bboxes") or []
        caption_orig = r.get("caption") or ""
        if light_compress:
            w_new, h_new = get_resized_size(w_orig, h_orig, MIN_PIXELS, LIGHT_MAX_PIXELS)
            scaled_bboxes = scale_bboxes(bboxes_orig, w_orig, h_orig, w_new, h_new)
        elif compress:
            w_new, h_new = get_resized_size(w_orig, h_orig)
            scaled_bboxes = scale_bboxes(bboxes_orig, w_orig, h_orig, w_new, h_new)
        else:
            w_new, h_new = w_orig, h_orig
            scaled_bboxes = bboxes_orig
        caption_scaled = replace_bboxes_in_caption(caption_orig, scaled_bboxes)

        # LLaMA-Factory sharegpt: list of conversations; each message can have image in content
        conv = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{USER_PROMPT}",
                    "image": str(image_path.resolve()),
                },
                {
                    "from": "gpt",
                    "value": build_assistant_content(
                        r, bboxes=scaled_bboxes, caption=caption_scaled
                    ),
                },
            ],
            "images": [str(image_path.resolve())],
        }
        out_list.append(conv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(VLM_TRAIN_JSON, "w", encoding="utf-8") as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(out_list)} conversations to {VLM_TRAIN_JSON}")


if __name__ == "__main__":
    main()
