"""
读取 inference_results.json，生成赛题提交用 CSV（UTF-8 编码）。
字段：image_name, label, location(RLE), explanation。
若推理时 VLM 使用了压缩，可通过 --vlm_compress 将 explanation 中的坐标还原到原图尺寸。
运行后先等待 3 小时再执行生成逻辑。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from src.utils import extract_bboxes_from_caption
from prepare_vlm_data import get_resized_size, scale_bboxes, replace_bboxes_in_caption

# 与 inference.py 一致，用于计算压缩后尺寸
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
LIGHT_MAX_PIXELS = 1280 * 1280 # 与 inference / train_vlm / prepare_vlm_data 一致
NO_COMPRESS_MAX_PIXELS = 8192 * 8192
PATCH_SIZE = 28

# ============ 可在此修改默认路径 ============
CONFIG = {
    "input_json": str(ROOT / "data" / "inference_results.json"),
    "output_csv": str(ROOT / "data" / "submission.csv"),
    "test_dir": str(ROOT / "ForgeryAnalysis_Stage_1_Test" / "Image"),
}
# ===========================================


def _restore_explanation_coordinates(
    explanation: str,
    image_path: Path,
    vlm_compress: str,
) -> str:
    """
    将 explanation 中的 [x1,y1,x2,y2] 坐标从 VLM 压缩图空间还原到原图尺寸。
    vlm_compress: no | light | heavy。为 no 时仍会计算缩放（通常 1:1）。
    图片不存在或无坐标时返回原 explanation。
    """
    if not explanation or vlm_compress not in ("no", "light", "heavy"):
        return explanation
    bboxes = extract_bboxes_from_caption(explanation)
    if not bboxes:
        return explanation
    if not image_path.exists():
        return explanation
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            w_orig, h_orig = im.size[0], im.size[1]
    except Exception:
        return explanation
    if vlm_compress == "no":
        min_p, max_p = PATCH_SIZE * PATCH_SIZE, NO_COMPRESS_MAX_PIXELS
    elif vlm_compress == "light":
        min_p, max_p = MIN_PIXELS, LIGHT_MAX_PIXELS
    else:
        min_p, max_p = MIN_PIXELS, MAX_PIXELS
    w_comp, h_comp = get_resized_size(w_orig, h_orig, min_pixels=min_p, max_pixels=max_p)
    bboxes_orig = scale_bboxes(bboxes, w_comp, h_comp, w_orig, h_orig)
    return replace_bboxes_in_caption(explanation, bboxes_orig)


def main() -> None:
    delay_seconds = 5 * 3600
    print(f"[Delay] 等待 {delay_seconds / 3600:.1f} 小时后生成 CSV…", flush=True)
    time.sleep(delay_seconds)
    print("[Delay] 等待结束，开始生成 CSV。", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=CONFIG["input_json"], help="推理结果 JSON（来自 inference.py --output）")
    parser.add_argument("--output", type=str, default=CONFIG["output_csv"], help="提交 CSV 路径，UTF-8 编码")
    parser.add_argument("--test_dir", type=str, default=CONFIG["test_dir"], help="测试图目录，用于读取原图尺寸")
    parser.add_argument("--vlm_compress", type=str, choices=("no", "light", "heavy"), default="no",
                        help="推理时 VLM 压缩模式：no=不压缩，light=轻压缩(1280²)，heavy=强压缩(1280×28²)。非 no 时会将 explanation 中坐标还原到原图")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    test_dir = Path(args.test_dir)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "label", "location", "explanation"])
        for r in results:
            image_name = r.get("image_name", "")
            label = r.get("label", 0)
            location = r.get("location", "")
            explanation = r.get("explanation", "")
            image_path = test_dir / image_name
            explanation = _restore_explanation_coordinates(explanation, image_path, args.vlm_compress)
            writer.writerow([image_name, label, location, explanation])

    print(f"Wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
