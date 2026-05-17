# Copyright (c) DCIC. Utils for RLE, bbox parsing, VLM output parsing, image loading.

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


# --- RLE encoding (submission format) ---

def mask_to_rle(mask: np.ndarray, threshold: int = 127) -> str:
    """Convert binary mask to RLE JSON string for submission `location` field."""
    if mask_utils is None:
        raise ImportError("pycocotools is required: pip install pycocotools")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) if mask.shape[2] == 3 else mask[:, :, 0]
    binary = (mask.astype(np.uint8) > threshold).astype(np.uint8)
    binary_fortran = np.asfortranarray(binary)
    rle_dict = mask_utils.encode(binary_fortran)
    if isinstance(rle_dict, list):
        # multiple RLEs; merge into one mask then encode (or take first - check competition spec)
        rle_dict = rle_dict[0] if rle_dict else {"size": list(binary.shape), "counts": []}
    if isinstance(rle_dict.get("counts"), bytes):
        rle_dict["counts"] = rle_dict["counts"].decode("utf-8")
    return json.dumps(rle_dict, ensure_ascii=False)


def rle_to_mask(rle_str: str) -> np.ndarray:
    """Decode RLE JSON string to binary mask (for evaluation)."""
    if mask_utils is None:
        raise ImportError("pycocotools is required")
    rle = json.loads(rle_str)
    if isinstance(rle.get("counts"), list):
        return mask_utils.decode(rle)
    return mask_utils.decode(rle)


# --- Bbox extraction from caption text ---

BBOX_PATTERN = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")


def extract_bboxes_from_caption(text: str) -> list[list[int]]:
    """Extract all [x1, y1, x2, y2] bboxes from caption text."""
    out = []
    for m in BBOX_PATTERN.finditer(text):
        out.append([int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))])
    return out


# --- VLM output parsing (structured format: [Detection], [Bboxes], [Explanation]) ---

DETECTION_PATTERN = re.compile(r"\[Detection\]\s*(\d+)", re.IGNORECASE)
BBOXES_PATTERN = re.compile(r"\[Bboxes\]\s*(\[[^\]]*(?:\[[^\]]*\])*[^\]]*\])", re.IGNORECASE | re.DOTALL)
EXPLANATION_PATTERN = re.compile(r"\[Explanation\]\s*(.+)", re.IGNORECASE | re.DOTALL)

# 当模型未输出 [Detection] 时的回退：根据 explanation 正文推断 0/1
# 伪造类：覆盖"伪造图像""伪造收据""经过数字篡改""数字拼接合成""后期添加"等常见表述
# 注：先经 LABEL_NEGATION_PATTERN 移除"未发现伪造"等，故剩余"伪造/篡改"可视为正面表述
LABEL_FORGERY_KEYWORDS = re.compile(
    r"存在伪造|系.*伪造|人为伪造|确系.*伪造|数字伪造|经过.*篡改|存在.*篡改|人为.*篡改|"
    r"伪造的|伪造品|伪造内容|伪造图像|伪造收据|系人为|系数字|不具备.*真实性|不具备.*可信度|"
    r"数字拼接|数字合成|后期添加|人为添加|篡改痕迹|数字篡改|"
    r"伪造|篡改",  # 单独"伪造/篡改"（否定语境已先移除）
    re.IGNORECASE,
)
# 真实类：明确表示未发现伪造、真实拍摄等
LABEL_REAL_KEYWORDS = re.compile(
    r"未发现伪造|不存在伪造|无伪造|未发现.*篡改|未发现.*异常|"
    r"真实拍摄|真实.*照片|未发现.*伪造痕迹|未检测到伪造|未发现数字伪造|未发现.*伪造|"
    r"未发现数字伪造或后期篡改",
    re.IGNORECASE,
)
# 否定语境：这些短语中的"伪造/篡改"不应计为伪造证据（先移除再检查伪造）
# 覆盖"未发现数字伪造或后期篡改的痕迹"等完整否定句；末尾 [^。]* 贪婪匹配以移除整句
LABEL_NEGATION_PATTERN = re.compile(
    r"未发现[^。]*?(?:伪造|篡改)[^。]*|不存在[^。]*?(?:伪造|篡改)[^。]*|无[^。]*?(?:伪造|篡改)[^。]*",
    re.IGNORECASE,
)


def _infer_label_from_text(text: str) -> int:
    """
    当缺少 [Detection] 时，根据 explanation 正文推断 label：1=伪造，0=未伪造。
    优先从 explanation 推断，确保 label 与解释内容一致。
    """
    if not (text or "").strip():
        return 0
    # 先检查真实类（在原文上），若明确说"未发现伪造/真实拍摄"则判 0
    has_real = bool(LABEL_REAL_KEYWORDS.search(text))
    # 移除否定语境后再检查伪造类，避免"未发现数字伪造"被误判为伪造
    text_for_forgery_check = LABEL_NEGATION_PATTERN.sub("", text)
    has_forgery = bool(LABEL_FORGERY_KEYWORDS.search(text_for_forgery_check))
    if has_real and not has_forgery:
        return 0
    if has_forgery:
        return 1
    # 都不匹配时默认 0（保守）
    return 0


def parse_vlm_output(raw: str) -> dict[str, Any]:
    """
    Parse VLM response into label, bboxes, explanation.
    Returns dict with keys: label (0|1), bboxes (list of [x1,y1,x2,y2]), explanation (str).
    若回复中无 [Detection] 标记，则根据正文关键词回退推断 label。
    """
    label = 0
    bboxes: list[list[int]] = []
    explanation = ""

    m = DETECTION_PATTERN.search(raw)
    if m:
        label = int(m.group(1).strip())
        if label not in (0, 1):
            label = 1 if label else 0
    else:
        # 模型未按格式输出 [Detection]，从全文推断
        label = _infer_label_from_text(raw)

    m = BBOXES_PATTERN.search(raw)
    if m:
        bbox_str = m.group(1).strip()
        # Parse nested list e.g. [[1431, 539, 2253, 730], [x2,y2,x2,y2]]
        try:
            parsed = json.loads(bbox_str)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, (list, tuple)) and len(item) >= 4:
                        bboxes.append([int(item[0]), int(item[1]), int(item[2]), int(item[3])])
        except json.JSONDecodeError:
            for bm in BBOX_PATTERN.finditer(bbox_str):
                bboxes.append([int(bm.group(1)), int(bm.group(2)), int(bm.group(3)), int(bm.group(4))])

    m = EXPLANATION_PATTERN.search(raw)
    if m:
        explanation = m.group(1).strip()

    if not explanation and "[Explanation]" not in raw:
        # Fallback: treat rest of text as explanation
        explanation = raw.strip()

    return {"label": label, "bboxes": bboxes, "explanation": explanation}


# --- Image loading ---

def load_image(path: str | Path) -> np.ndarray:
    """Load image as RGB numpy array (H, W, 3)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def find_image_path(root: Path, stem: str) -> Path | None:
    """Find image file by stem (no extension) under root; supports .jpg and .png."""
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        p = root / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def find_mask_path(root: Path, stem: str) -> Path | None:
    """Find mask file by stem under root; typically .png."""
    for ext in (".png", ".jpg", ".bmp"):
        p = root / f"{stem}{ext}"
        if p.exists():
            return p
    return None
