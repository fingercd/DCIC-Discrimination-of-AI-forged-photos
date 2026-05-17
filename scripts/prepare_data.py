"""
Parse all .md captions under Black/White, extract bboxes, build dataset.json.
Paths: DCIC/ForgeryAnalysis_Stage_1_Train/{Black|White}/Caption/*.md, Image, Mask.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BBOX_PATTERN = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")


def extract_bboxes(text: str) -> list[list[int]]:
    return [[int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))] for m in BBOX_PATTERN.finditer(text)]

TRAIN_DIR = "ForgeryAnalysis_Stage_1_Train"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    train_base = ROOT / TRAIN_DIR
    if not train_base.exists():
        raise FileNotFoundError(f"Train dir not found: {train_base}")

    records = []

    for split in ("Black", "White"):
        caption_dir = train_base / split / "Caption"
        image_dir = train_base / split / "Image"
        mask_dir = train_base / split / "Mask"
        if not caption_dir.exists():
            continue
        is_forged = split == "Black"

        for md_path in sorted(caption_dir.glob("*.md")):
            stem = md_path.stem
            text = md_path.read_text(encoding="utf-8").strip()
            bboxes = extract_bboxes(text)

            # Resolve image path (jpg or png)
            img_path = None
            for ext in (".jpg", ".jpeg", ".png"):
                p = image_dir / f"{stem}{ext}"
                if p.exists():
                    img_path = str(p.relative_to(ROOT))
                    break
            if img_path is None:
                continue

            # Mask only for Black
            mask_path = None
            if is_forged and mask_dir.exists():
                p = mask_dir / f"{stem}.png"
                if p.exists():
                    mask_path = str(p.relative_to(ROOT))

            records.append({
                "id": stem,
                "split": split,
                "label": 1 if is_forged else 0,
                "caption": text,
                "bboxes": bboxes,
                "image_path": img_path,
                "mask_path": mask_path,
            })

    out_path = DATA_DIR / "dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
