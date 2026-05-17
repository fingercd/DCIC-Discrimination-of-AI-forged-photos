# Plan B: Dataset for UNet (image + binary mask)
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ForgeryMaskDataset(Dataset):
    """Image + binary mask for UNet. Black=forged (mask), White=real (all-zero mask)."""

    def __init__(
        self,
        root: Path | str,
        records: list[dict],
        image_size: tuple[int, int] = (512, 512),
    ):
        self.root = Path(root)
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, rel_path: str) -> np.ndarray:
        p = self.root / rel_path
        if not p.exists():
            raise FileNotFoundError(p)
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Failed to read {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_mask(self, rel_path: Optional[str], h: int, w: int) -> np.ndarray:
        if rel_path is None:
            return np.zeros((h, w), dtype=np.float32)
        p = self.root / rel_path
        if not p.exists():
            return np.zeros((h, w), dtype=np.float32)
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return np.zeros((h, w), dtype=np.float32)
        # Binarize: >127 -> 1
        m = (m.astype(np.float32) > 127).astype(np.float32)
        return m

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self.records[idx]
        img = self._load_image(r["image_path"])
        mask = self._load_mask(r.get("mask_path"), img.shape[0], img.shape[1])

        # Resize to image_size
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)

        # To tensor: image [3,H,W] float [0,1], mask [1,H,W] float [0,1]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return {"image": img, "mask": mask}
