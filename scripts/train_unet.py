"""
Train UNet for forgery mask prediction (Plan B).
Uses dataset.json: Black (image+mask) + White (image + all-zero mask).
Loss: BCE + Dice. Saves checkpoint to checkpoints/unet_best.pt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.unet_model import build_unet, load_unet_checkpoint
from src.unet_dataset import ForgeryMaskDataset


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    pred = pred.sigmoid()
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    return 1 - (2 * intersection + smooth) / (union + smooth)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(ROOT / "data" / "dataset.json"))
    parser.add_argument("--root", type=str, default=str(ROOT))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32, help="Default 8 for DeepLabV3+ ResNet50 on 24GB; increase if using smaller model")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--save", type=str, default=str(ROOT / "checkpoints" / "unet_best.pt"))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--architecture", type=str, default="deeplabv3plus", choices=["unet", "deeplabv3plus"])
    parser.add_argument("--encoder", type=str, default="resnet50")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold for mask; saved in checkpoint for inference")
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Only use samples that have image (and mask for Black)
    dataset = ForgeryMaskDataset(Path(args.root), records, (args.size, args.size))
    n = len(dataset)
    n_val = max(1, int(n * args.val_ratio))
    n_train = n - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        architecture=args.architecture,
    ).to(device)
    load_unet_checkpoint(model, args.resume)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = -1.0
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    n_batches = len(train_loader)
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            logits = model(img)
            bce = F.binary_cross_entropy_with_logits(logits, mask)
            dice = dice_loss(logits, mask)
            loss = bce + dice
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_item = loss.item()
            bce_item = bce.item()
            dice_item = dice.item()
            train_loss += loss_item
            lr = opt.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{args.epochs} Batch {batch_idx+1}/{n_batches} loss={loss_item:.4f} bce={bce_item:.4f} dice={dice_item:.4f} lr={lr:.2e}")
        scheduler.step()
        train_loss /= n_batches

        model.eval()
        val_dice_sum = 0.0
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                mask = batch["mask"].to(device)
                logits = model(img)
                bce = F.binary_cross_entropy_with_logits(logits, mask)
                dice = dice_loss(logits, mask)
                val_loss_sum += (bce.item() + dice.item()) * logits.size(0)
                pred = logits.sigmoid()
                for i in range(pred.size(0)):
                    p = pred[i].view(-1)
                    t = mask[i].view(-1)
                    inter = (p * t).sum()
                    u = p.sum() + t.sum()
                    d = (2 * inter + 1e-6) / (u + 1e-6)
                    val_dice_sum += d.item()
                    val_n += 1
        val_dice = val_dice_sum / max(val_n, 1)
        val_loss = val_loss_sum / max(val_n, 1)
        if val_dice > best_val:
            best_val = val_dice
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "val_loss": val_loss,
                    "architecture": args.architecture,
                    "encoder_name": args.encoder,
                    "size": args.size,
                    "threshold": args.threshold,
                },
                args.save,
            )
        print(f"Epoch {epoch+1}/{args.epochs} [summary] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f} best_val_dice={best_val:.4f}")
    print(f"Best model saved to {args.save}")


if __name__ == "__main__":
    main()
