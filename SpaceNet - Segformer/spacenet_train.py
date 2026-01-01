from __future__ import annotations

from pathlib import Path

import numpy as np
import time
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from spacenet_dataset import get_dataloaders


def _iou_from_logits(logits, target) -> float:
    pred = logits.argmax(1)
    inter = ((pred == 1) & (target == 1)).sum().item()
    union = ((pred == 1) | (target == 1)).sum().item()
    return inter / (union + 1e-6)


def _tp_fp_fn_from_logits(logits, target):
    pred = logits.argmax(1)
    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    return tp, fp, fn


def _correct_total_from_logits(logits, target):
    pred = logits.argmax(1)
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct, total

def _soft_dice_loss(logits, target, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    target_1 = (target == 1).float()
    prob_1 = probs[:, 1]
    intersection = (prob_1 * target_1).sum(dim=(1, 2))
    union = prob_1.sum(dim=(1, 2)) + target_1.sum(dim=(1, 2))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def _format_roots(data_root: str | Path) -> str:
    if isinstance(data_root, Path):
        return str(data_root)
    return str(data_root)


def train_segformer(
    data_root: str | Path,
    ckpt_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 8,
    epochs: int = 25,
    lr: float = 3e-5,
    weight_decay: float = 1e-2,
    skip_overexposed_train: bool = False,
    init_from: str | Path | None = None,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"device={device} | epochs={epochs} | batch_size={batch_size} | "
        f"num_workers={num_workers} | skip_overexposed_train={skip_overexposed_train}",
        flush=True,
    )
    print(f"data_root={_format_roots(data_root)}", flush=True)
    print("Building dataloaders...", flush=True)
    train_loader, val_loader = get_dataloaders(
        data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        skip_overexposed_train=skip_overexposed_train,
    )

    print("Loading SegFormer checkpoint...", flush=True)
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=2,
        ignore_mismatched_sizes=True,
    ).to(device)
    print("Model loaded.", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_iou = -1.0
    start_epoch = 0
    latest_path = ckpt_dir / "latest_segformer_vegas.pth"
    best_path = ckpt_dir / "best_segformer_vegas.pth"

    if init_from:
        init_path = Path(init_from)
        if init_path.exists():
            state = torch.load(init_path, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"], strict=False)
            else:
                model.load_state_dict(state, strict=False)
            print(f"Initialized weights from {init_path}", flush=True)
        else:
            print(f"Init checkpoint not found: {init_path}", flush=True)
    elif latest_path.exists():
        state = torch.load(latest_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"], strict=False)
        opt.load_state_dict(state["optimizer_state_dict"])
        scaler.load_state_dict(state["scaler"])
        best_iou = float(state.get("best_iou", best_iou))
        start_epoch = int(state.get("epoch", 0))
        print(f"Resuming from epoch {start_epoch} | best_iou={best_iou:.4f}")
    else:
        print("No checkpoint found. Training from scratch.")

    print(f"Starting training at epoch {start_epoch + 1}/{epochs}...")
    for epoch in range(start_epoch + 1, epochs + 1):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_steps = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(pixel_values=imgs)
                logits = F.interpolate(
                    out.logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                ce = F.cross_entropy(logits, masks)
                dice = _soft_dice_loss(logits, masks)
                loss = 0.5 * ce + 0.5 * dice
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_loss += loss.item()
            train_steps += 1
            if train_steps % 10 == 0:
                print(
                    f"\rEpoch {epoch} "
                    f"Batch {train_steps}/{len(train_loader)} "
                    f"Loss: {loss.item():.4f}",
                    end="",
                )

        model.eval()
        tp_total = 0
        fp_total = 0
        fn_total = 0
        correct_total = 0
        total_pixels = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(pixel_values=imgs)
                logits = F.interpolate(
                    out.logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                tp, fp, fn = _tp_fp_fn_from_logits(logits, masks)
                tp_total += tp
                fp_total += fp
                fn_total += fn
                correct, total = _correct_total_from_logits(logits, masks)
                correct_total += correct
                total_pixels += total

        denom = tp_total + fp_total + fn_total + 1e-6
        miou = tp_total / denom
        acc = correct_total / max(total_pixels, 1)
        scheduler.step(miou)
        current_lr = opt.param_groups[0]["lr"]
        avg_train_loss = train_loss / max(train_steps, 1)
        duration = time.time() - start_time
        print(
            f"\nEpoch {epoch} | Time: {duration:.0f}s | "
            f"Train Loss: {avg_train_loss:.4f} | Val IoU: {miou:.4f} | "
            f"Val Acc: {acc:.4f} | LR: {current_lr:.2e}"
        )

        if miou > best_iou:
            best_iou = miou
            torch.save(model.state_dict(), best_path)
            print(f"saved best -> {best_path}")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "best_iou": best_iou,
                "epoch": epoch,
            },
            latest_path,
        )

    return best_iou
