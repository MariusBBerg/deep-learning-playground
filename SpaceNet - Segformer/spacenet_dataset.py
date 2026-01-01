from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset


class BuildingsTiles(Dataset):
    def __init__(
        self,
        img_dir: str | Path,
        msk_dir: str | Path,
        augment=None,
        skip_overexposed: bool = False,
    ):
        img_paths = sorted(Path(img_dir).glob("*.png"))
        self.msk_dir = Path(msk_dir)
        self.augment = augment
        self._filtered = 0
        if skip_overexposed:
            kept = []
            for p in img_paths:
                if self._is_overexposed(p):
                    self._filtered += 1
                else:
                    kept.append(p)
            img_paths = kept
        self.img_paths = img_paths

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, i: int):
        p = self.img_paths[i]
        img = np.array(Image.open(p).convert("RGB"))
        msk = np.array(Image.open(self.msk_dir / p.name))
        msk = (msk > 0).astype(np.uint8)

        if self.augment:
            out = self.augment(image=img, mask=msk)
            img, msk = out["image"], out["mask"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        msk = torch.from_numpy(msk).long()
        return img, msk

    @staticmethod
    def _is_overexposed(path: Path) -> bool:
        arr = np.array(Image.open(path))
        p01 = float(np.percentile(arr, 1))
        p50 = float(np.percentile(arr, 50))
        return p50 >= 250 and p01 >= 150


def build_train_aug():
    return A.Compose(
        [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ]
    )


def _normalize_roots(data_root: str | Path):
    if isinstance(data_root, Path):
        return [data_root]
    roots = [Path(p.strip()) for p in str(data_root).split(",") if p.strip()]
    return roots or [Path(str(data_root))]


def get_dataloaders(
    data_root: str | Path,
    batch_size: int = 16,
    num_workers: int = 4,
    skip_overexposed_train: bool = False,
):
    roots = _normalize_roots(data_root)
    train_sets = []
    val_sets = []
    for root in roots:
        print(f"Loading train dataset from {root}/train/images", flush=True)
        train_sets.append(
            BuildingsTiles(
                root / "train/images",
                root / "train/masks",
                augment=build_train_aug(),
                skip_overexposed=skip_overexposed_train,
            )
        )
        print(f"Loading val dataset from {root}/val/images", flush=True)
        val_sets.append(
            BuildingsTiles(
                root / "val/images",
                root / "val/masks",
                augment=None,
            )
        )

    train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
    val_ds = ConcatDataset(val_sets) if len(val_sets) > 1 else val_sets[0]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    if skip_overexposed_train:
        filtered = sum(getattr(ds, "_filtered", 0) for ds in train_sets)
        print(f"Filtered overexposed train tiles: {filtered} (kept {len(train_ds)})")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    return train_loader, val_loader
