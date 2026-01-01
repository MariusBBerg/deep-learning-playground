from __future__ import annotations

import modal

app = modal.App("spacenet-segformer")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "albumentations",
        "numpy",
        "Pillow",
        "opencv-python-headless",
    )
    .add_local_file("spacenet_dataset.py", remote_path="/root/spacenet_dataset.py")
    .add_local_file("spacenet_train.py", remote_path="/root/spacenet_train.py")
)

data_vol = modal.Volume.from_name("spacenet-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("spacenet-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="L40S",
    cpu=8.0,
    memory=32768,
    timeout=86400,
    volumes={"/data": data_vol, "/checkpoints": ckpt_vol},
)
def train(
    data_root: str = "/data/tiles_vegas_rgb",
    ckpt_dir: str = "/checkpoints",
    batch_size: int = 32,
    num_workers: int = 8,
    epochs: int = 25,
    lr: float = 8e-6,
    weight_decay: float = 1e-2,
    skip_overexposed_train: bool = False,
    init_from: str | None = None,
):
    import sys

    sys.path.append("/root")
    from spacenet_train import train_segformer

    best_iou = train_segformer(
        data_root=data_root,
        ckpt_dir=ckpt_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        skip_overexposed_train=skip_overexposed_train,
        init_from=init_from,
    )

    ckpt_vol.commit()
    return best_iou


@app.local_entrypoint()
def main(
    data_root: str = "/data/tiles_vegas_rgb",
    ckpt_dir: str = "/checkpoints",
    batch_size: int = 16,
    num_workers: int = 8,
    epochs: int = 50,
    lr: float = 3e-5,
    weight_decay: float = 1e-2,
    skip_overexposed_train: bool = False,
    init_from: str | None = None,
):
    train.remote(
        data_root=data_root,
        ckpt_dir=ckpt_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        skip_overexposed_train=skip_overexposed_train,
        init_from=init_from,
    )
