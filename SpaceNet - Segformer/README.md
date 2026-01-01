## SpaceNet (Vegas + Paris)

Quick status:
- Trained a SegFormer-based model on **Vegas + Paris tiles**.
- **Best val IoU ~0.85** on the mixed validation set (about 0.856).
- Training ran on Modal (L40S) with tiles at `/data/tiles_vegas_rgb` and `/data/tiles_paris_rgb`.

Run example (mix, init from Vegas-best):
```
modal run modal_train.py \
  --data-root /data/tiles_vegas_rgb,/data/tiles_paris_rgb \
  --ckpt-dir /checkpoints/mix \
  --init-from /checkpoints/best_segformer_vegas.pth \
  --epochs 30 \
  --batch-size 32 \
  --num-workers 12
```
