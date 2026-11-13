# RoofSat training (roof/building mask)

RoofSat provides per-image `building_masks/` suitable for training a binary segmentation model.

Important: RoofSat is **research-only** (Airbus imagery). Confirm license fit before using in production.

## 1) Download + unzip

From `services/segmenter/train_roofsat/`:

```bash
curl -L -o Roofsat.zip "http://www-sop.inria.fr/members/Florent.Lafarge/benchmark/Roofsat.zip"
unzip -q Roofsat.zip -d .
```

You should now have `Roofsat/img_color/` and `Roofsat/building_masks/`.

## 2) Install training deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../requirements-train.txt
```

## 3) Train

```bash
python train.py --data-dir ./Roofsat --out-dir ./runs/roofsat_unet
```

Train until a target IoU (early stop):

```bash
python train.py --data-dir ./Roofsat --out-dir ./runs/roofsat_unet --target-iou 0.95
```

More "white roof" robustness (synthetic augmentation):

```bash
python train.py --data-dir ./Roofsat --out-dir ./runs/roofsat_unet --white-roof-p 0.35
```

Mix datasets (repeat `--data-dir`):

```bash
python train.py \
  --data-dir ./Roofsat \
  --data-dir /path/to/another_building_dataset \
  --out-dir ./runs/mixed_unet \
  --target-iou 0.95
```

This writes:
- `runs/roofsat_unet/best.pt` (PyTorch, the epoch with the best validation IoU)
- `runs/roofsat_unet/test_metrics.json` and `test_images.txt` (see below)

On Windows, add `--num-workers 0`: DataLoader workers are started with spawn there and cannot pickle the augmentation function.

### Held-out test split

Before the train/val split, `--test-frac` (default `0.15`) holds out a fixed test set chosen from the sorted image paths with `--test-seed` (default `2026`). It stays the same when you change `--seed`, `--val-frac` or any hyperparameter, so runs are comparable. Training and checkpoint selection only see train and val. After training, `best.pt` is scored once on the test set (pixel IoU, mean per-image IoU, Dice, precision, recall) and the result is written to `test_metrics.json`, with the test file names in `test_images.txt`. `--test-frac 0` restores the original train/val-only split.

Choose settings by validation IoU, not test IoU, or the test score stops being a fair estimate.

### Experiment tracking (MLflow)

Tracking is off unless `--mlflow-uri` is set. With it, each run logs its config, split sizes, per-epoch `train_loss`, `val_loss`, `val_iou` and `best_iou`, the final test metrics, and `config.json`, `test_metrics.json` and `test_images.txt` as artifacts. A crash marks the run `FAILED`.

```bash
python train.py --data-dir ./Roofsat --out-dir ./runs/baseline --mlflow-uri sqlite:///mlflow.db --run-name baseline
python train.py --data-dir ./Roofsat --out-dir ./runs/no_white_roof --white-roof-p 0 --mlflow-uri sqlite:///mlflow.db --run-name no-white-roof
python train.py --data-dir ./Roofsat --out-dir ./runs/epochs40 --epochs 40 --mlflow-uri sqlite:///mlflow.db --run-name epochs40

mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5055   # then open http://127.0.0.1:5055
```

Results of those three runs on RoofSat (550 images: 398 train, 70 val, 82 test; one seed each; RTX 4070 Laptop GPU, November 2026):

| Run | Best epoch | Val IoU | Test pixel IoU | Test Dice | Test precision / recall | Train time |
|---|---|---|---|---|---|---|
| baseline (defaults) | 25 / 25 | 0.918 | 0.933 | 0.965 | 0.966 / 0.964 | 9.8 min |
| `--white-roof-p 0` | 21 / 25 | 0.918 | 0.934 | 0.966 | 0.971 / 0.960 | 9.8 min |
| `--epochs 40` | 35 / 40 | 0.928 | 0.940 | 0.969 | 0.973 / 0.965 | 13.9 min |

Training longer helps (picked by val IoU). White-roof augmentation makes no measurable difference on RoofSat, which has few bright commercial roofs; its value would need a test set that has them. Differences of about 0.001 are within run-to-run noise.

## 4) Export ONNX for the FastAPI segmenter

```bash
python export_onnx.py --checkpoint ./runs/roofsat_unet/best.pt --out ../models/roofsat_unet.onnx
```

## 5) Run the segmenter using ONNX

From `services/segmenter/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-onnx.txt

export ROOFSEG_ONNX_PATH="$PWD/models/roofsat_unet.onnx"
uvicorn main:app --host 0.0.0.0 --port 8000
```

In the Next.js app `.env.local`:

```bash
SEGMENT_SERVICE_URL=http://localhost:8000
```

## Notes

- The model predicts *building/roof pixels* for the whole tile. The app chooses the *correct* roof using the address-matched `osmFootprint` it sends in `meta`.
- If `ROOFSEG_ONNX_PATH` is set, the segmenter tries ONNX first and falls back to SAM (if configured) or the OSM footprint.

## Improving apartments / multifamily coverage

- Add denser urban building-mask data (apartments/condos) to reduce failures on large/complex roofs.
- This training code will read any images in `img_color/` with matching binary masks in `building_masks/` (supports `.png`, `.jpg`, `.tif`).
- For very large aerial tiles, training now takes random `--input-size` crops to preserve native resolution (instead of shrinking the entire tile).
- Candidate public datasets to augment RoofSat (review licenses/terms before use):
  - Inria Aerial Image Labeling: https://project.inria.fr/aerialimagelabeling/
  - SpaceNet building datasets: https://spacenet.ai/datasets/
  - Google Open Buildings: https://sites.research.google/open-buildings/
