import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np

from dataset import RoofSatSample, build_dataset_index, load_img_mask, make_augmenter, split_train_val_test


def _require_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.utils.data as tud  # type: ignore

        return torch, nn, tud
    except Exception as e:
        raise RuntimeError(f"PyTorch is required for training: {e}")


@dataclass(frozen=True)
class TrainConfig:
    data_dirs: List[str]
    out_dir: str
    encoder: str
    input_size: int
    batch_size: int
    lr: float
    epochs: int
    val_frac: float
    seed: int
    amp: bool
    target_iou: Optional[float]
    patience: int
    white_roof_p: float
    test_frac: float
    test_seed: int
    num_workers: int


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    try:
        import random

        random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _to_tensor(img_rgb: np.ndarray, mask: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    import cv2

    if img_rgb.shape[0] != size or img_rgb.shape[1] != size:
        img = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        img = img_rgb
    if mask.shape[0] != size or mask.shape[1] != size:
        m = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    else:
        m = mask

    x = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))  # 3xSxS

    y = (m > 0).astype(np.float32)[None, ...]  # 1xSxS
    return x, y


def _maybe_crop(img_rgb: np.ndarray, mask: np.ndarray, size: int, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """Crop to size when the source tile is larger.

    This preserves native resolution for large aerial tiles (urban / multifamily scenes)
    vs. downscaling the whole tile.
    """

    h, w = mask.shape[:2]
    if h < size or w < size:
        return img_rgb, mask

    mode = (mode or "").strip().lower()
    if mode == "center":
        x0 = int((w - size) // 2)
        y0 = int((h - size) // 2)
    else:
        # default: random. Prefer crops that contain positive pixels.
        if int(mask.sum()) > 0 and float(np.random.rand()) < 0.7:
            ys, xs = np.nonzero(mask > 0)
            j = int(np.random.randint(0, len(xs)))
            cx = int(xs[j])
            cy = int(ys[j])
            x0 = int(np.clip(cx - size // 2, 0, w - size))
            y0 = int(np.clip(cy - size // 2, 0, h - size))
        else:
            x0 = int(np.random.randint(0, w - size + 1))
            y0 = int(np.random.randint(0, h - size + 1))

    img_c = img_rgb[y0 : y0 + size, x0 : x0 + size]
    mask_c = mask[y0 : y0 + size, x0 : x0 + size]
    return img_c, mask_c


def _white_roof_augment(img_rgb: np.ndarray, mask: np.ndarray, p: float) -> np.ndarray:
    """Simulate high-albedo (white) roofs on the positive class.

    This is a targeted domain-augmentation for bright apartment/commercial roofs.
    """

    if p <= 0:
        return img_rgb
    if float(np.random.rand()) > float(p):
        return img_rgb

    if img_rgb.dtype != np.uint8:
        img = np.clip(img_rgb, 0, 255).astype(np.uint8)
    else:
        img = img_rgb

    m = (mask > 0)
    if int(m.sum()) < 80:
        return img

    import cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    # OpenCV HSV: H 0..179, S 0..255, V 0..255
    sat_scale = float(np.random.uniform(0.05, 0.28))
    val_scale = float(np.random.uniform(1.12, 1.45))
    val_bias = float(np.random.uniform(0.0, 18.0))

    hsv[..., 1][m] = np.clip(hsv[..., 1][m] * sat_scale, 0, 255)
    hsv[..., 2][m] = np.clip(hsv[..., 2][m] * val_scale + val_bias, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Keep some original texture.
    alpha = float(np.random.uniform(0.55, 0.9))
    blended = (alpha * out.astype(np.float32) + (1.0 - alpha) * img.astype(np.float32)).clip(0, 255)
    return blended.astype(np.uint8)


def _iou_from_logits(logits, y_true, eps: float = 1e-6) -> float:
    import torch  # type: ignore

    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).float()
    inter = (pred * y_true).sum(dim=(1, 2, 3))
    union = (pred + y_true - pred * y_true).sum(dim=(1, 2, 3))
    iou = (inter + eps) / (union + eps)
    return float(iou.mean().detach().cpu().item())


class _RoofSatTorchDataset:
    def __init__(self, samples: List[RoofSatSample], input_size: int, aug_kind: str, crop_mode: str, white_roof_p: float):
        self.samples = samples
        self.input_size = input_size
        self.augment = make_augmenter(aug_kind)
        self.crop_mode = crop_mode
        self.white_roof_p = float(max(0.0, min(1.0, white_roof_p)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img, mask = load_img_mask(self.samples[idx])
        if self.augment is not None:
            img, mask = self.augment(img, mask)
        img, mask = _maybe_crop(img, mask, self.input_size, mode=self.crop_mode)
        img = _white_roof_augment(img, mask, self.white_roof_p)
        x, y = _to_tensor(img, mask, self.input_size)
        return x, y


class _MlflowRun:
    """Optional MLflow logging. Every method is a no-op when no tracking URI is given."""

    def __init__(self, tracking_uri: Optional[str], experiment: str, run_name: Optional[str]):
        self._mlflow = None
        if not tracking_uri:
            return
        try:
            import mlflow  # type: ignore
        except Exception as e:
            raise RuntimeError(f"mlflow is required for --mlflow-uri: {e}")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name)
        self._mlflow = mlflow

    def params(self, values: dict) -> None:
        if self._mlflow:
            self._mlflow.log_params(values)

    def metrics(self, values: dict, step: Optional[int] = None) -> None:
        if self._mlflow:
            self._mlflow.log_metrics({k: float(v) for k, v in values.items()}, step=step)

    def tag(self, key: str, value: str) -> None:
        if self._mlflow:
            self._mlflow.set_tag(key, value)

    def artifact(self, path: str) -> None:
        if self._mlflow:
            self._mlflow.log_artifact(path)

    def end(self, status: str = "FINISHED") -> None:
        if self._mlflow:
            self._mlflow.end_run(status=status)


def _score_split(model, samples: List[RoofSatSample], input_size: int, device) -> dict:
    """Pixel-level IoU, Dice, precision and recall over center crops, plus mean per-image IoU."""
    import torch  # type: ignore

    ds = _RoofSatTorchDataset(samples, input_size, aug_kind="none", crop_mode="center", white_roof_p=0.0)
    tp = fp = fn = 0
    per_image: List[float] = []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            logits = model(torch.as_tensor(x).unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits)[0, 0] >= 0.5).cpu().numpy()
            truth = np.asarray(y)[0] > 0.5
            inter = int((pred & truth).sum())
            extra = int((pred & ~truth).sum())
            missed = int((~pred & truth).sum())
            tp, fp, fn = tp + inter, fp + extra, fn + missed
            per_image.append(inter / max(1, inter + extra + missed))
    return {
        "test_pixel_iou": tp / max(1, tp + fp + fn),
        "test_mean_image_iou": float(np.mean(per_image)) if per_image else 0.0,
        "test_dice": 2 * tp / max(1, 2 * tp + fp + fn),
        "test_precision": tp / max(1, tp + fp),
        "test_recall": tp / max(1, tp + fn),
    }


def _evaluate_test(cfg: TrainConfig, test_s: List[RoofSatSample], best_path: str, device, tracker: _MlflowRun) -> None:
    """Score best.pt once on the held-out test split and save the results next to it."""
    import segmentation_models_pytorch as smp  # type: ignore
    import torch  # type: ignore

    ckpt = torch.load(best_path, map_location="cpu")
    model = smp.Unet(encoder_name=ckpt["encoder"], encoder_weights=None, in_channels=3, classes=1, activation=None)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    results = {**_score_split(model, test_s, cfg.input_size, device), "best_epoch": int(ckpt["epoch"]), "best_val_iou": float(ckpt["val_iou"])}

    metrics_path = os.path.join(cfg.out_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({**results, "n_test": len(test_s)}, f, indent=2, sort_keys=True)
    images_path = os.path.join(cfg.out_dir, "test_images.txt")
    with open(images_path, "w", encoding="utf-8") as f:
        f.write("\n".join(os.path.basename(s.img_path) for s in test_s) + "\n")

    tracker.metrics(results)
    tracker.artifact(metrics_path)
    tracker.artifact(images_path)
    print(json.dumps({"event": "test", **{k: round(v, 5) if isinstance(v, float) else v for k, v in results.items()}}))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        action="append",
        required=True,
        help="Dataset root. Can be provided multiple times to mix datasets.",
    )
    p.add_argument("--out-dir", required=True, help="Output run directory")
    p.add_argument("--encoder", default="resnet34", help="SMP encoder name")
    p.add_argument("--input-size", type=int, default=512, help="Training crop size")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--amp", action="store_true", help="Use mixed precision")
    p.add_argument("--target-iou", type=float, default=None, help="Stop early when val IoU >= this value")
    p.add_argument("--patience", type=int, default=7, help="Early stop after N epochs without improvement")
    p.add_argument("--white-roof-p", type=float, default=0.25, help="Prob of synthetic white-roof augmentation (0..1)")
    p.add_argument("--test-frac", type=float, default=0.15, help="Fraction held out as a fixed test set (0 disables)")
    p.add_argument("--test-seed", type=int, default=2026, help="Seed for the test split (independent of --seed)")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (use 0 on Windows)")
    p.add_argument("--mlflow-uri", default=None, help="MLflow tracking URI, e.g. sqlite:///mlflow.db (off when unset)")
    p.add_argument("--mlflow-experiment", default="sunnyview-roofseg", help="MLflow experiment name")
    p.add_argument("--run-name", default=None, help="MLflow run name")
    args = p.parse_args()

    cfg = TrainConfig(
        data_dirs=[str(d) for d in (args.data_dir or [])],
        out_dir=args.out_dir,
        encoder=args.encoder,
        input_size=args.input_size,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        val_frac=args.val_frac,
        seed=args.seed,
        amp=bool(args.amp),
        target_iou=float(args.target_iou) if args.target_iou is not None else None,
        patience=int(args.patience),
        white_roof_p=float(args.white_roof_p),
        test_frac=float(args.test_frac),
        test_seed=int(args.test_seed),
        num_workers=max(0, int(args.num_workers)),
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)

    _seed_all(cfg.seed)
    torch, nn, tud = _require_torch()

    try:
        import segmentation_models_pytorch as smp  # type: ignore
    except Exception as e:
        raise RuntimeError(f"segmentation-models-pytorch is required: {e}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        # Apple Silicon acceleration when available.
        try:
            device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        except Exception:
            device = torch.device("cpu")

    samples: List[RoofSatSample] = []
    for d in cfg.data_dirs:
        samples.extend(build_dataset_index(d))
    train_s, val_s, test_s = split_train_val_test(samples, cfg.val_frac, cfg.test_frac, seed=cfg.seed, test_seed=cfg.test_seed)
    print(json.dumps({"event": "split", "train": len(train_s), "val": len(val_s), "test": len(test_s)}))

    tracker = _MlflowRun(args.mlflow_uri, args.mlflow_experiment, args.run_name)
    tracker.params({k: v for k, v in asdict(cfg).items() if k not in ("data_dirs", "out_dir")})
    tracker.params({"n_train": len(train_s), "n_val": len(val_s), "n_test": len(test_s)})
    tracker.tag("device", torch.cuda.get_device_name(0) if device.type == "cuda" else device.type)
    tracker.artifact(os.path.join(cfg.out_dir, "config.json"))

    train_ds = _RoofSatTorchDataset(train_s, cfg.input_size, aug_kind="train", crop_mode="random", white_roof_p=cfg.white_roof_p)
    val_ds = _RoofSatTorchDataset(val_s, cfg.input_size, aug_kind="none", crop_mode="center", white_roof_p=0.0)

    train_loader = tud.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = tud.DataLoader(val_ds, batch_size=max(1, cfg.batch_size // 2), shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = smp.Unet(
        encoder_name=cfg.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
    model = model.to(device)

    bce = nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def loss_fn(logits, y_true):
        return 0.55 * bce(logits, y_true) + 0.45 * dice(logits, y_true)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    best_iou: float = -1.0
    best_epoch: int = 0
    best_path = os.path.join(cfg.out_dir, "best.pt")
    last_path = os.path.join(cfg.out_dir, "last.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb_t = xb.to(device) if torch.is_tensor(xb) else torch.from_numpy(xb).to(device)
            yb_t = yb.to(device) if torch.is_tensor(yb) else torch.from_numpy(yb).to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(xb_t)
                loss = loss_fn(logits, yb_t)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_loss += float(loss.detach().cpu().item())

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb_t = xb.to(device) if torch.is_tensor(xb) else torch.from_numpy(xb).to(device)
                yb_t = yb.to(device) if torch.is_tensor(yb) else torch.from_numpy(yb).to(device)
                logits = model(xb_t)
                loss = loss_fn(logits, yb_t)
                val_loss += float(loss.detach().cpu().item())
                val_iou += _iou_from_logits(logits, yb_t)

        val_loss /= max(1, len(val_loader))
        val_iou /= max(1, len(val_loader))

        # Save last.
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "encoder": cfg.encoder,
                "input_size": cfg.input_size,
                "val_iou": val_iou,
            },
            last_path,
        )

        # Save best.
        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "encoder": cfg.encoder,
                    "input_size": cfg.input_size,
                    "val_iou": val_iou,
                },
                best_path,
            )

        msg = {
            "epoch": epoch,
            "device": str(device),
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_iou": round(val_iou, 5),
            "best_iou": round(best_iou, 5),
            "best_epoch": int(best_epoch),
        }
        print(json.dumps(msg))
        tracker.metrics({"train_loss": train_loss, "val_loss": val_loss, "val_iou": val_iou, "best_iou": best_iou}, step=epoch)

        if cfg.target_iou is not None and val_iou >= cfg.target_iou:
            print(json.dumps({"event": "target_reached", "epoch": epoch, "val_iou": round(val_iou, 5), "target": cfg.target_iou}))
            tracker.tag("stop_event", "target_reached")
            break

        if cfg.patience > 0 and epoch - best_epoch >= cfg.patience:
            print(json.dumps({"event": "early_stop", "epoch": epoch, "best_epoch": best_epoch, "best_iou": round(best_iou, 5)}))
            tracker.tag("stop_event", "early_stop")
            break

    if test_s:
        _evaluate_test(cfg, test_s, best_path, device, tracker)
    tracker.end()


def _fail_active_mlflow_run() -> None:
    mlflow = sys.modules.get("mlflow")
    if mlflow is not None and mlflow.active_run() is not None:
        mlflow.end_run(status="FAILED")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        _fail_active_mlflow_run()
        raise
