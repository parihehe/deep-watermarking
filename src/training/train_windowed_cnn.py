"""Train the experimental windowed 1D-CNN blind watermark extractor.

This is the training pipeline for the paper-inspired **windowed** decoder. It is
a separate experimental track from the production Phase 8 decoder; nothing in
this file touches the frozen Phase 6 embedder or the production app.

Pipeline
--------
1. Build ``WindowedExtractionDataset`` for train + validation. Every sample is
   a real processed DIV2K image watermarked on the fly by the frozen embedder,
   then turned into per-bit 15-value windows with ground-truth labels from the
   actual embedded bits.
2. Train ``WindowedCNNExtractor`` per-bit with ``BCEWithLogitsLoss`` (Adam).
3. After each epoch, evaluate on the fixed validation set (loss / bit-accuracy /
   BER / exact payload-match / NC).
4. Save ``<run>_last.pt`` every epoch and ``<run>_best.pt`` whenever validation
   bit-accuracy improves. Early-stop on patience.
5. --resume allows continuing a run from a previous checkpoint instead of
   overwriting it; the checkpoint is inspected first and never clobbered.
6. Write ``training_log.csv`` + ``training_summary.json``.

Run
---
    python -m src.training.train_windowed_cnn --config configs/windowed_cnn.yaml

Fast CPU smoke check:
    python -m src.training.train_windowed_cnn --config configs/windowed_cnn.yaml \
        --epochs 3 --train-limit 16 --val-limit 8 --batch-size 8 --bit-length 8
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.models.windowed_cnn import (
    WindowedCNNConfig,
    WindowedCNNExtractor,
    load_windowed_checkpoint,
    save_windowed_checkpoint,
)
from src.training.windowed_dataset import (
    WindowedExtractionConfig,
    WindowedExtractionDataset,
)
from src.watermark.embed import EmbedConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]

__all__ = ["WindowedTrainConfig", "load_config", "main", "train"]


@dataclass
class WindowedTrainConfig:
    # data / payload
    processed_root: str = "data/processed/div2k_256"
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    train_limit: int | None = None
    val_limit: int | None = None
    image_size: int = 256
    bit_length: int = 64

    # frozen Phase 6 embedder parameters used to synthesise training data
    wavelet: str = "haar"
    mode: str = "symmetric"
    subband: str = "LL"
    extra_subbands: tuple[str, ...] = ()
    alpha: float = 0.02
    start_sv_index: int = 0

    # model (windowed CNN, paper architecture)
    window_size: int = 15
    conv1_channels: int = 32
    conv2_channels: int = 64
    kernel_size: int = 3
    fc_units: int = 64
    dropout_conv: float = 0.3
    dropout_fc: float = 0.5

    # optimisation
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stop_patience: int = 10
    num_workers: int = 0
    device: str = "auto"
    seed: int = 20260906
    resume: str | None = None

    # output
    checkpoint_dir: str = "models/experimental/windowed_cnn"
    results_dir: str = "results/windowed_cnn"
    run_name: str = "windowed_cnn"

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def embed_config(self) -> EmbedConfig:
        return EmbedConfig(
            wavelet=self.wavelet,
            subband=self.subband,
            extra_subbands=tuple(self.extra_subbands),
            alpha=self.alpha,
            bit_length=self.bit_length,
            start_sv_index=self.start_sv_index,
            mode=self.mode,
        )

    def model_config(self) -> WindowedCNNConfig:
        return WindowedCNNConfig(
            bit_length=self.bit_length,
            window_size=self.window_size,
            start_sv_index=self.start_sv_index,
            conv1_channels=self.conv1_channels,
            conv2_channels=self.conv2_channels,
            kernel_size=self.kernel_size,
            fc_units=self.fc_units,
            dropout_conv=self.dropout_conv,
            dropout_fc=self.dropout_fc,
            wavelet=self.wavelet,
            mode=self.mode,
            subband=self.subband,
            image_size=self.image_size,
        )


def load_config(path: str | Path, overrides: dict | None = None) -> WindowedTrainConfig:
    """Load ``configs/windowed_cnn.yaml`` (``windowed_cnn:`` block) then apply CLI overrides."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["windowed_cnn"]
    embed = raw.get("embed", {})
    payload = raw.get("payload", {})
    data = raw.get("data", {})
    model = raw.get("model", {})
    train_cfg = raw.get("train", {})
    output = raw.get("output", {})

    cfg = WindowedTrainConfig(
        processed_root=data.get("processed_root", "data/processed/div2k_256"),
        train_split=data.get("train_split", "train"),
        val_split=data.get("validation_split", "validation"),
        test_split=data.get("test_split", "test"),
        train_limit=data.get("train_limit"),
        val_limit=data.get("val_limit"),
        image_size=int(data.get("image_size", 256)),
        bit_length=int(payload.get("bit_length", 64)),
        wavelet=embed.get("wavelet", "haar"),
        mode=embed.get("mode", "symmetric"),
        subband=embed.get("subband", "LL"),
        extra_subbands=tuple(embed.get("extra_subbands", []) or []),
        alpha=float(embed.get("alpha", 0.02)),
        start_sv_index=int(embed.get("start_sv_index", 0)),
        window_size=int(model.get("window_size", 15)),
        conv1_channels=int(model.get("conv1_channels", 32)),
        conv2_channels=int(model.get("conv2_channels", 64)),
        kernel_size=int(model.get("kernel_size", 3)),
        fc_units=int(model.get("fc_units", 64)),
        dropout_conv=float(model.get("dropout_conv", 0.3)),
        dropout_fc=float(model.get("dropout_fc", 0.5)),
        epochs=int(train_cfg.get("epochs", 50)),
        batch_size=int(train_cfg.get("batch_size", 64)),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        early_stop_patience=int(train_cfg.get("early_stop_patience", 10)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        device=str(train_cfg.get("device", "auto")),
        seed=int(train_cfg.get("seed", 20260906)),
        resume=Path(train_cfg.get("resume")) if train_cfg.get("resume") else None,
        checkpoint_dir=output.get("checkpoint_dir", "models/experimental/windowed_cnn"),
        results_dir=output.get("results_dir", "results/windowed_cnn"),
        run_name=output.get("run_name", "windowed_cnn"),
    )
    for key, value in (overrides or {}).items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class EvalMetrics:
    loss: float
    bit_accuracy: float
    ber: float
    exact_match: float
    nc: float
    n_windows: int

    def as_dict(self) -> dict:
        return asdict(self)


def _evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str
) -> EvalMetrics:
    model.eval()
    total_loss = 0.0
    total_bits = 0
    correct_bits = 0
    exact = 0  # images whose whole payload is recovered
    nc_sum = 0.0
    n_images = 0
    with torch.no_grad():
        for windows, targets in loader:
            # windows: (B, bit_length, window_size); targets: (B, bit_length)
            windows = windows.to(device)
            targets = targets.to(device)
            b, nbits, wsize = windows.shape
            flat_w = windows.reshape(b * nbits, wsize)
            flat_t = targets.reshape(b * nbits)
            logits = model(flat_w)  # (B*nbits, 1)
            total_loss += criterion(logits, flat_t.unsqueeze(1)).item() * b

            preds = (torch.sigmoid(logits) > 0.5).float().reshape(b, nbits)
            matches = preds.eq(targets)
            correct_bits += int(matches.sum().item())
            total_bits += targets.numel()
            exact += int(matches.all(dim=1).sum().item())

            ref = targets * 2.0 - 1.0
            rec = preds * 2.0 - 1.0
            denom = ref.norm(dim=1) * rec.norm(dim=1)
            nc = torch.where(denom > 0, (ref * rec).sum(dim=1) / denom, torch.zeros_like(denom))
            nc_sum += float(nc.sum().item())
            n_images += b

    bit_acc = correct_bits / max(total_bits, 1)
    return EvalMetrics(
        loss=total_loss / max(n_images, 1),
        bit_accuracy=bit_acc,
        ber=1.0 - bit_acc,
        exact_match=exact / max(n_images, 1),
        nc=nc_sum / max(n_images, 1),
        n_windows=n_images,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _checkpoint_best_val_bit_acc(path: Path, device: str) -> float:
    """Best validation bit-accuracy recorded in ``path``, or ``-1`` if missing."""
    if not path.is_file():
        return -1.0
    try:
        _, meta = load_windowed_checkpoint(path, map_location=device)
        return float(meta.get("best_val_bit_acc", -1.0))
    except Exception:  # noqa: BLE001 - a corrupt best checkpoint must not block resume
        return -1.0


def _checkpoint_best_epoch(path: Path, device: str) -> int:
    if not path.is_file():
        return -1
    try:
        _, meta = load_windowed_checkpoint(path, map_location=device)
        return int(meta.get("best_epoch", -1))
    except Exception:  # noqa: BLE001
        return -1


def build_datasets(
    cfg: WindowedTrainConfig,
) -> tuple[WindowedExtractionDataset, WindowedExtractionDataset]:
    root = str((PROJECT_ROOT / cfg.processed_root).resolve())
    common = {
        "processed_root": root,
        "bit_length": cfg.bit_length,
        "model_config": cfg.model_config(),
        "embed_config": cfg.embed_config(),
        "image_size": cfg.image_size,
        "seed": cfg.seed,
    }
    train_ds = WindowedExtractionDataset(
        WindowedExtractionConfig(limit=cfg.train_limit, **common),
        cfg.train_split,
        deterministic=False,
    )
    val_ds = WindowedExtractionDataset(
        WindowedExtractionConfig(limit=cfg.val_limit, **common),
        cfg.val_split,
        deterministic=True,
    )
    return train_ds, val_ds


def build_model(cfg: WindowedTrainConfig) -> WindowedCNNExtractor:
    return WindowedCNNExtractor(cfg.model_config())


def train(cfg: WindowedTrainConfig, *, verbose: bool = True) -> dict:
    """Run the full windowed-CNN training loop and return a JSON-able summary."""
    _seed_everything(cfg.seed)
    device = cfg.resolved_device()

    train_ds, val_ds = build_datasets(cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    model = build_model(cfg)
    start_epoch = 0
    best_bit_acc = -1.0
    best_epoch = -1

    checkpoint_dir = (PROJECT_ROOT / cfg.checkpoint_dir).resolve()
    results_dir = (PROJECT_ROOT / cfg.results_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / f"{cfg.run_name}_last.pt"
    best_path = checkpoint_dir / f"{cfg.run_name}_best.pt"
    log_path = results_dir / f"{cfg.run_name}_training_log.csv"
    summary_path = results_dir / f"{cfg.run_name}_training_summary.json"

    # Resume: inspect the existing checkpoint, honour its config, and load
    # weights/epoch/best without overwriting anything it produced. The true
    # running best is read from the surviving best checkpoint (if any), not
    # from the resumed ``_last.pt`` whose metadata may lag by an epoch.
    if cfg.resume:
        if not Path(cfg.resume).is_file():
            raise FileNotFoundError(f"--resume checkpoint not found: {cfg.resume}")
        model, meta = load_windowed_checkpoint(cfg.resume, map_location=device)
        model = model.to(device)
        start_epoch = int(meta.get("epoch", 0)) + 1  # resume AFTER the saved epoch
        best_bit_acc = max(
            float(meta.get("best_val_bit_acc", -1.0)),
            _checkpoint_best_val_bit_acc(best_path, device),
        )
        best_epoch = max(
            int(meta.get("best_epoch", -1)),
            _checkpoint_best_epoch(best_path, device),
        )
        if verbose:
            print(
                f"[windowed] resuming from {cfg.resume} at epoch {start_epoch} "
                f"(prior best val_bit_acc={best_bit_acc:.4f} at epoch {best_epoch})"
            )
    else:
        model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if verbose:
        print(
            f"[windowed] device={device} params={model.num_parameters():,} "
            f"bit_length={cfg.bit_length} window_size={cfg.model_config().window_size} "
            f"alpha={cfg.alpha} train={len(train_ds)} val={len(val_ds)} epochs={cfg.epochs}"
        )

    rows: list[dict] = []
    baseline = _evaluate(model, val_loader, criterion, device)
    if verbose:
        print(
            f"[windowed] epoch {0:2d} (init)  val_loss={baseline.loss:.4f} "
            f"val_bit_acc={baseline.bit_accuracy:.4f} val_ber={baseline.ber:.4f}"
        )

    common_extra = {
        "train_config": asdict(cfg),
        "image_size": cfg.image_size,
        "bit_length": cfg.bit_length,
        "window_size": cfg.model_config().window_size,
    }
    best_metrics = baseline
    if best_bit_acc < 0:
        best_bit_acc = baseline.bit_accuracy
        best_epoch = 0
        best_metrics = baseline

    no_improve = 0
    early_stopped = False
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        started = time.time()
        running_loss = 0.0
        running_correct = 0
        running_bits = 0
        n_batches = 0
        for windows, targets in train_loader:
            b, nbits, wsize = windows.shape
            windows = windows.to(device)
            targets = targets.to(device)
            flat_w = windows.reshape(b * nbits, wsize)
            flat_t = targets.reshape(b * nbits)

            optimizer.zero_grad(set_to_none=True)
            logits = model(flat_w).squeeze(1)
            loss = criterion(logits, flat_t)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            running_loss += loss.item() * b
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                running_correct += int(preds.eq(flat_t).sum().item())
                running_bits += flat_t.numel()
            n_batches += 1

        train_loss = running_loss / max(n_batches * cfg.batch_size, 1)
        train_bit_acc = running_correct / max(running_bits, 1)
        val = _evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - started

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_bit_acc": round(train_bit_acc, 6),
            "val_loss": round(val.loss, 6),
            "val_bit_acc": round(val.bit_accuracy, 6),
            "val_ber": round(val.ber, 6),
            "val_exact_match": round(val.exact_match, 6),
            "val_nc": round(val.nc, 6),
            "seconds": round(elapsed, 2),
        }
        rows.append(row)
        if verbose:
            print(
                f"[windowed] epoch {epoch:2d}/{cfg.epochs}  "
                f"train_loss={train_loss:.4f} train_bit_acc={train_bit_acc:.4f}  "
                f"val_loss={val.loss:.4f} val_bit_acc={val.bit_accuracy:.4f} "
                f"val_ber={val.ber:.4f} val_exact={val.exact_match:.4f}  ({elapsed:.1f}s)"
            )

        # Persist a best checkpoint on the FIRST epoch even if it does not beat
        # the random-init baseline, so `best.pt` always exists (and resume never
        # reads an absent file); afterwards, improve-only.
        improved = not best_path.is_file() or val.bit_accuracy > best_bit_acc + 1e-6
        if improved:
            best_bit_acc = val.bit_accuracy
            best_epoch = epoch
            best_metrics = val
            no_improve = 0
            save_windowed_checkpoint(
                str(best_path),
                model,
                extra={
                    **common_extra,
                    "epoch": epoch,
                    "val_metrics": val.as_dict(),
                    "best_val_bit_acc": best_bit_acc,
                    "best_epoch": best_epoch,
                },
            )
            if verbose:
                print(f"[windowed]   -> new best val_bit_acc={best_bit_acc:.4f} at epoch {epoch}")
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                early_stopped = True
                if verbose:
                    print(
                        f"[windowed] early stopping after {epoch} epochs (patience {cfg.early_stop_patience})"
                    )
                break

        # `last` is saved AFTER the best-update so its metadata reflects the
        # current best (used by --resume to avoid regressing best.pt).
        save_windowed_checkpoint(
            str(last_path),
            model,
            extra={
                **common_extra,
                "epoch": epoch,
                "val_metrics": val.as_dict(),
                "best_val_bit_acc": best_bit_acc,
                "best_epoch": best_epoch,
            },
        )

    with log_path.open("w", newline="", encoding="utf-8") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "phase": "experimental-windowed-cnn",
        "description": "Paper-style windowed 1D-CNN blind watermark extractor — training run",
        "config": asdict(cfg),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": bool(torch.cuda.is_available()),
            "device": device,
        },
        "model_parameters": model.num_parameters(),
        "n_train_images": len(train_ds),
        "n_val_images": len(val_ds),
        "init_val_metrics": baseline.as_dict(),
        "best_epoch": best_epoch,
        "best_val_metrics": best_metrics.as_dict(),
        "final_val_metrics": rows[-1] if rows else None,
        "early_stopped": early_stopped,
        "resumed_from": str(cfg.resume) if cfg.resume else None,
        "checkpoints": {"last": str(last_path), "best": str(best_path)},
        "training_log": str(log_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if verbose:
        print(f"[windowed] wrote {log_path}")
        print(f"[windowed] wrote {summary_path}")
        print(
            f"[windowed] best epoch {best_epoch}: val_bit_acc={best_metrics.bit_accuracy:.4f} "
            f"val_ber={best_metrics.ber:.4f} val_exact={best_metrics.exact_match:.4f} -> {best_path}"
        )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the experimental windowed 1D-CNN blind extractor"
    )
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "windowed_cnn.yaml"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, dest="batch_size", default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--bit-length", type=int, dest="bit_length", default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--window-size", type=int, dest="window_size", default=None)
    p.add_argument("--train-limit", type=int, dest="train_limit", default=None)
    p.add_argument("--val-limit", type=int, dest="val_limit", default=None)
    p.add_argument("--num-workers", type=int, dest="num_workers", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-name", dest="run_name", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--early-stop-patience", type=int, dest="early_stop_patience", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    cfg = load_config(args.config, overrides)
    train(cfg)


if __name__ == "__main__":
    main()
