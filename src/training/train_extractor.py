"""Phase 8 — training + validation pipeline for the blind CNN watermark extractor.

Pipeline
--------
1. Build a :class:`~src.training.watermark_dataset.WatermarkExtractionDataset`
   for ``train`` and ``validation``. Every sample is a real processed DIV2K
   image watermarked on the fly by the **frozen Phase 6 DWT-SVD embedder** with
   a random payload.
2. Train :class:`~src.models.cnn_extractor.BlindCNNExtractor` with per-bit
   ``BCEWithLogitsLoss`` (Adam).
3. After each epoch, evaluate on the fixed validation set and record
   loss / bit-accuracy / BER / exact-match rate / normalised correlation.
4. Save ``last.pt`` every epoch and ``best.pt`` whenever validation bit-accuracy
   improves. Both checkpoints are self-describing (they embed the
   ``ExtractorConfig``).
5. Write ``training_log.csv`` and ``training_summary.json`` under the results
   directory.

Nothing in this file touches the frozen baseline or the Phase 7 web app.

Run
---
    .venv/Scripts/python.exe -m src.training.train_extractor --config configs/cnn_extractor.yaml

Common overrides for a fast CPU check::

    .venv/Scripts/python.exe -m src.training.train_extractor \
        --config configs/cnn_extractor.yaml \
        --epochs 3 --train-limit 64 --val-limit 16 --batch-size 8 --bit-length 16
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

from src.models.cnn_extractor import BlindCNNExtractor, ExtractorConfig, save_checkpoint
from src.training.watermark_dataset import (
    WatermarkExtractionConfig,
    WatermarkExtractionDataset,
)
from src.watermark.embed import EmbedConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]

__all__ = ["TrainConfig", "load_config", "main", "train"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # data / payload
    processed_root: str = "data/processed/div2k_256"
    train_split: str = "train"
    val_split: str = "validation"
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

    # model
    sv_features: int = 128
    conv_channels: int = 64
    conv_layers: int = 4
    kernel_size: int = 7
    dropout: float = 0.1

    # optimisation
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 0
    device: str = "auto"
    seed: int = 20260901

    # output
    checkpoint_dir: str = "models/phase8_cnn"
    results_dir: str = "results/phase8_cnn"
    run_name: str = "phase8_cnn"

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


def load_config(path: str | Path, overrides: dict | None = None) -> TrainConfig:
    """Load ``configs/cnn_extractor.yaml`` (``cnn_extractor:`` block) into a
    :class:`TrainConfig`, then apply optional CLI overrides."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["cnn_extractor"]
    embed = raw.get("embed", {})
    payload = raw.get("payload", {})
    data = raw.get("data", {})
    model = raw.get("model", {})
    train_cfg = raw.get("train", {})
    output = raw.get("output", {})

    cfg = TrainConfig(
        processed_root=data.get("processed_root", "data/processed/div2k_256"),
        train_split=data.get("train_split", "train"),
        val_split=data.get("validation_split", "validation"),
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
        sv_features=int(model.get("sv_features", 128)),
        conv_channels=int(model.get("conv_channels", 64)),
        conv_layers=int(model.get("conv_layers", 4)),
        kernel_size=int(model.get("kernel_size", 7)),
        dropout=float(model.get("dropout", 0.1)),
        epochs=int(train_cfg.get("epochs", 20)),
        batch_size=int(train_cfg.get("batch_size", 16)),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        device=str(train_cfg.get("device", "auto")),
        seed=int(train_cfg.get("seed", 20260901)),
        checkpoint_dir=output.get("checkpoint_dir", "models/phase8_cnn"),
        results_dir=output.get("results_dir", "results/phase8_cnn"),
        run_name=output.get("run_name", "phase8_cnn"),
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
    n_samples: int

    def as_dict(self) -> dict:
        return asdict(self)


def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> EvalMetrics:
    model.eval()
    total_loss = 0.0
    total_bits = 0
    correct_bits = 0
    exact = 0
    nc_sum = 0.0
    n_samples = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            total_loss += criterion(logits, targets).item() * images.size(0)

            preds = (torch.sigmoid(logits) > 0.5).float()
            matches = preds.eq(targets)
            correct_bits += int(matches.sum().item())
            total_bits += targets.numel()
            exact += int(matches.all(dim=1).sum().item())

            # normalised correlation on bipolar {-1, +1}, per sample
            ref = targets * 2.0 - 1.0
            rec = preds * 2.0 - 1.0
            denom = ref.norm(dim=1) * rec.norm(dim=1)
            nc = torch.where(denom > 0, (ref * rec).sum(dim=1) / denom, torch.zeros_like(denom))
            nc_sum += float(nc.sum().item())
            n_samples += images.size(0)

    bit_acc = correct_bits / max(total_bits, 1)
    return EvalMetrics(
        loss=total_loss / max(n_samples, 1),
        bit_accuracy=bit_acc,
        ber=1.0 - bit_acc,
        exact_match=exact / max(n_samples, 1),
        nc=nc_sum / max(n_samples, 1),
        n_samples=n_samples,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_datasets(cfg: TrainConfig) -> tuple[WatermarkExtractionDataset, WatermarkExtractionDataset]:
    root = str((PROJECT_ROOT / cfg.processed_root).resolve())
    common = {
        "processed_root": root,
        "bit_length": cfg.bit_length,
        "embed_config": cfg.embed_config(),
        "image_size": cfg.image_size,
        "seed": cfg.seed,
    }
    train_ds = WatermarkExtractionDataset(
        WatermarkExtractionConfig(limit=cfg.train_limit, **common),
        cfg.train_split,
        deterministic=False,
    )
    val_ds = WatermarkExtractionDataset(
        WatermarkExtractionConfig(limit=cfg.val_limit, **common),
        cfg.val_split,
        deterministic=True,
    )
    return train_ds, val_ds


def build_model(cfg: TrainConfig) -> BlindCNNExtractor:
    return BlindCNNExtractor(
        ExtractorConfig(
            bit_length=cfg.bit_length,
            sv_features=cfg.sv_features,
            conv_channels=cfg.conv_channels,
            conv_layers=cfg.conv_layers,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
            image_size=cfg.image_size,
        )
    )


def train(cfg: TrainConfig, *, verbose: bool = True) -> dict:
    """Run the full Phase 8 training loop and return a JSON-able summary."""
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

    model = build_model(cfg).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    checkpoint_dir = (PROJECT_ROOT / cfg.checkpoint_dir).resolve()
    results_dir = (PROJECT_ROOT / cfg.results_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / f"{cfg.run_name}_last.pt"
    best_path = checkpoint_dir / f"{cfg.run_name}_best.pt"
    log_path = results_dir / f"{cfg.run_name}_training_log.csv"
    summary_path = results_dir / f"{cfg.run_name}_training_summary.json"

    if verbose:
        print(
            f"[phase8] device={device} params={model.num_parameters():,} "
            f"bit_length={cfg.bit_length} alpha={cfg.alpha} "
            f"train={len(train_ds)} val={len(val_ds)} epochs={cfg.epochs}"
        )

    baseline = _evaluate(model, val_loader, criterion, device)
    if verbose:
        print(
            f"[phase8] epoch  0 (init)  val_loss={baseline.loss:.4f} "
            f"val_bit_acc={baseline.bit_accuracy:.4f} val_ber={baseline.ber:.4f}"
        )

    rows: list[dict] = []
    best_bit_acc = -1.0
    best_epoch = 0
    best_metrics = baseline

    common_extra = {
        "train_config": asdict(cfg),
        "image_size": cfg.image_size,
        "bit_length": cfg.bit_length,
    }

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        started = time.time()
        running_loss = 0.0
        running_correct = 0
        running_bits = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                running_correct += int(preds.eq(targets).sum().item())
                running_bits += targets.numel()

        train_loss = running_loss / max(len(train_ds), 1)
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
                f"[phase8] epoch {epoch:2d}/{cfg.epochs}  "
                f"train_loss={train_loss:.4f} train_bit_acc={train_bit_acc:.4f}  "
                f"val_loss={val.loss:.4f} val_bit_acc={val.bit_accuracy:.4f} "
                f"val_ber={val.ber:.4f} val_exact={val.exact_match:.3f}  ({elapsed:.1f}s)"
            )

        save_checkpoint(
            str(last_path),
            model,
            extra={**common_extra, "epoch": epoch, "val_metrics": val.as_dict()},
        )
        if val.bit_accuracy > best_bit_acc:
            best_bit_acc = val.bit_accuracy
            best_epoch = epoch
            best_metrics = val
            save_checkpoint(
                str(best_path),
                model,
                extra={**common_extra, "epoch": epoch, "val_metrics": val.as_dict()},
            )

    with log_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "phase": 8,
        "description": "Blind CNN watermark extractor — training run",
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
        "checkpoints": {
            "last": str(last_path),
            "best": str(best_path),
        },
        "training_log": str(log_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if verbose:
        print(f"[phase8] wrote {log_path}")
        print(f"[phase8] wrote {summary_path}")
        print(
            f"[phase8] best epoch {best_epoch}: val_bit_acc={best_metrics.bit_accuracy:.4f} "
            f"val_ber={best_metrics.ber:.4f} -> {best_path}"
        )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 8 — train the blind CNN watermark extractor")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "cnn_extractor.yaml"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, dest="batch_size", default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--bit-length", type=int, dest="bit_length", default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--train-limit", type=int, dest="train_limit", default=None)
    p.add_argument("--val-limit", type=int, dest="val_limit", default=None)
    p.add_argument("--num-workers", type=int, dest="num_workers", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-name", dest="run_name", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    cfg = load_config(args.config, overrides)
    train(cfg)


if __name__ == "__main__":
    main()
