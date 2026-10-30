"""Langkah 3 - Melatih FlareFormer.

Model terbaik (TSS validasi tertinggi) disimpan ke <run_dir>/best.pt beserta
config dan threshold-nya; riwayat per epoch disimpan ke <run_dir>/history.csv.

Contoh:
    python train.py                                  # pengaturan default
    python train.py --n_frames 1 --run_dir runs/single_frame   # tanpa dimensi waktu
"""
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from flare.config import config_from_args, get_parser
from flare.data import BalancedSampler, MagnetogramDataset, build_samples, load_index
from flare.engine import get_device, make_loader, make_optimizer, predict_proba, set_seed, train_one_epoch
from flare.metrics import best_threshold, scores
from flare.model import build_model


def describe(name: str, samples: pd.DataFrame):
    n_pos = int(samples["label"].sum())
    print(f"  {name:5s}: {len(samples):7,} sampel | flare: {n_pos:6,} "
          f"({100 * n_pos / max(len(samples), 1):.1f}%) | AR: {samples['ar'].nunique()}")


def load_data(cfg):
    """Sampel train (1 per train_stride_h jam) dan validasi (1 per eval_stride_h jam)."""
    train_s = build_samples(load_index(cfg, "train"), cfg, cfg.train_stride_h)
    val_s = build_samples(load_index(cfg, "val"), cfg, cfg.eval_stride_h)
    print(f"Target: flare >= {cfg.flare_class}1.0 dalam 24 jam | {cfg.n_frames} frame / sampel")
    describe("train", train_s)
    describe("val", val_s)
    if train_s["label"].sum() == 0 or val_s["label"].sum() == 0:
        raise SystemExit("Tidak ada sampel flare di train/val. Unduh lebih banyak data.")
    return train_s, val_s


def validate(model, val_dl, val_ds, device, amp) -> dict:
    """Skor validasi pada threshold terbaiknya (threshold dipilih dari validasi itu sendiri)."""
    prob = predict_proba(model, val_dl, device, amp)
    threshold = min(best_threshold(val_ds.labels, prob), 1.0)   # roc_curve bisa memberi inf -> batasi 1
    return scores(val_ds.labels, prob, threshold)


def main():
    cfg = config_from_args(get_parser(__doc__).parse_args())
    set_seed(cfg.seed)
    device = get_device()
    amp = cfg.amp and device.type == "cuda"
    torch.backends.cudnn.benchmark = True       # ukuran input selalu sama -> cari algoritma konvolusi tercepat
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    gpu = f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    print(f"Device: {device}{gpu} | AMP: {amp}")

    # 1. Data: train memakai augmentasi + sampler seimbang, validasi apa adanya
    train_s, val_s = load_data(cfg)
    train_ds = MagnetogramDataset(train_s, cfg.image_dir, augment=True)
    val_ds = MagnetogramDataset(val_s, cfg.image_dir)
    sampler = BalancedSampler(train_ds.labels, cfg.neg_ratio, cfg.seed)
    train_dl = make_loader(train_ds, cfg, sampler=sampler)
    val_dl = make_loader(val_ds, cfg)

    # 2. Model, loss (kesalahan pada flare diberi bobot lebih), optimizer
    model = build_model(cfg).to(device)
    print(f"Parameter model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} juta")
    pos_weight = torch.tensor(sampler.n_neg / len(sampler.pos), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer, scheduler = make_optimizer(model, cfg, cfg.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    # 3. Loop training: simpan model dengan TSS validasi tertinggi, berhenti bila tidak membaik
    best_tss, bad_epochs, history = -1.0, 0, []
    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        loss = train_one_epoch(model, train_dl, optimizer, loss_fn, scaler, device, amp)
        scheduler.step()
        val = validate(model, val_dl, val_ds, device, amp)

        history.append({"epoch": epoch, "train_loss": loss,
                        **{f"val_{k}": val[k] for k in ("TSS", "HSS", "AUC", "recall", "precision")},
                        "threshold": val["threshold"]})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        improved = val["TSS"] > best_tss
        print(f"Epoch {epoch:2d} | loss {loss:.4f} | val TSS {val['TSS']:.3f} "
              f"HSS {val['HSS']:.3f} AUC {val['AUC']:.3f} | thr {val['threshold']:.2f} | "
              f"{time.time() - start:.0f}s {'*' if improved else ''}")

        if improved:
            best_tss, bad_epochs = val["TSS"], 0
            torch.save({"model": model.state_dict(), "config": asdict(cfg),
                        "threshold": val["threshold"], "epoch": epoch, "val": val},
                       run_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Early stopping: TSS tidak naik selama {cfg.patience} epoch.")
                break

    print(f"Selesai. TSS validasi terbaik = {best_tss:.3f} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
