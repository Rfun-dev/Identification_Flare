"""Loop training dan prediksi yang dipakai bersama oleh skrip-skrip training dan evaluasi."""
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from flare.data import load_sequence


def set_seed(seed: int):
    """Samakan titik awal semua pengacak agar hasil training bisa diulang."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(dataset, cfg, sampler=None) -> DataLoader:
    return DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler,
                      num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=cfg.num_workers > 0)


def make_optimizer(model, cfg, epochs: int):
    """AdamW dengan dua learning rate, plus jadwal cosine.

    CNN sudah pretrained, jadi cukup disetel pelan (backbone_lr). Transformer dan head
    masih baru, jadi belajar lebih cepat (lr). Learning rate turun perlahan sampai `epochs`.
    """
    cnn_params = list(model.cnn.parameters())
    other_params = [p for name, p in model.named_parameters() if not name.startswith("cnn.")]
    optimizer = torch.optim.AdamW([{"params": cnn_params, "lr": cfg.backbone_lr},
                                   {"params": other_params, "lr": cfg.lr}],
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return optimizer, scheduler


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, amp) -> float:
    """Satu kali melewati seluruh data training. Kembali: rata-rata loss."""
    model.train()
    total, n = 0.0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp):
            loss = loss_fn(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # cegah lonjakan gradien
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def predict_proba(model, loader, device, amp):
    """Probabilitas flare (0-1) untuk setiap sampel di loader, urut sesuai dataset."""
    model.eval()
    probs = []
    for x, _ in tqdm(loader, desc="predict", leave=False):
        with torch.autocast(device.type, enabled=amp):
            logits = model(x.to(device, non_blocking=True))
        probs.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(probs).numpy()


@torch.no_grad()
def predict_logits(model, loader, device):
    """Keluaran mentah model (tanpa sigmoid), mis. untuk model regresi di train_xclass.py."""
    model.eval()
    outputs = []
    for x, _ in tqdm(loader, desc="predict", leave=False):
        outputs.append(model(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(outputs).numpy()


@torch.no_grad()
def predict_one(model, paths, device) -> float:
    """Skor flare (0-1) untuk satu urutan PNG (urut lama -> baru)."""
    x = load_sequence(paths).to(device)
    return torch.sigmoid(model(x)).item()
