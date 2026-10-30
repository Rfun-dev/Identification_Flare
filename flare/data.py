"""Membaca magnetogram, menyusun urutan frame, augmentasi, dan sampler seimbang."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

# Batas bawah fluks puncak sinar-X GOES (W/m^2) untuk tiap kelas flare.
CLASS_FLUX = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
FLARE_THRESHOLD = {"C": 1e-6, "M": 1e-5, "X": 1e-4}
M_FLUX = FLARE_THRESHOLD["M"]
X_FLUX = FLARE_THRESHOLD["X"]


def goes_to_flux(goes: str) -> float:
    """Kelas GOES -> fluks puncak (W/m^2). 'M2.5' -> 2.5e-5 ; '0' -> 0 (tidak ada flare >= C1.0)."""
    goes = goes.strip()
    if goes in ("", "0"):
        return 0.0
    return CLASS_FLUX[goes[0].upper()] * float(goes[1:])


def load_index(cfg, split: str) -> pd.DataFrame:
    """Ambil baris data/index.csv untuk satu split ("train", "val", atau "test")."""
    df = pd.read_csv(cfg.index_csv, parse_dates=["time"])
    return df[df["split"] == split]


# ---------------------------------------------------------------- membaca gambar

def load_magnetogram(path) -> np.ndarray:
    """PNG 8-bit -> medan magnet ternormalisasi (kira-kira B / 2550 Gauss).

    Dataset menyimpan piksel = (B + 2550) / 20, sehingga piksel/127.5 - 1 = B/2550.
    Hasilnya: 0 = tanpa medan, positif = kutub utara, negatif = kutub selatan.
    """
    return np.asarray(Image.open(path), dtype=np.float32) / 127.5 - 1.0


def stack_frames(paths) -> torch.Tensor:
    """Beberapa PNG (urut lama -> baru) -> tensor [T, 1, H, W]."""
    frames = [load_magnetogram(p) for p in paths]
    return torch.from_numpy(np.stack(frames))[:, None]


def load_sequence(paths) -> torch.Tensor:
    """Satu sampel siap masuk model: tensor [1, T, 1, H, W] (batch berisi 1 sampel)."""
    return stack_frames(paths)[None]


# ---------------------------------------------------------------- menyusun sampel

def to_minutes(times: pd.Series) -> np.ndarray:
    """Waktu -> jumlah menit (bilangan bulat), agar mudah dihitung selisihnya."""
    return times.to_numpy().astype("datetime64[m]").astype(np.int64)


def nearest(sorted_vals: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Indeks elemen sorted_vals yang paling dekat dengan setiap target.

    Contoh: nearest([0, 12, 24], [11, 20]) -> [1, 2]. Bila sama dekat, dipilih yang kanan.
    """
    right = np.searchsorted(sorted_vals, targets).clip(0, len(sorted_vals) - 1)
    left = (right - 1).clip(0)
    use_left = np.abs(targets - sorted_vals[left]) < np.abs(sorted_vals[right] - targets)
    return np.where(use_left, left, right)


def frame_indices(t_min: np.ndarray, anchors: np.ndarray, cfg):
    """Untuk tiap anchor, cari n_frames frame (urut lama -> baru) berjarak frame_gap_min.

    t_min   : waktu semua frame satu AR dalam menit (terurut).
    anchors : indeks frame terakhir tiap sampel.
    Kembali : idx [n_anchor, n_frames] dan ok [n_anchor] (False = ada frame yang hilang).

    Contoh dengan 4 frame berjarak 72 menit: waktu target = anchor - 216, -144, -72, -0 menit.
    """
    offsets = cfg.frame_gap_min * np.arange(cfg.n_frames - 1, -1, -1)
    targets = t_min[anchors][:, None] - offsets
    idx = nearest(t_min, targets)
    ok = (np.abs(t_min[idx] - targets) <= cfg.frame_tol_min).all(axis=1)
    return idx, ok


def build_samples(df: pd.DataFrame, cfg, stride_h: float) -> pd.DataFrame:
    """Ubah tabel gambar menjadi tabel sampel.

    Satu sampel = n_frames magnetogram dari AR yang sama, diakhiri pada waktu 'anchor'.
    Label = 1 bila ada flare >= cfg.flare_class dalam 24 jam setelah anchor.
    Anchor diambil satu per `stride_h` jam karena frame berjarak 12 menit hampir identik.
    """
    threshold = FLARE_THRESHOLD[cfg.flare_class]
    stride_min = max(int(stride_h * 60), 1)
    rows = []
    for ar, g in df.sort_values("time").groupby("ar"):
        t = to_minutes(g["time"])
        _, anchors = np.unique(t // stride_min, return_index=True)    # frame pertama tiap blok stride
        idx, ok = frame_indices(t, anchors, cfg)
        paths, flux, times = g["path"].to_numpy(), g["flux"].to_numpy(), g["time"].to_numpy()
        for a, frames in zip(anchors[ok], idx[ok]):
            rows.append((ar, times[a], tuple(paths[frames]), flux[a], int(flux[a] >= threshold)))
    return pd.DataFrame(rows, columns=["ar", "time", "paths", "flux", "label"])


# ---------------------------------------------------------------- dataset untuk PyTorch

def augment(x: torch.Tensor) -> torch.Tensor:
    """Augmentasi yang tidak mengubah sifat fisis AR; sama untuk semua frame.

    Flip & rotasi 90 derajat: orientasi AR di gambar tidak menentukan flare.
    Tukar polaritas (x -> -x): kompleksitas medan tetap sama walau kutubnya ditukar.
    """
    if torch.rand(1) < 0.5:
        x = x.flip(-1)
    if torch.rand(1) < 0.5:
        x = x.flip(-2)
    x = torch.rot90(x, int(torch.randint(4, (1,))), dims=(-2, -1))
    if torch.rand(1) < 0.5:
        x = -x
    return x


class MagnetogramDataset(Dataset):
    """Mengembalikan (x, y): x berbentuk [T, 1, H, W], y = label (0/1, atau target lain)."""

    def __init__(self, samples: pd.DataFrame, image_dir, augment: bool = False):
        self.paths = samples["paths"].tolist()
        self.labels = samples["label"].to_numpy(np.float32)
        self.image_dir = Path(image_dir)
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = stack_frames([self.image_dir / p for p in self.paths[i]])
        if self.augment:
            x = augment(x)
        return x, self.labels[i]


class BalancedSampler(Sampler):
    """Setiap epoch: SEMUA sampel positif + sampel negatif acak (neg_ratio kali lipat).

    Negatif diundi ulang tiap epoch, jadi epoch tetap pendek tetapi lama-lama
    model melihat banyak variasi AR tenang.
    """

    def __init__(self, labels: np.ndarray, neg_ratio: float, seed: int = 0):
        self.pos = np.flatnonzero(labels == 1)
        self.neg = np.flatnonzero(labels == 0)
        self.n_neg = min(len(self.neg), int(round(neg_ratio * len(self.pos))))
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.pos) + self.n_neg

    def __iter__(self):
        neg = self.rng.choice(self.neg, self.n_neg, replace=False)
        return iter(self.rng.permutation(np.concatenate([self.pos, neg])).tolist())
