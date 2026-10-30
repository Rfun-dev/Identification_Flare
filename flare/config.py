"""Semua pengaturan proyek di satu tempat.

Setiap nilai bisa diubah dari command line tanpa mengedit kode, contoh:
    python train.py --epochs 20 --n_frames 6 --batch_size 8
"""
import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class Config:
    # --- Lokasi file ---
    data_dir: str = "data"
    image_dir: str = "data/Lat60_Lon60_Nans0_png_224"
    index_csv: str = "data/index.csv"
    run_dir: str = "runs/flareformer"

    # --- Definisi tugas: apakah AR akan menghasilkan flare >= kelas ini dalam 24 jam? ---
    flare_class: str = "M"            # "C", "M", atau "X"

    # --- Urutan waktu: satu sampel = beberapa magnetogram berurutan dari AR yang sama ---
    n_frames: int = 4                 # jumlah frame per sampel (1 = hanya gambar terakhir)
    frame_gap_min: int = 72           # jarak antar frame (menit), mengikuti Li dkk. (2025)
    frame_tol_min: int = 24           # toleransi bila frame di waktu persis itu tidak ada
    train_stride_h: float = 1.0       # jarak antar sampel training dalam satu AR (jam)
    eval_stride_h: float = 3.0        # jarak antar sampel validasi/test (jam)

    # --- Model FlareFormer ---
    d_model: int = 128                # ukuran vektor token di Transformer
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    pretrained: bool = True           # pakai bobot ResNet18 hasil latih ImageNet

    # --- Training ---
    epochs: int = 15
    batch_size: int = 16
    lr: float = 3e-4                  # learning rate Transformer + head
    backbone_lr: float = 1e-4         # learning rate CNN (lebih kecil, sudah pretrained)
    weight_decay: float = 0.05
    neg_ratio: float = 3.0            # per epoch: jumlah sampel non-flare = neg_ratio x sampel flare
    patience: int = 5                 # early stopping bila TSS validasi tidak naik
    num_workers: int = 4
    amp: bool = False                 # float16; hanya cepat di GPU ber-tensor core (RTX), 4x lebih lambat di GTX 1650 Ti
    seed: int = 42


def _str2bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes", "y")


def get_parser(description: str = "") -> argparse.ArgumentParser:
    """Membuat argumen --nama_field otomatis untuk setiap field di Config."""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(Config):
        parser.add_argument(f"--{f.name}", default=f.default,
                            type=_str2bool if f.type is bool else f.type)
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(**{f.name: getattr(args, f.name) for f in fields(Config)})


def load_config(run_dir) -> Config:
    """Config yang dipakai saat model di run_dir dilatih (dibaca dari config.json)."""
    return Config(**json.loads((Path(run_dir) / "config.json").read_text()))
