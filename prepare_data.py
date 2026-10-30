"""Langkah 2 - Membuat indeks data (data/index.csv).

Menggabungkan file label + daftar AR validasi/test menjadi satu tabel:
    ar, time, path, goes, flux, split
Hanya gambar yang benar-benar ada di disk yang dimasukkan, jadi unduhan
sebagian (--max_gb) tetap bisa dipakai.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from flare.config import config_from_args, get_parser
from flare.data import goes_to_flux

LABEL_FILE = "C1.0_24hr_224_png_Labels.txt"
# Nama file mis. 1158_hmi.M_720s.20110214_120000_TAI.1.magnetogram_224.png -> AR 1158, waktu 20110214_120000
NAME_PATTERN = r"^(\d{4})_hmi\.M_720s\.(\d{8}_\d{6})_TAI"


def read_labels(data_dir: Path) -> pd.DataFrame:
    """File label (nama gambar, kelas GOES) -> tabel dengan kolom ar, time, path, goes, flux."""
    df = pd.read_csv(data_dir / LABEL_FILE, header=None, names=["file", "goes"], dtype=str)
    parts = df["file"].str.extract(NAME_PATTERN)          # kolom 0 = nomor AR, kolom 1 = waktu
    df = df[parts[0].notna()].copy()
    parts = parts.dropna()
    df["ar"] = parts[0].astype(int)
    df["time"] = pd.to_datetime(parts[1], format="%Y%m%d_%H%M%S")
    df["path"] = parts[0] + "/" + df["file"]
    df["flux"] = df["goes"].map(goes_to_flux)
    return df


def read_ar_list(data_dir: Path, keyword: str) -> set:
    """Baca daftar nomor AR dari file List_of_AR_in_<keyword>...csv (tidak peka huruf besar)."""
    files = [p for p in data_dir.glob("List_of_AR_in_*") if keyword in p.name.lower()]
    if not files:
        raise FileNotFoundError(f"Daftar AR '{keyword}' tidak ditemukan di {data_dir}. "
                                "Jalankan download_data.py untuk petunjuk.")
    tokens = files[0].read_text().replace(",", " ").split()
    return {int(tok) for tok in tokens if tok.isdigit()}


def list_images(image_dir: Path) -> set:
    """Semua path 'NNNN/nama.png' yang ada di disk (os.scandir jauh lebih cepat dari exists())."""
    found = set()
    for ar_dir in os.scandir(image_dir):
        if not ar_dir.is_dir():
            continue
        for f in os.scandir(ar_dir.path):
            if f.name.endswith(".png"):
                found.add(f"{ar_dir.name}/{f.name}")
    return found


def main():
    cfg = config_from_args(get_parser(__doc__).parse_args())
    data_dir = Path(cfg.data_dir)

    df = read_labels(data_dir)
    df = df[df["path"].isin(list_images(Path(cfg.image_dir)))]
    print(f"{len(df):,} gambar berlabel ditemukan di disk.")

    # setiap AR masuk tepat satu split; AR yang tidak ada di daftar val/test masuk train
    val_ars = read_ar_list(data_dir, "validation")
    test_ars = read_ar_list(data_dir, "test")
    df["split"] = np.select([df["ar"].isin(val_ars), df["ar"].isin(test_ars)],
                            ["val", "test"], default="train")

    df = df.sort_values(["ar", "time"])[["ar", "time", "path", "goes", "flux", "split"]]
    df.to_csv(cfg.index_csv, index=False)

    summary = df.groupby("split").agg(images=("path", "size"), ARs=("ar", "nunique"),
                                      pct_C_up=("flux", lambda f: 100 * (f >= 1e-6).mean()),
                                      pct_M_up=("flux", lambda f: 100 * (f >= 1e-5).mean()))
    print(summary.round(2).to_string())
    for split in ("train", "val", "test"):
        if split not in summary.index:
            print(f"PERINGATAN: split '{split}' kosong (unduh lebih banyak gambar).")
    print(f"Indeks disimpan ke {cfg.index_csv}")


if __name__ == "__main__":
    main()
