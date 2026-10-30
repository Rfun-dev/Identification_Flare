"""Membuat dataset SINTETIS kecil dengan format yang sama persis dengan dataset asli.

Gunanya menguji seluruh pipeline (prepare -> train -> evaluate -> predict) dalam
hitungan menit di CPU tanpa mengunduh 12 GB. Hasil latihnya TIDAK bermakna ilmiah.

"Fisika mainan": AR yang akan menghasilkan flare besar punya pasangan kutub +/-
yang makin kuat dan makin rapat (garis netral makin tajam) menjelang flare.

    python make_dummy_data.py --data_dir data_dummy
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

CADENCE_MIN = 12                                   # cadence HMI: 1 gambar / 12 menit
FLARE_PER_AR = ["0", "C3.4", "M1.6", "0", "M5.2", "X1.3"]
YY, XX = np.mgrid[0:224, 0:224].astype(np.float32)


def blob(x0, y0, size):
    """Bintik Gaussian bernilai 1 di (x0, y0)."""
    return np.exp(-((XX - x0) ** 2 + (YY - y0) ** 2) / (2 * size ** 2))


def bipole(cx, cy, sep, angle, strength, size):
    """Sepasang kutub magnet (positif & negatif) berjarak `sep` piksel, dalam Gauss."""
    dx, dy = np.cos(angle) * sep / 2, np.sin(angle) * sep / 2
    return strength * (blob(cx + dx, cy + dy, size) - blob(cx - dx, cy - dy, size))


def split_of(k: int) -> str:
    """AR ke-k masuk split mana (1 dari 7 AR untuk validasi, 1 dari 7 untuk test)."""
    if k % 7 == 5:
        return "Validation"
    if k % 7 == 6:
        return "Test"
    return "Train"


def max_complexity(flare: str) -> float:
    """Seberapa rumit medan AR menjelang flare: 0 = tanpa flare, 0,4 = kelas C, 1 = kelas M/X."""
    if flare[0] == "0":
        return 0.0
    if flare[0] == "C":
        return 0.4
    return 1.0


def complexity(amp: float, hours_to_flare: float) -> float:
    """Naik perlahan dalam 24 jam menjelang flare, lalu meluruh setelah flare."""
    if hours_to_flare >= 0:
        return amp * np.clip(1.3 - hours_to_flare / 24, 0, 1)
    return amp * np.exp(hours_to_flare / 1.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="data_dummy")
    parser.add_argument("--n_ars", type=int, default=36)
    parser.add_argument("--hours", type=float, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    data_dir = Path(args.data_dir)
    img_root = data_dir / "Lat60_Lon60_Nans0_png_224"
    n_frames = int(args.hours * 60 / CADENCE_MIN)
    labels, splits = [], {"Train": [], "Validation": [], "Test": []}

    for k in range(args.n_ars):
        ar = 9001 + k
        splits[split_of(k)].append(ar)
        flare = FLARE_PER_AR[k % len(FLARE_PER_AR)]
        # Urutan pemanggilan rng di bawah menentukan gambar yang dihasilkan; jangan diubah.
        t_flare = rng.uniform(0.5, 1.0) * args.hours          # jam puncak flare
        amp = max_complexity(flare)
        cx, cy = rng.uniform(80, 144, 2)
        angle = rng.uniform(0, np.pi)
        background = 0
        for _ in range(3):                                     # 3 pasangan kutub acak sebagai latar
            x0, y0 = rng.uniform(40, 184, 2)
            sep = rng.uniform(20, 50)
            bg_angle = rng.uniform(0, np.pi)
            strength = rng.uniform(200, 600)
            background = background + bipole(x0, y0, sep, bg_angle, strength, 8)
        start = datetime(2015, 1, 1) + timedelta(days=3 * k)
        (img_root / str(ar)).mkdir(parents=True, exist_ok=True)

        for i in range(n_frames):
            hour = i * CADENCE_MIN / 60
            c = complexity(amp, t_flare - hour)
            field = (background + bipole(cx, cy, 50 - 35 * c, angle, 600 + 1800 * c, 10)
                     + rng.normal(0, 10, XX.shape))
            pixels = np.clip(np.round((field + 2550) / 20), 0, 255).astype(np.uint8)
            stamp = start + timedelta(hours=hour)
            name = f"{ar}_hmi.M_720s.{stamp:%Y%m%d_%H%M%S}_TAI.1.magnetogram_224.png"
            Image.fromarray(pixels).save(img_root / str(ar) / name)
            label = flare if 0 < t_flare - hour <= 24 else "0"   # flare dalam 24 jam setelah gambar
            labels.append(f"{name},{label}")

    (data_dir / "C1.0_24hr_224_png_Labels.txt").write_text("\n".join(labels) + "\n")
    for split, ars in splits.items():
        case = "data" if split == "Validation" else "Data"   # ikuti penamaan file di Dryad
        (data_dir / f"List_of_AR_in_{split}_{case}_by_AR.csv").write_text(
            "\n".join(map(str, ars)) + "\n")
    print(f"{len(labels)} gambar sintetis dibuat di {img_root}")


if __name__ == "__main__":
    main()
