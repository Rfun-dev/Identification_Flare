"""Langkah 5 - Prediksi flare untuk satu active region (AR).

Dua cara memberi input:
    python predict.py --ar 1158 --time "2011-02-14 12:00"     # frame dicari otomatis dari index.csv
    python predict.py --images a.png b.png c.png d.png        # urut lama -> baru, sebanyak n_frames

Tambahkan --saliency untuk menyimpan peta bagian magnetogram yang paling memengaruhi prediksi.
Fungsi predict_images() juga dipakai oleh fetch_realtime.py.
"""
import argparse
from pathlib import Path

import matplotlib
import pandas as pd
import torch
import torch.nn.functional as F

from flare.data import frame_indices, load_sequence, nearest, to_minutes
from flare.engine import get_device
from flare.model import load_checkpoint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def find_sequence(cfg, ar: int, time: str):
    """n_frames path magnetogram AR `ar` yang berakhir paling dekat dengan `time`, plus label aslinya."""
    df = pd.read_csv(cfg.index_csv, parse_dates=["time"])
    g = df[df["ar"] == ar].sort_values("time")
    if g.empty:
        raise SystemExit(f"AR {ar} tidak ada di {cfg.index_csv}")
    t = to_minutes(g["time"])
    anchor = nearest(t, to_minutes(pd.Series([pd.Timestamp(time)])))
    idx, ok = frame_indices(t, anchor, cfg)
    if not ok[0]:
        raise SystemExit("Frame sebelumnya tidak lengkap di sekitar waktu itu, coba waktu lain.")
    rows = g.iloc[idx[0]]
    print(f"Frame dipakai: {', '.join(rows['time'].dt.strftime('%Y-%m-%d %H:%M'))}")
    return [Path(cfg.image_dir) / p for p in rows["path"]], rows["goes"].iloc[-1]


# ---------------------------------------------------------------- peta saliency

def saliency(model, x: torch.Tensor, n: int = 16, noise: float = 0.05):
    """SmoothGrad: rata-rata |d skor / d piksel| pada frame terakhir dari n salinan input
    yang diberi noise kecil. Hasilnya: piksel mana yang paling mengubah prediksi."""
    model.requires_grad_(False)
    grad = torch.zeros_like(x[0, -1, 0])
    for _ in range(n):
        xi = (x + noise * torch.randn_like(x)).requires_grad_(True)
        with torch.enable_grad():
            model(xi).sum().backward()
        grad += xi.grad[0, -1, 0].abs()
    s = F.avg_pool2d(grad[None, None], 9, stride=1, padding=4)[0, 0]   # haluskan
    return (s / s.max().clamp_min(1e-12)).cpu().numpy()


def save_saliency(x, sal, title, path):
    last = x[0, -1, 0].cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(8, 4.2))
    for a in ax:
        a.imshow(last, cmap="gray", vmin=-0.5, vmax=0.5)
        a.axis("off")
    ax[0].set_title("Magnetogram terakhir\n(putih = kutub +, hitam = kutub -)")
    ax[1].imshow(sal, cmap="inferno", alpha=0.55)
    ax[1].set_title("Saliency: area paling berpengaruh")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- prediksi

def report(model, cfg, ckpt, paths, device, run_dir: Path, with_saliency=False, true_label=None) -> float:
    """Hitung skor untuk satu urutan gambar, cetak keputusan, dan (opsional) simpan saliency."""
    if len(paths) != cfg.n_frames:
        raise SystemExit(f"Model ini butuh {cfg.n_frames} gambar, diberi {len(paths)}.")

    x = load_sequence(paths).to(device)
    with torch.no_grad():
        score = torch.sigmoid(model(x)).item()
    threshold = ckpt["threshold"]
    decision = "FLARE" if score >= threshold else "TIDAK ADA FLARE"

    print(f"Skor flare >= {cfg.flare_class}1.0 dalam 24 jam : {score:.3f}")
    print(f"Keputusan (threshold {threshold:.2f})       : {decision}")
    if true_label is not None:
        print(f"Label sebenarnya (flare terbesar 24 jam) : {true_label}")

    if with_saliency:
        out = run_dir / "saliency.png"
        save_saliency(x, saliency(model, x), f"Skor {score:.3f} -> {decision}", out)
        print(f"Peta saliency disimpan ke {out}")
    return score


def predict_images(run_dir, paths, with_saliency=False) -> float:
    """Prediksi dari daftar PNG (urut lama -> baru) dengan model di run_dir."""
    run_dir = Path(run_dir)
    device = get_device()
    model, cfg, ckpt = load_checkpoint(run_dir / "best.pt", device)
    return report(model, cfg, ckpt, [Path(p) for p in paths], device, run_dir, with_saliency)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", default="runs/flareformer")
    parser.add_argument("--ar", type=int, help="nomor AR 4 digit, mis. 1158")
    parser.add_argument("--time", help='waktu prediksi, mis. "2011-02-14 12:00"')
    parser.add_argument("--images", nargs="+", help="path PNG, urut lama -> baru")
    parser.add_argument("--saliency", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = get_device()
    model, cfg, ckpt = load_checkpoint(run_dir / "best.pt", device)

    if args.images:
        paths, true_label = [Path(p) for p in args.images], None
    elif args.ar is not None and args.time:
        paths, true_label = find_sequence(cfg, args.ar, args.time)
    else:
        raise SystemExit("Beri --images, atau --ar dan --time.")
    report(model, cfg, ckpt, paths, device, run_dir, args.saliency, true_label)


if __name__ == "__main__":
    main()
