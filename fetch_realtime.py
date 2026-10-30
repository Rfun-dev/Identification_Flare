"""Langkah 6 - Ambil magnetogram SDO/HMI terbaru dari JSOC lalu prediksi flare 24 jam ke depan.

    python fetch_realtime.py --ar 4556 --predict                      # data terbaru (near-real-time)
    python fetch_realtime.py --ar 1158 --time "2011-02-14 12:00" --verify --predict

--ar      nomor NOAA AR 4 digit seperti di dataset (AR 14556 -> 4556, AR 11158 -> 1158).
--time    waktu frame terakhir (TAI). Kosong = rekaman terbaru di JSOC.
--verify  bandingkan hasil dengan PNG dataset pada waktu yang sama (cek pipeline cocok per piksel).
--predict langsung prediksi (Langkah 5) + peluang flare kelas X (Progress 18).

Hasil: 4 PNG di data_realtime/<ar>/, format dan nama file sama dengan dataset, siap untuk predict.py.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from flare import hmi, jsoc, noaa
from flare.config import Config, load_config
from flare.engine import get_device, predict_one
from flare.features import history_from_events, prepare_flares, sequence_features
from flare.model import load_checkpoint
from predict import predict_images

FLARE_LEVELS = ["tidak ada", "C", "M", "X"]


def pick_series(end) -> str:
    """Data final (hmi.M_720s) bila sudah tersedia, selain itu near-real-time (hmi.M_720s_nrt)."""
    if end is not None and end <= jsoc.latest_time("hmi.M_720s"):
        return "hmi.M_720s"
    return "hmi.M_720s_nrt"


def make_images(args, cfg, ar: int):
    """Cari 4 frame, tentukan posisi AR, lalu buat PNG. Kembali: (tabel frame, daftar path PNG)."""
    end = pd.Timestamp(args.time) if args.time else None
    series = pick_series(end) if args.series == "auto" else args.series
    if end is None:
        end = jsoc.latest_time(series)
    try:
        frames = jsoc.query_frames(series, end, cfg.n_frames, cfg.frame_gap_min, cfg.frame_tol_min)
        print(f"Series {series}, frame: {', '.join(frames['time'].dt.strftime('%Y-%m-%d %H:%M'))} TAI")
        if args.position:
            lon, lat = (float(v) for v in args.position.split(","))
            source = "manual"
        else:
            lon, lat, source = noaa.ar_position(ar, frames["time"].iloc[-1])
        print(f"Posisi AR {ar}: bujur Carrington {lon:.2f}, lintang {lat:g}. Sumber: {source}")
        result = hmi.save_sequence(frames, lon, lat, Path(args.out_dir) / str(ar), ar, series)
    except ValueError as e:
        raise SystemExit(str(e))

    for (path, earth_lon, earth_lat), t in zip(result, frames["time"]):
        print(f"  {t:%Y-%m-%d %H:%M}  bujur {earth_lon:6.1f}, lintang {earth_lat:6.1f}  ->  {path}")
    return frames, [path for path, _, _ in result]


def verify(out_paths, frames, ar, cfg, run_dir):
    """Bandingkan PNG buatan pipeline dengan PNG dataset pada AR dan waktu yang sama,
    per piksel dan lewat skor model."""
    df = pd.read_csv(cfg.index_csv, parse_dates=["time"])
    df = df[df["ar"] == ar].set_index("time")
    missing = [t for t in frames["time"] if t not in df.index]
    if missing:
        print(f"\nVerifikasi dilewati: frame {', '.join(map(str, missing))} tidak ada di dataset.")
        return
    ref_paths = [Path(cfg.image_dir) / df.loc[t, "path"] for t in frames["time"]]

    print("\nVerifikasi terhadap dataset:")
    for ours_path, ref_path, t in zip(out_paths, ref_paths, frames["time"]):
        ours = np.asarray(Image.open(ours_path), dtype=np.int16)
        diff = np.abs(ours - np.asarray(Image.open(ref_path), dtype=np.int16))
        print(f"  {t}: sama persis {np.mean(diff == 0) * 100:5.1f}% piksel, selisih rata-rata {diff.mean():.3f}, "
              f"maks {diff.max()} level (dari 255)")

    device = get_device()
    model, _, _ = load_checkpoint(Path(run_dir) / "best.pt", device)
    score_ref = predict_one(model, ref_paths, device)
    score_ours = predict_one(model, out_paths, device)
    print(f"  Skor model: gambar dataset {score_ref:.3f}, gambar pipeline {score_ours:.3f} "
          f"(selisih {abs(score_ref - score_ours):.3f})")


def x_probability(out_paths, ar: int, last: pd.Timestamp, xclass_dir: Path):
    """Peluang flare X bila flare >= M terjadi: skor komposit fitur fisis + riwayat flare (Progress 18)."""
    with open(xclass_dir / "model.pkl", "rb") as fh:
        saved = pickle.load(fh)
    # riwayat flare AR ini 48 jam terakhir, dari daftar flare SWPC di HEK
    hek = noaa.hek_search("fl", "SWPC", last - pd.Timedelta(days=3), last + pd.Timedelta(hours=1),
                          "ar_noaanum,event_peaktime,fl_goescls")
    flares = prepare_flares(hek)
    hist = history_from_events(flares[flares["ar"] % 10000 == ar], noaa.noaa_reports(ar, last), last)

    features = pd.DataFrame([{**sequence_features(out_paths), **hist}])
    name = saved["final"]
    p = float(saved["models"][name].predict_proba(features[saved["feature_sets"][name]])[0, 1])
    latest = flares["peak"].max() if len(flares) else None
    return p, saved["base_rate"], hist, latest


def print_x_probability(out_paths, ar, last, xclass_dir: Path):
    p, base, hist, latest = x_probability(out_paths, ar, last, xclass_dir)
    print(f"\nJika flare >= M terjadi, peluang kelas X  : {p:.0%} "
          f"(rata-rata semua flare >= M: {base:.0%}; {p / base:.1f}x rata-rata)")
    latest_note = f" (daftar flare HEK terakhir: {latest:%Y-%m-%d %H:%M} UT)" if latest is not None else ""
    print(f"  riwayat flare AR ini: 24 jam terakhir {FLARE_LEVELS[int(hist['hist24'])]}, "
          f"24-48 jam {FLARE_LEVELS[int(hist['hist48'])]}" + latest_note)
    print("  Ini peluang, bukan kepastian: AUC X vs M ±0,79 pada data 2021-2025 (Progress 18).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ar", type=int, required=True, help="nomor NOAA AR 4 digit, mis. 4556")
    parser.add_argument("--time", help='waktu frame terakhir (TAI), mis. "2011-02-14 12:00"; kosong = terbaru')
    parser.add_argument("--series", choices=["auto", "hmi.M_720s", "hmi.M_720s_nrt"], default="auto")
    parser.add_argument("--position", help="posisi manual 'BUJUR_CARRINGTON,LINTANG' (derajat), mis. 39,-20")
    parser.add_argument("--out_dir", default="data_realtime")
    parser.add_argument("--run_dir", default="runs/flareformer")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--predict", action="store_true", help="langsung prediksi")
    parser.add_argument("--saliency", action="store_true")
    parser.add_argument("--xclass_dir", default="runs/xclass_feat", help="model peluang X (xclass_features.py)")
    args = parser.parse_args()

    ar = args.ar % 10000
    has_config = (Path(args.run_dir) / "config.json").exists()
    cfg = load_config(args.run_dir) if has_config else Config()

    frames, out_paths = make_images(args, cfg, ar)
    if args.verify:
        verify(out_paths, frames, ar, cfg, args.run_dir)

    last = frames["time"].iloc[-1]
    age_h = (pd.Timestamp.now(tz="UTC").tz_localize(None) - last) / pd.Timedelta(hours=1)
    print(f"\nPrediksi berlaku untuk 24 jam setelah {last:%Y-%m-%d %H:%M} TAI"
          + (f" (frame terakhir berumur {age_h:.1f} jam)." if age_h < 48 else "."))
    print("Catatan: label latih memakai skala GOES-15. Sejak 2020 NOAA menghapus faktor 0,7, "
          "jadi M1.0 lama setara +-M1.4 di laporan sekarang.")

    if not args.predict:
        cmd = [sys.executable, "predict.py", "--run_dir", args.run_dir, "--images", *map(str, out_paths)]
        if args.saliency:
            cmd.append("--saliency")
        print("Jalankan prediksi dengan:\n  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        return

    print(flush=True)
    predict_images(args.run_dir, out_paths, args.saliency)
    xclass_dir = Path(args.xclass_dir)
    if (xclass_dir / "model.pkl").exists():
        print_x_probability(out_paths, ar, last, xclass_dir)


if __name__ == "__main__":
    main()
