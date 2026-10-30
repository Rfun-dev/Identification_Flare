"""Langkah 7 - Uji model pada data SETELAH dataset (2021 ke atas, siklus Matahari 25).

    python evaluate_recent.py                                   # 2021-2025, 100 positif + 100 negatif
    python evaluate_recent.py --n_pos 30 --n_neg 30             # uji cepat

Dataset latih berakhir 2018, jadi ini menjawab: "apakah model masih bekerja pada Matahari sekarang?"

Cara kerja:
1. Ambil daftar flare (SWPC) dan laporan posisi harian AR (NOAA) dari HEK.
2. Sampel positif: waktu acak 0,5-23,5 jam sebelum flare >= M (maks. 3 per AR).
   Sampel negatif: waktu acak dari laporan harian AR yang berada dalam +-45 derajat.
3. Label dengan aturan dataset: flare terbesar dari AR itu yang puncaknya dalam 24 jam setelah
   frame terakhir. Flare 2021+ diukur GOES-16/18 tanpa faktor 0,7, jadi dikali 0,7 sebelum
   dibandingkan dengan batas M1.0 skala lama (= M1.43 skala sekarang).
4. Buat 4 PNG per sampel dengan pipeline fetch_realtime (data final hmi.M_720s), skor dengan model,
   hitung TSS/AUC pada threshold validasi, plus selang kepercayaan 95% (bootstrap per AR).

Hasil: data_recent/ (PNG + daftar sampel), <run_dir>/recent_predictions.csv, recent_metrics.json.
Bisa dihentikan dan dijalankan ulang: sampel yang PNG-nya sudah ada tidak diunduh lagi.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from flare.config import load_config
from flare.data import FLARE_THRESHOLD
from flare.engine import get_device, predict_one
from flare.features import HOURS_24, max_old_flux, to_old_scale
from flare.metrics import bootstrap_ci_by_ar, scores
from flare.model import load_checkpoint
from flare.recent import build_all, load_events, load_or_create


def make_samples(flares, reports, start, end, n_pos, n_neg, threshold_flux, rng):
    """Daftar (ar, time) yang akan diuji, beserta labelnya."""
    day = pd.Timedelta(days=1)          # jaga jarak dari ujung periode agar jendela 24 jam tidak terpotong

    # positif: waktu acak sebelum flare besar
    big = flares[(to_old_scale(flares) >= threshold_flux) & flares["peak"].between(start + day, end)]
    big = big.sample(frac=1, random_state=rng).groupby("ar").head(3)        # maks. 3 flare per AR
    big = big.sample(min(len(big), n_pos), random_state=rng)
    pos_times = big["peak"] - pd.to_timedelta(rng.uniform(0.5, 23.5, len(big)), unit="h")

    # negatif: waktu acak dari laporan harian AR yang dekat pusat piringan
    rep = reports[(reports["lon"].abs() <= 45) & reports["date"].between(start, end - day)]
    rep = rep.sample(min(len(rep), n_neg), random_state=rng)
    neg_times = rep["date"] + pd.to_timedelta(rng.uniform(0, 12, len(rep)), unit="h")

    s = pd.DataFrame({"ar": np.r_[big["ar"].to_numpy(), rep["ar"].to_numpy()],
                      "time": np.r_[pos_times.to_numpy(), neg_times.to_numpy()],
                      "drawn_as": ["positif"] * len(big) + ["negatif"] * len(rep)})
    s["time"] = s["time"].dt.floor("12min")                                  # ikuti jadwal HMI 12 menit
    s = s.drop_duplicates(["ar", "time"]).reset_index(drop=True)

    # label = flare terbesar dari AR itu dengan puncak dalam (time, time + 24 jam], skala lama
    s["flux_old_scale"] = [max_old_flux(flares[flares["ar"] == ar], t, t + HOURS_24)
                           for ar, t in zip(s["ar"], s["time"])]
    s["label"] = (s["flux_old_scale"] >= threshold_flux).astype(int)
    return s


def metrics_with_ci(df: pd.DataFrame, threshold: float) -> dict:
    """Metrik pada threshold validasi, plus selang kepercayaan 95% TSS dan AUC (bootstrap per AR)."""
    stats = {"TSS_CI95": lambda s: scores(s["label"], s["prob"], threshold)["TSS"],
             "AUC_CI95": lambda s: scores(s["label"], s["prob"], threshold)["AUC"]}
    return {**scores(df["label"], df["prob"], threshold), **bootstrap_ci_by_ar(df, "label", stats)}


def print_comparison(recent, base, ok, test, thr):
    print(f"\n{'':22s}{'2021+ (baru)':>16s}{'test 2010-2018':>18s}")
    for k in ["TSS", "AUC", "recall"]:
        print(f"{k:22s}{recent[k]:16.3f}{base[k]:18.3f}")
    print(f"{'false alarm rate':22s}{recent['FP'] / max(recent['FP'] + recent['TN'], 1):16.3f}"
          f"{base['FP'] / max(base['FP'] + base['TN'], 1):18.3f}")
    for k in ["TSS_CI95", "AUC_CI95"]:
        print(f"{k:22s}{'%.2f-%.2f' % tuple(recent[k]):>16s}{'%.2f-%.2f' % tuple(base[k]):>18s}")
    print(f"{'sampel (positif)':22s}{'%d (%d)' % (len(ok), ok['label'].sum()):>16s}"
          f"{'%d (%d)' % (len(test), test['label'].sum()):>18s}")
    print(f"\nThreshold {thr:.3f} dari validasi. Precision/HSS tidak dibandingkan: proporsi flare di sampel "
          "ini sengaja dibuat seimbang, berbeda dari kenyataan.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", default="runs/flareformer")
    parser.add_argument("--out_dir", default="data_recent")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--n_pos", type=int, default=100)
    parser.add_argument("--n_neg", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2, help="sampel diunduh bersamaan (x4 koneksi)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir, out_dir = Path(args.run_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(run_dir)
    threshold_flux = FLARE_THRESHOLD[cfg.flare_class]
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)

    # 1-2. daftar flare dan sampel
    flares, reports = load_events(out_dir, start, end)
    samples_csv = out_dir / "samples.csv"
    rng = np.random.RandomState(args.seed)
    samples = load_or_create(samples_csv, lambda: make_samples(flares, reports, start, end, args.n_pos,
                                                               args.n_neg, threshold_flux, rng))
    print(f"{len(samples)} sampel ({samples['label'].sum()} positif menurut label 24 jam)")

    # 3. gambar
    samples = build_all(samples, reports, cfg, out_dir, samples_csv, args.workers)
    ok = samples[samples["status"] == "ok"].copy()
    print(f"\nSiap dinilai: {len(ok)} sampel; dilewati: {samples['status'].str.startswith('dilewati').sum()}; "
          f"error (jalankan ulang untuk mencoba lagi): {samples['status'].str.startswith('error').sum()}")

    # 4. skor model dan metrik
    device = get_device()
    model, _, ckpt = load_checkpoint(run_dir / "best.pt", device)
    ok["prob"] = [predict_one(model, paths, device) for paths in ok["paths"].str.split("|")]
    ok[["ar", "time", "drawn_as", "flux_old_scale", "label", "prob"]].to_csv(run_dir / "recent_predictions.csv",
                                                                            index=False)
    thr = ckpt["threshold"]
    recent = metrics_with_ci(ok, thr)
    recent.update({"n_samples": len(ok), "n_positive": int(ok["label"].sum()), "n_ar": int(ok["ar"].nunique()),
                   "period": f"{args.start}..{args.end}"})
    test = pd.read_csv(run_dir / "test_predictions.csv")
    base = metrics_with_ci(test, thr)
    (run_dir / "recent_metrics.json").write_text(json.dumps({"recent": recent, "test_2010_2018": base}, indent=2))
    print_comparison(recent, base, ok, test, thr)


if __name__ == "__main__":
    main()
