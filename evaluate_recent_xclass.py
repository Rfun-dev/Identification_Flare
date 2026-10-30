"""Langkah 8c - Konfirmasi independen pemisah M vs X pada data 2021-2025 (siklus Matahari 25).

    python evaluate_recent_xclass.py --n_x 120 --n_m 120

Skor komposit di xclass_features.py dipilih setelah melihat hasil pada data 2010-2018, dan test hanya
punya 3 AR berflare X. Di sini semua model diuji pada AR 2021-2025 yang tidak pernah dilihat oleh model
mana pun. Sampel: waktu acak 0,5-23,5 jam sebelum flare X dan sebelum flare M (maks. 3 per AR),
gambar dibuat dengan pipeline fetch_realtime (data final hmi.M_720s), riwayat flare dari daftar
flare SWPC (HEK), fluks GOES-16+ dikali 0,7 agar sama dengan skala label latih.

Proporsi X di sampel ini sengaja dibuat tinggi, jadi yang bermakna hanya AUC (pemisahan),
bukan peluang absolut atau BSS.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from flare.config import load_config
from flare.data import M_FLUX, X_FLUX
from flare.engine import get_device, predict_one
from flare.features import HOURS_24, history_from_events, max_old_flux, sequence_features, to_old_scale
from flare.model import load_checkpoint
from flare.recent import build_all, load_events, load_or_create
from flare.xclass import evaluate_x_vs_m

SINGLE_FEATURES = ["log_R", "lsg_mm", "log_usflux", "spot_area_mm2", "max_grad", "hist24"]


def sample_xm(flares, start, end, n_x, n_m, rng) -> pd.DataFrame:
    """Waktu acak 0,5-23,5 jam sebelum flare X dan sebelum flare M (maks. 3 flare per AR)."""
    old = to_old_scale(flares)
    in_period = flares["peak"].between(start + pd.Timedelta(days=1), end)
    picks = []
    for selected, n in ((in_period & (old >= X_FLUX), n_x),
                        (in_period & (old >= M_FLUX) & (old < X_FLUX), n_m)):
        f = flares[selected].sample(frac=1, random_state=rng).groupby("ar").head(3)
        picks.append(f.sample(min(len(f), n), random_state=rng))
    f = pd.concat(picks)
    times = f["peak"] - pd.to_timedelta(rng.uniform(0.5, 23.5, len(f)), unit="h")
    s = pd.DataFrame({"ar": f["ar"].to_numpy(), "time": times.to_numpy()})
    s["time"] = s["time"].dt.floor("12min")
    return s.drop_duplicates(["ar", "time"]).reset_index(drop=True)


def create_samples(flares, start, end, n_x, n_m, rng) -> pd.DataFrame:
    """Sampel + label (flare terbesar 24 jam, skala lama); tahap 2 hanya untuk sampel yang memang >= M."""
    s = sample_xm(flares, start, end, n_x, n_m, rng)
    s["flux_old_scale"] = [max_old_flux(flares[flares["ar"] == ar], t, t + HOURS_24)
                           for ar, t in zip(s["ar"], s["time"])]
    return s[s["flux_old_scale"] >= M_FLUX].reset_index(drop=True)


def describe(s: pd.DataFrame) -> str:
    n_x, n_m = s["X"].sum(), (1 - s["X"]).sum()
    return (f"{n_x} X dari {s.loc[s.X == 1, 'ar'].nunique()} AR, "
            f"{n_m} M dari {s.loc[s.X == 0, 'ar'].nunique()} AR")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1_dir", default="runs/flareformer")
    parser.add_argument("--xclass_dir", default="runs/xclass_feat")
    parser.add_argument("--out_dir", default="data_recent")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--n_x", type=int, default=120)
    parser.add_argument("--n_m", type=int, default=120)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    out_dir, xclass_dir = Path(args.out_dir), Path(args.xclass_dir)
    cfg = load_config(args.stage1_dir)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)

    # 1-2. daftar flare dan sampel
    flares, reports = load_events(out_dir, start, end)
    csv = out_dir / "samples_xm.csv"
    rng = np.random.RandomState(args.seed)
    s = load_or_create(csv, lambda: create_samples(flares, start, end, args.n_x, args.n_m, rng))
    s["X"] = (s["flux_old_scale"] >= X_FLUX).astype(int)
    print(f"{len(s)} sampel >= M: {describe(s)}")

    # 3. gambar
    s = build_all(s, reports, cfg, out_dir, csv, args.workers)
    ok = s[s["status"] == "ok"].copy().reset_index(drop=True)
    print(f"\nSiap dinilai: {len(ok)} sampel ({describe(ok)})")
    paths = ok["paths"].str.split("|")

    # 4. skor tahap 1 dan fitur
    device = get_device()
    model, _, _ = load_checkpoint(Path(args.stage1_dir) / "best.pt", device)
    ok["prob"] = [predict_one(model, p, device) for p in paths]
    rows = []
    for p, ar, t in zip(paths, ok["ar"], ok["time"]):
        hist = history_from_events(flares[flares["ar"] == ar], reports[reports["ar"] == ar], t)
        rows.append({**sequence_features(p), **hist})
    ok = pd.concat([ok, pd.DataFrame(rows)], axis=1)

    # 5. nilai semua model, skor tahap 1, gabungannya, dan fitur tunggal
    with open(xclass_dir / "model.pkl", "rb") as fh:
        saved = pickle.load(fh)
    res = {}
    for name, m in saved["models"].items():
        ok[f"p_{name}"] = m.predict_proba(ok[saved["feature_sets"][name]])[:, 1]
        res[name] = evaluate_x_vs_m(ok, f"p_{name}")
    res["skor tahap 1"] = evaluate_x_vs_m(ok, "prob")
    ok["combo"] = (rankdata(ok["prob"]) + rankdata(ok[f"p_{saved['final']}"])) / (2 * len(ok))
    res[f"tahap 1 + {saved['final']} (rata-rata peringkat)"] = evaluate_x_vs_m(ok, "combo")
    for c in SINGLE_FEATURES:
        res[f"fitur tunggal: {c}"] = evaluate_x_vs_m(ok, c)

    meta = {"n": len(ok), "n_X": int(ok["X"].sum()), "n_ar_X": int(ok.loc[ok.X == 1, "ar"].nunique()),
            "n_ar_M": int(ok.loc[ok.X == 0, "ar"].nunique()), "period": f"{args.start}..{args.end}"}
    (xclass_dir / "recent_metrics.json").write_text(
        json.dumps({"meta": meta, "AUC_X_vs_M": res}, indent=2, ensure_ascii=False))
    ok.drop(columns="paths").to_csv(xclass_dir / "recent_predictions.csv", index=False)

    print(f"\nAUC X vs M pada data 2021-2025 (0,5 = acak):\n  {'':52s} {'AUC':>6s} {'95%':>11s} {'AUC-AR':>7s}")
    for name, r in res.items():
        print(f"  {name:52s} {r['AUC']:6.3f} {r['AUC_CI95'][0]:5.2f}-{r['AUC_CI95'][1]:4.2f} {r['AUC_AR']:7.3f}")


if __name__ == "__main__":
    main()
