"""Langkah 8b - Peluang flare X (bila flare >= M terjadi) dari fitur fisis + riwayat flare.

Pengganti model dalam di train_xclass.py (Progress 17), yang terlalu besar untuk 18 AR berflare X.
Di sini: model kecil dengan ~11 fitur yang didukung riset (lihat flare/features.py dan flare/xclass.py).
Keluarannya PELUANG P(X | >= M), bukan kelas pasti, karena kekuatan flare sebagian bersifat acak
(distribusi power-law; Wheatland 2005).

    python xclass_features.py

Pengujian sama dengan train_xclass.py agar bisa dibandingkan: cross-validation per AR pada AR train+val
(lipatan sama), lalu model akhir diuji sekali pada AR test, dibandingkan dengan skor tahap 1.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tqdm import tqdm

from flare.config import load_config
from flare.data import M_FLUX, X_FLUX, build_samples, load_index
from flare.features import history, sequence_features
from flare.metrics import best_threshold
from flare.xclass import (FEATURE_SETS, FINAL, assign_folds, confusion_table, evaluate_x_vs_m,
                          make_model, true_class, two_stage_class)


def add_features(samples: pd.DataFrame, index: pd.DataFrame, cfg, cache: Path) -> pd.DataFrame:
    """Tambah kolom fitur ke tiap sampel; hasil disimpan ke `cache` agar tidak dihitung ulang."""
    if cache.exists():
        features = pd.read_csv(cache, parse_dates=["time"])
        return samples.merge(features, on=["ar", "time"], how="left")

    # fitur fisis dari gambar
    image_dir = Path(cfg.image_dir)
    phys = [sequence_features([image_dir / p for p in paths])
            for paths in tqdm(samples["paths"], desc=f"fitur {cache.stem}")]
    features = pd.concat([samples[["ar", "time"]].reset_index(drop=True), pd.DataFrame(phys)], axis=1)

    # riwayat flare dari label dataset, dihitung per AR
    hist_parts = []
    for ar, g in features.groupby("ar", sort=False):
        h = history(index[index["ar"] == ar].sort_values("time"), g["time"])
        h.index = g.index
        hist_parts.append(h)
    features = pd.concat([features, pd.concat(hist_parts).sort_index()], axis=1)

    features.to_csv(cache, index=False)
    return samples.merge(features, on=["ar", "time"], how="left")


def load_samples(cfg, stride_h: float, run_dir: Path):
    """Sampel >= M dari AR train+val (untuk CV), dan semua sampel test (untuk uji akhir)."""
    idx = {s: load_index(cfg, s) for s in ("train", "val", "test")}
    dev_index = pd.concat([idx["train"], idx["val"]])
    dev = build_samples(dev_index, cfg, stride_h)
    dev = add_features(dev[dev["flux"] >= M_FLUX].reset_index(drop=True), dev_index, cfg,
                       run_dir / "features_dev.csv")
    test_all = add_features(build_samples(idx["test"], cfg, cfg.eval_stride_h), idx["test"], cfg,
                            run_dir / "features_test_all.csv")
    for d in (dev, test_all):
        d["X"] = (d["flux"] >= X_FLUX).astype(int)
    return dev, test_all


def cross_validate(dev: pd.DataFrame, n_folds: int, seed: int) -> dict:
    """Tiap model diuji pada AR yang tidak dilihatnya saat dilatih. Kolom p_<nama> ditambahkan ke dev."""
    dev["fold"] = assign_folds(dev, n_folds, seed)
    results = {}
    for name, cols in FEATURE_SETS.items():
        dev[f"p_{name}"] = np.nan
        for f in range(n_folds):
            train, held_out = dev["fold"] != f, dev["fold"] == f
            m = make_model(name).fit(dev.loc[train, cols], dev.loc[train, "X"])
            dev.loc[held_out, f"p_{name}"] = m.predict_proba(dev.loc[held_out, cols])[:, 1]
        results[name] = evaluate_x_vs_m(dev, f"p_{name}", base_rate=dev["X"].mean())

    for name in ("x", "logflux"):          # model dalam dari Progress 17, lipatan yang sama
        oof = Path(f"runs/xclass_{name}/oof_predictions.csv")
        if oof.exists():
            o = pd.read_csv(oof).rename(columns={"score": "s"})
            o["X"] = (o["flux"] >= X_FLUX).astype(int)
            results[f"model dalam ({name}, Progress 17)"] = evaluate_x_vs_m(o, "s")
    return results


def test_final_models(dev: pd.DataFrame, test: pd.DataFrame):
    """Latih tiap model pada semua AR train+val, lalu uji sekali pada AR test."""
    results, finals = {}, {}
    for name, cols in FEATURE_SETS.items():
        finals[name] = make_model(name).fit(dev[cols], dev["X"])
        test[f"p_{name}"] = finals[name].predict_proba(test[cols])[:, 1]
        results[name] = evaluate_x_vs_m(test, f"p_{name}", base_rate=dev["X"].mean())

    # pembanding: skor tahap 1 (bersih, karena tahap 1 tidak pernah melihat AR test)
    results["skor tahap 1 (pembanding)"] = evaluate_x_vs_m(test, "prob")
    # gabungan tanpa bobot yang dipelajari: rata-rata peringkat skor tahap 1 dan peluang fitur
    test["combo"] = (rankdata(test["prob"]) + rankdata(test[f"p_{FINAL}"])) / (2 * len(test))
    results[f"tahap 1 + {FINAL} (rata-rata peringkat)"] = evaluate_x_vs_m(test, "combo")
    return results, finals


def print_report(results, base_rate, thr_x, table):
    print(f"\nAUC X vs M (0,5 = acak). Klimatologi: P(X | >= M) = {base_rate:.3f}")
    for part in ("cv_per_ar", "test"):
        print(f"\n[{part}]  {'AUC':>6s} {'95%':>13s} {'AUC-AR':>7s} {'BSS':>6s}")
        for name, r in results[part].items():
            bss = f"{r['BSS_vs_klimatologi']:6.3f}" if "BSS_vs_klimatologi" in r else "     -"
            print(f"  {name:48s} {r['AUC']:6.3f} {r['AUC_CI95'][0]:5.2f}-{r['AUC_CI95'][1]:4.2f}   "
                  f"{r['AUC_AR']:6.3f} {bss}")
    print("\nKoefisien (fitur distandarkan):", results["koefisien_fisis+riwayat"])
    print(f"\nGabungan dua tahap pada semua sampel test (threshold P(X) = {thr_x:.3f}):\n{table.to_string()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1_dir", default="runs/flareformer")
    parser.add_argument("--run_dir", default="runs/xclass_feat")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--stride_h", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_dir, stage1_dir = Path(args.run_dir), Path(args.stage1_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(stage1_dir)

    # data (sama dengan train_xclass.py)
    dev, test_all = load_samples(cfg, args.stride_h, run_dir)
    test = test_all[test_all["flux"] >= M_FLUX].reset_index(drop=True)
    print(f"Train+val >= M: {len(dev)} sampel, {dev['ar'].nunique()} AR, AR X: {dev.loc[dev.X == 1, 'ar'].nunique()}")
    print(f"Test >= M     : {len(test)} sampel, {test['ar'].nunique()} AR, AR X: {test.loc[test.X == 1, 'ar'].nunique()}")

    # 1. cross-validation per AR
    results = {"cv_per_ar": cross_validate(dev, args.folds, args.seed)}

    # 2. model akhir -> test
    stage1 = pd.read_csv(stage1_dir / "test_predictions.csv", parse_dates=["time"])[["ar", "time", "prob"]]
    test = test.merge(stage1, on=["ar", "time"], how="left")
    test_all = test_all.merge(stage1, on=["ar", "time"], how="left")
    results["test"], finals = test_final_models(dev, test)

    # koefisien LR biasa (fitur distandarkan): tanda berlawanan pada fitur yang mirip = tanda overfitting
    lr = finals["fisis+riwayat"][-1]
    coef = pd.Series(lr.coef_[0], index=FEATURE_SETS["fisis+riwayat"]).sort_values(key=abs, ascending=False)
    results["koefisien_fisis+riwayat"] = coef.round(3).to_dict()

    # 3. gabungan dua tahap pada SEMUA sampel test; threshold P(X) dipilih dari CV
    thr_x = best_threshold(dev["X"], dev[f"p_{FINAL}"])
    thr1 = float(json.loads((stage1_dir / "test_metrics.json").read_text())["threshold"])
    test_all["p_x"] = finals[FINAL].predict_proba(test_all[FEATURE_SETS[FINAL]])[:, 1]
    table = confusion_table(true_class(test_all["flux"]),
                            two_stage_class(test_all["prob"], thr1, test_all["p_x"], thr_x))
    results["gabungan"] = {"threshold_P(X)_dari_CV": thr_x, "threshold_tahap1": thr1, "tabel": table.to_dict()}

    # simpan
    base_rate = float(dev["X"].mean())
    with open(run_dir / "model.pkl", "wb") as fh:
        pickle.dump({"models": finals, "feature_sets": FEATURE_SETS, "final": FINAL,
                     "threshold": thr_x, "base_rate": base_rate}, fh)
    dev.drop(columns="paths").to_csv(run_dir / "oof_predictions.csv", index=False)
    test.drop(columns="paths").to_csv(run_dir / "test_predictions.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print_report(results, base_rate, thr_x, table)


if __name__ == "__main__":
    main()
