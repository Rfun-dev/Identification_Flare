"""Langkah 8 (eksperimen) - Model dalam tahap 2: membedakan flare kelas M dari kelas X.

Hasilnya negatif (README Progress 17); pendekatan yang dipakai adalah xclass_features.py (Progress 18).

Dipakai SETELAH model tahap 1 menyatakan "akan ada flare >= M". Tahap 2 dilatih dan diuji hanya pada
sampel yang memang diikuti flare >= M, dan menjawab: kekuatan flarenya M atau X?

    python train_xclass.py --target x --run_dir runs/xclass_x
    python train_xclass.py --target logflux --run_dir runs/xclass_logflux

--target x         klasifikasi biner X vs M (BCE, X diberi bobot lebih)
--target logflux   regresi log10(fluks): M1 -> 0, X1 -> 1; skor >= 1 berarti X. Memakai seluruh kekuatan
                   flare M1..M9, bukan hanya 18 AR berlabel X.

Kelas X hanya ada di 18 AR (13 train, 2 val, 3 test), jadi validasi biasa tidak cukup. Alurnya:
1. Cross-validation per AR (--folds) pada AR train+val: tiap AR diuji oleh model yang belum
   melihatnya. Dari sini didapat AUC (dengan selang kepercayaan) dan threshold X vs M.
2. Model akhir dilatih pada semua AR train+val, lalu diuji SEKALI pada 17+3 AR test.
3. Gabungan dengan model tahap 1 diuji pada SEMUA sampel test (tiga kelas: tenang / M / X).
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch import nn

from flare.config import load_config
from flare.data import M_FLUX, X_FLUX, BalancedSampler, MagnetogramDataset, build_samples, load_index
from flare.engine import get_device, make_loader, make_optimizer, predict_logits, set_seed, train_one_epoch
from flare.metrics import best_threshold, scores
from flare.model import build_model
from flare.xclass import assign_folds, auc_ci, confusion_table, true_class, two_stage_class


def target_of(flux: np.ndarray, kind: str) -> np.ndarray:
    """Target training: 1/0 untuk X/M (kind="x"), atau log10(fluks) dengan M1 = 0, X1 = 1 (kind="logflux")."""
    if kind == "x":
        return (flux >= X_FLUX).astype(np.float32)
    return np.clip(np.log10(np.maximum(flux, 1e-9)) + 5, 0, 2.2).astype(np.float32)


# ---------------------------------------------------------------- melatih & memprediksi

def fit(train_s: pd.DataFrame, cfg, kind: str, epochs: int, device, seed: int):
    """Latih satu model tahap 2 pada sampel train_s (semuanya >= M)."""
    set_seed(seed)
    is_x = train_s["flux"].to_numpy() >= X_FLUX
    targets = target_of(train_s["flux"].to_numpy(), kind)
    ds = MagnetogramDataset(train_s.assign(label=targets), cfg.image_dir, augment=True)
    sampler = BalancedSampler(is_x.astype(np.float32), cfg.neg_ratio, seed)     # semua X + M acak
    loader = make_loader(ds, cfg, sampler=sampler)
    model = build_model(cfg).to(device)
    if kind == "x":
        pos_weight = torch.tensor(sampler.n_neg / len(sampler.pos), device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.SmoothL1Loss()
    optimizer, scheduler = make_optimizer(model, cfg, epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=False)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, False)
        scheduler.step()
        print(f"    epoch {ep}/{epochs} loss {loss:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return model


def predict_samples(model, samples: pd.DataFrame, cfg, device) -> np.ndarray:
    ds = MagnetogramDataset(samples.assign(label=0.0), cfg.image_dir)
    return predict_logits(model, make_loader(ds, cfg), device)


# ---------------------------------------------------------------- metrik

def summarize(df: pd.DataFrame, kind: str) -> dict:
    """Metrik X vs M pada sampel >= M dengan kolom ar, flux, X, score."""
    per_ar = df.groupby("ar").agg(X=("X", "max"), score=("score", "median"))
    out = {"n_samples": len(df), "n_X_samples": int(df["X"].sum()), "n_ar": int(df["ar"].nunique()),
           "n_X_ar": int(per_ar["X"].sum()),
           "AUC_sample": float(roc_auc_score(df["X"], df["score"])),
           "AUC_sample_CI95": auc_ci(df, "score"),
           "AUC_ar": float(roc_auc_score(per_ar["X"], per_ar["score"]))}
    if kind == "logflux":
        out["spearman_logflux"] = float(spearmanr(df["score"], np.log10(df["flux"])).statistic)
        error = df["score"].clip(0, 2.2) - target_of(df["flux"].to_numpy(), kind)
        out["MAE_logflux"] = float(np.mean(np.abs(error)))
    return out


def tss_at(y, score, thr) -> dict:
    """TSS X vs M pada satu threshold."""
    m = scores(y, score, thr)
    return {"threshold": float(thr), "TSS": m["TSS"], "recall_X": m["recall"],
            "false_alarm_M_as_X": m["FP"] / max(m["FP"] + m["TN"], 1),
            "TP": m["TP"], "FN": m["FN"], "FP": m["FP"], "TN": m["TN"]}


def scalars(d: dict) -> dict:
    """Hanya isi yang bukan dict (untuk dicetak ringkas)."""
    return {k: v for k, v in d.items() if not isinstance(v, dict)}


# ---------------------------------------------------------------- tiga bagian eksperimen

def cross_validate(dev_mx, cfg, args, device, run_dir):
    """1. Cross-validation per AR. Kembali: (metrik CV, threshold X vs M dari CV)."""
    dev_mx["fold"] = assign_folds(dev_mx, args.folds, args.seed)
    oof = []
    for f in range(args.folds):
        tr, te = dev_mx[dev_mx["fold"] != f], dev_mx[dev_mx["fold"] == f]
        n_x_ar = te.loc[te["flux"] >= X_FLUX, "ar"].nunique()
        print(f"Lipatan {f + 1}/{args.folds}: latih {len(tr)} sampel / {tr['ar'].nunique()} AR, "
              f"uji {len(te)} sampel / {te['ar'].nunique()} AR (AR X: {n_x_ar})", flush=True)
        model = fit(tr, cfg, args.target, args.epochs, device, args.seed + f)
        o = te[["ar", "time", "flux", "fold"]].copy()
        o["score"] = predict_samples(model, te, cfg, device)
        oof.append(o)
        del model
        torch.cuda.empty_cache()
    oof = pd.concat(oof, ignore_index=True)
    oof["X"] = (oof["flux"] >= X_FLUX).astype(int)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)

    cv = summarize(oof, args.target)
    thr = best_threshold(oof["X"], oof["score"])
    cv["threshold_from_cv"] = thr
    cv["at_threshold (optimistis: threshold dipilih dari data yang sama)"] = tss_at(oof["X"], oof["score"], thr)
    print("\nCross-validation per AR:", json.dumps(scalars(cv), indent=1))
    return cv, thr


def test_final(dev_mx, test_mx, cfg, args, device, run_dir, thr, stage1):
    """2. Model akhir dilatih pada semua AR train+val, diuji sekali pada sampel test >= M."""
    print("\nMelatih model akhir pada semua AR train+val ...", flush=True)
    final = fit(dev_mx, cfg, args.target, args.epochs, device, args.seed)
    torch.save({"model": final.state_dict(), "config": asdict(cfg), "target": args.target, "threshold": thr,
                "stage1_dir": args.stage1_dir}, run_dir / "best.pt")
    test_mx["score"] = predict_samples(final, test_mx, cfg, device)
    test_mx["X"] = (test_mx["flux"] >= X_FLUX).astype(int)
    test = summarize(test_mx, args.target)
    test["at_cv_threshold"] = tss_at(test_mx["X"], test_mx["score"], thr)

    both = test_mx.merge(stage1, on=["ar", "time"])          # bandingkan dengan skor tahap 1 pada sampel sama
    if len(both) and both["X"].nunique() == 2:
        test["baseline_stage1_score_AUC_sample"] = float(roc_auc_score(both["X"], both["prob"]))
        test["model_AUC_same_samples"] = float(roc_auc_score(both["X"], both["score"]))
    print("\nTest (sampel >= M):", json.dumps(scalars(test), indent=1))
    print("  pada threshold dari CV:", json.dumps(test["at_cv_threshold"]))
    test_mx.drop(columns="paths").to_csv(run_dir / "test_xclass_predictions.csv", index=False)
    return final, test


def combined(final, test_s_all, cfg, args, device, run_dir, thr, stage1) -> pd.DataFrame:
    """3. Gabungan dua tahap (tenang / M / X) pada SEMUA sampel test."""
    test_all = test_s_all.copy()
    test_all["score2"] = predict_samples(final, test_all, cfg, device)
    c = test_all.merge(stage1, on=["ar", "time"])
    thr1 = torch.load(Path(args.stage1_dir) / "best.pt", map_location="cpu", weights_only=True)["threshold"]
    true = true_class(c["flux"])
    pred = two_stage_class(c["prob"], thr1, c["score2"], thr)
    table = confusion_table(true, pred)
    print("\nGabungan dua tahap pada semua sampel test:\n", table.to_string())
    c.drop(columns="paths").assign(true_class=true, pred_class=pred).to_csv(
        run_dir / "test_combined_predictions.csv", index=False)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=["x", "logflux"], default="x")
    parser.add_argument("--stage1_dir", default="runs/flareformer")
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--stride_h", type=float, default=2.0, help="jarak antar sampel training (jam)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="uji cepat: 2 lipatan, 1 epoch, data dipangkas")
    args = parser.parse_args()
    run_dir = Path(args.run_dir or f"runs/xclass_{args.target}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.stage1_dir)
    device = get_device()
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device} | target: {args.target} | epochs: {args.epochs}")

    # data: semua sampel dari AR train+val dan test; tahap 2 hanya memakai yang >= M
    idx = {s: load_index(cfg, s) for s in ("train", "val", "test")}
    dev_s = build_samples(pd.concat([idx["train"], idx["val"]]), cfg, args.stride_h)
    test_s_all = build_samples(idx["test"], cfg, cfg.eval_stride_h)
    dev_mx = dev_s[dev_s["flux"] >= M_FLUX].reset_index(drop=True)
    test_mx = test_s_all[test_s_all["flux"] >= M_FLUX].reset_index(drop=True)
    for name, d in [("train+val", dev_mx), ("test", test_mx)]:
        print(f"  {name:10s}: {len(d):6,} sampel >= M | X: {(d['flux'] >= X_FLUX).sum():5,} | "
              f"AR: {d['ar'].nunique()} (X: {d.loc[d['flux'] >= X_FLUX, 'ar'].nunique()})")
    if args.smoke:
        keep = dev_mx.sample(frac=1, random_state=0).groupby("ar").head(6)     # acak agar sampel X ikut
        dev_mx, args.folds, args.epochs = keep.reset_index(drop=True), 2, 1
        test_mx = test_mx.sample(60, random_state=0).reset_index(drop=True)
        test_s_all = pd.concat([test_s_all.sample(200, random_state=0), test_s_all[test_s_all["flux"] >= X_FLUX].head(30)])
        test_s_all = test_s_all.drop_duplicates(["ar", "time"]).reset_index(drop=True)

    stage1 = pd.read_csv(Path(args.stage1_dir) / "test_predictions.csv", parse_dates=["time"])[["ar", "time", "prob"]]
    cv, thr = cross_validate(dev_mx, cfg, args, device, run_dir)
    final, test = test_final(dev_mx, test_mx, cfg, args, device, run_dir, thr, stage1)
    table = combined(final, test_s_all, cfg, args, device, run_dir, thr, stage1)
    (run_dir / "xclass_metrics.json").write_text(json.dumps(
        {"target": args.target, "epochs": args.epochs, "cv_per_ar": cv, "test_final": test,
         "combined_confusion": table.to_dict()}, indent=2))
    print(f"\nSelesai -> {run_dir}")


if __name__ == "__main__":
    main()
