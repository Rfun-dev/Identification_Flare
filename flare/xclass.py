"""Tahap 2: peluang flare kelas X bila flare >= M terjadi, dan alat evaluasi M vs X.

Dipakai oleh:
    xclass_features.py         melatih model peluang X
    evaluate_recent_xclass.py  menguji model pada data 2021-2025
    fetch_realtime.py          mencetak peluang X saat prediksi
    train_xclass.py            eksperimen model dalam (Progress 17, hasil negatif)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from flare.data import M_FLUX, X_FLUX
from flare.features import EVOL, HIST, PHYS
from flare.metrics import bootstrap_ci_by_ar

QUIET, M_CLASS, X_CLASS = "tenang(<M)", "M", "X"

# Fitur yang menurut riset makin besar -> flare makin besar (arah ditentukan sebelum melihat data).
# Fitur perubahan (EVOL) tidak dimasukkan karena arah pengaruhnya belum mapan di literatur.
COMPOSITE = ["log_usflux", "log_R", "lsg_mm", "log_gwill", "spot_area_mm2", "max_grad", "hist24", "hist48"]

# Model yang dibandingkan: nama -> kolom fitur yang diberikan ke model.
FEATURE_SETS = {
    "riwayat": HIST,
    "fisis": PHYS + EVOL,
    "fisis+riwayat": PHYS + EVOL + HIST,
    "fisis+riwayat (C=0,01)": PHYS + EVOL + HIST,
    "komposit bobot sama": PHYS + EVOL + HIST,      # kolom yang dipakai ditentukan oleh COMPOSITE
}
FINAL = "komposit bobot sama"                         # model yang dipakai untuk prediksi


class Composite:
    """Skor komposit berbobot sama: rata-rata z-score fitur, lalu satu regresi logistik 1 variabel
    untuk mengubah skor menjadi peluang. Hanya 2 parameter dipelajari, jadi sulit menghafal
    (cara standar bila contoh sangat sedikit; bobot sama sering menyaingi bobot hasil fitting)."""

    def __init__(self, cols):
        self.cols = cols

    def score(self, X):
        z = (X[self.cols] - self.mu) / self.sd        # z-score tiap fitur
        return z.mean(axis=1).to_numpy()[:, None]     # rata-rata -> satu angka per sampel

    def fit(self, X, y):
        self.mu = X[self.cols].mean()
        self.sd = X[self.cols].std().replace(0, 1)
        self.lr = LogisticRegression(max_iter=1000).fit(self.score(X), y)
        return self

    def predict_proba(self, X):
        return self.lr.predict_proba(self.score(X))


def make_model(name: str):
    """Model baru (belum dilatih) sesuai nama di FEATURE_SETS."""
    if name.startswith("komposit"):
        return Composite(COMPOSITE)
    C = 0.01 if name.endswith("(C=0,01)") else 1.0          # C kecil = regularisasi kuat
    # Tanpa class_weight agar keluarannya peluang yang terkalibrasi (bukan sekadar skor).
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


# ---------------------------------------------------------------- cross-validation per AR

def assign_folds(samples: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    """Nomor lipatan cross-validation untuk tiap sampel, ditentukan per AR.

    Semua sampel satu AR masuk lipatan yang sama (agar tidak bocor). AR yang pernah berflare X
    dan AR yang hanya berflare M disebar terpisah, supaya tiap lipatan kebagian AR X sama banyak.
    """
    ar_has_x = samples.groupby("ar")["flux"].max().ge(X_FLUX)
    rng = np.random.default_rng(seed)
    fold_of = {}
    for has_x in (True, False):
        ars = rng.permutation(ar_has_x.index[ar_has_x == has_x].to_numpy())
        for i, ar in enumerate(ars):
            fold_of[ar] = i % n_folds
    return samples["ar"].map(fold_of)


# ---------------------------------------------------------------- evaluasi

def auc_ci(df: pd.DataFrame, col: str) -> list:
    """Selang kepercayaan 95% untuk AUC X vs M (bootstrap per AR)."""
    stat = {"AUC": lambda s: roc_auc_score(s["X"], s[col])}
    return bootstrap_ci_by_ar(df, "X", stat)["AUC"]


def evaluate_x_vs_m(df: pd.DataFrame, col: str, base_rate: float | None = None) -> dict:
    """Seberapa baik kolom skor `col` memisahkan sampel X (kolom X = 1) dari sampel M.

    AUC    : per sampel (0,5 = acak, 1 = sempurna), plus selang kepercayaan 95%.
    AUC_AR : per AR, memakai median skor tiap AR.
    BSS    : Brier skill score terhadap klimatologi (Barnes dkk. 2016), bila base_rate diberikan.
    """
    per_ar = df.groupby("ar").agg(X=("X", "max"), s=(col, "median"))
    out = {"AUC": float(roc_auc_score(df["X"], df[col])),
           "AUC_CI95": auc_ci(df, col),
           "AUC_AR": float(roc_auc_score(per_ar["X"], per_ar["s"]))}
    if base_rate is not None:
        brier = brier_score_loss(df["X"], df[col])
        brier_climatology = brier_score_loss(df["X"], np.full(len(df), base_rate))
        out["BSS_vs_klimatologi"] = float(1 - brier / brier_climatology)
    return out


# ---------------------------------------------------------------- gabungan dua tahap (tiga kelas)

def true_class(flux) -> np.ndarray:
    """Kelas sebenarnya dari fluks flare terbesar 24 jam: 'X', 'M', atau 'tenang(<M)'."""
    flux = np.asarray(flux)
    return np.select([flux >= X_FLUX, flux >= M_FLUX], [X_CLASS, M_CLASS], default=QUIET)


def two_stage_class(stage1_prob, stage1_threshold, x_score, x_threshold) -> np.ndarray:
    """Prediksi tiga kelas: tahap 1 memutuskan ada flare >= M atau tidak, lalu tahap 2 memutuskan M atau X."""
    stage1_prob, x_score = np.asarray(stage1_prob), np.asarray(x_score)
    return np.select([stage1_prob < stage1_threshold, x_score >= x_threshold], [QUIET, X_CLASS], default=M_CLASS)


def confusion_table(true, pred) -> pd.DataFrame:
    """Tabel 3x3: baris = kelas sebenarnya, kolom = kelas prediksi."""
    order = [QUIET, M_CLASS, X_CLASS]
    table = pd.crosstab(pd.Series(true, name="sebenarnya"), pd.Series(pred, name="prediksi"))
    return table.reindex(index=order, columns=order, fill_value=0)
