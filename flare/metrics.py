"""Metrik evaluasi prediksi flare.

Data flare sangat tidak seimbang (flare >= M hanya ~3% sampel), sehingga akurasi
menyesatkan: model yang selalu menjawab "tidak ada flare" sudah 97% akurat.
Karena itu metrik utama adalah TSS (True Skill Statistic), standar di riset flare.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def best_threshold(y, score) -> float:
    """Ambang skor yang memaksimalkan TSS (= TPR - FPR) di sepanjang kurva ROC."""
    fpr, tpr, thr = roc_curve(y, score)
    return float(thr[np.argmax(tpr - fpr)])


def scores(y, prob, threshold: float) -> dict:
    """Semua metrik pada satu threshold. y = label 0/1, prob = skor model."""
    y = np.asarray(y).astype(bool)
    prob = np.asarray(prob)
    pred = prob >= threshold
    tp, fn = int(np.sum(pred & y)), int(np.sum(~pred & y))
    fp, tn = int(np.sum(pred & ~y)), int(np.sum(~pred & ~y))

    recall = tp / max(tp + fn, 1)       # dari semua flare, berapa yang tertangkap
    fpr = fp / max(fp + tn, 1)          # dari semua kondisi tenang, berapa yang salah alarm
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return {
        "threshold": float(threshold),
        "TSS": recall - fpr,
        "HSS": 2 * (tp * tn - fn * fp) / max(hss_den, 1),
        "recall": recall,
        "precision": tp / max(tp + fp, 1),
        "accuracy": (tp + tn) / max(len(y), 1),
        "AUC": float(roc_auc_score(y, prob)) if 0 < y.sum() < len(y) else float("nan"),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


def bootstrap_ci_by_ar(df: pd.DataFrame, label_col: str, stats: dict, n=1000, seed=0) -> dict:
    """Selang kepercayaan 95% untuk beberapa metrik, dengan bootstrap per AR.

    Sampel dari AR yang sama hampir identik, jadi yang diundi ulang adalah AR-nya (semua
    sampel satu AR ikut terbawa), bukan sampel satu per satu.

    stats   : {nama: fungsi(tabel_hasil_undian) -> angka}
    Kembali : {nama: [batas bawah, batas atas]}
    """
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("ar")]
    values = {name: [] for name in stats}
    for _ in range(n):
        picked = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[i] for i in picked])
        if 0 < sample[label_col].sum() < len(sample):        # perlu kedua kelas agar AUC terdefinisi
            for name, fn in stats.items():
                values[name].append(fn(sample))
    return {name: percentile_95(v) for name, v in values.items()}


def percentile_95(values) -> list:
    """Batas bawah (persentil 2,5) dan atas (97,5) dari hasil bootstrap."""
    if not values:
        return [float("nan"), float("nan")]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
