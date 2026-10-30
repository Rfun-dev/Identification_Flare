"""Langkah 4 - Evaluasi model terbaik pada data TEST (AR yang tidak pernah dilihat model).

Threshold yang dipakai adalah threshold terbaik dari data VALIDASI, bukan dari test,
supaya skor test benar-benar jujur.

Hasil di <run_dir>/: test_metrics.json, test_predictions.csv, roc_curve.png, history.png

    python evaluate.py --run_dir runs/flareformer
"""
import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.metrics import roc_curve

from flare.data import MagnetogramDataset, build_samples, load_index
from flare.engine import get_device, make_loader, predict_proba
from flare.metrics import scores
from flare.model import load_checkpoint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_roc(y, prob, m, path):
    fpr, tpr, _ = roc_curve(y, prob)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {m['AUC']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="tebakan acak")
    ax.scatter([m["FP"] / max(m["FP"] + m["TN"], 1)], [m["recall"]], color="red", zorder=3,
               label=f"threshold {m['threshold']:.2f} (TSS {m['TSS']:.3f})")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="Kurva ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_history(history_csv, path):
    h = pd.read_csv(history_csv)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(h["epoch"], h["train_loss"], marker="o")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Loss training")
    for k in ("val_TSS", "val_HSS", "val_AUC"):
        ax[1].plot(h["epoch"], h[k], marker="o", label=k.replace("val_", ""))
    ax[1].set(xlabel="epoch", title="Skor validasi")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", default="runs/flareformer")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = get_device()
    model, cfg, ckpt = load_checkpoint(run_dir / "best.pt", device)
    cfg.num_workers = args.num_workers
    amp = cfg.amp and device.type == "cuda"

    samples = build_samples(load_index(cfg, args.split), cfg, cfg.eval_stride_h)
    ds = MagnetogramDataset(samples, cfg.image_dir)
    prob = predict_proba(model, make_loader(ds, cfg), device, amp)

    m = scores(ds.labels, prob, ckpt["threshold"])
    print(f"Hasil pada {args.split} ({len(ds):,} sampel, {int(ds.labels.sum()):,} flare), "
          f"model epoch {ckpt['epoch']}:")
    for name, value in m.items():
        if isinstance(value, float):
            print(f"  {name:9s} = {value:.3f}")
        else:
            print(f"  {name:9s} = {value:,}")      # bilangan bulat (TP, FP, TN, FN)

    (run_dir / f"{args.split}_metrics.json").write_text(json.dumps(m, indent=2))
    predictions = samples.drop(columns="paths").assign(prob=prob)
    predictions.to_csv(run_dir / f"{args.split}_predictions.csv", index=False)
    plot_roc(ds.labels, prob, m, run_dir / "roc_curve.png")
    if (run_dir / "history.csv").exists():
        plot_history(run_dir / "history.csv", run_dir / "history.png")
    print(f"Hasil disimpan di {run_dir}/")


if __name__ == "__main__":
    main()
