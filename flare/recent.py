"""Alat bantu evaluasi pada data SETELAH dataset (2021+), dipakai evaluate_recent.py dan
evaluate_recent_xclass.py.

Alurnya selalu sama:
1. load_events     : daftar flare SWPC + laporan posisi AR dari HEK (disimpan ke CSV, diunduh sekali).
2. load_or_create  : daftar sampel (ar, time); bila CSV-nya sudah ada, dilanjutkan.
3. build_all       : buat 4 PNG per sampel lewat JSOC; sampel yang sudah selesai tidak diunduh lagi.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from flare import hmi, jsoc, noaa
from flare.features import prepare_flares

SERIES = "hmi.M_720s"        # data final HMI (data near-real-time tidak dipakai untuk evaluasi)


def load_events(data_dir: Path, start, end):
    """Flare SWPC (dengan nomor AR) dan laporan posisi AR dari HEK; disimpan agar tidak diunduh ulang."""
    flares_csv, reports_csv = data_dir / "hek_flares.csv", data_dir / "hek_ar_reports.csv"
    if not flares_csv.exists():
        print("Mengambil daftar flare SWPC dari HEK ...")
        flares = noaa.hek_search("fl", "SWPC", start, end, "ar_noaanum,event_peaktime,fl_goescls")
        flares.to_csv(flares_csv, index=False)
    if not reports_csv.exists():
        print("Mengambil laporan posisi harian AR dari HEK ...")
        margin = pd.Timedelta(days=14)
        noaa.hek_ar_reports(start - margin, end + margin).to_csv(reports_csv, index=False)
    flares = prepare_flares(pd.read_csv(flares_csv))
    reports = pd.read_csv(reports_csv, parse_dates=["date"])
    return flares, reports


def load_or_create(csv_path: Path, create) -> pd.DataFrame:
    """Baca daftar sampel yang sudah ada (agar pekerjaan bisa dilanjutkan), atau buat baru dengan create().
    Kolom status: '' = belum, 'ok', 'dilewati: ...' (tidak memenuhi syarat), 'error: ...' (coba lagi)."""
    if csv_path.exists():
        samples = pd.read_csv(csv_path, parse_dates=["time"])
    else:
        samples = create()
        samples["status"] = ""
        samples["paths"] = ""
    samples["status"] = samples["status"].fillna("").astype(str)
    samples["paths"] = samples["paths"].fillna("").astype(str)
    return samples


def build_sample(row, reports: pd.DataFrame, cfg, out_dir: Path):
    """4 PNG untuk satu sampel. Kembali: (daftar path, status)."""
    ar4 = int(row["ar"]) % 10000
    folder = out_dir / str(ar4)
    try:
        frames = jsoc.query_frames(SERIES, row["time"], cfg.n_frames, cfg.frame_gap_min, cfg.frame_tol_min)
        paths = [folder / hmi.png_name(ar4, SERIES, t) for t in frames["time"]]
        if all(p.exists() for p in paths):
            return paths, "ok"
        lon, lat, _ = noaa.ar_position(ar4, row["time"], reports)
        result = hmi.save_sequence(frames, lon, lat, folder, ar4, SERIES)
        return [p for p, _, _ in result], "ok"
    except ValueError as e:              # tidak memenuhi syarat dataset (di luar +-60 derajat, NaN, celah data)
        return [], f"dilewati: {e}"
    except Exception as e:               # gangguan jaringan dll.; sampel bisa dicoba lagi nanti
        return [], f"error: {type(e).__name__}: {e}"


def build_all(samples: pd.DataFrame, reports: pd.DataFrame, cfg, out_dir: Path, csv_path: Path, workers: int):
    """Buat PNG untuk semua sampel yang belum selesai, beberapa sampel sekaligus.
    Kemajuan disimpan ke csv_path tiap 10 sampel, jadi program bisa dihentikan lalu dilanjutkan."""
    todo = samples.index[~samples["status"].str.startswith(("ok", "dilewati"))]
    with ThreadPoolExecutor(workers) as pool:
        jobs = {pool.submit(build_sample, samples.loc[i], reports, cfg, out_dir): i for i in todo}
        for n, job in enumerate(as_completed(jobs), 1):
            i = jobs[job]
            paths, status = job.result()
            samples.loc[i, "status"] = status
            samples.loc[i, "paths"] = "|".join(map(str, paths))
            print(f"[{n}/{len(todo)}] AR {samples.loc[i, 'ar']} {samples.loc[i, 'time']:%Y-%m-%d %H:%M}: {status}",
                  flush=True)
            if n % 10 == 0:
                samples.to_csv(csv_path, index=False)
    samples.to_csv(csv_path, index=False)
    return samples
