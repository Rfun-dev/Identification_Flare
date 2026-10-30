"""Memotong magnetogram SDO/HMI menjadi PNG 224x224 yang formatnya sama dengan dataset.

Langkah pembuatan dataset Boucheron dkk. (2023) yang ditiru (dicek cocok 100% per piksel pada AR 11158):
1. Series hmi.M_720s (magnetogram line-of-sight, 1 gambar / 12 menit), filter QUALITY >= 0.
2. Kotak 600x600 piksel berpusat di titik Carrington TETAP selama AR terlihat (lihat
   noaa.ar_position). Titik itu diikuti dari frame ke frame karena Matahari berputar.
3. Diputar 180 derajat (array HMI mentah terbalik; di dataset kutub utara Matahari di atas).
4. Resize ke 224x224 (skimage, order=1, anti_aliasing), lalu
   piksel = round(clip(B + 2550, 0, 5100) * 255 / 5100).
5. Hanya dipakai bila pusat AR dalam +-60 derajat dan potongan tanpa NaN.
"""
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import astropy.units as u
import drms
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from PIL import Image
from skimage.transform import resize
from sunpy.coordinates import frames

from flare.jsoc import fetch_rows

BOX = 600            # ukuran potongan di resolusi penuh HMI (piksel)
OUT = 224            # ukuran gambar dataset
MAX_DEG = 60         # dataset hanya memakai AR dalam +-60 derajat dari pusat piringan


def png_name(ar: int, series: str, time: pd.Timestamp) -> str:
    """Nama file dengan format dataset, mis. 1158_hmi.M_720s.20110214_120000_TAI.1.magnetogram_224.png"""
    return f"{ar}_{series}.{time:%Y%m%d_%H%M%S}_TAI.1.magnetogram_224.png"


# ---------------------------------------------------------------- geometri

def _observer(k):
    """Waktu, posisi pengamat (satelit SDO), dan jari-jari Matahari untuk satu rekaman JSOC."""
    t = drms.to_datetime(k["T_OBS"])
    obs = SkyCoord(k["CRLN_OBS"] * u.deg, k["CRLT_OBS"] * u.deg, k["DSUN_OBS"] * u.m,
                   frame=frames.HeliographicCarrington, obstime=t, observer="self")
    obs = obs.transform_to(frames.HeliographicStonyhurst(obstime=t))
    rsun = k["RSUN_REF"] * u.m
    return t, obs, rsun


def carrington_to_pixel(k, lon: float, lat: float):
    """Titik Carrington di permukaan Matahari -> (x, y) piksel array FITS (mulai 0),
    plus bujur/lintang Stonyhurst (dilihat dari Bumi) untuk cek batas +-60 derajat.

    Alurnya: koordinat Carrington -> sudut di langit (helioprojective, detik busur)
    -> putar & skala sesuai metadata gambar (CROTA2, CDELT, CRPIX) -> piksel.
    """
    t, obs, rsun = _observer(k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        point = SkyCoord(lon * u.deg, lat * u.deg, rsun,
                         frame=frames.HeliographicCarrington(observer=obs, obstime=t, rsun=rsun))
        sky = point.transform_to(frames.Helioprojective(observer=obs, obstime=t, rsun=rsun))
        earth_view = point.transform_to(frames.HeliographicStonyhurst(obstime=t))
    tx = sky.Tx.to_value(u.arcsec) - k["CRVAL1"]
    ty = sky.Ty.to_value(u.arcsec) - k["CRVAL2"]
    angle = np.deg2rad(k["CROTA2"])
    dx = tx * np.cos(angle) + ty * np.sin(angle)
    dy = -tx * np.sin(angle) + ty * np.cos(angle)
    x = dx / k["CDELT1"] + k["CRPIX1"] - 1
    y = dy / k["CDELT2"] + k["CRPIX2"] - 1
    return x, y, earth_view.lon.deg, earth_view.lat.deg


# ---------------------------------------------------------------- potong & konversi

def to_dataset_png(crop: np.ndarray) -> np.ndarray:
    """Potongan 600x600 (Gauss) -> gambar 8-bit 224x224 seperti dataset (resize, skala, putar 180)."""
    small = resize(crop, (OUT, OUT), order=1, mode="reflect", clip=True, preserve_range=True, anti_aliasing=True)
    img = np.round(np.clip(small + 2550, 0, 5100) * 255 / 5100).astype(np.uint8)
    return np.ascontiguousarray(img[::-1, ::-1])


def make_png(k, lon: float, lat: float):
    """Satu rekaman JSOC -> gambar 224x224 berformat dataset, untuk AR di titik Carrington (lon, lat).

    Kembali: (gambar uint8, bujur Stonyhurst, lintang). ValueError bila tidak memenuhi syarat dataset.
    """
    x, y, earth_lon, earth_lat = carrington_to_pixel(k, lon, lat)
    if abs(earth_lon) >= MAX_DEG or abs(earth_lat) >= MAX_DEG:
        raise ValueError(f"AR di bujur {earth_lon:.1f}, lintang {earth_lat:.1f} dilihat dari Bumi; "
                         "model hanya dilatih untuk AR dalam +-60 derajat dari pusat piringan")

    x0 = int(round(x - (BOX - 1) / 2))            # pojok kiri atas kotak
    y0 = int(round(y - (BOX - 1) / 2))
    rows = fetch_rows(k["path"], y0, BOX)         # hanya 600 baris yang dibutuhkan
    if x0 < 0 or x0 + BOX > rows.shape[1]:
        raise ValueError("kotak 600x600 keluar dari batas gambar")
    crop = rows[:, x0:x0 + BOX]

    n_nan = int(np.isnan(crop).sum())
    if n_nan:
        raise ValueError(f"potongan berisi {n_nan} piksel NaN (dataset hanya memakai potongan tanpa NaN)")
    return to_dataset_png(crop), earth_lon, earth_lat


def save_sequence(frames_df: pd.DataFrame, lon: float, lat: float, out_dir: Path, ar: int, series: str):
    """Buat dan simpan PNG untuk semua frame. Keempat frame diunduh bersamaan (paralel).
    Kembali: daftar (path, bujur Stonyhurst, lintang) per frame, urut lama -> baru."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def make_and_save(k):
        img, earth_lon, earth_lat = make_png(k, lon, lat)
        path = out_dir / png_name(ar, series, k["time"])
        Image.fromarray(img).save(path)
        return path, earth_lon, earth_lat

    records = [k for _, k in frames_df.iterrows()]
    with ThreadPoolExecutor(len(records)) as pool:
        return list(pool.map(make_and_save, records))
