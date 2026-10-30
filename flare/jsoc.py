"""Mengambil magnetogram SDO/HMI dari JSOC (Joint Science Operations Center, Stanford).

Dua hal yang dilakukan modul ini:
1. query_frames : mencari rekaman berjarak 72 menit beserta metadatanya (posisi, rotasi, skala).
2. fetch_rows   : mengunduh HANYA baris gambar yang dibutuhkan (±2 MB), bukan file penuh (±14 MB).

Tidak perlu akun atau email: file diambil langsung dari arsip JSOC.
"""
import time
import urllib.request

import drms
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.io.fits.hdu.compressed._codecs import Rice1

JSOC = "http://jsoc.stanford.edu"
# Metadata yang dibutuhkan untuk menghitung posisi AR di gambar (lihat hmi.carrington_to_pixel).
KEYS = "T_REC,T_OBS,QUALITY,CRPIX1,CRPIX2,CRVAL1,CRVAL2,CDELT1,CDELT2,CROTA2,CRLN_OBS,CRLT_OBS,DSUN_OBS,RSUN_REF"
FITS_BLOCK = 2880        # header FITS tersusun dari blok 2880 byte, tiap baris (card) 80 byte

_client = None


def client() -> drms.Client:
    """Satu koneksi drms dipakai bersama (dibuat saat pertama kali dibutuhkan)."""
    global _client
    if _client is None:
        _client = drms.Client()
    return _client


def tai(t: pd.Timestamp) -> str:
    """Timestamp -> format waktu JSOC, mis. 2011.02.14_12:00:00_TAI."""
    return t.strftime("%Y.%m.%d_%H:%M:%S_TAI")


def latest_time(series: str) -> pd.Timestamp:
    """Waktu rekaman terbaru yang sudah tersedia di series JSOC."""
    keys = client().query(f"{series}[$]", key="T_REC")
    return drms.to_datetime(keys.T_REC[0])


# ---------------------------------------------------------------- mencari rekaman

def _query_records(series: str, start: pd.Timestamp, stop: pd.Timestamp) -> pd.DataFrame:
    """Semua rekaman berkualitas baik antara start dan stop, urut waktu, dengan kolom path & time."""
    keys, segments = client().query(f"{series}[{tai(start)}-{tai(stop)}][? QUALITY >= 0 ?]",
                                    key=KEYS, seg="magnetogram")
    if not keys.empty:                    # saat celah data, JSOC memberi tabel kosong tanpa kolom
        keys["path"] = segments["magnetogram"]
        keys = keys[keys["path"].str.startswith("/SUM")].copy()     # buang rekaman tanpa file
    if keys.empty:
        raise ValueError(f"Tidak ada rekaman {series} antara {start} dan {stop}.")
    keys["time"] = drms.to_datetime(keys["T_REC"])
    return keys.sort_values("time").reset_index(drop=True)


def query_frames(series: str, end: pd.Timestamp, n_frames: int, gap_min: int, tol_min: int) -> pd.DataFrame:
    """Cari n_frames rekaman (urut lama -> baru) berjarak gap_min menit yang berakhir di `end`.

    Aturannya sama dengan flare.data.frame_indices: tiap frame diambil dari rekaman terdekat
    dengan waktu targetnya, dan harus berselisih paling banyak tol_min menit.
    """
    start = end - pd.Timedelta(minutes=gap_min * (n_frames - 1) + tol_min)
    stop = end + pd.Timedelta(minutes=tol_min)
    records = _query_records(series, start, stop)

    rows = []
    for k in range(n_frames - 1, -1, -1):                    # dari frame paling lama ke paling baru
        target = end - pd.Timedelta(minutes=gap_min * k)
        i = (records["time"] - target).abs().idxmin()
        if abs(records["time"][i] - target) > pd.Timedelta(minutes=tol_min):
            raise ValueError(f"Frame sekitar {target} tidak tersedia di {series} (toleransi {tol_min} menit).")
        rows.append(records.loc[i])
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------- mengunduh sebagian file

def _download_range(url: str, start: int, end: int) -> bytes:
    """Byte ke-start sampai ke-end (inklusif) dari sebuah URL (HTTP Range request), dicoba 5 kali."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            if len(data) == end - start + 1:
                return data
        except OSError:
            if attempt == 4:
                raise
        time.sleep(2 ** attempt)
    raise OSError(f"unduhan tidak lengkap: {url} bytes {start}-{end}")


def _read_header(buf: bytes, pos: int):
    """Header FITS mulai dari byte `pos` -> (Header, posisi awal data sesudahnya)."""
    end = pos
    while True:
        block = buf[end:end + FITS_BLOCK]
        if len(block) < FITS_BLOCK:
            raise ValueError("header FITS terpotong")
        end += FITS_BLOCK
        cards = [block[i:i + 80] for i in range(0, FITS_BLOCK, 80)]
        if any(card.rstrip() == b"END" for card in cards):
            return fits.Header.fromstring(buf[pos:end].decode("ascii")), end


def _decode_rows(header, table: np.ndarray, chunk: np.ndarray, chunk_offset: int) -> np.ndarray:
    """Dekompresi baris-baris gambar (RICE) dari potongan heap -> medan magnet (Gauss), NaN = kosong."""
    params = {header[f"ZNAME{i}"]: header[f"ZVAL{i}"] for i in range(1, 10) if f"ZNAME{i}" in header}
    rice = Rice1(blocksize=params.get("BLOCKSIZE", 32), bytepix=params.get("BYTEPIX", 4),
                 tilesize=header["ZTILE1"])
    blank = header.get("ZBLANK", header.get("BLANK"))     # nilai bilangan bulat yang berarti "tidak ada data"
    rows = np.empty((len(table), header["ZNAXIS1"]), dtype=np.float64)
    for i, (n_bytes, offset) in enumerate(table):
        start = offset - chunk_offset
        raw = np.asarray(rice.decode(chunk[start:start + n_bytes])).reshape(-1).astype(np.int64)
        row = raw * header.get("BSCALE", 1.0) + header.get("BZERO", 0.0)
        if blank is not None:
            row[raw == blank] = np.nan
        rows[i] = row
    return rows


def fetch_rows(path: str, y0: int, n_rows: int) -> np.ndarray:
    """Baca hanya baris y0 .. y0+n_rows-1 dari magnetogram HMI di JSOC.

    File HMI dikompresi per baris (1 tile = 1 baris 4096 piksel), dan sebuah tabel di awal file
    mencatat posisi setiap baris. Jadi cukup:
      1. unduh header + tabel (±38 KB),
      2. unduh potongan file yang berisi baris-baris yang dibutuhkan (±2 MB untuk 600 baris),
      3. dekompresi baris-baris itu.
    """
    url = JSOC + path

    # 1. header + tabel posisi baris
    buf = _download_range(url, 0, 65535)
    _, pos = _read_header(buf, 0)                     # header utama (kosong, dilewati)
    header, data_start = _read_header(buf, pos)       # header gambar terkompresi
    assert header["ZCMPTYPE"] == "RICE_1" and header["ZTILE2"] == 1, "format kompresi HMI tak terduga"
    table_size = header["NAXIS1"] * header["NAXIS2"]
    if data_start + table_size > len(buf):
        buf = _download_range(url, 0, data_start + table_size - 1)
    int_type = ">i8" if header["TFORM1"].startswith("1Q") else ">i4"
    table = np.frombuffer(buf[data_start:data_start + table_size], dtype=int_type).reshape(-1, 2)
    table = table[y0:y0 + n_rows]                     # kolom: (jumlah byte, posisi di heap)
    if y0 < 0 or len(table) < n_rows:
        raise ValueError("kotak 600x600 keluar dari batas gambar")

    # 2. potongan heap yang memuat semua baris yang dibutuhkan
    heap_start = data_start + header.get("THEAP", table_size)
    lo = int(table[:, 1].min())
    hi = int((table[:, 1] + table[:, 0]).max())
    chunk = np.frombuffer(_download_range(url, heap_start + lo, heap_start + hi - 1), dtype=np.uint8)

    # 3. dekompresi
    return _decode_rows(header, table, chunk, lo)
