"""Posisi active region (AR) dari laporan harian NOAA, untuk menentukan pusat potongan gambar.

Sumber laporan:
- daftar region SWPC (hanya 30 hari terakhir, paling baru),
- HEK / Heliophysics Event Knowledgebase (salinan laporan NOAA untuk semua tahun).

Aturan pusat kotak yang meniru dataset (dicek pada 4 AR, README Progress 15):
pakai laporan saat AR paling dekat dengan meridian tengah, lalu ubah ke koordinat Carrington
(koordinat yang ikut berputar bersama Matahari): L0(waktu laporan) + bujur + 0,09 derajat.
"""
import functools
import json
import urllib.parse
import urllib.request

import pandas as pd
from sunpy.coordinates import sun

SWPC_REGIONS = "https://services.swpc.noaa.gov/json/solar_regions.json"
HEK = "https://www.lmsal.com/hek/her"

# Kalibrasi empiris: selisih yang sama (0.086-0.104 derajat) terukur pada AR 11158, 12192,
# 12297, dan 12673 (2011-2017). Asalnya tidak diketahui (konvensi internal JSOC).
CARRINGTON_OFFSET_DEG = 0.09


def parse_location(loc: str):
    """'S07W84' -> (bujur Stonyhurst, lintang) = (84, -7). Barat positif, utara positif."""
    lat = int(loc[1:3]) * (1 if loc[0] == "N" else -1)
    lon = int(loc[4:6]) * (1 if loc[3] == "W" else -1)
    return lon, lat


# ---------------------------------------------------------------- sumber laporan

@functools.cache
def _swpc_regions() -> pd.DataFrame:
    """Daftar region SWPC (diunduh sekali per program)."""
    with urllib.request.urlopen(SWPC_REGIONS, timeout=60) as r:
        return pd.DataFrame(json.load(r))


def hek_search(event_type: str, frm_name: str, start: pd.Timestamp, end: pd.Timestamp, fields: str) -> pd.DataFrame:
    """Cari event di HEK, mis. flare (event_type="fl") atau AR ("ar").

    Pencarian dipecah per bulan agar hasil tiap permintaan tidak terpotong oleh batas HEK.
    """
    month_starts = list(pd.date_range(start, end, freq="MS"))
    edges = sorted(set([start] + month_starts + [end]))
    parts = []
    for a, b in zip(edges[:-1], edges[1:]):
        params = {"cosec": 2, "cmd": "search", "type": "column", "event_type": event_type,
                  "event_starttime": a.isoformat(), "event_endtime": b.isoformat(),
                  "event_coordsys": "helioprojective", "x1": -1200, "x2": 1200, "y1": -1200, "y2": 1200,
                  "param0": "FRM_NAME", "op0": "=", "value0": frm_name,
                  "return": fields, "result_limit": 5000}
        with urllib.request.urlopen(HEK + "?" + urllib.parse.urlencode(params), timeout=300) as r:
            result = json.load(r)
        if result.get("overmax"):
            raise RuntimeError(f"HEK: hasil {a}..{b} terpotong, perkecil rentang waktu")
        parts.append(pd.DataFrame(result["result"]))
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates() if not df.empty else df


def hek_ar_reports(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Laporan posisi harian NOAA dari HEK. Kolom: ar (5 digit), date, lon, lat (Stonyhurst)."""
    df = hek_search("ar", "NOAA SWPC Observer", start, end, "ar_noaanum,event_starttime,hgs_x,hgs_y")
    if df.empty:
        return pd.DataFrame(columns=["ar", "date", "lon", "lat"])
    return pd.DataFrame({"ar": df["ar_noaanum"].astype(int).to_numpy(),
                         "date": pd.to_datetime(df["event_starttime"]).to_numpy(),
                         "lon": df["hgs_x"].to_numpy(), "lat": df["hgs_y"].to_numpy()})


def noaa_reports(ar: int, time: pd.Timestamp) -> pd.DataFrame:
    """Laporan posisi harian NOAA untuk AR `ar` (4 digit) di sekitar `time`.

    Kolom: date (waktu berlakunya posisi), lon, lat (Stonyhurst, derajat), source.
    """
    regions = _swpc_regions()
    g = regions[regions["region"] % 10000 == ar].dropna(subset=["location"])
    swpc_is_recent = not g.empty and pd.to_datetime(g["observed_date"]).max() >= time.normalize() - pd.Timedelta(days=1)
    if swpc_is_recent:
        lonlat = g["location"].map(parse_location)
        # Posisi SWPC berlaku pukul 24:00 tanggal observasi ("valid at DD/2400Z") = 00:00 hari berikutnya.
        # Dicek: dengan +1 hari, L0 + bujur sama dengan kolom carrington_longitude SWPC.
        date = pd.to_datetime(g["observed_date"]) + pd.Timedelta(days=1)
        return pd.DataFrame({"date": date.to_numpy(), "lon": [p[0] for p in lonlat],
                             "lat": [p[1] for p in lonlat], "source": "SWPC"})

    day = time.normalize()
    found = hek_ar_reports(day - pd.Timedelta(days=14), day + pd.Timedelta(days=14))
    found = found[found["ar"] % 10000 == ar]
    return found.drop(columns="ar").assign(source="HEK")


# ---------------------------------------------------------------- pusat kotak

def _warnings(row, reports) -> str:
    """Peringatan bila laporan yang dipilih mungkin kurang tepat."""
    text = ""
    if abs(row["lon"]) > 7:
        text += "\n  PERINGATAN: AR belum dekat meridian tengah, posisi bisa bergeser dari aturan dataset"
    if abs(row["lat"] - reports["lat"].median()) > 3:
        text += (f"\n  PERINGATAN: lintang laporan ini ({row['lat']:+.0f}) jauh dari lintang laporan lain "
                 f"(median {reports['lat'].median():+.0f}); mungkin salah ketik di laporan NOAA. "
                 "Pertimbangkan --position.")
    return text


def ar_position(ar: int, time: pd.Timestamp, reports: pd.DataFrame | None = None):
    """Titik Carrington (bujur, lintang) pusat kotak, dengan aturan yang sama seperti dataset.

    Bila AR belum melewati meridian tengah (data real-time), dipakai laporan terdekat ke meridian
    yang sudah ada; posisinya bisa sedikit berbeda dari yang akan dipakai dataset.
    `reports` (kolom ar, date, lon, lat) bisa diberikan agar tidak perlu bertanya ke SWPC/HEK lagi.
    Kembali: (bujur Carrington, lintang, keterangan sumber).
    """
    if reports is None:
        rep = noaa_reports(ar % 10000, time)
    else:
        same_ar = reports["ar"] % 10000 == ar % 10000
        near_time = (reports["date"] - time).abs() <= pd.Timedelta(days=14)
        rep = reports[same_ar & near_time].assign(source="HEK")
    if rep.empty:
        raise ValueError(f"Laporan posisi NOAA untuk AR {ar} sekitar {time.date()} tidak ditemukan. "
                         "Beri posisi manual dengan --position BUJUR_CARRINGTON,LINTANG.")

    row = rep.iloc[rep["lon"].abs().argmin()]                # laporan paling dekat meridian tengah
    l0 = sun.L0(row["date"]).deg                            # bujur Carrington meridian tengah saat itu
    lon = (l0 + row["lon"] + CARRINGTON_OFFSET_DEG) % 360
    note = (f"laporan NOAA {row['date']:%Y-%m-%d %H:%M} ({row['source']}), "
            f"bujur {int(row['lon']):+d} dari meridian tengah")
    return float(lon), float(row["lat"]), note + _warnings(row, rep)
