"""Fitur fisis magnetogram dan riwayat flare, untuk memperkirakan KEKUATAN flare (M vs X).

Dasar riset:
- Fluks tak bertanda & luas medan kuat: AR yang lebih besar menghasilkan flare lebih besar
  (Sammis, Tang & Zirin 2000; USFLUX pada Bobra & Couvidat 2015).
- Nilai R (Schrijver 2007): fluks di sekitar garis netral bermedan kuat (>150 G), proksi energi bebas.
- Panjang garis netral bergradien tinggi (Falconer dkk. 2003, >50 G/Mm) dan GWILL
  (Mason & Hoeksema 2010): proksi energi bebas dari magnetogram line-of-sight.
- Perubahan fitur selama 3,6 jam: munculnya fluks baru sebelum flare besar.
- Riwayat flare 24-48 jam sebelumnya: indikator terkuat flare berikutnya (Wheatland 2005;
  Falconer dkk. 2012).

Skala piksel: 224 piksel = 600 piksel HMI x 0,504" -> 1,35"/piksel ~ 0,98 Mm (proyeksi diabaikan,
AR dalam +-60 derajat). Nilai PNG terpotong di +-2550 G, jadi inti bintik terkuat ikut terpotong.
"""
import numpy as np
import pandas as pd
from scipy import ndimage

from flare.data import goes_to_flux, load_magnetogram, to_minutes

MM_PER_PX = 600 * 0.504 / 224 * 0.725        # ~0.98 Mm per piksel 224x224
PX_AREA_CM2 = (MM_PER_PX * 1e8) ** 2
B_NOISE, B_STRONG, B_SPOT = 100.0, 150.0, 1000.0     # Gauss
GRAD_MIN = 50.0                                       # G/Mm (Falconer dkk. 2003)
R_FWHM_MM = 15.0                                      # Schrijver 2007

# Nama-nama fitur
PHYS = ["log_usflux", "log_R", "lsg_mm", "log_gwill", "spot_area_mm2", "max_grad"]
EVOL = ["d_log_usflux", "d_log_R"]
HIST = ["hist24", "hist48", "hist_known"]

# Skala GOES: label dataset memakai skala GOES-15 (dikali 0,7). Sejak era GOES-16 (±2020) faktor itu
# dihapus, jadi fluks baru dikali 0,7 agar sebanding dengan label latih.
GOES16_FACTOR = 0.7
GOES16_START = pd.Timestamp("2020-01-01")
DEG_PER_DAY = 13.2      # rotasi Matahari dilihat dari Bumi
HOURS_24 = pd.Timedelta(hours=24)


# ---------------------------------------------------------------- fitur fisis dari gambar

def field(path) -> np.ndarray:
    """PNG dataset -> medan magnet line-of-sight (Gauss)."""
    return load_magnetogram(path).astype(np.float64) * 2550.0


def physics(B: np.ndarray) -> dict:
    """Enam fitur fisis dari satu magnetogram B (Gauss)."""
    absB = np.abs(B)

    # fluks magnet tak bertanda (Mx), hanya piksel di atas derau
    usflux = absB[absB > B_NOISE].sum() * PX_AREA_CM2

    # garis netral bermedan kuat: daerah + dan - (>150 G) yang dilebarkan 1 piksel lalu bertumpuk
    kernel = np.ones((3, 3), bool)
    near_pos = ndimage.binary_dilation(B > B_STRONG, kernel)
    near_neg = ndimage.binary_dilation(B < -B_STRONG, kernel)
    pil = near_pos & near_neg

    # nilai R: fluks yang diberi bobot Gaussian (lebar 15 Mm) di sekitar garis netral
    sigma = R_FWHM_MM / 2.355 / MM_PER_PX
    weight = ndimage.gaussian_filter(pil.astype(float), sigma)
    R = (absB * weight).sum() * PX_AREA_CM2

    # gradien medan (G/Mm); garis netral yang gradiennya tajam
    gy, gx = np.gradient(B, MM_PER_PX)
    grad = np.hypot(gx, gy)
    sharp_pil = pil & (grad > GRAD_MIN)

    return {"log_usflux": np.log10(usflux + 1e18),
            "log_R": np.log10(R + 1e16),
            "lsg_mm": sharp_pil.sum() * MM_PER_PX,
            "log_gwill": np.log10((grad * sharp_pil).sum() * MM_PER_PX + 1.0),
            "spot_area_mm2": (absB > B_SPOT).sum() * MM_PER_PX ** 2,
            "max_grad": float(np.percentile(grad, 99.9))}


def sequence_features(paths) -> dict:
    """Fitur dari urutan frame (lama -> baru): fisis frame terakhir + perubahan sejak frame pertama."""
    first = physics(field(paths[0]))
    last = physics(field(paths[-1]))
    out = dict(last)
    out["d_log_usflux"] = last["log_usflux"] - first["log_usflux"]
    out["d_log_R"] = last["log_R"] - first["log_R"]
    return out


# ---------------------------------------------------------------- riwayat flare

def flare_level(flux) -> float:
    """Kelas flare terbesar sebagai angka: 0 = tidak ada >=C, 1 = C, 2 = M, 3 = X.
    Sengaja kasar agar bisa dihitung sama persis dari label dataset maupun daftar flare SWPC."""
    return float(np.digitize(flux, [1e-6, 1e-5, 1e-4]))


def _level_near(t_all: np.ndarray, flux: np.ndarray, target: int, tol_min: int) -> float:
    """flare_level dari label gambar terdekat dengan waktu `target` (menit), atau NaN bila tidak ada
    gambar dalam toleransi. Bila dua gambar sama dekat, dipilih yang lebih awal."""
    i = np.searchsorted(t_all, target).clip(0, len(t_all) - 1)
    candidates = [j for j in (i - 1, i) if 0 <= j < len(t_all)]
    j = min(candidates, key=lambda j: abs(t_all[j] - target))
    if abs(t_all[j] - target) > tol_min:
        return np.nan
    return flare_level(flux[j])


def history(index_ar: pd.DataFrame, times: pd.Series, tol_min: int = 24) -> pd.DataFrame:
    """Riwayat flare AR sebelum tiap waktu, dari label dataset (satu AR).

    Label gambar pada waktu t-24 jam = flare terbesar dengan puncak dalam (t-24 jam, t), jadi
    tidak tumpang tindih dengan label sampel (t, t+24 jam]. hist_known = 0 bila AR belum terlihat
    24 jam sebelumnya (mis. baru muncul dari tepi timur); nilai yang tidak diketahui diisi 0.
    """
    t_all = to_minutes(index_ar["time"])
    flux = index_ar["flux"].to_numpy()
    rows = []
    for t in to_minutes(times):
        h24 = _level_near(t_all, flux, t - 24 * 60, tol_min)
        h48 = _level_near(t_all, flux, t - 48 * 60, tol_min)
        rows.append({"hist24": h24, "hist48": h48, "hist_known": float(not np.isnan(h24))})
    return pd.DataFrame(rows).fillna(0.0)


# ---------------------------------------------------------------- riwayat dari daftar flare SWPC

def prepare_flares(hek: pd.DataFrame) -> pd.DataFrame:
    """Daftar flare SWPC dari HEK -> kolom ar (5 digit), peak, fl_goescls, flux (skala GOES saat itu).
    Hanya flare dengan nomor AR dan kelas GOES (sebagian entri memakai kelas H-alpha, mis. "SF")."""
    if hek.empty:
        return pd.DataFrame(columns=["ar", "peak", "fl_goescls", "flux"])
    has_ar = hek["ar_noaanum"] > 0
    is_goes_class = hek["fl_goescls"].astype(str).str.match(r"^[ABCMX]\d")
    f = hek[has_ar & is_goes_class].copy()
    f["flux"] = f["fl_goescls"].map(goes_to_flux)
    f["peak"] = pd.to_datetime(f["event_peaktime"])
    return f.rename(columns={"ar_noaanum": "ar"})[["ar", "peak", "fl_goescls", "flux"]]


def to_old_scale(fl: pd.DataFrame) -> pd.Series:
    """Fluks tiap flare dalam skala label latih (GOES-15): flare era GOES-16+ dikali 0,7."""
    scale = np.where(fl["peak"] >= GOES16_START, GOES16_FACTOR, 1.0)
    return fl["flux"] * scale


def max_old_flux(fl_ar: pd.DataFrame, t0, t1) -> float:
    """Fluks flare terbesar dengan puncak dalam (t0, t1], dalam skala label latih. 0 bila tidak ada."""
    f = fl_ar[(fl_ar["peak"] > t0) & (fl_ar["peak"] <= t1)]
    if f.empty:
        return 0.0
    return float(to_old_scale(f).max())


def _visible(reports_ar: pd.DataFrame, t) -> bool:
    """Apakah AR dalam +-60 derajat pada waktu t (syarat ada gambar di dataset)?
    Diperkirakan dari laporan posisi terakhir sebelum t, ditambah rotasi Matahari sejak itu."""
    earlier = reports_ar[reports_ar["date"] <= t]
    if earlier.empty:
        return False
    last = earlier.iloc[earlier["date"].argmax()]
    lon_now = last["lon"] + DEG_PER_DAY * (t - last["date"]) / pd.Timedelta(days=1)
    return abs(lon_now) < 60


def history_from_events(fl_ar: pd.DataFrame, reports_ar: pd.DataFrame, t) -> dict:
    """Fitur riwayat seperti history(), tetapi dari daftar flare SWPC dan laporan posisi NOAA satu AR.
    Dipakai untuk data setelah dataset (2021+) dan untuk prediksi real-time."""
    known24 = _visible(reports_ar, t - HOURS_24)
    known48 = _visible(reports_ar, t - 2 * HOURS_24)
    h24 = flare_level(max_old_flux(fl_ar, t - HOURS_24, t)) if known24 else 0.0
    h48 = flare_level(max_old_flux(fl_ar, t - 2 * HOURS_24, t - HOURS_24)) if known48 else 0.0
    return {"hist24": h24, "hist48": h48, "hist_known": float(known24)}
