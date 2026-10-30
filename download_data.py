"""Langkah 1 - Unduh dataset magnetogram active region (Boucheron dkk. 2023).

- Gambar PNG 224x224 (13 GB) diunduh dari Zenodo dengan beberapa koneksi sekaligus
  (default 8). Arsip dipecah menjadi potongan 128 MB di data/zenodo_parts/. Bila koneksi
  putus, potongan itu dilanjutkan otomatis; bila program berhenti (Ctrl+C, listrik mati),
  jalankan ulang perintah yang sama dan unduhan DILANJUTKAN, bukan diulang dari awal.
- Setelah lengkap, MD5 arsip dicek, file .png diekstrak ke data/, lalu potongan dihapus.
  Butuh ruang kosong sementara +-26 GB (13 GB arsip + 13 GB hasil ekstrak).
- File label & daftar split ada di Dryad. Dryad memblokir unduhan otomatis,
  jadi skrip ini hanya mengecek keberadaannya dan memberi petunjuk unduh manual.

Contoh:
    python download_data.py                  # unduh semua gambar (13 GB), 8 koneksi
    python download_data.py --connections 4  # pakai 4 koneksi
    python download_data.py --max_gb 1       # unduh sebagian saja untuk uji coba
    python download_data.py --tar D:/Downloads/Lat60_Lon60_Nans0_png_224.tar.gz
"""
import argparse
import hashlib
import http.client
import shutil
import tarfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

ZENODO_URL = ("https://zenodo.org/api/records/7775776/files/"
              "Lat60_Lon60_Nans0_png_224.tar.gz/content")
ZENODO_MD5 = "c6498435e19705710ec4e1ec0c041a1d"
# 128 MB per potongan -> +-97 request untuk seluruh arsip, di bawah rate limit Zenodo (133 request/menit).
PIECE_SIZE = 128 * 1024 ** 2
DRYAD_URL = "https://doi.org/10.5061/dryad.jq2bvq898"
DRYAD_FILES = ["C1.0_24hr_224_png_Labels.txt",
               "List_of_AR_in_Validation_data_by_AR.csv",
               "List_of_AR_in_Test_Data_by_AR.csv"]

STOP = threading.Event()        # diset saat Ctrl+C / error, agar semua koneksi berhenti


class CountingReader:
    """Membungkus stream untuk menampilkan progres dan berhenti setelah `limit` byte."""

    def __init__(self, raw, limit=None):
        self.raw = raw
        self.limit = limit
        self.count = 0              # byte yang sudah dibaca
        self._next_report = 0       # cetak progres lagi setelah byte ke-sekian

    def read(self, n=-1):
        if self.limit and self.count >= self.limit:
            return b""
        data = self.raw.read(n)
        self.count += len(data)
        if self.count >= self._next_report:
            print(f"\r  {self.count / 1e9:6.2f} GB diproses", end="", flush=True)
            self._next_report += 100_000_000
        return data


class PartsReader:
    """Membaca file-file potongan berurutan seolah-olah satu file .tar.gz."""

    def __init__(self, paths):
        self.paths = iter(paths)
        self.file = None            # file potongan yang sedang dibaca

    def read(self, n=-1):
        while True:
            if self.file is None:
                path = next(self.paths, None)
                if path is None:
                    return b""
                self.file = open(path, "rb")
            data = self.file.read(n)
            if data:
                return data
            self.file.close()
            self.file = None


class Progress:
    """Menjumlahkan byte dari semua koneksi, lalu mencetak kecepatan dan perkiraan sisa waktu."""

    def __init__(self, done, total):
        self.done = done                     # byte yang sudah ada (termasuk dari unduhan sebelumnya)
        self.total = total
        self.start_done = done
        self.start = time.time()
        self.lock = threading.Lock()         # beberapa koneksi menambah angka bersamaan
        self.last_print = 0.0

    def add(self, n):
        with self.lock:
            self.done += n
            now = time.time()
            if now - self.last_print >= 2:
                speed = (self.done - self.start_done) / max(now - self.start, 1e-9)
                eta_min = (self.total - self.done) / max(speed, 1) / 60
                print(f"\r  {self.done / 1e9:6.2f} / {self.total / 1e9:.2f} GB | "
                      f"{speed / 1e6:5.2f} MB/s | sisa +-{eta_min:.0f} menit   ", end="", flush=True)
                self.last_print = now


def file_size(path: Path) -> int:
    """Ukuran file dalam byte, 0 bila belum ada."""
    return path.stat().st_size if path.exists() else 0


def wait_before_retry(attempt: int, error=None):
    """Jeda sebelum mencoba lagi: makin lama tiap percobaan (maks 60 detik).
    HTTP 429 (terlalu banyak request) menunggu sesuai permintaan server."""
    if isinstance(error, urllib.error.HTTPError) and error.code == 429:
        STOP.wait(int(error.headers.get("Retry-After", 60)))
    else:
        STOP.wait(min(2 ** attempt, 60))


def remote_size(url):
    """Ukuran arsip di server, atau None bila server tidak mendukung unduhan per potongan (Range)."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        if resp.status != 206:
            return None
        return int(resp.headers["Content-Range"].split("/")[-1])


def download_piece(url, path: Path, start: int, end: int, progress: Progress, retries: int = 30):
    """Unduh byte [start, end) ke `path`. Bila file sudah terisi sebagian, lanjutkan dari situ.

    Koneksi yang putus dicoba lagi (jeda makin lama, maks 60 detik) mulai dari byte terakhir.
    """
    size = end - start
    for attempt in range(retries):
        have = file_size(path)
        if have > size:                 # tidak semestinya terjadi: ulang potongan ini
            path.unlink()
            have = 0
        if have == size or STOP.is_set():
            return
        req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end - 1}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(path, "ab") as f:
                if resp.status != 206:
                    raise OSError(f"server membalas HTTP {resp.status}, bukan 206")
                while not STOP.is_set():
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    progress.add(len(chunk))
        except urllib.error.HTTPError as e:
            wait_before_retry(attempt, e)
        except (OSError, http.client.HTTPException):      # koneksi putus, timeout, dll.
            wait_before_retry(attempt)
    if file_size(path) != size and not STOP.is_set():
        raise RuntimeError(f"{path.name} gagal diunduh setelah {retries} percobaan")


def download_archive(url, parts_dir: Path, connections: int, limit=None):
    """Unduh arsip (atau `limit` byte pertamanya) sebagai potongan dengan beberapa koneksi.

    Kembali: (daftar path potongan, ukuran arsip penuh), atau (None, None) bila server
    tidak mendukung unduhan per potongan.
    """
    full_size = remote_size(url)
    if full_size is None:
        return None, None
    total = min(full_size, limit) if limit else full_size
    parts_dir.mkdir(parents=True, exist_ok=True)
    # tiap potongan: (file, byte awal, byte akhir)
    pieces = []
    for i, start in enumerate(range(0, total, PIECE_SIZE)):
        pieces.append((parts_dir / f"part_{i:04d}", start, min(start + PIECE_SIZE, total)))
    done = sum(min(file_size(p), end - start) for p, start, end in pieces)

    free = shutil.disk_usage(parts_dir).free
    need = (total - done) + total            # sisa unduhan + ruang untuk hasil ekstrak
    if free < need:
        print(f"  PERINGATAN: ruang kosong {free / 1e9:.0f} GB, perkiraan kebutuhan +-{need / 1e9:.0f} GB.")
    print(f"  {total / 1e9:.2f} GB = {len(pieces)} potongan, {connections} koneksi. "
          f"Sudah terunduh sebelumnya: {done / 1e9:.2f} GB")

    progress = Progress(done, total)
    pool = ThreadPoolExecutor(connections)
    pending = {pool.submit(download_piece, url, p, s, e, progress) for p, s, e in pieces}
    try:
        while pending:
            finished, pending = wait(pending, timeout=1)   # timeout agar Ctrl+C tetap tertangkap
            for future in finished:
                future.result()                            # munculkan error dari potongan yang gagal
    except BaseException:
        STOP.set()
        pool.shutdown(cancel_futures=True)
        print("\n  Unduhan terhenti. Potongan yang sudah terunduh tetap tersimpan; "
              "jalankan ulang perintah yang sama untuk melanjutkan.")
        raise
    pool.shutdown()
    print()
    return [p for p, _, _ in pieces], full_size


def md5_of(parts) -> str:
    """MD5 seluruh arsip (dibaca dari potongan-potongannya berurutan)."""
    md5, reader = hashlib.md5(), PartsReader(parts)
    while True:
        chunk = reader.read(16 * 1024 ** 2)
        if not chunk:
            break
        md5.update(chunk)
    return md5.hexdigest()


def extract(stream, data_dir: Path, limit=None) -> int:
    """Ekstrak file .png dari stream tar.gz satu per satu.

    Gambar yang sedang ditulis saat stream terputus dihapus. Tanpa ini, gambar setengah jadi
    akan tertinggal dan dilewati pada ekstraksi berikutnya karena dianggap sudah ada.
    """
    reader = CountingReader(stream, limit)
    n_new = 0
    current = None                  # gambar yang sedang ditulis (dihapus bila terputus)
    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as tar:
            for member in tar:
                if member.isfile() and member.name.endswith(".png"):
                    target = data_dir / member.name
                    if not target.exists():
                        current = target
                        tar.extract(member, path=data_dir, filter="data")
                        current = None
                        n_new += 1
    except BaseException as e:
        if current is not None:
            current.unlink(missing_ok=True)
        # arsip yang terpotong karena --max_gb memang berakhir dengan error baca: itu wajar
        reached_limit = isinstance(e, (tarfile.ReadError, EOFError)) and limit and reader.count >= limit
        if not reached_limit:
            raise
        print("\n  Batas --max_gb tercapai.")
    print(f"\n  {n_new} gambar baru diekstrak.")
    return n_new


def download_from_zenodo(data_dir: Path, connections: int, limit=None):
    print(f"Mengunduh gambar dari Zenodo:\n  {ZENODO_URL}")
    parts_dir = data_dir / "zenodo_parts"
    parts, full_size = download_archive(ZENODO_URL, parts_dir, connections, limit)
    if parts is None:
        print("  Server tidak mendukung unduhan per potongan, jadi resume & multi-koneksi tidak "
              "bisa dipakai. Mengunduh dengan 1 koneksi sambil mengekstrak.")
        with urllib.request.urlopen(ZENODO_URL) as resp:
            extract(resp, data_dir, limit)
        return

    complete = sum(file_size(p) for p in parts) == full_size
    if complete:
        print("  Mengecek MD5 arsip ...")
        if md5_of(parts) != ZENODO_MD5:
            raise SystemExit(f"  MD5 tidak cocok, arsip rusak. Hapus folder {parts_dir} lalu ulangi.")
        print("  MD5 cocok.")
    print("  Mengekstrak ...")
    extract(PartsReader(parts), data_dir, None if complete else limit)
    if complete:
        shutil.rmtree(parts_dir)
        print(f"  Potongan arsip di {parts_dir} dihapus.")
    else:
        print(f"  Potongan disimpan di {parts_dir}; unduhan penuh nanti akan melanjutkannya.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--tar", help="pakai file .tar.gz yang sudah diunduh manual")
    parser.add_argument("--max_gb", type=float, help="hanya unduh N GB pertama")
    parser.add_argument("--connections", type=int, default=8, help="jumlah koneksi sekaligus")
    parser.add_argument("--skip_images", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    limit = int(args.max_gb * 1e9) if args.max_gb else None

    if not args.skip_images:
        if args.tar:
            print(f"Mengekstrak {args.tar} ...")
            with open(args.tar, "rb") as f:
                extract(f, data_dir, limit)
        else:
            download_from_zenodo(data_dir, args.connections, limit)

    missing = [name for name in DRYAD_FILES if not (data_dir / name).exists()]
    if missing:
        print("\nFile berikut belum ada. Unduh manual lewat browser dari:")
        print(f"  {DRYAD_URL}")
        for name in missing:
            print(f"  - {name}  ->  simpan ke {data_dir / name}")
    else:
        print("\nSemua file label sudah lengkap. Lanjut: python prepare_data.py")


if __name__ == "__main__":
    main()
