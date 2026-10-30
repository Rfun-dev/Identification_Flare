# FlareFormer: Prediksi Solar Flare dari Citra Magnetogram

Proyek ini memprediksi **apakah sebuah active region (AR) di Matahari akan menghasilkan flare kelas M atau lebih besar dalam 24 jam ke depan**, hanya dari urutan citra magnetogram SDO/HMI.

**Istilah dasar:**
- **Solar flare** adalah ledakan energi di atmosfer Matahari. Kekuatannya diukur dari fluks sinar‑X oleh satelit GOES, lalu dikelompokkan menjadi kelas A, B, C, M, dan X. Setiap kelas 10× lebih kuat dari kelas sebelumnya. Flare kelas M dan X bisa mengganggu satelit, GPS, dan komunikasi radio.
- **Magnetogram** adalah "peta medan magnet" permukaan Matahari. Di gambar, putih berarti kutub positif dan hitam kutub negatif. Flare besar biasanya muncul di AR yang medannya kuat dan rumit, terutama bila kutub + dan − saling berdesakan sehingga membentuk *garis netral* yang tajam.
- **Active region (AR)** adalah daerah aktif di Matahari (tempat bintik Matahari). NOAA memberi setiap AR nomor, misalnya AR 11158 (di dataset ditulis 4 digit: `1158`).

---

## Daftar isi
1. [Cara menjalankan](#cara-menjalankan)
2. [Struktur folder](#struktur-folder)
3. [Peta kode: urutan membaca](#peta-kode-urutan-membaca)
4. [Log progress](#log-progress)
5. [Hasil eksperimen](#hasil-eksperimen)
6. [Ide pengembangan](#ide-pengembangan)
7. [Template progress berikutnya](#template-progress-berikutnya)
8. [Referensi](#referensi)

---

## Cara menjalankan

### 0. Instalasi (virtual environment `.venv` + CUDA)
Proyek ini memakai virtual environment sendiri di folder `.venv`, sehingga tidak bergantung pada Python global laptop. Sekali jalan, perintah ini membuat `.venv`, memasang semua library (termasuk PyTorch versi CUDA 12.6), lalu mengecek GPU:
```powershell
powershell -ExecutionPolicy Bypass -File setup_env.ps1
```
Hasil yang benar diakhiri dengan `CUDA tersedia: True` dan `GPU: NVIDIA GeForce GTX 1650 Ti`.

Menjalankan program dengan Python milik `.venv`:
```powershell
.\.venv\Scripts\Activate.ps1      # aktifkan sekali per terminal; prompt diawali (.venv)
python train.py                   # sekarang otomatis memakai GPU
```
Jika `Activate.ps1` diblokir Windows, jalankan sekali `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, atau panggil langsung `.\.venv\Scripts\python.exe train.py`. Di VS Code, `.venv` sudah diset sebagai interpreter default lewat [.vscode/settings.json](.vscode/settings.json).

### 1. Uji cepat dengan data sintetis (sekitar 10 menit di CPU, tanpa unduh dataset)
```bash
python make_dummy_data.py --data_dir data_dummy
python prepare_data.py --data_dir data_dummy --image_dir data_dummy/Lat60_Lon60_Nans0_png_224 --index_csv data_dummy/index.csv
python train.py   --image_dir data_dummy/Lat60_Lon60_Nans0_png_224 --index_csv data_dummy/index.csv --run_dir runs/dummy --epochs 6 --train_stride_h 2 --eval_stride_h 1 --num_workers 2
python evaluate.py --run_dir runs/dummy
python predict.py  --run_dir runs/dummy --ar 9011 --time "2015-01-31 12:00" --saliency
```

### 2. Dengan data asli
```bash
python download_data.py            # unduh 13 GB gambar dari Zenodo, 8 koneksi; jika terputus, jalankan ulang untuk melanjutkan
#  -> unduh manual 3 file label dari Dryad sesuai petunjuk yang dicetak skrip, simpan di data/
python prepare_data.py             # buat data/index.csv
python train.py                    # latih model -> runs/flareformer/best.pt
python evaluate.py                 # skor pada data test
python predict.py --ar 1158 --time "2011-02-14 12:00" --saliency
```

### 3. Prediksi real-time dari data SDO terbaru
```bash
python fetch_realtime.py --ar 4556 --predict --saliency       # AR 14556, data terbaru (±1 jam lalu)
python fetch_realtime.py --ar 1158 --time "2011-02-14 12:00" --verify   # cek pipeline vs dataset
python evaluate_recent.py --n_pos 150 --n_neg 150             # uji pada data 2021-2025 (±40 menit)
python xclass_features.py                                     # latih peluang X (beberapa menit)
python evaluate_recent_xclass.py --n_x 150 --n_m 150          # konfirmasi peluang X pada 2021-2025
```
Dengan `--predict`, `fetch_realtime.py` juga mencetak **peluang flare X bila flare ≥M terjadi** (Progress 18).
Nomor AR yang sedang aktif bisa dilihat di https://www.swpc.noaa.gov (Solar Regions). Tulis 4 digit terakhir (AR 14556 → `4556`). Tidak perlu akun atau email JSOC. Detailnya ada di Progress 15.

Semua pengaturan ada di [flare/config.py](flare/config.py) dan bisa diubah dari command line, misalnya `python train.py --n_frames 6 --batch_size 8`.

---

## Struktur folder
```
project 2/
├── README.md              <- dokumen ini (log progress)
├── requirements.txt       <- daftar library + versi (PyTorch CUDA 12.6)
├── setup_env.ps1          <- buat .venv + pasang library + cek GPU, sekali jalan
├── gpu_running.py         <- cek apakah PyTorch bisa memakai GPU
├── .venv/                 <- virtual environment Python khusus proyek ini
├── .vscode/settings.json  <- VS Code otomatis memakai Python dari .venv
├── download_data.py       <- Langkah 1: unduh gambar (Zenodo) + cek file label (Dryad)
├── prepare_data.py        <- Langkah 2: buat tabel indeks + label + split
├── train.py               <- Langkah 3: latih model
├── evaluate.py            <- Langkah 4: uji pada data test, buat grafik
├── predict.py             <- Langkah 5: prediksi satu AR + peta saliency
├── fetch_realtime.py      <- Langkah 6: ambil magnetogram terbaru dari JSOC -> PNG -> prediksi
├── evaluate_recent.py     <- Langkah 7: uji model pada data 2021+ (setelah dataset)
├── train_xclass.py        <- Langkah 8 (eksperimen): pemisah M vs X; hasil negatif, lihat Progress 17
├── xclass_features.py     <- Langkah 8b: peluang X dari fitur fisis + riwayat (komposit), Progress 18
├── evaluate_recent_xclass.py <- Langkah 8c: konfirmasi peluang X pada data 2021-2025
├── make_dummy_data.py     <- data sintetis untuk uji pipeline
├── data_realtime/         <- PNG hasil fetch_realtime.py, satu folder per AR
├── data_recent/           <- PNG + daftar sampel + daftar flare HEK untuk evaluate_recent.py
└── flare/                 <- "mesin" proyek (dipakai oleh skrip di atas)
    ├── config.py          <- semua hyperparameter + load_config()
    ├── data.py            <- baca gambar, susun urutan frame, augmentasi, sampler, kelas GOES
    ├── model.py           <- arsitektur FlareFormer
    ├── engine.py          <- loop training, prediksi, optimizer, seed
    ├── metrics.py         <- TSS, HSS, threshold, selang kepercayaan per AR (bootstrap)
    ├── jsoc.py            <- cari rekaman HMI di JSOC + unduh hanya baris yang dibutuhkan
    ├── noaa.py            <- posisi AR dari laporan NOAA (SWPC / HEK)
    ├── hmi.py             <- potong magnetogram -> PNG persis seperti dataset
    ├── recent.py          <- alat bantu evaluasi data 2021+ (daftar flare, unduh, lanjutkan)
    ├── features.py        <- fitur fisis (R, garis netral, fluks) + riwayat flare
    └── xclass.py          <- model peluang X (skor komposit) + alat evaluasi M vs X
```
**Kenapa dipisah begini?** Setiap skrip di folder utama adalah satu langkah yang dijalankan berurutan, sehingga alurnya mudah diikuti. Logika yang dipakai berulang (model, data, metrik) disimpan di folder `flare/` supaya tidak ada kode ganda. Skrip tidak saling mengimpor, kecuali `fetch_realtime.py` yang memakai fungsi `predict_images()` dari `predict.py` (Langkah 5).

---

## Peta kode: urutan membaca

Kode ini paling mudah dipahami bila dibaca **mengikuti alur data**, bukan per file:

**A. Dari gambar ke model (inti proyek)**
1. [flare/config.py](flare/config.py): semua angka pengaturan ada di satu tempat.
2. [prepare_data.py](prepare_data.py): nama file gambar + label diubah menjadi tabel `data/index.csv`.
3. [flare/data.py](flare/data.py): `build_samples()` menyusun 4 frame berjarak 72 menit menjadi satu sampel; `MagnetogramDataset` membaca PNG menjadi tensor.
4. [flare/model.py](flare/model.py): `FlareFormer.forward()`, 8 baris yang menjadi inti model.
5. [flare/engine.py](flare/engine.py): `train_one_epoch()` dan `predict_proba()`, loop standar PyTorch.
6. [train.py](train.py): menggabungkan semuanya; bacalah `main()` dari atas ke bawah (data -> model -> loop).
7. [flare/metrics.py](flare/metrics.py) dan [evaluate.py](evaluate.py): cara model dinilai (TSS, threshold dari validasi).
8. [predict.py](predict.py): `report()` memprediksi satu AR.

**B. Data real-time (Progress 15)**
1. [fetch_realtime.py](fetch_realtime.py): `main()` -> `make_images()` -> `verify()` / `predict_images()`.
2. [flare/jsoc.py](flare/jsoc.py): `query_frames()` mencari 4 rekaman; `fetch_rows()` mengunduh sebagian file.
3. [flare/noaa.py](flare/noaa.py): `ar_position()` menentukan pusat kotak.
4. [flare/hmi.py](flare/hmi.py): `make_png()` memotong, memutar, dan mengubah skala gambar.

**C. Evaluasi data 2021+ dan peluang kelas X (Progress 16-18)**
1. [flare/recent.py](flare/recent.py) -> [evaluate_recent.py](evaluate_recent.py).
2. [flare/features.py](flare/features.py) -> [flare/xclass.py](flare/xclass.py) -> [xclass_features.py](xclass_features.py) -> [evaluate_recent_xclass.py](evaluate_recent_xclass.py).
3. [train_xclass.py](train_xclass.py) hanya arsip eksperimen yang gagal (Progress 17); boleh dilewati.

**Kebiasaan penulisan kode:**
- Setiap file diawali docstring: apa gunanya, cara menjalankan, dan dasar risetnya.
- Fungsi `main()` di tiap skrip dibagi menjadi langkah bernomor (`# 1. ...`, `# 2. ...`).
- Nama variabel memakai bahasa Inggris (konvensi Python), sedangkan komentar dan pesan memakai bahasa Indonesia.

---

## Log progress

Setiap progress dicatat dengan format: **Tujuan** (apa yang ingin dicapai), **Kenapa** (alasan memilih cara ini), dan **Penjelasan sederhana**.

### Progress 1 — Riset literatur & memilih pendekatan

**Tujuan:** Menentukan arsitektur model berdasarkan riset terbaru (2025–2026), bukan sekadar menebak.

**Temuan riset:**

| Riset | Yang dilakukan | Pelajaran yang diambil |
|---|---|---|
| Li dkk. 2025 (ApJS 278:63) | Video transformer (MViT) pada 16 magnetogram berjarak 72 menit (±24 jam) | **Model spatio-temporal (melihat urutan waktu) mengalahkan model 1 gambar**, dan Transformer mengalahkan CNN. TSS ≥M: MViT 0,694–0,775; CNN 0,491–0,620. |
| Cao & Zhao 2026 (arXiv:2609.10772) | Hybrid ResNet50 + Transformer | **Gabungan CNN + Transformer** lebih baik daripada CNN saja maupun SVM. |
| Riggi dkk. 2025 (Astronomy & Computing) | Fine-tuning foundation model (SigLIP2, VideoMAE) pada dataset Boucheron 2023 | TSS ≈ 0,65 untuk ≥M. **Split harus per AR**, pilih model berdasarkan **TSS**, tangani data tak seimbang dengan loss berbobot/sampler, augmentasi flip + rotasi 90°. Variasi skor antar-split besar (±0,165). |
| Surya (NASA/IBM dkk. 2025, arXiv:2508.14112) | Foundation model heliofisika 366 juta parameter | Arah riset terdepan, tetapi **terlalu besar untuk GPU 4 GB**. Menjadi ide pengembangan. |

**Keputusan:** Membuat **FlareFormer**, yaitu CNN ringan (ResNet18) untuk membaca pola lokal di tiap frame, lalu Transformer untuk menghubungkan pola antar-lokasi dan antar-waktu. Desain ini menggabungkan temuan utama Li dkk. (spatio-temporal + transformer) dan Cao & Zhao (hybrid CNN-Transformer), dengan ukuran yang muat di GTX 1650 Ti.

**Kenapa:** Model 366 juta parameter atau ResNet50 + video transformer tidak realistis untuk laptop 4 GB. Kita ambil *ide*-nya, tetapi dengan komponen yang lebih kecil (11,6 juta parameter).

**Penjelasan sederhana:** CNN berperan seperti "mata" yang mengenali bentuk (kutub, garis netral). Transformer berperan seperti "otak" yang bertanya: *"Bagian mana yang penting, dan bagaimana ia berubah dari jam ke jam?"*

---

### Progress 2 — Memilih & mengunduh dataset

**Tujuan:** Mendapatkan data magnetogram berlabel yang siap pakai dan diakui dalam riset.

**Pilihan:** *Active Region Magnetogram Image Dataset* (Boucheron dkk., Scientific Data 2023), versi resolusi rendah:
- 950.047 magnetogram HMI (PNG 224×224, 8-bit) dari 1.570 AR, Mei 2010 – Desember 2018, satu gambar setiap 12 menit.
- Gambar (12 GB) ada di **Zenodo** dan bisa diunduh otomatis dengan `download_data.py`.
- Label dan daftar split ada di **Dryad**. Dryad memblokir unduhan otomatis, jadi 3 file berikut harus diunduh manual dari https://doi.org/10.5061/dryad.jq2bvq898 dan disimpan ke folder `data/`:
  `C1.0_24hr_224_png_Labels.txt`, `List_of_AR_in_Validation_data_by_AR.csv`, `List_of_AR_in_Test_Data_by_AR.csv`.

**Kenapa dataset ini:**
1. **Dipakai riset terbaru** (Riggi dkk. 2025), jadi hasil kita bisa dibandingkan langsung.
2. **Label sudah jadi**: kelas GOES flare terbesar dalam 24 jam setelah setiap gambar.
3. **Split resmi per AR** (train 1.256 AR, validasi 157 AR, test 157 AR), sehingga tidak ada kebocoran data.
4. Versi 224×224 PNG hanya 12 GB, dibanding 375 GB untuk versi FITS penuh. Ini realistis untuk laptop.
5. Nama folder `Lat60_Lon60_Nans0` menandakan AR dibatasi ±60° dari pusat piringan Matahari dan tanpa piksel kosong. Di dekat tepi piringan, magnetogram line-of-sight terdistorsi.

**Efisiensi:** `download_data.py` mengunduh dengan 8 koneksi sekaligus dan bisa melanjutkan unduhan yang terputus (detailnya di Progress 13). Opsi `--max_gb 1` mengunduh sebagian saja untuk uji coba. Arsip yang sudah diunduh lewat browser bisa diekstrak dengan `python download_data.py --tar <file.tar.gz>`.

**Penjelasan sederhana:** Bayangkan album foto setiap AR, satu foto tiap 12 menit, dan setiap foto diberi catatan "flare apa yang terjadi 24 jam setelah foto ini diambil".

---

### Progress 3 — Membuat indeks data, label, dan split per AR

**Tujuan:** Menggabungkan file label dan daftar split menjadi satu tabel `data/index.csv` (kolom `ar, time, path, goes, flux, split`). File: [prepare_data.py](prepare_data.py).

**Yang dilakukan:**
- Nomor AR dan waktu dibaca dari nama file, misalnya `1158_hmi.M_720s.20110214_120000_TAI...png` → AR 1158, 14 Feb 2011 12:00.
- Kelas GOES diubah menjadi angka fluks: `M2.5` → 2,5×10⁻⁵ W/m². Dengan begitu, target bisa diganti hanya lewat `--flare_class C|M|X`.
- **Target default: flare ≥ M1.0 dalam 24 jam.** Ini standar di hampir semua paper flare forecasting, termasuk ketiga paper di Progress 1.
- Setiap AR dimasukkan seluruhnya ke **satu** split: train, val, atau test.
- Hanya gambar yang benar-benar ada di disk yang dimasukkan, sehingga unduhan sebagian tetap bisa dipakai.

**Kenapa split per AR, bukan acak per gambar?** Gambar dari AR yang sama dengan jarak 12 menit hampir identik. Jika dibagi acak, model akan "melihat soal ujian" saat belajar, sehingga skornya tinggi palsu. Ini disebut *data leakage* dan diperingatkan oleh Ahmadzadeh dkk. (2021) serta Riggi dkk. (2025).

**Penjelasan sederhana:** Seperti ujian. Soal ujian (AR test) harus soal baru, bukan soal latihan yang hanya diganti sedikit angkanya.

**Catatan:** Label dataset memakai skala GOES 13–15. Satelit GOES 16+ mengukur sekitar 1/0,7 kali lebih besar. Ini penting jika nanti model dibandingkan dengan data flare setelah 2020.

---

### Progress 4 — Dari gambar menjadi sampel: normalisasi & urutan frame

**Tujuan:** Mengubah piksel menjadi angka bermakna fisis, lalu menyusun beberapa frame berurutan menjadi satu sampel. File: [flare/data.py](flare/data.py).

**4a. Normalisasi.** Dataset menyimpan medan magnet B (Gauss) sebagai `piksel = (B + 2550) / 20`. Kode kita menghitung `piksel / 127.5 − 1`, yang setara dengan **B / 2550**.
- **Kenapa:** Nilai 0 tetap berarti "tanpa medan", positif berarti kutub utara, dan negatif kutub selatan. Tanda (+/−) ini penting, karena flare terjadi di tempat kutub berlawanan bertemu. Normalisasi ImageNet biasa (mean/std RGB) akan menghilangkan makna ini.

**4b. Urutan waktu.** Satu sampel = **4 frame dengan jarak 72 menit** (mencakup ±3,6 jam terakhir). Label diambil dari frame terakhir.
- **Kenapa:** Li dkk. (2025) menunjukkan bahwa melihat *evolusi* medan (munculnya fluks baru, kutub yang makin berdesakan) lebih baik daripada satu foto. Jarak 72 menit mengikuti mereka. Jumlah frame dikurangi dari 16 menjadi 4 agar muat di GPU 4 GB.
- Bila ada frame yang hilang (data bolong lebih dari ±24 menit), sampel dilewati agar model tidak menerima urutan yang salah.

**4c. Jarak antar sampel (stride).** Training mengambil 1 sampel per jam per AR, dan validasi/test 1 sampel per 3 jam.
- **Kenapa:** Frame berjarak 12 menit hampir identik. Memakai semuanya membuat training 5× lebih lama tanpa tambahan informasi berarti. Ini penghematan waktu terbesar di proyek ini.

**Penjelasan sederhana:** Daripada menilai cuaca dari satu foto awan, kita melihat "time-lapse" 4 foto untuk tahu apakah awannya sedang menebal.

---

### Progress 5 — Augmentasi data

**Tujuan:** Memperbanyak variasi data latih tanpa data baru, supaya model tidak menghafal. File: [flare/data.py](flare/data.py), fungsi `augment`.

**Yang dilakukan** (hanya saat training, transformasi yang sama untuk keempat frame):
- Cermin kiri-kanan dan atas-bawah, serta rotasi 90°/180°/270°. Ini dipakai Riggi dkk. (2025).
- **Tukar polaritas** (+ menjadi −). Kompleksitas medan, seperti panjang dan ketajaman garis netral, tidak berubah walaupun kutubnya ditukar.

**Kenapa transformasinya sama untuk semua frame:** Kalau frame 1 dicerminkan tetapi frame 2 tidak, model akan melihat "perubahan" palsu antar waktu.

**Kenapa tidak memakai augmentasi warna/kontras biasa:** Nilai piksel adalah ukuran fisis (Gauss). Mengubah kontras sama saja dengan memalsukan kekuatan medan magnet.

**Penjelasan sederhana:** AR yang difoto terbalik tetaplah AR yang sama berbahayanya.

---

### Progress 6 — Arsitektur FlareFormer

**Tujuan:** Membangun model hybrid CNN + Transformer yang ringan. File: [flare/model.py](flare/model.py).

```
4 magnetogram [4 × 1 × 224 × 224]
   │  ResNet18 (pretrained ImageNet), per frame
   ▼
peta fitur 7×7 per frame  ──►  49 "token" per frame × 4 frame = 196 token
   │  + penanda posisi (di mana) + penanda waktu (kapan) + token [CLS]
   ▼
Transformer encoder (2 lapis, 4 head): setiap token "memperhatikan" semua token lain
   ▼
token [CLS] ──► Linear ──► skor flare (0–1)
```

**Kenapa setiap komponen dipilih:**
- **ResNet18 pretrained:** Model ini sudah pandai mengenali tepi dan bentuk dari jutaan foto (transfer learning), sehingga butuh lebih sedikit data dan waktu. ResNet18 (11 juta parameter) dipilih alih-alih ResNet50 (Cao & Zhao 2026) karena jauh lebih ringan untuk GPU 4 GB.
- **Konversi ke 1 channel:** ResNet aslinya untuk gambar RGB (3 channel), sedangkan magnetogram hanya 1 channel. Bobot filter ketiga channel dijumlahkan menjadi 1, sehingga pengetahuan pretrained tetap terpakai.
- **Token spasial (7×7), bukan satu vektor per gambar:** Transformer bisa menghubungkan *bagian-bagian* AR, misalnya kutub + di kiri dengan kutub − di kanan, dan bagaimana keduanya bergerak antar frame. Inilah inti "spatio-temporal" seperti pada video transformer Li dkk.
- **Token [CLS]:** Token khusus yang mengumpulkan ringkasan seluruh urutan untuk keputusan akhir, teknik standar dari ViT/BERT.
- **Transformer kecil** (d_model 128, 2 lapis): Dengan hanya 196 token, transformer kecil sudah cukup dan cepat.

**Total: 11,6 juta parameter.** Untuk perbandingan, Surya punya 366 juta.

**Penjelasan sederhana:** ResNet memotong tiap foto menjadi 49 kotak kecil dan mendeskripsikan isi tiap kotak. Transformer lalu membaca 196 deskripsi (49 kotak × 4 foto) sekaligus dan memutuskan kotak mana yang mencurigakan.

---

### Progress 7 — Menangani data yang sangat tidak seimbang

**Tujuan:** Mencegah model malas yang selalu menjawab "tidak ada flare". File: [flare/data.py](flare/data.py) (`BalancedSampler`) dan [train.py](train.py) (`pos_weight`).

**Masalahnya:** Di data test resmi, hanya **sekitar 3,5%** gambar yang diikuti flare ≥M (Riggi dkk. 2025). Model yang selalu menjawab "aman" sudah 96,5% akurat, padahal tidak berguna.

**Solusi (dua lapis):**
1. **BalancedSampler:** Setiap epoch memakai **semua** sampel flare ditambah sampel non-flare acak sebanyak 3× (`neg_ratio`). Sampel non-flare diundi ulang setiap epoch, jadi epoch tetap pendek dan cepat, tetapi lama-kelamaan model tetap melihat banyak variasi AR tenang.
2. **Loss berbobot (`pos_weight`):** Kesalahan pada sampel flare diberi bobot 3× lebih besar, sehingga kedua kelas sama pentingnya bagi model.

**Kenapa:** Metrik utama kita, TSS, sama dengan *recall flare + recall non-flare − 1*, sehingga kedua kelas dianggap sama penting. Dengan membuat loss seimbang, tujuan training searah dengan metrik evaluasi. Pendekatan loss berbobot juga dipakai Li dkk. (2025) dan Riggi dkk. (2025).

**Penjelasan sederhana:** Seperti dokter yang belajar mendeteksi penyakit langka. Kasus langka harus sering ditunjukkan dan kesalahannya dihukum lebih berat. Kalau tidak, dokter akan selalu menjawab "sehat".

---

### Progress 8 — Training loop

**Tujuan:** Melatih model secara stabil dan efisien. File: [train.py](train.py), [flare/engine.py](flare/engine.py).

| Teknik | Kenapa |
|---|---|
| **AdamW** + weight decay 0,05 | Optimizer standar untuk Transformer. Weight decay mencegah bobot membesar (overfitting). |
| **Learning rate berbeda**: CNN 1e-4, Transformer 3e-4 | CNN sudah pintar (pretrained), jadi cukup disetel pelan. Transformer masih baru, jadi belajar lebih cepat. |
| **Cosine LR schedule** | LR turun perlahan menjelang akhir training, sehingga model "mendarat" dengan halus di solusi yang baik. |
| **Mixed precision (AMP)**, default **mati** | Hitungan float16 hanya mempercepat GPU yang punya *tensor core* (seri RTX). Di GTX 1650 Ti justru 4× lebih lambat (lihat Progress 12). Aktifkan dengan `--amp true` jika memakai GPU RTX. |
| **Gradient clipping** (norma 1,0) | Mencegah lonjakan gradien yang membuat training Transformer tidak stabil. |
| **Model terbaik dipilih dari TSS validasi** | Sama seperti Li dkk. dan Riggi dkk. Yang disimpan adalah model dengan TSS tertinggi, bukan model epoch terakhir. |
| **Early stopping** (5 epoch) | Berhenti otomatis jika TSS validasi tidak naik lagi, untuk menghemat waktu dan mencegah overfitting. |

Output training: `runs/flareformer/best.pt` (bobot + config + threshold), `history.csv` (skor per epoch), dan `config.json`.

---

### Progress 9 — Metrik evaluasi & pemilihan threshold

**Tujuan:** Mengukur kualitas prediksi dengan cara yang adil dan standar. File: [flare/metrics.py](flare/metrics.py), [evaluate.py](evaluate.py).

**Metrik yang dilaporkan:**
- **TSS (True Skill Statistic)** = recall − false positive rate. Rentangnya −1 sampai 1: 0 berarti setara tebakan acak, 1 sempurna. TSS **tidak terpengaruh proporsi kelas**, sehingga direkomendasikan untuk flare forecasting (Bloomfield dkk. 2012). Ini metrik utama.
- **HSS (Heidke Skill Score):** Seberapa jauh lebih baik dari tebakan acak, dengan memperhitungkan proporsi kelas.
- **Recall:** Dari semua flare, berapa persen yang tertangkap.
- **Precision:** Dari semua alarm, berapa persen yang benar-benar flare.
- **AUC:** Kualitas ranking skor, tidak bergantung pada threshold.

**Kenapa bukan akurasi:** Lihat Progress 7. Model "selalu aman" mendapat akurasi 96,5% tetapi TSS-nya 0.

**Threshold:** Model mengeluarkan skor 0–1. Batas skor untuk menyatakan "FLARE" **dipilih pada data validasi** (titik pada kurva ROC dengan TSS maksimum), lalu **dipakai apa adanya pada data test**. Threshold tidak boleh dipilih dari data test, karena itu sama dengan mengintip kunci jawaban.

**Penjelasan sederhana:** TSS menjawab pertanyaan "berapa banyak flare yang tertangkap, dikurangi berapa sering alarm palsu berbunyi".

Output `evaluate.py`: `test_metrics.json`, `test_predictions.csv` (skor per sampel untuk analisis lanjut), `roc_curve.png`, dan `history.png`.

---

### Progress 10 — Prediksi & interpretabilitas (peta saliency)

**Tujuan:** Memakai model untuk satu AR dan menunjukkan **bagian magnetogram mana** yang membuat model memutuskan demikian. File: [predict.py](predict.py).

```bash
python predict.py --ar 1158 --time "2011-02-14 12:00" --saliency
python predict.py --images f1.png f2.png f3.png f4.png      # urut lama -> baru
```

**Peta saliency (SmoothGrad):** Peta ini mengukur seberapa besar skor berubah jika suatu piksel diubah sedikit. Hasilnya dirata-rata dari 16 salinan input yang diberi noise kecil, agar petanya bersih (Smilkov dkk. 2017).

**Kenapa:** Model "kotak hitam" sulit dipercaya untuk keperluan operasional. Li dkk. (2025) juga menganalisis bagian yang diperhatikan model (dengan Grad-CAM). Jika model sehat, area terang semestinya berada di sekitar **garis netral** antara kutub + dan − yang kuat.

**Catatan penting:** Skor model **bukan peluang yang terkalibrasi**. Karena training diseimbangkan (Progress 7), skornya cenderung tinggi. Gunakan **keputusan berdasarkan threshold**, jangan membaca skor 0,8 sebagai "80% pasti flare".

---

### Progress 11 — Uji seluruh pipeline dengan data sintetis

**Tujuan:** Memastikan semua kode berjalan dari awal sampai akhir sebelum menghabiskan waktu mengunduh 12 GB dan melatih berjam-jam. File: [make_dummy_data.py](make_dummy_data.py).

**Cara kerja data sintetis:** 36 AR palsu, 30 jam per AR, satu gambar per 12 menit (total 5.400 gambar), dengan format nama file dan label **persis** seperti dataset asli. AR yang akan menghasilkan flare M/X diberi pasangan kutub +/− yang makin kuat dan makin rapat menjelang flare.

**Kenapa:** Bug pada pipeline (format nama file, label, split, bentuk tensor) lebih murah ditemukan dalam 10 menit dengan data kecil daripada setelah 5 jam training.

**Hasil (CPU, 6 epoch, ±8 menit):**

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| TSS validasi | 0,563 | 0,827 | 0,887 | 0,903 | 0,935 | 0,968 |

Test (130 sampel, 41 flare): **TSS 0,576**, HSS 0,630, AUC 0,958, recall 0,610, precision 0,893. Prediksi contoh AR 9011 (label asli M5.2) menghasilkan skor 0,98 (FLARE), dan AR 9001 (tanpa flare) menghasilkan skor 0,004. Peta saliency menyorot garis netral di antara kutub kuat, sesuai harapan.

**Pelajaran yang didapat:**
1. **Percobaan pertama gagal:** TSS test −0,32. Penyebabnya ada di generator data sintetis, bukan di model: sinyal "kompleksitas" baru muncul ±10 jam sebelum flare, padahal label positif mencakup 24 jam. Akibatnya separuh sampel positif memang tidak bisa dibedakan. Setelah sinyal dibuat konsisten selama 24 jam, model belajar dengan baik. Pelajarannya: **label harus benar-benar bisa diprediksi dari input**. Pada data asli pun, sebagian flare memang tidak terlihat dari magnetogram saja (Riggi dkk. 2025).
2. **TSS validasi (0,97) jauh di atas TSS test (0,58)**, padahal AUC test 0,96. Validasi hanya berisi 5 AR, sehingga threshold yang dipilih kurang mewakili. Fenomena yang sama dilaporkan Riggi dkk. (2025): skor bisa bervariasi ±0,165 antar-split. Pada data asli dengan 157 AR validasi, efek ini jauh lebih kecil.
3. Hasil sintetis ini **tidak bermakna ilmiah**. Fungsinya hanya membuktikan bahwa pipeline benar.

**Kecepatan terukur:** Di CPU laptop ini, 1 batch (16 sampel × 4 frame) butuh ±3,3 detik. Untuk data asli (perkiraan ±20 ribu sampel per epoch), itu berarti ±1 jam per epoch di CPU. Dengan GPU, angka ini turun drastis (lihat Progress 12).

---

### Progress 12 — Virtual environment sendiri + GPU (CUDA)

**Tujuan:** Menjalankan proyek di environment Python milik sendiri (`.venv` di folder proyek) dengan PyTorch yang memakai GPU GTX 1650 Ti. File: [requirements.txt](requirements.txt), [setup_env.ps1](setup_env.ps1), [.vscode/settings.json](.vscode/settings.json), [gpu_running.py](gpu_running.py), [flare/config.py](flare/config.py), [train.py](train.py).

**Yang dilakukan:**
1. **Memasang PyTorch 2.14.1 build CUDA 12.6** ke `.venv`. Sebelumnya torch di `.venv` adalah versi CPU, karena `pip install torch` dari PyPI di Windows selalu memberi versi CPU.
2. **Mengunci versi di `requirements.txt`** (torch `2.14.1+cu126`, pandas `3.0.6`, dll.) dan menambahkan baris `--extra-index-url` ke server PyTorch, sehingga pemasangan ulang selalu menghasilkan versi CUDA dengan versi library yang sama persis.
3. **`setup_env.ps1`**: satu perintah untuk membuat `.venv`, memasang semua library, lalu menjalankan `gpu_running.py` untuk mengecek GPU.
4. **`.vscode/settings.json`**: VS Code otomatis memakai Python dari `.venv`.
5. **torchaudio dicopot.** Proyek ini tidak memakainya, dan versi CUDA terakhirnya (2.11) tidak cocok dengan torch 2.14.
6. **AMP dimatikan secara default**, dan `cudnn.benchmark` dinyalakan (lihat Hasil).

**Kenapa memakai virtual environment:** Proyek tidak bergantung pada Python global laptop. Jika library global di-update untuk proyek lain, proyek ini tidak ikut rusak. Proyek juga mudah dipindah ke laptop lain: cukup jalankan `setup_env.ps1`.

**Kenapa CUDA 12.6:** GTX 1650 Ti adalah GPU Turing (compute capability 7.5). Build CUDA 12.6 adalah pilihan paling aman untuk GPU generasi ini, dan didukung oleh driver laptop (driver mendukung sampai CUDA 13.3).

**Penjelasan sederhana:** `.venv` seperti "kotak perkakas" khusus proyek ini, terpisah dari perkakas lain di laptop. Isinya sekarang termasuk "mesin" PyTorch yang bisa memakai GPU. GPU mengerjakan ribuan perkalian secara paralel, sedangkan CPU mengerjakannya beberapa sekaligus.

**Hasil (data sintetis, batch 16 sampel × 4 frame):**

| Mode | Waktu per batch training | Waktu per epoch | Total 6 epoch |
|---|---|---|---|
| CPU | ±3,3 detik | ±75 detik | ±8 menit |
| GPU + AMP (float16) | ±1,4 detik | ±36 detik | ±5 menit |
| **GPU float32 (default sekarang)** | **±0,35–0,45 detik** | **±8–14 detik** | **±2 menit** |

Puncak memori GPU float32 sekitar 2,9 GB, masih muat di 4 GB. Epoch pertama lebih lama (±45 detik) karena worker data dan pencarian algoritma konvolusi (`cudnn.benchmark`) disiapkan sekali di awal.

**Temuan penting, AMP justru memperlambat:** Seri GTX 16xx tidak punya *tensor core*, unit khusus yang membuat float16 cepat di kartu RTX. Hasil ukur langsung menunjukkan float16 di GTX 1650 Ti **4× lebih lambat** daripada float32. Pelajarannya: **teknik percepatan harus diukur di hardware sendiri, jangan diasumsikan.**

**Catatan suhu:** Saat diuji, GPU mencapai 83 °C dan clock-nya turun dari maks 2100 MHz ke 1200 MHz (*thermal throttling*). Untuk training panjang dengan data asli, colokkan charger dan pastikan ventilasi laptop lancar. Perkiraan waktu dengan data asli: ±8–10 menit per epoch di GPU, dibanding ±1 jam di CPU.

**Catatan hasil uji ulang:** TSS validasi tetap 0,978, tetapi TSS test kali ini 0,343 (AUC 0,934), sedangkan uji CPU sebelumnya mendapat 0,576. AUC yang tetap tinggi menunjukkan model sama-sama bagus dalam mengurutkan sampel. Perbedaan datang dari threshold yang dipilih dari validasi yang hanya berisi 5 AR, persis seperti pelajaran nomor 2 di Progress 11. Ini bukan efek dari GPU.

---

### Progress 13 — Unduhan Zenodo bisa dilanjutkan & memakai 8 koneksi

**Tujuan:** Unduhan arsip gambar (13 GB) tidak perlu diulang dari awal bila terputus, dan lebih cepat karena memakai beberapa koneksi sekaligus. File: [download_data.py](download_data.py).

**Yang dilakukan:**
1. **Arsip dipecah menjadi potongan 128 MB** (±97 potongan) di `data/zenodo_parts/`. Setiap potongan diunduh dengan permintaan *HTTP Range* ("kirim byte ke-X sampai ke-Y saja").
2. **8 koneksi sekaligus** (`--connections`, default 8). Setiap koneksi mengerjakan potongan yang berbeda.
3. **Resume dua lapis:**
   - Bila koneksi putus di tengah potongan, potongan itu otomatis dicoba lagi (maks. 30 kali, jeda makin lama) mulai dari byte terakhir yang diterima.
   - Bila program berhenti (Ctrl+C, laptop mati, crash), **jalankan ulang perintah yang sama**. Skrip memeriksa ukuran setiap potongan di disk dan hanya mengunduh yang kurang. Status unduhan dibaca dari file potongan itu sendiri, jadi tidak ada file catatan yang bisa rusak.
4. **Cek MD5** setelah lengkap. Hasilnya dibandingkan dengan MD5 resmi dari Zenodo, untuk memastikan tidak ada satu byte pun yang salah.
5. Setelah ekstraksi selesai, potongan dihapus otomatis. Pada unduhan sebagian (`--max_gb`), potongan disimpan supaya unduhan penuh nanti bisa melanjutkannya.
6. **Perbaikan bug ekstraksi:** Jika ekstraksi terhenti di tengah file (arsip parsial habis, atau Ctrl+C), gambar yang sedang ditulis kini dihapus. Sebelumnya tertinggal file PNG 0 byte yang akan dilewati pada ekstraksi berikutnya (dianggap "sudah ada") dan bisa membuat training error.

**Kenapa:**
- **Resume:** Unduhan 13 GB memakan waktu berjam-jam. Satu kali koneksi putus pada cara lama berarti mengulang dari 0 GB, dan ini sudah terjadi saat mengunduh PyTorch (Progress 12).
- **8 koneksi:** Zenodo membatasi kecepatan **per koneksi**. Hasil ukur: 1 koneksi ±0,35 MB/s, 8 koneksi ±1,05 MB/s, jadi sekitar 3× lebih cepat. Perkiraan waktu unduh penuh ±3–4 jam, dibanding ±10 jam dengan 1 koneksi. Kecepatan bisa berbeda tergantung waktu dan kondisi server.
- **Potongan 128 MB, bukan lebih kecil:** Zenodo membatasi 133 request per menit (*rate limit*). Dengan ±97 potongan, jumlah request tetap aman. Jika server tetap membalas "terlalu banyak request" (HTTP 429), skrip menunggu sesuai waktu yang diminta server.

**Konsekuensi (trade-off):** Cara lama mengekstrak sambil mengunduh, tanpa menyimpan arsip. Cara itu hemat disk, tetapi tidak bisa dilanjutkan, karena proses dekompresi gzip harus dimulai dari byte pertama. Cara baru menyimpan arsip dulu, sehingga **butuh ruang kosong sementara ±26 GB** (13 GB arsip + 13 GB hasil ekstrak). Disk D: saat ini punya 92 GB kosong. Skrip memberi peringatan jika ruang tidak cukup. Jika server suatu saat tidak mendukung Range, skrip otomatis kembali ke cara lama (1 koneksi) dan memberi tahu.

**Penjelasan sederhana:** Seperti memindahkan 97 kardus dengan 8 orang sekaligus, bukan satu orang membawa satu truk. Kalau ada yang terhenti, kardus yang sudah sampai tidak perlu diangkut ulang; cukup lanjutkan kardus yang belum.

**Hasil uji (80 MB pertama, potongan diperkecil menjadi 8 MB agar 8 koneksi terpakai):**
- Program **dimatikan paksa** setelah 25 detik, saat 8 potongan sedang diunduh bersamaan dan 18 MB sudah terkumpul.
- Saat dijalankan ulang, program mencetak "Sudah terunduh sebelumnya: 0.02 GB" lalu **melanjutkan**, tidak mengulang.
- MD5 hasilnya **identik** dengan unduhan `curl` biasa untuk rentang yang sama (`5a26ef78…d750`).
- Setelah perbaikan bug ekstraksi: 6.146 gambar diekstrak, 0 rusak.
- Belum teruji langsung: cek MD5 arsip penuh dan penghapusan potongan setelah ekstraksi. Keduanya baru jalan setelah unduhan 13 GB lengkap.

---

### Progress 14 — Training pertama dengan data asli & evaluasi pada data test

**Tujuan:** Mengukur kualitas FlareFormer pada data asli, terutama pada AR test yang tidak pernah dilihat model saat training maupun saat memilih threshold.

**Yang dilakukan:** `python train.py` dengan pengaturan default (4 frame, target flare ≥M dalam 24 jam), lalu `python evaluate.py`. Threshold (0,07) dipilih dari data validasi dan dipakai apa adanya pada data test.

**Hasil:**

| | Validasi | Test |
|---|---|---|
| TSS | 0,771 | **0,633** |
| HSS | 0,196 | 0,164 |
| AUC | 0,943 | 0,906 |
| Recall | 0,956 | 0,854 |
| Precision | 0,136 | 0,123 |

Pada test (6.067 sampel, 213 di antaranya diikuti flare ≥M): 182 flare tertangkap (TP), 31 terlewat (FN), 1.295 alarm palsu (FP), 4.559 benar tenang (TN). Training berhenti otomatis setelah epoch 6 karena TSS validasi terbaik justru terjadi di epoch 1 dan tidak terlampaui selama 5 epoch berikutnya.

**Cara membaca hasil:**
- **TSS test 0,633 berada di kisaran literatur** (Riggi dkk. 2025: 0,646 untuk 1 magnetogram; Li dkk. 2025: 0,49–0,78). Protokol evaluasinya tidak persis sama, jadi ini perbandingan kasar.
- **Turun dari 0,771 (validasi) ke 0,633 (test) itu wajar.** Epoch terbaik dan threshold dipilih dari validasi yang hanya punya 182 sampel flare, sehingga skor validasi cenderung terlalu optimis. Angka test adalah angka yang jujur untuk dilaporkan.
- **Precision rendah (12,3%).** Threshold 0,07 dipilih karena memaksimalkan TSS, yaitu menangkap hampir semua flare dengan harga banyak alarm palsu (22% kondisi tenang ikut dialarmkan). Karena flare ≥M hanya 3,5% dari sampel, dari sekitar 8 alarm hanya 1 yang benar-benar diikuti flare. Threshold yang lebih tinggi menukar recall dengan precision.
- **Overfitting mulai epoch 2.** Loss training terus turun (0,55 → 0,29), sedangkan skor validasi tidak membaik (AUC 0,943 → 0,871 di epoch 6). Model mulai menghafal data training.

**Penjelasan sederhana:** Model ini seperti penjaga yang sangat waspada. Hampir tidak ada flare yang lolos (85% tertangkap), tetapi alarmnya sering salah (sekitar 1 dari 5 hari tenang tetap dialarmkan).

**Langkah berikutnya:** (1) ablasi `--n_frames 1` untuk menguji apakah informasi waktu membantu, (2) kurangi overfitting dengan learning rate lebih kecil atau dropout lebih besar. Perbandingan antar pengaturan dilakukan pada skor **validasi**; data test hanya dipakai sekali untuk model akhir, supaya tidak ikut "dituning".

---

### Progress 15 — Prediksi real-time dari data SDO terbaru

**Tujuan:** Memakai model untuk Matahari *hari ini*: ambil 4 magnetogram HMI terbaru dari JSOC, ubah ke format yang **persis sama** dengan dataset latih, lalu prediksi flare ≥M dalam 24 jam ke depan. File: [fetch_realtime.py](fetch_realtime.py), [flare/hmi.py](flare/hmi.py).

**Cara dataset dibuat (dari paper Boucheron dkk. 2023 + kode penulis di GitHub, lalu dicek sendiri):**
1. Series `hmi.M_720s` dari JSOC, filter `QUALITY >= 0`.
2. Fitur *im_patch* JSOC: kotak 600×600 piksel yang **mengikuti rotasi Matahari** (tracking Carrington).
3. Resize ke 224×224 (scikit-image, `order=1`, `anti_aliasing`), lalu `piksel = round(clip(B + 2550, 0, 5100) × 255/5100)`.
4. Hanya AR dalam ±60° dan potongan tanpa NaN.

**Hal yang tidak tertulis di paper, ditemukan dengan mencocokkan PNG dataset dengan FITS asli:**
- Gambar dataset **diputar 180°** dari array FITS mentah (array HMI terbalik; di dataset kutub utara di atas).
- Urutannya potong → resize → 8-bit. Dengan posisi yang tepat, hasilnya **sama 100% per piksel**.
- Pusat kotak adalah titik Carrington yang **tetap** selama AR terlihat, yaitu posisi NOAA pada laporan harian **saat AR paling dekat dengan meridian tengah**, ditambah 0,09°. Angka 0,09° adalah kalibrasi empiris yang konsisten pada 4 AR (2011–2017); asal fisisnya belum diketahui.
- Posisi di daftar region SWPC berlaku **pukul 24:00 tanggal observasi** (00:00 hari berikutnya). Tanpa koreksi ini, kotak meleset ±13° (satu hari rotasi). Ini terdeteksi karena gambar AR 14556 awalnya kosong di tengah.

**Alur `fetch_realtime.py`:**
```
SWPC/HEK (posisi AR) ─┐
JSOC (4 frame, 72 menit) ─► potong 600×600 + tracking ─► putar 180° ─► 224×224 ─► 8-bit PNG ─► predict.py
```
- Data **terbaru** memakai `hmi.M_720s_nrt` (near-real-time). Data final `hmi.M_720s` baru tersedia ±4,5 hari kemudian. Skrip memilih otomatis.
- Data diambil **langsung dari arsip JSOC tanpa email/akun**. File FITS HMI (±14 MB) dikompresi per baris, dan tabel di awal file mencatat posisi tiap baris. Jadi skrip hanya mengunduh header + tabel (±38 KB) dan 600 baris di sekitar AR (±2 MB) lewat *HTTP Range request*. Hasilnya dicek identik dengan membaca file penuh, dan 4 frame diunduh bersamaan. Waktu per prediksi turun dari ±4–6 menit menjadi ±25 detik.
- Posisi AR: daftar region SWPC (30 hari terakhir) atau HEK (lebih lama). Bisa diganti manual dengan `--position BUJUR_CARRINGTON,LINTANG`.
- Skrip menolak AR di luar ±60° dan potongan ber-NaN, karena model tidak pernah dilatih dengan gambar seperti itu.

**Kenapa:** Model hanya bisa dipercaya bila input real-time berasal dari "distribusi" yang sama dengan data latih. Perbedaan kecil seperti orientasi terbalik atau skala piksel berbeda bisa membuat prediksi tidak bermakna tanpa ada error yang terlihat. Karena itu pipeline diverifikasi langsung terhadap PNG dataset, bukan sekadar "jalan".

**Hasil verifikasi** (`--verify`, 4 AR dari 4 tahun, termasuk 1 AR test):

| AR | Waktu frame terakhir | Frame sama 100% | Skor dataset | Skor pipeline |
|---|---|---|---|---|
| 11158 | 2011-02-14 12:00 | 2 dari 4 | 0,853 | 0,855 |
| 12192 | 2014-10-23 11:00 | 2 dari 4 | 0,988 | 0,988 |
| 12297 (test) | 2015-03-13 06:00 | 2 dari 4 | 0,789 | 0,781 |
| 12673 | 2017-09-04 00:36 | 0 dari 4 | 0,854 | 0,860 |

Frame yang tidak sama persis hanya **bergeser ±1 piksel** (selisih rata-rata ±0,5 dari 255 level) karena pembulatan posisi sub-piksel. Skor model berbeda paling banyak 0,009 dan keputusannya selalu sama.

**Uji real-time (2026-10-17, frame terakhir 07:36 TAI, berumur 0,9 jam):** AR 14556 skor 0,012 dan AR 14557 skor 0,036 (dengan lintang dikoreksi manual ke N07), keduanya "TIDAK ADA FLARE". Peta saliency menyorot garis netral di tengah AR bipolar.

**Batasan penting:**
1. Model dilatih dengan data 2010–2018 (siklus Matahari 24). Uji pada data 2021–2025 ada di Progress 16 (TSS 0,59).
2. Label latih memakai skala GOES-15. Sejak 2020 NOAA menghapus faktor 0,7, jadi M1.0 lama ≈ **M1.4** di laporan sekarang. Pakai batas ini saat mencocokkan prediksi dengan flare yang benar-benar terjadi.
3. Data NRT dikalibrasi sedikit berbeda dari data final.
4. Bila AR belum melewati meridian tengah, posisi kotak memakai laporan terdekat yang ada sehingga bisa sedikit berbeda dari aturan dataset. Skrip memberi peringatan.
5. Laporan NOAA kadang salah ketik. Contoh: AR 14557 tercatat N07 selama 4 hari lalu N17 pada 2026-10-17. Skrip memberi peringatan bila lintang melompat >3°.

**Penjelasan sederhana:** Seperti memotret ulang dengan kamera, lensa, dan sudut yang sama persis dengan foto-foto yang dipakai untuk melatih model. Kalau kameranya beda, model bisa salah menilai walaupun Mataharinya sama.

---

### Progress 16 — Uji model pada data 2021–2025 (siklus Matahari 25)

**Tujuan:** Menjawab batasan nomor 1 di Progress 15: apakah model yang dilatih dengan data 2010–2018 masih bekerja pada Matahari sekarang? File: [evaluate_recent.py](evaluate_recent.py).

**Cara kerja:**
1. Daftar flare SWPC (beserta nomor AR) dan laporan posisi harian AR diambil dari HEK untuk 2021–2025.
2. **Sampel positif:** 150 waktu acak, 0,5–23,5 jam sebelum flare ≥M (maks. 3 flare per AR supaya tidak didominasi satu AR besar). **Sampel negatif:** 150 waktu acak dari laporan harian AR yang berada dalam ±45°.
3. **Label** memakai aturan dataset: flare terbesar dari AR itu yang puncaknya dalam 24 jam setelah frame terakhir. Fluks GOES-16/18 dikali 0,7 agar sama dengan skala label latih, sehingga positif berarti ≥M1.43 skala sekarang.
4. Gambar dibuat dengan pipeline Progress 15 (data final `hmi.M_720s`). Threshold 0,07 dari validasi dipakai apa adanya.
5. **Selang kepercayaan 95%** dihitung dengan bootstrap **per AR**: yang diundi adalah AR-nya, bukan sampelnya, karena sampel dari AR yang sama saling berkorelasi. Selang test 2010–2018 dihitung dengan cara yang sama.

**Efisiensi:** Unduhan sebagian dari Progress 15 (±2 MB per frame, 4 frame paralel, 3 sampel bersamaan) membuat 300 sampel selesai dalam ±40 menit. Dengan unduhan penuh, waktunya ±20 jam.

**Hasil:**

| | 2021–2025 (baru) | Test 2010–2018 (Progress 14) |
|---|---|---|
| TSS | **0,593** (95%: 0,48–0,70) | 0,633 (95%: 0,43–0,75) |
| AUC | 0,869 (95%: 0,81–0,92) | 0,906 (95%: 0,83–0,95) |
| Recall | 0,837 | 0,854 |
| False alarm rate | 0,244 | 0,221 |
| Sampel (positif) | 221 (86) dari 196 AR | 6.067 (213) |

79 dari 300 sampel dilewati: 51 positif karena AR-nya di luar ±60° (flare dekat tepi piringan; dataset juga membuang AR seperti ini), 16 karena frame tidak tersedia, dan 12 karena potongan berisi NaN atau berada di celah data.

**Cara membaca hasil:**
- **Model masih bekerja pada siklus 25.** TSS turun 0,04 dan AUC turun 0,04, tetapi selang kepercayaannya tumpang tindih lebar. Dengan jumlah sampel ini, penurunan tersebut tidak bisa dibedakan dari variasi acak. TSS 0,59 masih di dalam kisaran literatur (0,49–0,78).
- **Distribusi skor mirip.** Median skor sampel tenang 0,023 (test: 0,017). Median skor sampel flare 0,67 (test: 0,81), jadi sedikit lebih ragu pada flare, tapi kebanyakan masih jauh di atas threshold 0,07.
- **Alarm palsu terkonsentrasi di AR yang aktif.** Negatif yang diikuti flare kelas C (atau M lemah di bawah batas) dialarmkan 44%, sedangkan negatif yang benar-benar tenang 20%. Ini wajar: AR yang sedang aktif memang tampak "berbahaya".
- **Semua 5 sampel sebelum flare X tertangkap.**
- **Per tahun** (2021–22: TSS 0,57; 2023: 0,38; 2024: 0,62; 2025: 0,55) angkanya terlalu sedikit untuk disimpulkan. Misalnya 2023 hanya punya 8 sampel positif.

**Batasan evaluasi ini:**
1. **Protokolnya tidak identik dengan test resmi.** Test memakai semua sampel tiap 3 jam, sedangkan di sini sampelnya acak dan seimbang. TSS dan AUC tidak bergantung pada proporsi kelas, jadi masih bisa dibandingkan, tetapi precision dan HSS tidak.
2. Negatif diambil dari AR dalam ±45° saat laporan, sedangkan positif sampai ±60°. Posisi negatif jadi sedikit lebih dekat ke pusat piringan daripada di dataset.
3. Koreksi skala GOES hampir tidak berpengaruh di sini (86 vs 87 positif), karena positif memang diambil dari flare ≥M1.43. Koreksi ini tetap penting saat membandingkan prediksi harian dengan laporan flare SWPC.
4. Data final (`hmi.M_720s`) yang dipakai di sini sedikit berbeda dari data NRT pada prediksi real-time.

**Penjelasan sederhana:** Seperti dokter yang belajar dari pasien tahun 2010–2018 lalu diuji pada pasien tahun 2021–2025. Ketepatannya sedikit menurun, tapi masih dalam batas wajar, jadi ilmunya tidak "kedaluwarsa".

---

### Progress 17 — Model tahap 2: membedakan flare M dari X (hasil negatif)

**Tujuan:** Menambah pemisah M vs X di belakang model tahap 1, supaya keluarannya tiga kelas (tenang / M / X). File: [train_xclass.py](train_xclass.py).

**Masalah utama: data kelas X sangat sedikit per AR.**

| | Sampel ≥M | Sampel X | AR dengan X |
|---|---|---|---|
| train | 23.395 | 2.254 | 13 |
| val | 2.816 | 238 | 2 |
| test | 3.312 | 351 | 3 |

Gambar dari satu AR hampir identik, sehingga jumlah gambar menyesatkan. Informasi yang sebenarnya hanya **18 AR**. Validasi dengan 2 AR X tidak bisa dipakai untuk memilih model atau threshold.

**Cara menguji agar jujur:**
1. **Cross-validation per AR** (5 lipatan, 3 AR X per lipatan) pada AR train+val. Setiap AR diuji oleh model yang belum pernah melihatnya. Selang kepercayaan dihitung dengan bootstrap per AR.
2. Model akhir dilatih pada semua AR train+val, lalu diuji **sekali** pada AR test.
3. Dua target dibandingkan: **klasifikasi X vs M** (BCE) dan **regresi log-fluks** (M1→0, X1→1, memakai seluruh informasi kekuatan M1–M9). Kedua model mulai dari bobot ImageNet, bukan dari model tahap 1, agar tidak ada kebocoran.
4. **Pembanding:** skor model tahap 1 sendiri. Skor ini sudah lebih tinggi pada flare X (AUC 0,81 pada test).

**Hasil (AUC X vs M; 0,5 = tebakan acak):**

| | Klasifikasi X | Regresi log-fluks | Skor tahap 1 (pembanding) |
|---|---|---|---|
| Cross-validation, sampel | 0,690 (95%: 0,52–0,80) | 0,683 (95%: 0,56–0,80) | – |
| Cross-validation, tingkat AR | 0,658 | 0,768 | – |
| Test, sampel | 0,744 (95%: 0,55–0,97) | 0,691 (95%: 0,35–0,95) | **0,810** |
| Test, tingkat AR (3 AR X) | 0,881 | 0,857 | 0,857 |

AUC antar lipatan sangat bervariasi, dari 0,42 (di bawah tebakan acak) sampai 0,83.

**Gabungan dua tahap pada semua sampel test (5.854 sampel tenang, 189 M, 24 X):**

| Model tahap 2 | X terdeteksi | M salah disebut X | Tenang salah disebut X | Ketepatan alarm X |
|---|---|---|---|---|
| Klasifikasi X | 24 dari 24 | 99 dari 189 | 926 | 2,3% |
| Regresi log-fluks | 16 dari 24 | 68 dari 189 | 223 | 5,2% |

**Kesimpulan: model tahap 2 tidak berguna secara operasional.**
- Tidak ada model yang mengalahkan skor tahap 1 sebagai pembeda X dari M. Dengan 18 AR, model tidak belajar pola magnetik khusus kelas X. Ia lebih banyak menghafal AR yang kebetulan berflare X.
- Selang kepercayaan lebar dan hasil per lipatan tidak stabil. Perbedaan antar model tidak bisa disimpulkan.
- Bila dipakai, alarm "X" hampir selalu salah (ketepatan 2–5%).

**Kenapa:** Flare X dipicu oleh medan yang paling kuat dan rumit, tetapi hanya ada 18 contoh independen. Model dengan 11,6 juta parameter mudah menghafal 18 contoh itu. Ini batas data, bukan bug.

**Tidak dimasukkan ke `predict.py` / `fetch_realtime.py`.** Keluaran tiga kelas yang tidak andal lebih menyesatkan daripada tidak ada.

**Jalan keluar yang mungkin:**
1. Pakai skor tahap 1 yang sudah ada sebagai indikator kekuatan ("skor sangat tinggi, flare besar lebih mungkin"), tanpa menyebutnya prediksi kelas X.
2. Tambah data X dengan data siklus 25 (2021+): AR berflare X di sana dapat ditambahkan lewat HEK + pipeline `fetch_realtime`. Ini cara paling realistis untuk menaikkan jumlah AR X, tetapi perlu label ulang dan pengujian ulang.
3. Fitur yang jauh lebih sedikit parameternya, misalnya 29 fitur kompleksitas magnetik dari Dryad dengan regresi logistik, kurang rawan menghafal.

**Penjelasan sederhana:** Seperti meminta orang membedakan badai "sangat dahsyat" dari "dahsyat" setelah hanya melihat 18 badai dahsyat. Ia lebih mungkin mengingat badai-badai itu satu per satu daripada mengerti apa yang membuatnya berbeda.

**Lanjutan:** lihat Progress 18. Pendekatan berbasis riset dengan skor komposit berhasil, dan sudah dikonfirmasi pada data independen.

---

### Progress 18 — Peluang flare X dari fitur fisis + riwayat flare (berbasis riset)

**Tujuan:** Mengganti model dalam yang gagal di Progress 17 dengan pendekatan yang didukung riset, lalu memasangnya ke prediksi real-time. File: [flare/features.py](flare/features.py), [xclass_features.py](xclass_features.py), [evaluate_recent_xclass.py](evaluate_recent_xclass.py), [fetch_realtime.py](fetch_realtime.py).

**Temuan riset yang dipakai:**

| Riset | Temuan | Dipakai sebagai |
|---|---|---|
| Wheatland 2005 | Ukuran flare mengikuti power-law (indeks 1,7–1,9). Dari semua flare ≥M, hanya ±7–16% mencapai X. Skor ramalan untuk X (0,066) jauh di bawah M-X (0,272). | Target berupa **peluang** P(X \| ≥M), bukan kelas pasti. Klimatologi di data kita 9,7%, cocok dengan teori. |
| Wheatland 2005; Falconer dkk. 2012 | Riwayat flare adalah indikator terkuat flare berikutnya; menambah produktivitas flare terkini memperbaiki ramalan. | Fitur `hist24`, `hist48`: kelas flare terbesar 0–24 dan 24–48 jam sebelumnya. |
| Sammis, Tang & Zirin 2000; Bobra & Couvidat 2015 | AR yang lebih besar (luas bintik, fluks tak bertanda) menghasilkan flare lebih besar. | `log_usflux`, `spot_area_mm2` |
| Schrijver 2007 | Fluks di sekitar garis netral bermedan kuat (nilai R) adalah proksi energi bebas dan berkorelasi kuat dengan flare M/X. | `log_R` (ambang 150 G, Gaussian 15 Mm) |
| Falconer dkk. 2003; Mason & Hoeksema 2010 | Panjang garis netral bergradien tinggi (>50 G/Mm) adalah proksi energi bebas dari magnetogram line-of-sight. | `lsg_mm`, `log_gwill`, `max_grad` |
| Sinha dkk. 2022 | Pada parameter magnetik, regresi logistik mengungguli model yang lebih rumit. | Model kecil, bukan jaringan saraf |

Semua fitur dihitung dari 4 PNG yang sama dengan model tahap 1. Riwayat flare diambil dari label dataset 24 dan 48 jam sebelumnya. Untuk data 2021+ dan real-time, riwayat diambil dari daftar flare SWPC di HEK.

**Diagnosis, kenapa regresi logistik biasa gagal (AUC CV 0,50):** Setiap fitur sendiri-sendiri bermakna (AUC 0,70–0,75). Tapi fitur-fiturnya sangat berkorelasi, dan dengan hanya 14 AR X regresi memberi bobot berlawanan pada fitur yang hampir sama (log_R +1,08, log_gwill −0,97). Ini overfitting.

**Solusi: skor komposit berbobot sama.** Fitur distandarkan, lalu dirata-rata dengan arah yang ditentukan riset (makin besar, flare makin kuat). Hanya 2 parameter yang dipelajari, yaitu regresi logistik 1 variabel untuk mengubah skor menjadi peluang. Fitur perubahan 3,6 jam tidak dipakai karena arah pengaruhnya belum mapan di literatur. Metode ini dipilih **setelah** melihat hasil CV, jadi wajib dikonfirmasi pada data independen.

**Hasil (AUC X vs M, 0,5 = acak, 95% dari bootstrap per AR):**

| Model | CV 2010–2018 (14 AR X) | Test (3 AR X) | **2021–2025 independen (12 AR X, 77 AR M)** |
|---|---|---|---|
| Model dalam, klasifikasi X (Progress 17) | 0,69 | 0,74 | – |
| Regresi logistik biasa, fisis+riwayat | 0,50 | 0,65 | 0,74 |
| Regresi logistik C=0,01 | 0,69 | 0,71 | 0,77 |
| Hanya riwayat flare | 0,69 | 0,63 | 0,71 |
| **Komposit bobot sama** | **0,75** (0,60–0,86) | 0,79 (0,62–0,98) | **0,79** (0,64–0,89) |
| Skor tahap 1 | – | 0,81 | 0,78 |
| Tahap 1 + komposit (rata-rata peringkat) | – | 0,85 | 0,80 |

Di data 2021–2025, fitur tunggal juga konsisten: nilai R 0,77, gradien maks 0,76, garis netral 0,74, fluks 0,74, luas bintik 0,71, riwayat 0,70. **Arah dan besarnya cocok dengan riset di tiga kumpulan data berbeda.**

**Gabungan dua tahap pada semua sampel test** (tahap 1 menyatakan FLARE, lalu peluang X ≥ 0,10 dianggap alarm X): 14 dari 24 sampel X tertangkap, dan 14 dari 116 alarm X benar (12%). Model dalam di Progress 17 hanya 2–5%. Klimatologi 9,7% berarti alarm X 12% hanya sedikit lebih baik dari menebak semua flare sebagai X dengan peluang rata-rata. Karena itu **keluarannya disajikan sebagai peluang, bukan alarm.**

**Cara membaca hasil:**
- Skor komposit konsisten ±0,75–0,79 di tiga kumpulan data, termasuk data yang tidak pernah dilihat. Ini mengalahkan model dalam (0,69) dan setara dengan skor tahap 1.
- Skor tahap 1 sudah membawa informasi kekuatan flare: AR yang paling "berbahaya" cenderung menghasilkan flare terbesar. Menggabungkan keduanya sedikit lebih baik (0,80–0,85), tetapi selisihnya masih dalam selang kepercayaan.
- Sesuai Wheatland (2005), kekuatan flare sebagian besar bersifat acak. Peluang di atas 2–3× rata-rata adalah sinyal berguna, tapi bukan kepastian.

**Pemasangan ke prediksi real-time:** `fetch_realtime.py --predict` sekarang mencetak:
```
Skor flare >= M1.0 dalam 24 jam : 0.986
Keputusan (threshold 0.07)       : FLARE
Jika flare >= M terjadi, peluang kelas X  : 27% (rata-rata semua flare >= M: 10%; 2.8x rata-rata)
  riwayat flare AR ini: 24 jam terakhir X, 24-48 jam M (daftar flare HEK terakhir: 2017-09-06 09:10 UT)
```
Contoh di atas adalah AR 12673 pukul 10:00, dua jam sebelum flare X9.3. AR ini termasuk data latih, jadi hanya ilustrasi. Riwayat flare real-time diambil dari HEK (jeda ±7 jam). Faktor 0,7 skala GOES hanya diterapkan pada flare 2020 ke atas.

**Batasan:**
1. Validasi independen hanya punya 12 AR X, jadi selang kepercayaannya masih lebar (0,64–0,89).
2. PNG dataset memotong medan di ±2550 G, sehingga inti bintik terkuat (yang paling relevan untuk flare X) ikut terpotong. Fitur dari magnetogram FITS asli mungkin lebih kuat.
3. Peluang dikalibrasi pada data 2010–2018 (BSS terhadap klimatologi hanya +0,03 di CV dan +0,06 di test), jadi angka persennya bersifat kasar.
4. Riwayat flare real-time bergantung pada jeda HEK; flare beberapa jam terakhir bisa belum tercatat.

**Penjelasan sederhana:** Daripada meminta model besar menghafal 18 contoh, kita memakai "aturan praktis" fisikawan Matahari: AR besar, garis netral panjang dan tajam, dan sudah sering ber-flare cenderung menghasilkan flare besar. Aturan-aturan ini dijumlahkan sama rata. Hasilnya tidak sempurna, karena kekuatan flare memang sebagian acak, tapi konsisten di data lama maupun data baru.

---

### Progress 19 — Kode disederhanakan agar mudah dibaca (hasil tidak berubah)

**Tujuan:** Membuat kode lebih mudah dibaca dan dipelajari oleh mahasiswa dengan kemampuan coding menengah, **tanpa mengubah satu pun hasil atau nilai** program.

**Kenapa:** Setelah 18 progress, kode tumbuh dengan tiga masalah:
1. **Kode ganda.** "Muat 4 PNG lalu hitung skor model" ditulis di 5 tempat, bootstrap per AR di 3 tempat, dan pembagian lipatan CV di 2 tempat. Bila satu diubah, yang lain mudah terlupa.
2. **Fungsi `main()` terlalu panjang** (sampai 99 baris) dan mencampur banyak langkah.
3. **Skrip mengimpor skrip lain** (`evaluate_recent_xclass.py` mengambil fungsi dari dua skrip lain), dan satu file `flare/hmi.py` (321 baris) mengurus tiga hal berbeda.

**Yang dilakukan:**

| Perubahan | Contoh |
|---|---|
| `flare/hmi.py` dipecah menurut tugasnya | `jsoc.py` (unduh dari JSOC), `noaa.py` (posisi AR), `hmi.py` (potong gambar) |
| Kode bersama dipindah ke `flare/` | `recent.py` (evaluasi 2021+), `xclass.py` (model peluang X, lipatan CV, tabel tiga kelas) |
| Satu fungsi untuk satu tugas yang berulang | `predict_one()`, `load_sequence()`, `make_optimizer()`, `set_seed()`, `bootstrap_ci_by_ar()`, `load_config()`, `goes_to_flux()` |
| `main()` dipecah menjadi langkah bernama | `train.py`: `load_data()`, `validate()`; `xclass_features.py`: `cross_validate()`, `test_final_models()` |
| Penulisan ringkas yang sulit dibaca diganti | operator `:=`, `np.where` bersarang, lambda di dalam fungsi, `dict.get` sebagai pengganti `if` |
| `fetch_realtime.py` memanggil `predict_images()` langsung | sebelumnya menjalankan `predict.py` sebagai proses terpisah |
| Panduan membaca | `flare/__init__.py` berisi peta paket, README bagian "Peta kode" |

Fungsi terpanjang turun dari 99 menjadi 62 baris. Baris kode murni naik dari 1.773 menjadi 1.909 (+8%), karena perulangan ditulis eksplisit dan ada fungsi-fungsi kecil baru. Komentar dan docstring bertambah lebih banyak lagi.

**Bukti hasil tidak berubah (uji regresi):** Setiap program dijalankan dengan kode lama, lalu dengan kode baru, dan semua keluarannya dibandingkan per angka (toleransi 1e-6) atau per byte:

| Yang diuji | Hasil |
|---|---|
| `make_dummy_data.py` (5.400 gambar) dan `prepare_data.py` (data sintetis dan asli) | identik per byte |
| `evaluate.py` pada 6.067 sampel test | identik (TSS 0,633) |
| `predict.py` (`--ar` / `--images` / jumlah gambar salah) | identik |
| `fetch_realtime.py` (verifikasi AR 11158, posisi manual, peluang X AR 12673) | PNG identik per byte, keluaran identik |
| `evaluate_recent.py` dan `evaluate_recent_xclass.py` | identik |
| `xclass_features.py`, fitur fisis dihitung ulang dari nol | identik |
| `download_data.py` (unduh 20 MB dari Zenodo + ekstrak arsip lokal) | gambar identik per byte |
| `train.py` di CPU, 2 epoch | riwayat dan **seluruh 11,6 juta bobot model identik bit demi bit** |
| `train_xclass.py --smoke` di CPU (3 kali training) | metrik, semua prediksi, dan bobot model akhir identik |

**Pelajaran dari uji ini:** Training di **GPU tidak deterministik**. Kode lama yang dijalankan dua kali dengan seed sama pun memberi loss sedikit berbeda (0,5264 vs 0,5253), karena cuDNN memilih algoritma secara otomatis (`cudnn.benchmark`) dan memakai operasi atomik yang urutan penjumlahannya acak. Jadi perbandingan training yang adil harus dilakukan di CPU. Di Windows, mematikan GPU harus memakai `CUDA_VISIBLE_DEVICES=-1`; nilai kosong (`""`) dianggap tidak ada dan GPU tetap terpakai.

**Catatan:** `runs/xclass_feat/model.pkl` dibuat ulang (`python xclass_features.py`), karena kelas `Composite` pindah dari `flare/features.py` ke `flare/xclass.py` dan file pickle menyimpan lokasi kelas. Isinya dicek identik: metrik, prediksi CV, dan prediksi test sama persis.

**Penjelasan sederhana:** Seperti merapikan buku catatan: halaman yang isinya sama digabung, bab yang terlalu panjang dibagi menjadi subbab, dan ditambah daftar isi. Semua isinya, termasuk setiap angka, tetap sama.

---

## Hasil eksperimen

Isi tabel ini setelah training dengan data asli. Sebagai pembanding, literatur melaporkan TSS ≥M sekitar **0,49–0,78** untuk model berbasis magnetogram (Riggi dkk. 2025: 0,646; Li dkk. 2025: 0,49–0,78 tergantung model dan filter data).

| Eksperimen | Perintah | TSS val | TSS test | HSS test | AUC test | Catatan |
|---|---|---|---|---|---|---|
| FlareFormer 4 frame | `python train.py` | 0,771 | 0,633 | 0,164 | 0,906 | model utama; berhenti di epoch 6, terbaik epoch 1, threshold 0,07; recall 0,854, precision 0,123 |
| FlareFormer 4 frame, data 2021–2025 | `python evaluate_recent.py` | – | 0,593 | – | 0,869 | sampel acak seimbang (221, 86 positif, 196 AR); TSS 95%: 0,48–0,70; lihat Progress 16 |
| Tahap 2: M vs X (klasifikasi / regresi) | `python train_xclass.py --target x` | – | – | – | 0,744 / 0,691 | AUC X vs M pada test; **di bawah** skor tahap 1 (0,810); 18 AR X; lihat Progress 17 |
| Peluang X: komposit fisis+riwayat | `python xclass_features.py` | – | – | – | 0,785 | AUC X vs M; CV 0,748; **2021–2025 independen 0,786** (12 AR X); lihat Progress 18 |
| Ablasi: 1 frame | `python train.py --n_frames 1 --run_dir runs/single` | | | | | apakah dimensi waktu membantu? |
| Ablasi: 6 frame | `python train.py --n_frames 6 --batch_size 8 --run_dir runs/f6` | | | | | riwayat lebih panjang |

Perbandingan 1 frame vs 4 frame penting untuk laporan, karena menjawab pertanyaan "apakah informasi evolusi waktu benar-benar berguna?" pada data ini.

---

## Ide pengembangan

1. **Lebih banyak frame / riwayat lebih panjang.** Li dkk. memakai 16 frame (24 jam). Coba `--n_frames 8` atau lebih jika GPU memungkinkan.
2. **Tambahkan data EUV (SDO/AIA).** Riggi dkk. menyimpulkan bahwa morfologi magnetik *perlu tetapi tidak cukup*. Informasi korona bisa menambah skor.
3. **Gabungkan fitur fisis.** Dataset Dryad menyediakan 29 fitur kompleksitas magnetik per gambar (`..._features.csv`) yang bisa digabung dengan token [CLS].
4. **Cross-validation per AR (5 fold)** untuk melaporkan TSS rata-rata ± standar deviasi, seperti Riggi dkk.
5. **Kalibrasi skor** (Platt scaling / isotonic) agar skor bisa dibaca sebagai peluang.
6. **Fine-tuning foundation model Surya dengan LoRA**, jika tersedia GPU yang lebih besar.

---

## Template progress berikutnya

Salin blok ini setiap kali ada perubahan:

```markdown
### Progress N — <judul singkat>

**Tujuan:** <apa yang ingin dicapai>

**Kenapa:** <alasan memilih cara ini; rujuk paper/eksperimen bila ada>

**Penjelasan sederhana:** <analogi atau penjelasan tanpa istilah teknis>

**File yang berubah:** <daftar file>

**Hasil:** <angka TSS/HSS atau pengamatan; bandingkan dengan progress sebelumnya>
```

---

## Referensi

1. Boucheron, L. E., Vincent, T., Grajeda, J. A., & Wuest, E. (2023). *Solar active region magnetogram image dataset for studies of space weather.* Scientific Data. arXiv:2305.09492. Data: https://doi.org/10.5061/dryad.jq2bvq898, gambar: https://doi.org/10.5281/zenodo.7775776, kode: https://github.com/DuckDuckPig/AR-flares
2. Li, X., Ye, H., Zheng, Y., dkk. (2025). *Intelligent Forecasting for Solar Flares Using Magnetograms from SDO/SHARP, SDO/HMI, and ASO-S/FMG.* ApJS 278(2), 63. https://doi.org/10.3847/1538-4365/add149
3. Riggi, S., Romano, P., Pilzer, A., & Becciani, U. (2025). *Solar flare forecasting with foundational transformer models across image, video, and time-series modalities.* Astronomy and Computing. arXiv:2510.23400. Kode: https://github.com/inaf-oact-ai/solar-flare-forecaster
4. Cao, M., & Zhao, J. (2026). *Solar Flare Prediction Using a Hybrid Convolutional Neural Network and Transformer Model.* arXiv:2609.10772.
5. NASA, IBM, dkk. (2025). *Surya: Foundation Model for Heliophysics.* arXiv:2508.14112.
6. Bloomfield, D. S., Higgins, P. A., McAteer, R. T. J., & Gallagher, P. T. (2012). *Toward reliable benchmarking of solar flare forecasting methods.* ApJL 747, L41.
7. Ahmadzadeh, A., dkk. (2021). *How to train your flare prediction model: Revisiting robust sampling of rare events.* ApJS 254, 23.
8. Smilkov, D., dkk. (2017). *SmoothGrad: removing noise by adding noise.* arXiv:1706.03825.
9. He, K., dkk. (2016). *Deep Residual Learning for Image Recognition.* CVPR. — Vaswani, A., dkk. (2017). *Attention Is All You Need.* NeurIPS.
10. Wheatland, M. S. (2005). *A statistical solar flare forecast method.* Space Weather 3, S07003. arXiv:astro-ph/0505311
11. Schrijver, C. J. (2007). *A characteristic magnetic field pattern associated with all major solar flares and its use in flare forecasting.* ApJL 655, L117. Lihat juga arXiv:0710.0562.
12. Falconer, D. A., Moore, R. L., & Gary, G. A. (2003). *A measure from line-of-sight magnetograms for prediction of coronal mass ejections.* JGR 108, 1380. — Falconer dkk. (2012). *Prior flaring as a complement to free magnetic energy for forecasting solar eruptions.* ApJ 757, 32.
13. Mason, J. P., & Hoeksema, J. T. (2010). *Testing automated solar flare forecasting with 13 years of MDI synoptic magnetograms.* ApJ 723, 634.
14. Sammis, I., Tang, F., & Zirin, H. (2000). *The dependence of large flare occurrence on the magnetic structure of sunspots.* ApJ 540, 583.
15. Bobra, M. G., & Couvidat, S. (2015). *Solar flare prediction using SDO/HMI vector magnetic field data with a machine-learning algorithm.* ApJ 798, 135.
16. Barnes, G., dkk. (2016). *A comparison of flare forecasting methods, I: results from the "All-Clear" workshop.* ApJ 829, 89. arXiv:1608.06319
17. Sinha, S., dkk. (2022). *A comparative analysis of machine learning models for solar flare forecasting: identifying high performing active region flare indicators.* ApJ 935, 45.
