"""Paket prediksi solar flare dari citra magnetogram SDO/HMI.

Isi paket, dalam urutan membaca yang disarankan:

    Inti model (dipakai train.py, evaluate.py, predict.py)
        config.py    semua pengaturan: lokasi file, ukuran model, learning rate, dll.
        data.py      membaca PNG, menyusun urutan 4 frame, augmentasi, sampler seimbang
        model.py     arsitektur FlareFormer (ResNet18 + Transformer)
        engine.py    loop training, prediksi, dan optimizer
        metrics.py   TSS, HSS, AUC, pemilihan threshold, selang kepercayaan per AR

    Data real-time (dipakai fetch_realtime.py dan evaluasi 2021+)
        jsoc.py      mengambil magnetogram HMI dari JSOC (hanya baris yang dibutuhkan)
        noaa.py      posisi active region dari laporan NOAA (SWPC / HEK)
        hmi.py       memotong magnetogram menjadi PNG berformat dataset
        recent.py    alat bantu evaluasi pada data 2021+ (daftar flare, unduh, lanjutkan)

    Peluang flare kelas X (tahap 2)
        features.py  fitur fisis magnetogram + riwayat flare
        xclass.py    model skor komposit dan alat evaluasi M vs X
"""
