# Membuat virtual environment .venv di folder proyek dan memasang semua library,
# termasuk PyTorch versi CUDA 12.6. Jalankan dari folder proyek:
#     powershell -ExecutionPolicy Bypass -File setup_env.ps1

if (-not (Test-Path .venv)) {
    python -m venv .venv
}
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Error "Instalasi gagal."; exit 1 }
.\.venv\Scripts\python.exe gpu_running.py
