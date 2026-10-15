# Monitoring APD Konstruksi — YOLO + Streamlit

Aplikasi web untuk memeriksa gambar dan video menggunakan:
- `best.pt`: model deteksi APD hasil training Anda
- `yolov8n.pt`: model YOLO COCO untuk deteksi manusia

Kelas APD yang diharapkan: `Helmet`, `Mask`, `Safety Vest`, `boots`, `gloves`.

> Status "APD terdeteksi lengkap" hanya berarti kelima kelas berhasil dideteksi dan diasosiasikan oleh heuristik bounding box. Hasil harus diverifikasi oleh petugas; bukan jaminan kepatuhan keselamatan.

## 1. Struktur proyek

```text
apd_monitoring_web/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/
    └── best.pt              # salin bobot model Anda ke sini
```

## 2. Menjalankan secara lokal

Disarankan Python 3.10–3.12. Buat virtual environment:

### Windows

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Buka URL lokal yang dicetak Streamlit, biasanya `http://localhost:8501`.

Salin `best.pt` Anda ke `models/best.pt`. Jika file berada di lokasi lain, ubah "Path model APD" pada sidebar atau atur variabel lingkungan `PPE_MODEL_PATH`.

Model person `yolov8n.pt` biasanya diunduh otomatis oleh Ultralytics saat pertama kali dijalankan, sehingga komputer memerlukan internet pada saat awal jika bobot belum ada di cache.

## 3. Publikasi gratis dengan Hugging Face Spaces

1. Buat akun di Hugging Face.
2. Buat Space baru, pilih SDK **Docker** jika ingin mengontrol dependensi sistem, atau pilih Streamlit jika paket yang diperlukan tersedia.
3. Unggah `app.py`, `requirements.txt`, dan file konfigurasi deployment yang diperlukan.
4. Tempatkan `best.pt` pada lokasi yang bisa dibaca aplikasi, atau unduh saat startup dari penyimpanan model yang Anda kendalikan.
5. Pastikan lisensi dataset dan model mengizinkan distribusi publik.
6. Uji dengan gambar kecil dan video pendek terlebih dahulu.

### Pilihan Docker (lebih konsisten)

Buat file `Dockerfile` dengan isi:

```dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py .
COPY models ./models

EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

Untuk Space berbasis Docker, unggah juga `Dockerfile`. Periksa instruksi Hugging Face Spaces terbaru terkait port dan resource yang tersedia.

## 4. Model weights dan ukuran file

Jangan commit `best.pt` ke GitHub publik sebelum memastikan Anda memang boleh mendistribusikannya. File model bisa besar dan berisi hasil pekerjaan/dataset Anda. Opsi:
- simpan model di repositori model privat/publik yang sesuai;
- gunakan Git LFS bila platform mendukungnya;
- unduh bobot saat startup dari URL penyimpanan yang Anda kontrol, idealnya dengan checksum.

Untuk mengatur path model di hosting, set environment variable `PPE_MODEL_PATH`, misalnya `/app/models/best.pt`.

## 5. Batasan penting

- Asosiasi APD ke manusia saat ini memakai posisi bounding box, bukan pose estimation atau tracking.
- Orang jauh, tertutup, atau membungkuk bisa menyebabkan APD terlewat atau salah terasosiasi.
- Video diproses frame demi frame; ringkasan yang ditampilkan adalah frame terakhir yang dianalisis, bukan jumlah pekerja unik selama video.
- Hosting gratis dapat memiliki batas CPU/RAM, durasi proses, penyimpanan, dan tidur saat tidak aktif.
- Untuk monitoring keselamatan nyata, lakukan validasi dengan data berlabel dan verifikasi manusia.
