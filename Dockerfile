# ── Base image ─────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies (dibutuhkan oleh opencv & matplotlib) ─
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies dulu (biar layer cache efektif) ─
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy seluruh source code aplikasi ──────────────────────
# model2.h5 TIDAK perlu ada di sini -- akan didownload otomatis
# dari Hugging Face Hub saat container start (lihat app.py)
COPY . .

# ── Hugging Face Spaces WAJIB pakai port 7860 ──────────────
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
EXPOSE 7860

# ── Jalankan dengan Gunicorn ────────────────────────────────
CMD ["sh", "-c", "gunicorn -w 1 -b 0.0.0.0:${PORT} --timeout 180 app:app"]
