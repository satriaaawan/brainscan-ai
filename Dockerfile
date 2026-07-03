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
# Pastikan folder ini juga berisi: templates/index.html, model2.h5, dll
COPY . .

# ── Environment ─────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

# ── Jalankan dengan Gunicorn (bukan flask dev server) ──────
# Timeout dinaikkan karena inference model + Grad-CAM bisa agak lama
CMD ["sh", "-c", "gunicorn -w 1 -b 0.0.0.0:${PORT} --timeout 120 app:app"]
