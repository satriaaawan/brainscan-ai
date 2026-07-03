"""
BrainScan AI - Flask Edition
Klasifikasi Tumor Otak | Xception + Grad-CAM | 15 Kelas
"""

import os
import io
import base64
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, render_template, request, jsonify
import tensorflow as tf

# ── Config ─────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

MODEL_PATH = "model2.h5"
IMG_SIZE   = (299, 299)

# ── Download model dari Hugging Face Hub jika belum ada lokal ──
# Ganti REPO_ID sesuai repo model kamu di HF Hub, contoh: "satriaaawan/brainscan-ai"
HF_REPO_ID   = os.environ.get("HF_MODEL_REPO", "satriaaawan/brainscan-ai")
HF_FILENAME  = os.environ.get("HF_MODEL_FILENAME", "model2.h5")

def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] {MODEL_PATH} sudah ada, skip download.")
        return
    try:
        from huggingface_hub import hf_hub_download
        print(f"[INFO] Mengunduh model dari HF Hub: {HF_REPO_ID}/{HF_FILENAME} ...")
        downloaded_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
        # Buat symlink/copy ke MODEL_PATH agar path lokal konsisten
        import shutil
        shutil.copy(downloaded_path, MODEL_PATH)
        print(f"[INFO] Model berhasil diunduh ke {MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Gagal mengunduh model dari HF Hub: {e}")

ensure_model_downloaded()

CLASSES = [
    "Astrocytoma", "Carcinoma", "Ependymoma", "Ganglioglioma",
    "Germinoma", "Glioblastoma", "Granuloma", "Medulloblastoma",
    "Meningioma", "Neurocytoma", "Oligodendroglioma", "Papilloma",
    "Schwannoma", "Tuberculoma", "Otak Sehat",
]

# ── Load Model ─────────────────────────────────────────────
model = None

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] {MODEL_PATH} tidak ditemukan. Berjalan dalam mode demo.")
        inp = tf.keras.Input(shape=(299, 299, 3))
        out = tf.keras.layers.GlobalAveragePooling2D()(
            tf.keras.applications.MobileNetV2(include_top=False, weights=None)(inp))
        out = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(out)
        model = tf.keras.Model(inp, out)
        return False
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return True

model_loaded = load_model()

# ── Preprocessing ──────────────────────────────────────────
def preprocess(img_pil):
    img = img_pil.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Grad-CAM ───────────────────────────────────────────────
def get_gradcam(img_array):
    try:
        xception_layer = next(
            (l for l in model.layers if "xception" in l.name.lower()), None
        )
        if xception_layer is None:
            return None

        last_conv = next(
            (l for l in reversed(xception_layer.layers)
             if len(l.output.shape) == 4), None
        )
        if last_conv is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs=xception_layer.input,
            outputs=[xception_layer.get_layer(last_conv.name).output,
                     xception_layer.output],
        )

        img_tensor = tf.cast(img_array, tf.float32)
        with tf.GradientTape() as tape:
            conv_out, xception_out = grad_model(img_tensor)
            tape.watch(conv_out)
            x = xception_out
            head_started = False
            for layer in model.layers:
                if head_started:
                    x = layer(x, training=False)
                if layer.name == xception_layer.name:
                    head_started = True
            top_class = int(tf.argmax(x[0]))
            score = x[:, top_class]

        grads   = tape.gradient(score, conv_out)
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        print(f"[Grad-CAM Error] {e}")
        return None


def overlay_gradcam(img_pil, heatmap, alpha=0.45):
    img_rgb    = np.array(img_pil.convert("RGB").resize(IMG_SIZE))
    hm_resized = cv2.resize(heatmap, IMG_SIZE)
    hm_uint8   = np.uint8(255 * hm_resized)
    hm_color   = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    hm_color   = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    overlay    = (alpha * hm_color + (1 - alpha) * img_rgb).astype(np.uint8)
    return overlay


def pil_to_b64(img_pil):
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ndarray_to_b64(arr):
    return pil_to_b64(Image.fromarray(arr.astype(np.uint8)))


def plot_confidence_b64(probs, top_n=5):
    top_idx    = np.argsort(probs)[::-1][:top_n]
    top_probs  = probs[top_idx]
    top_labels = [CLASSES[i] for i in top_idx]

    colors = []
    for p in top_probs:
        if p >= 0.70:   colors.append("#0EA5A0")
        elif p >= 0.40: colors.append("#F59E0B")
        else:           colors.append("#EF4444")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6, max(2.5, top_n * 0.58)))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    ax.barh(top_labels[::-1], top_probs[::-1] * 100,
            color=colors[::-1], edgecolor="none", height=0.55)

    for i, (label, prob) in enumerate(zip(top_labels[::-1], top_probs[::-1])):
        ax.text(prob * 100 + 0.5, i, f"{prob*100:.1f}%",
                va="center", ha="left", fontsize=9,
                color="#374151", fontfamily="monospace")

    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", color="#6B7280", fontsize=9)
    ax.tick_params(colors="#374151", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(x=50, color="#E5E7EB", linewidth=1, linestyle="--")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#FAFAFA")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ── Routes ─────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           classes=CLASSES,
                           model_loaded=model_loaded,
                           model_path=MODEL_PATH)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang dikirim."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Tidak ada file yang dipilih."}), 400

    try:
        img_pil = Image.open(file.stream)
        arr     = preprocess(img_pil)

        probs      = model.predict(arr, verbose=0)[0]
        pred_idx   = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        # Confidence level
        if confidence >= 0.70:
            conf_level = "high"
        elif confidence >= 0.40:
            conf_level = "medium"
        else:
            conf_level = "low"

        # Top-5
        top5_idx = np.argsort(probs)[::-1][:5]
        top5 = [
            {"class": CLASSES[i], "prob": round(float(probs[i]) * 100, 2)}
            for i in top5_idx
        ]

        # All classes sorted
        all_sorted = [
            {"class": CLASSES[i], "prob": round(float(probs[i]) * 100, 2)}
            for i in np.argsort(probs)[::-1]
        ]

        # Gambar original (resize)
        img_resized = img_pil.convert("RGB").resize(IMG_SIZE)
        orig_b64    = pil_to_b64(img_resized)

        # Grad-CAM
        heatmap      = get_gradcam(arr)
        heatmap_b64  = None
        overlay_b64  = None
        if heatmap is not None:
            hm_uint8    = np.uint8(255 * cv2.resize(heatmap, IMG_SIZE))
            hm_color    = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
            hm_color    = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
            heatmap_b64 = ndarray_to_b64(hm_color)
            overlay_arr = overlay_gradcam(img_pil, heatmap)
            overlay_b64 = ndarray_to_b64(overlay_arr)

        # Chart
        chart_b64 = plot_confidence_b64(probs, top_n=5)

        return jsonify({
            "pred_class": pred_class,
            "confidence": round(confidence * 100, 2),
            "conf_level": conf_level,
            "pred_idx":   pred_idx + 1,
            "top5":       top5,
            "all":        all_sorted,
            "orig_b64":   orig_b64,
            "heatmap_b64":  heatmap_b64,
            "overlay_b64":  overlay_b64,
            "chart_b64":    chart_b64,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Hanya dipakai untuk testing lokal.
    # Di production, Gunicorn yang menjalankan `app` (lihat Dockerfile / Procfile).
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port)