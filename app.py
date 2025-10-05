# app.py — Photon OCR + AI Assistant (Google Vision + optional Keras + SocketIO)

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from dotenv import load_dotenv
import os, io, contextlib, numpy as np

# Google Vision
from google.cloud import vision
# OpenAI (for indentation / AI assistant)
from openai import OpenAI

# ── ENV SETUP ───────────────────────────────────────────────────────────────
load_dotenv()

# Use service account if available
gcp_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if gcp_key:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_key

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ── Optional Keras OCR model ────────────────────────────────────────────────
KERAS_AVAILABLE = True
keras_model = None
idx_to_label = {}
IMG_H, IMG_W, IMG_C = 28, 28, 1

try:
    import cv2
    import tensorflow as tf
    from tensorflow.keras.models import load_model as tfk_load_model

    MODEL_PATH = (os.getenv("PHOTON_OCR_MODEL") or "").strip()
    CLASSES_PATH = (os.getenv("PHOTON_OCR_CLASSES") or "").strip()

    if MODEL_PATH and os.path.isfile(MODEL_PATH):
        keras_model = tfk_load_model(MODEL_PATH, compile=False)
        shape = keras_model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        _, IMG_H, IMG_W, IMG_C = shape
        print(f"✅ Loaded Keras OCR model '{MODEL_PATH}' (input {IMG_H}x{IMG_W}x{IMG_C})")

    if CLASSES_PATH and os.path.isfile(CLASSES_PATH):
        classes = np.load(CLASSES_PATH, allow_pickle=True)
        if isinstance(classes, np.ndarray):
            classes = classes.tolist()
        idx_to_label = {i: str(lbl) for i, lbl in enumerate(classes)}
        print(f"✅ Loaded class labels '{CLASSES_PATH}'")

except Exception as e:
    KERAS_AVAILABLE = False
    keras_model = None
    print("ℹ️ TensorFlow/Keras not available; Vision-only OCR will be used.", e)

# ── APP + SOCKET.IO INIT ────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # allow all origins for stability
socketio = SocketIO(app, cors_allowed_origins="*")

# ── GOOGLE VISION INIT ─────────────────────────────────────────────────────
try:
    vision_client = vision.ImageAnnotatorClient()
    print("✅ Google Vision initialized")
except Exception as e:
    print("❌ Failed to init Google Vision:", e)
    raise e

# ── HELPERS ────────────────────────────────────────────────────────────────
def gpt_indent(raw_code: str) -> str:
    if not client:
        return raw_code
    msg = (
        "You are a Python formatter. Fix only the indentation of this code. "
        "Keep any typos or syntax errors.\n\n"
        f"```python\n{raw_code}\n```"
    )
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": msg}
        ],
        temperature=0,
        max_tokens=800,
    )
    text = resp.choices[0].message.content or ""
    return text.replace("```python", "").replace("```", "").strip()

def vision_document_text(image_bytes: bytes) -> tuple[str, float]:
    """High-accuracy Vision OCR using document_text_detection."""
    try:
        img = vision.Image(content=image_bytes)
        res = vision_client.document_text_detection(image=img)
        if res.error and res.error.message:
            return "", 0.0

        text, confs = "", []
        if getattr(res, "full_text_annotation", None) and res.full_text_annotation.text:
            text = res.full_text_annotation.text
            for p in res.full_text_annotation.pages:
                for b in p.blocks:
                    for para in b.paragraphs:
                        for w in para.words:
                            confs.append(w.confidence or 0.0)
        elif getattr(res, "text_annotations", None):
            text = res.text_annotations[0].description or ""

        avg_conf = (sum(confs) / len(confs)) if confs else 0.0
        return text.strip(), float(avg_conf)
    except Exception as e:
        print("Vision error:", e)
        return "", 0.0

def keras_ocr_text(image_bytes: bytes) -> str:
    """Fallback OCR using CNN model (if loaded)."""
    if not (KERAS_AVAILABLE and keras_model is not None):
        return ""
    try:
        import cv2
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ""
        img = cv2.resize(img, (int(IMG_W), int(IMG_H)))
        x = img.astype(np.float32) / 255.0
        if IMG_C == 1:
            x = x[..., np.newaxis]
        x = np.expand_dims(x, 0)
        preds = keras_model.predict(x, verbose=0)
        idx = int(np.argmax(preds, axis=1)[0])
        return idx_to_label.get(idx, str(idx))
    except Exception as e:
        print("Keras OCR error:", e)
        return ""

# ── ROUTES ─────────────────────────────────────────────────────────────────
@app.route("/process_images", methods=["POST"])
def process_images():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    auto_indent = (request.form.get("auto_indent", "false").lower() == "true")
    combined = []
    per_file = []

    for f in files:
        try:
            raw = f.read()
            v_text, v_conf = vision_document_text(raw)
            chosen = v_text or "# (No text detected)"
            per_file.append({"file": f.filename, "vision_conf": round(v_conf, 3)})
            combined.append(chosen)
        except Exception as e:
            per_file.append({"file": f.filename, "error": str(e)})
            combined.append(f"# Error processing {f.filename}: {str(e)}")

    merged = "\n".join(t for t in (s.strip() for s in combined) if t)
    if auto_indent and merged:
        try:
            merged = gpt_indent(merged) or merged
        except Exception as e:
            per_file.append({"error": f"indent_failed: {e}"})

    confs = [x.get("vision_conf", 0.0) for x in per_file]
    est = int(round(100 * (sum(confs) / max(1, len(confs))))) if confs else 0

    return jsonify({
        "extracted_text": merged,
        "estimated_accuracy": est,
        "details": per_file
    })

@app.route("/run_with_input", methods=["POST"])
def run_with_input():
    data = request.json or {}
    code = data.get("code", "")
    user_input = data.get("input", "")

    try:
        buffer = io.StringIO()
        input_lines = user_input.strip().splitlines()

        def input_patch(prompt=""):
            return input_lines.pop(0) if input_lines else ""

        exec_globals = {'input': input_patch}
        with contextlib.redirect_stdout(buffer):
            exec(code, exec_globals)
        return jsonify({"result": buffer.getvalue()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask_ai", methods=["POST"])
def ask_ai():
    data = request.json or {}
    prompt = data.get("message", "") or ""
    context_code = data.get("context_code", "") or ""
    if not client:
        return jsonify({'response': "OpenAI key not configured."})
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful Python tutor."},
                {"role": "user", "content": f"Code:\n```python\n{context_code}\n```\n\n{prompt}"}
            ],
            temperature=0.3,
            max_tokens=600,
        )
        return jsonify({'response': resp.choices[0].message.content})
    except Exception as e:
        return jsonify({'response': f"Error: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def home():
    return "✅ Photon OCR + AI Assistant backend is live!"

@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"}), 200

# ── BOOT ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
