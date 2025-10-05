# app.py — Photon OCR API (Vision-first, optional Keras fallback)

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os, io, re, contextlib

# Numeric / CV (Keras is optional)
import numpy as np

# Google Vision
from google.cloud import vision

# OpenAI (only for indent endpoint)
from openai import OpenAI

# ── ENV ───────────────────────────────────────────────────────────────────────
load_dotenv()

# Respect a provided service account path
gcp_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if gcp_key:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_key

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Optional Keras (will run fine without it)
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
        try:
            keras_model = tfk_load_model(MODEL_PATH, compile=False)
            shape = keras_model.input_shape
            if isinstance(shape, list):
                shape = shape[0]
            _, IMG_H, IMG_W, IMG_C = shape
            print(f"✅ Loaded Keras OCR model '{MODEL_PATH}' (input {IMG_H}x{IMG_W}x{IMG_C})")
        except Exception as e:
            print("⚠️ Could not load Keras model:", e)
            keras_model = None

    if CLASSES_PATH and os.path.isfile(CLASSES_PATH):
        classes = np.load(CLASSES_PATH, allow_pickle=True)
        if isinstance(classes, np.ndarray):
            classes = classes.tolist()
        idx_to_label = {i: str(lbl) for i, lbl in enumerate(classes)}
        print(f"✅ Loaded class labels '{CLASSES_PATH}'")

except Exception:
    KERAS_AVAILABLE = False
    keras_model = None
    print("ℹ️ TensorFlow/Keras not available; Vision-only OCR will be used.")

# ── APP ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=["https://photon-frontend-v2.onrender.com"])

# Google Vision client
try:
    vision_client = vision.ImageAnnotatorClient()
    print("✅ Google Vision initialized")
except Exception as e:
    print("❌ Failed to init Google Vision:", e)
    raise e

# ── HELPERS ───────────────────────────────────────────────────────────────────
def gpt_indent(raw_code: str) -> str:
    if not client:
        return raw_code
    msg = (
        "You are a Python formatter. Fix only the indentation of this code. "
        "Keep any typos, syntax errors, or incomplete lines.\n\n"
        f"```python\n{raw_code}\n```"
    )
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are helpful."},
                  {"role": "user", "content": msg}],
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
    """Very simple per-image classifier fallback (optional)."""
    if not (KERAS_AVAILABLE and keras_model is not None):
        return ""
    try:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ""
        img = cv2.resize(img, (int(IMG_W), int(IMG_H)))
        x = img.astype(np.float32) / 255.0
        if IMG_C == 1:
            x = x[..., np.newaxis]
        x = np.expand_dims(x, 0)  # (1, H, W, C)
        preds = keras_model.predict(x, verbose=0)
        idx = int(np.argmax(preds, axis=1)[0])
        return idx_to_label.get(idx, str(idx))
    except Exception as e:
        print("Keras OCR error:", e)
        return ""

CODE_CHARS = set(list("()[]{}<>:=+*-_/\\.,'\"#@%!&|^~`;$"))
CODE_KWS = {"def", "class", "import", "for", "while", "if", "elif", "else", "try", "except", "return", "print", "with", "as", "from"}

def code_score(s: str) -> float:
    if not s:
        return 0.0
    s = s.strip()
    if not s:
        return 0.0
    sym = sum(c in CODE_CHARS for c in s) / len(s)
    kws = sum(1 for k in CODE_KWS if k in s.lower())
    lines = [ln for ln in s.splitlines() if ln.strip()]
    colon = sum(ln.rstrip().endswith(":") for ln in lines) / max(1, len(lines))
    indent = sum(ln.startswith((" ", "\t")) for ln in lines) / max(1, len(lines))
    length = min(len(s) / 200.0, 1.0)
    return 0.5 * sym + 0.2 * colon + 0.15 * indent + 0.15 * length + 0.2 * (kws > 0)

def choose_text(vision_text: str, v_conf: float, keras_text: str) -> tuple[str, str, float]:
    """Pick the better of Vision or Keras (Vision wins with high conf)."""
    v_score = code_score(vision_text) + 0.25 * max(0.0, min(v_conf, 1.0))
    k_score = code_score(keras_text)
    if (v_conf >= 0.75 and len(vision_text) >= 6) or v_score >= k_score:
        return vision_text, "vision", v_conf
    return keras_text, "keras", 0.0

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/process_images", methods=["POST"])
def process_images():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    auto_indent = (request.form.get("auto_indent", "false").lower() == "true")
    engine = (request.form.get("engine") or "auto").lower()  # 'vision' | 'keras' | 'auto'

    combined = []
    per_file = []

    for f in files:
        try:
            raw = f.read()
            v_text, v_conf = ("", 0.0)
            k_text = ""

            if engine in ("vision", "auto"):
                v_text, v_conf = vision_document_text(raw)
            if engine in ("keras", "auto"):
                k_text = keras_ocr_text(raw)

            if engine == "vision":
                chosen, src, chosen_conf = v_text, "vision", v_conf
            elif engine == "keras":
                chosen, src, chosen_conf = k_text, "keras", 0.0
            else:
                chosen, src, chosen_conf = choose_text(v_text, v_conf, k_text)

            per_file.append({
                "file": f.filename,
                "engine_selected": src,
                "vision_conf": round(v_conf, 3),
                "vision_len": len(v_text or ""),
                "keras_len": len(k_text or ""),
                "chosen_len": len(chosen or ""),
            })

            if chosen:
                combined.append(chosen)
            else:
                combined.append(f"# (No text detected in {f.filename})")

        except Exception as e:
            per_file.append({"file": f.filename, "error": str(e)})
            combined.append(f"# Error processing {f.filename}: {str(e)}")

    merged = "\n".join(t for t in (s.strip() for s in combined) if t)

    if auto_indent and merged:
        try:
            merged = gpt_indent(merged) or merged
        except Exception as e:
            per_file.append({"error": f"indent_failed: {e}"})

    # Estimated accuracy: use mean Vision confidence when we used/checked Vision
    confs = [x.get("vision_conf", 0.0) for x in per_file]
    est = int(round(100 * (sum(confs) / max(1, len(confs))))) if confs else 0

    return jsonify({
        "extracted_text": merged,
        "engine": engine,
        "estimated_accuracy": est,    # 0–100
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
    context_input = data.get("context_input", "") or ""
    context_output = data.get("context_output", "") or ""
    if not client:
        return jsonify({'response': "OpenAI key not configured on server."})
    policy = (
        "You are a strict Python-only assistant. Refuse non-Python language requests."
    )
    user_prompt = (
        f"{policy}\n\n"
        f"User's current Python code:\n```python\n{context_code}\n```\n\n"
        + (f"Provided input:\n```\n{context_input}\n```\n\n" if context_input else "")
        + (f"Observed output:\n```\n{context_output}\n```\n\n" if context_output else "")
        + f"Question:\n{prompt}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful Python tutor."},
                      {"role": "user", "content": user_prompt}],
            temperature=0.3,
            max_tokens=600,
        )
        return jsonify({'response': resp.choices[0].message.content})
    except Exception as e:
        return jsonify({'response': f"Error: {str(e)}"}), 500

@app.route("/indent_with_gpt", methods=["POST"])
def indent_with_gpt_api():
    data = request.json or {}
    raw_code = data.get('code', '') or ""
    try:
        formatted = gpt_indent(raw_code)
        return jsonify({"formatted_code": formatted})
    except Exception as e:
        return jsonify({"formatted_code": raw_code, "error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "✅ Photon OCR + Python Assistant is live!"

@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"}), 200


# ── BOOT ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)


