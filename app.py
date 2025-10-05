# app.py — Photon OCR API (Vision-only mode, no Keras for low memory)

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os, io, re, contextlib
import numpy as np
from google.cloud import vision
from openai import OpenAI

# ── ENV ───────────────────────────────────────────────────────────────────────
load_dotenv()

# Google credentials and OpenAI setup
gcp_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if gcp_key:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_key

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

if OPENAI_API_KEY:
    try:
        client.models.list()  # test if key works
        print("✅ OpenAI initialized successfully")
    except Exception as e:
        print("❌ OpenAI init failed:", e)
else:
    print("⚠️ No OpenAI key found in environment")

# ── DISABLE KERAS ─────────────────────────────────────────────────────────────
KERAS_AVAILABLE = False
keras_model = None
idx_to_label = {}
IMG_H, IMG_W, IMG_C = 28, 28, 1
print("🚫 Skipping Keras OCR — running in Vision-only mode.")

# ── APP ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=["https://photon-frontend-v2.onrender.com"])

# ── GOOGLE VISION CLIENT ──────────────────────────────────────────────────────
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
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": msg},
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


# ── CODE SCORING ──────────────────────────────────────────────────────────────
CODE_CHARS = set(list("()[]{}<>:=+*-_/\\.,'\"#@%!&|^~`;$"))
CODE_KWS = {
    "def", "class", "import", "for", "while", "if", "elif", "else", "try",
    "except", "return", "print", "with", "as", "from"
}

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

# ── ROUTES ────────────────────────────────────────────────────────────────────
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

            per_file.append({
                "file": f.filename,
                "engine_selected": "vision",
                "vision_conf": round(v_conf, 3),
                "chosen_len": len(v_text or ""),
            })

            combined.append(v_text or f"# (No text detected in {f.filename})")

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
        "engine": "vision",
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
    context_input = data.get("context_input", "") or ""
    context_output = data.get("context_output", "") or ""
    if not client:
        return jsonify({'response': "OpenAI key not configured on server."})
    policy = "You are a strict Python-only assistant. Refuse non-Python language requests."
    user_prompt = (
        f"{policy}\n\nUser's current Python code:\n```python\n{context_code}\n```\n\n"
        + (f"Provided input:\n```\n{context_input}\n```\n\n" if context_input else "")
        + (f"Observed output:\n```\n{context_output}\n```\n\n" if context_output else "")
        + f"Question:\n{prompt}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful Python tutor."},
                {"role": "user", "content": user_prompt},
            ],
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
    return "✅ Photon OCR + Python Assistant (Vision-only) is live!"


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"}), 200


# ── BOOT ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)

