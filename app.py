# ============================================
# app.py — Photon Backend (OCR + AI Assistant + Vision)
# ============================================
# Requirements:
#   pip install flask flask-cors flask-socketio eventlet openai google-cloud-vision
# Environment variables (Render Dashboard):
#   GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account.json>
#   OPENAI_API_KEY=<your_openai_api_key>
#   PHOTON_OCR_MODEL=final_ocr_model_v5.keras
#   PHOTON_OCR_CLASSES=classes_final.npy
#   CORS_ORIGIN=https://photon-frontend-v2.onrender.com
# ============================================

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import vision
from openai import OpenAI
import eventlet

# --- Flask setup ---
app = Flask(__name__)

# ✅ Dynamic, Render-safe CORS
CORS(app, supports_credentials=True)

@app.after_request
def apply_cors(response):
    origin = request.headers.get("Origin")
    allowed = [
        "https://photon-frontend-v2.onrender.com",
        "http://localhost:3000"
    ]
    if origin in allowed:
        response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


# --- Environment setup ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY not found — check Render environment settings!")

GOOGLE_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not GOOGLE_CREDENTIALS:
    print("⚠️ GOOGLE_APPLICATION_CREDENTIALS not set — Vision API will fail if used.")


# --- Initialize clients ---
try:
    vision_client = vision.ImageAnnotatorClient()
except Exception as e:
    vision_client = None
    print(f"⚠️ Google Vision init failed: {e}")

try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    openai_client = None
    print(f"⚠️ OpenAI init failed: {e}")


# ============================================
# ROUTES
# ============================================

@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "Photon backend active"})


@app.route("/healthz")
def healthz():
    """Health check endpoint for Render"""
    return jsonify({"status": "healthy"}), 200


# --- Preflight handler for AI route ---
@app.route("/ask_ai", methods=["OPTIONS"])
def ask_ai_preflight():
    return '', 200


@app.route("/ask_ai", methods=["POST"])
def ask_ai():
    """Main AI assistant endpoint"""
    data = request.get_json()
    user_prompt = data.get("prompt", "").strip()
    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400

    if not openai_client:
        return jsonify({"error": "OpenAI not initialized"}), 500

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are Photon, an AI code assistant that helps debug and improve code concisely."},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = response.choices[0].message.content
        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- OCR (Google Vision) ---
@app.route("/ocr", methods=["POST"])
def ocr_image():
    """OCR using Google Vision"""
    if not vision_client:
        return jsonify({"error": "Google Vision client not initialized"}), 500

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        image_file = request.files["file"]
        content = image_file.read()
        image = vision.Image(content=content)

        response = vision_client.text_detection(image=image)
        texts = response.text_annotations

        if not texts:
            return jsonify({"text": "", "confidence": 0})

        detected_text = texts[0].description
        confidence = round(
            sum([d.confidence for d in response.text_annotations[1:]]) / max(1, len(response.text_annotations[1:])), 2
        )
        return jsonify({"text": detected_text, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Preflight + alias for process_images ---
@app.route("/process_images", methods=["POST", "OPTIONS"])
def process_images_alias():
    if request.method == "OPTIONS":
        return '', 200
    return ocr_image()


# ============================================
# MAIN ENTRY
# ============================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Photon backend running on port {port}")
    from eventlet import wsgi, listen
    wsgi.server(listen(("0.0.0.0", port)), app)



