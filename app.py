from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
import logging
import tempfile
import requests

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------
# Configuration
# ----------------------
MODEL_FILENAME = "best5.pt"
MODEL_PATH_OVERRIDE = os.environ.get("MODEL_PATH_OVERRIDE")

if MODEL_PATH_OVERRIDE:
    MODEL_PATH = os.path.abspath(MODEL_PATH_OVERRIDE)
else:
    MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)

# ★★★★★ YOUR GITHUB RELEASE URL ★★★★★
MODEL_URL = "https://github.com/yourname/yourrepo/releases/download/v2.0/best5.pt"

CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))

# ----------------------
# Download model
# ----------------------
def download_model():
    if MODEL_PATH_OVERRIDE and os.path.exists(MODEL_PATH):
        logger.info("Using overridden model path: %s", MODEL_PATH)
        return

    if os.path.exists(MODEL_PATH):
        logger.info("Model already exists locally.")
        return

    logger.info("Downloading model from %s", MODEL_URL)
    try:
        r = requests.get(MODEL_URL, stream=True)
        r.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("Model downloaded successfully.")
    except Exception as e:
        logger.exception("Error downloading model: %s", e)
        raise RuntimeError("Unable to download model")


# ----------------------
# Load YOLO model
# ----------------------
model = None
_model_loaded = False

try:
    download_model()

    model = YOLO(MODEL_PATH)   # ✔ Correct YOLO loading
    model.to("cpu")

    _model_loaded = True
    logger.info("YOLO model loaded successfully.")

except Exception as e:
    logger.exception("Failed to load YOLO model: %s", e)
    model = None


# ----------------------
# Prediction
# ----------------------
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400

    try:
        file = request.files['image']

        # Temp image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            file.save(tmp.name)
            img_path = tmp.name

        # YOLO Classification
        results = model.predict(
            source=img_path,
            imgsz=640,
            conf=0.25
        )

        r = results[0]
        probs = r.probs

        idx = probs.top1
        confidence = float(probs.top1conf)
        class_name = model.names[idx]

        # confidence gate
        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                'error': 'Low confidence.',
                'confidence': confidence,
                'probabilities': probs.data.tolist(),
                'classes': model.names
            }), 400

        return jsonify({
            "prediction": class_name,
            "confidence": confidence,
            "probabilities": probs.data.tolist(),
            "classes": model.names
        })

    except Exception as e:
        logger.exception("Prediction error: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        'model_loaded': _model_loaded,
        'model_path': MODEL_PATH,
        'model_url': MODEL_URL,
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
