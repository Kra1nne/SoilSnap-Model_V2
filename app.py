from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import tempfile
import requests
import io

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

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

MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/yourname/yourrepo/releases/download/v1.0/best5.pt"
)

CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))

# ----------------------
# Architecture
# ----------------------
class SoilModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224*224*3, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.model(x)


# ----------------------
# Download model
# ----------------------
def download_model():
    if MODEL_PATH_OVERRIDE and os.path.exists(MODEL_PATH):
        logger.info("Using overridden model path %s", MODEL_PATH)
        return

    if os.path.exists(MODEL_PATH):
        logger.info("Model already exists locally.")
        return

    logger.info("Downloading model from %s", MODEL_URL)
    try:
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logger.info("Model downloaded successfully.")
    except Exception as e:
        logger.exception("Error downloading model: %s", e)
        raise RuntimeError("Unable to download model")


# ----------------------
# Load model
# ----------------------
model = None
_model_loaded = False
try:
    download_model()

    model = SoilModel()
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    _model_loaded = True
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.exception("Failed to load PyTorch model: %s", e)
    model = None


# ----------------------
# Preprocessing
# ----------------------
def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))

    arr = np.array(img) / 255.0
    arr = arr.astype(np.float32).transpose(2, 0, 1)  # CHW
    return torch.tensor(arr).unsqueeze(0)


class_names = [
    'Clay', 'Clay Loam', 'Loam', 'Loamy Sand', 'Non-Soil', 'Sand', 'Sandy Loam', 'Silt', 'Silty Clay', 'Silty Loam'    
]


# ----------------------
# Prediction
# ----------------------
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file'}), 400

        image_bytes = request.files['image'].read()
        x = preprocess(image_bytes)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].numpy()

        idx = int(np.argmax(probs))
        predicted_class = class_names[idx]
        confidence = float(probs[idx])

        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                'error': 'Image does not appear to be soil.',
                'confidence': confidence,
                'probabilities': probs.tolist(),
                'classes': class_names
            }), 400

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': probs.tolist(),
            'classes': class_names
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
