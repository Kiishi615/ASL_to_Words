# pyrefly: ignore [missing-import]
import os
import pickle
import threading

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Lazy-load model — Flask binds to port immediately so HF health check passes
model = None
model_lock = threading.Lock()


def get_model():
    """Load model on first use (thread-safe)."""
    global model
    if model is None:
        with model_lock:
            if model is None:  # double-check after acquiring lock
                print("Loading model...")
                with open("model.p", "rb") as f:
                    model_dict = pickle.load(f)
                model = model_dict["model"]
                print("Model loaded successfully!")
    return model


# Start loading in background thread so it's ready faster
threading.Thread(target=get_model, daemon=True).start()


@app.route("/")
def index():
    """Serve the frontend HTML page."""
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    """Health check — always responds immediately."""
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept normalized hand landmarks from the browser-side MediaPipe
    and return a sign language prediction.

    Expects JSON: { "landmarks": [42 floats] }
    Returns JSON:  { "success": true, "prediction": "A" }
    """
    m = get_model()
    if m is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        landmarks = data.get("landmarks", [])

        if len(landmarks) != 42:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Expected 42 landmark values, got {len(landmarks)}",
                    }
                ),
                400,
            )

        prediction = m.predict([np.asarray(landmarks)])
        predicted_character = str(prediction[0])

        return jsonify({"success": True, "prediction": predicted_character})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting ASL Recognition API on port {port}...")
    app.run(host="0.0.0.0", port=port)
