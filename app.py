from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

# -------------------------
# 1. Load model
# -------------------------
MODEL_PATH = "./model.pkl"  # Replace with your model path

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# -------------------------
# 2. Create Flask app
# -------------------------
app = Flask(__name__)

# -------------------------
# 3. Health check endpoint
# -------------------------
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "Model API is running"}), 200

# -------------------------
# 4. Single prediction endpoint
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON:
    {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    """
    data = request.get_json()

    if not data or "features" not in data:
        return jsonify({"error": "No features provided"}), 400

    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)

    return jsonify({"prediction": prediction.tolist()}), 200

# -------------------------
# 5. Batch prediction endpoint
# -------------------------
@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Expected JSON:
    {
        "data": [
            [5.1, 3.5, 1.4, 0.2],
            [6.2, 3.4, 5.4, 2.3]
        ]
    }
    """
    data = request.get_json()

    if not data or "data" not in data:
        return jsonify({"error": "No data provided"}), 400

    features = np.array(data["data"])
    predictions = model.predict(features)

    return jsonify({"predictions": predictions.tolist()}), 200

# -------------------------
# 6. Run the Flask app
# -------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
