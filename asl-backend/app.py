import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from flask_cors import CORS
import os
import json

app = Flask(__name__)
CORS(app)

# Load trained ASL model
model = load_model("asl_model.h5")

# Load label map correctly from JSON
with open("dataset/label_map.json", "r") as f:
    label_map = json.load(f)

# Ensure label_map values are sorted properly for indexing
index_to_label = {v: k for k, v in label_map.items()}  # Reverse mapping
print(f"✅ Label Mapping Loaded: {index_to_label}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

@app.route("/detect", methods=["POST"])
def detect_asl():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files["image"]
    file_path = "uploaded_image.jpg"
    file.save(file_path)

    # Read and preprocess the image
    image = cv2.imread(file_path)
    if image is None:
        os.remove(file_path)
        return jsonify({"error": "Invalid image file"}), 400

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (64, 64)) / 255.0  # Resize & Normalize
    input_data = np.expand_dims(image_resized, axis=0)  # Shape (1, 64, 64, 3)

    # Predict ASL letter
    prediction = model.predict(input_data)
    predicted_index = np.argmax(prediction)

    print(f"🔍 Prediction Index: {predicted_index}")
    print(f"📌 Available Labels: {index_to_label}")

    # Ensure predicted index is valid
    if predicted_index not in index_to_label:
        os.remove(file_path)
        return jsonify({"error": "Prediction index out of range"}), 500

    predicted_letter = index_to_label[predicted_index]  # Convert index to letter

    print(f"🔥 Final Prediction: {predicted_letter} (Confidence: {np.max(prediction):.2f})")

    os.remove(file_path)
    return jsonify({"predicted_letter": predicted_letter})

if __name__ == "__main__":
    app.run(debug=True)