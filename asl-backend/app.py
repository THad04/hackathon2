import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load trained ASL model
model = load_model("asl_model.h5")

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

    # Read the image
    image = cv2.imread(file_path)
    if image is None:
        return jsonify({"error": "Invalid image file"}), 400

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Convert to grayscale & preprocess for model
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (28, 28)) / 255.0  # Normalize
            input_data = resized.reshape(1, 28, 28, 1)

            # Predict ASL letter
            prediction = model.predict(input_data)
            predicted_label = np.argmax(prediction)
            predicted_letter = chr(65 + predicted_label)  # Convert to A-Z letter

            # Delete image after processing
            os.remove(file_path)

            return jsonify({"predicted_letter": predicted_letter})

    # Delete image if no hand was detected
    os.remove(file_path)
    return jsonify({"error": "No hand detected"})

if __name__ == "__main__":
    app.run(debug=True)
