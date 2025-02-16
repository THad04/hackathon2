import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

# Load model
model = load_model('asl_model.h5')

# Load label map
with open('dataset/label_map.json', 'r') as f:
    label_map = json.load(f)
index_to_label = {v: k for k, v in label_map.items()}  # Reverse mapping

# Load test image
image_path = 'dataset/newdataset/asl_alphabet_train/asl_alphabet_train/A/A1.jpg'
image = cv2.imread(image_path)
if image is None:
    print(f"❌ Error: Could not load image from {image_path}")
    exit()

print(f"📸 Image Shape Before Processing: {image.shape}")

# Convert to RGB and resize
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (64, 64)) / 255.0
input_data = np.expand_dims(image_resized, axis=0)

# **Now we can print pixel values safely**
print(f"🖼️ First 5 Pixels of Processed Image: {input_data[0, 0, :5]}")

# Make prediction
prediction = model.predict(input_data)
predicted_index = np.argmax(prediction)

print(f"📌 Predicted Index: {predicted_index}")
print(f"🔥 Prediction: {index_to_label.get(predicted_index, 'Unknown')} (Confidence: {np.max(prediction):.2f})")
