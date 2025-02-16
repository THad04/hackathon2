import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import json

# Define dataset path
DATASET_DIR = "dataset/newdataset/asl_alphabet_train/asl_alphabet_train"

# Get class names from the dataset folder (filter out hidden files like .DS_Store)
class_names = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
print(f"🔍 Detected {len(class_names)} unique labels: {class_names}")

# Create a label map (indexing starts from 0)
label_map = {class_name: i for i, class_name in enumerate(class_names)}
print(f"📌 Label Mapping: {label_map}")

# Initialize lists for images and labels
images = []
labels = []

# Load images from each class folder
for class_name in class_names:
    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue  # Skip if it's not a directory

    print(f"📂 Processing {class_name}...")

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"❌ Skipping invalid image: {img_path}")
            continue

        # Resize to (64,64) to match model input shape
        image = cv2.resize(image, (64, 64))
        image = image / 255.0  # Normalize pixel values

        images.append(image)
        labels.append(label_map[class_name])  # Store class index

# Convert lists to NumPy arrays
images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# Ensure no empty dataset issues
if len(images) == 0:
    raise ValueError("❌ No valid images found! Check dataset directory structure.")

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)

print(f"✅ Dataset Loaded — Train: {X_train.shape}, Test: {X_test.shape}")

# Save processed data for training
np.savez("dataset/preprocessed_data.npz", 
         train_images=X_train, train_labels=y_train, 
         test_images=X_test, test_labels=y_test, 
         label_map=label_map)

# Save label map as JSON for easy debugging
with open("dataset/label_map.json", "w") as f:
    json.dump(label_map, f)
print("📁 Saved label map for inference.")

print("✅ Preprocessed dataset saved successfully!")