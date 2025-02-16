from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import json

# Load preprocessed dataset
data = np.load("dataset/preprocessed_data.npz", allow_pickle=True)

# Extract dataset
X_train, y_train = data["train_images"], data["train_labels"]
X_test, y_test = data["test_images"], data["test_labels"]
label_map = data["label_map"].item()  # Convert numpy object back to dictionary

# Check dataset shape
print(f"✅ Dataset Loaded - Train: {X_train.shape}, Test: {X_test.shape}")

# Ensure labels are integers
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Print all unique labels
unique_labels = np.unique(y_train)
print(f"🧑‍🏫 Unique Labels Found: {unique_labels}")

# Adjust num_classes dynamically
num_classes = len(unique_labels)  # Ensure correct number of classes
print(f"🔢 Adjusted num_classes: {num_classes}")

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Save label map for inference
with open("dataset/label_map.json", "w") as f:
    json.dump(label_map, f)
print("📁 Saved label map for inference.")

# Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),  # Adjusted input shape
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')  # Output dynamically adjusts to dataset
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define model checkpoint callback to save best model
checkpoint = ModelCheckpoint("asl_model_best.h5", monitor='val_accuracy', save_best_only=True, mode='max')

# Train model
print("🚀 Training started...")
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), callbacks=[checkpoint])

# Save final trained model
model.save("asl_model.h5")
print("✅ Model training complete! Saved as 'asl_model.h5'")