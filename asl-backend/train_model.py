import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

# Load preprocessed dataset
data = np.load("dataset/preprocessed_data.npz")
train_images, train_labels = data["train_images"], data["train_labels"]
test_images, test_labels = data["test_images"], data["test_labels"]

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(25, activation='softmax')  # 25 classes (A-Y, no J/Z due to motion)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Save the trained model
model.save("asl_model.h5")
print("✅ Model training complete! Saved as 'asl_model.h5'")
