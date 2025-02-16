import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
train_data = pd.read_csv("dataset/sign_mnist_train.csv")
test_data = pd.read_csv("dataset/sign_mnist_test.csv")

# Extract labels and images
train_labels = train_data.iloc[:, 0].values  # First column is the label (A-Z)
train_images = train_data.iloc[:, 1:].values  # Remaining columns are pixel values

test_labels = test_data.iloc[:, 0].values
test_images = test_data.iloc[:, 1:].values

# Normalize pixel values (0-1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape into 28x28 images (grayscale)
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Display a sample image
plt.imshow(train_images[0].reshape(28, 28), cmap="gray")
plt.title(f"Label: {chr(65 + train_labels[0])}")  # Convert number to A-Z letter
plt.show()

# Save processed data for training
np.savez("dataset/preprocessed_data.npz", train_images=train_images, train_labels=train_labels, 
         test_images=test_images, test_labels=test_labels)

print("✅ Dataset preprocessing complete. Ready for training!")
