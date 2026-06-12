import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from PIL import Image

# Define paths
base_folder = "face_images"
categories = ['faced', 'non_faced']
label_map = {'faced': 1, 'non_faced': 0}

# Image size
IMG_SIZE = 64

# Load and preprocess images
data = []
labels = []

for category in categories:
    folder_path = os.path.join(base_folder, category)
    valid_images = [img for img in os.listdir(folder_path) if img.endswith('.png') and img.split('.')[0].isdigit()]
    
    for image_name in valid_images:
        image_path = os.path.join(folder_path, image_name)
        try:
            img = Image.open(image_path).convert('L')  # Grayscale
            img = img.resize((IMG_SIZE, IMG_SIZE))     # Resize
            img_array = np.array(img)

            # Normalize: zero mean and unit variance
            img_array = (img_array - np.mean(img_array)) / (np.std(img_array) + 1e-8)
            img_flat = img_array.flatten()

            data.append(img_flat)
            labels.append(label_map[category])
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Build your OWN MLP model (custom) ---
np.random.seed(42)

input_size = X_train.shape[1]  # 64x64 = 4096
hidden_size1 = 256
hidden_size2 = 128
output_size = 1

# Initialize weights
W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2. / input_size)
b1 = np.zeros((1, hidden_size1))

W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2. / hidden_size1)
b2 = np.zeros((1, hidden_size2))

W3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2. / hidden_size2)
b3 = np.zeros((1, output_size))

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Training parameters
learning_rate = 0.001
epochs = 1000

# Training loop
for epoch in range(epochs):
    # Forward pass
    Z1 = np.dot(X_train, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)

    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)  # Final output

    # Loss (Binary Cross Entropy)
    loss = -np.mean(y_train * np.log(A3[:, 0] + 1e-8) + (1 - y_train) * np.log(1 - A3[:, 0] + 1e-8))

    # Backward pass
    dZ3 = A3 - y_train.reshape(-1, 1)
    dW3 = np.dot(A2.T, dZ3) / X_train.shape[0]
    db3 = np.mean(dZ3, axis=0, keepdims=True)

    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2) / X_train.shape[0]
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X_train.T, dZ1) / X_train.shape[0]
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    # Update weights
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Print loss occasionally
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss={loss:.4f}")

# --- Testing ---
# Predict on test set
Z1_test = np.dot(X_test, W1) + b1
A1_test = relu(Z1_test)

Z2_test = np.dot(A1_test, W2) + b2
A2_test = relu(Z2_test)

Z3_test = np.dot(A2_test, W3) + b3
A3_test = sigmoid(Z3_test)

y_pred_test = (A3_test[:, 0] > 0.5).astype(int)

# Accuracy
accuracy_test = accuracy_score(y_test, y_pred_test) * 100
print(f"\nTest Accuracy: {accuracy_test:.2f}%")

# Report (no warnings)
print(classification_report(y_test, y_pred_test, zero_division=0))

# --- Visualize ONLY Test Images ---
cols = 5
rows = (len(X_test) + cols - 1) // cols

plt.figure(figsize=(cols*2, rows*2))

for i in range(len(X_test)):
    img_array = X_test[i].reshape(IMG_SIZE, IMG_SIZE)
    prediction = y_pred_test[i]

    plt.subplot(rows, cols, i + 1)
    plt.imshow(img_array, cmap='gray')
    plt.title(f"{'Face' if prediction == 1 else 'Non-Face'}", fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.show()
