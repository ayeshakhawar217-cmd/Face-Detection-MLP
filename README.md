# Face vs Non-Face Classification (MLP from Scratch in NumPy)

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24-013243?logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

This project implements a fully connected Multi-Layer Perceptron (MLP) built entirely from scratch using NumPy to classify grayscale images into two classes: face and non-face. No deep learning libraries such as TensorFlow or PyTorch are used. The model includes manual implementation of forward propagation, backpropagation, and gradient descent to demonstrate the core mechanics of neural networks.

## Model Architecture
Input (4096) → Hidden Layer 1 (256, ReLU) → Hidden Layer 2 (128, ReLU) → Output Layer (1, Sigmoid)

## Dataset
Images are organized into two folders:
face_images/faced  
face_images/non_faced  

All images are converted to grayscale, resized to 64x64, normalized, and flattened into 4096-dimensional vectors.

## Key Steps
- Image preprocessing (grayscale conversion, resizing, normalization)
- Forward propagation through fully connected layers
- Binary cross-entropy loss computation
- Backpropagation for gradient calculation
- Gradient descent weight updates
- Model evaluation on test data


## Training Results

The model shows consistent convergence during training, with the loss decreasing steadily across epochs:

Epoch 0: Loss = 1.0934  
Epoch 100: Loss = 0.2223  
Epoch 200: Loss = 0.1300  
Epoch 300: Loss = 0.0903  
Epoch 400: Loss = 0.0682  
Epoch 500: Loss = 0.0542  
Epoch 600: Loss = 0.0446  
Epoch 700: Loss = 0.0377  
Epoch 800: Loss = 0.0325  
Epoch 900: Loss = 0.0284  

This indicates that the model successfully learned meaningful patterns from the training data and minimized the binary cross-entropy loss effectively.

## Evaluation Results

Test Accuracy: 80.49%

Classification Report:
- Class 0 (Non-Face): Precision = 0.76, Recall = 0.76, F1-score = 0.76  
- Class 1 (Face): Precision = 0.83, Recall = 0.83, F1-score = 0.83  

Overall performance shows balanced classification ability across both classes, with slightly better performance on face images. The model achieves around 80% accuracy on unseen test data, demonstrating that a simple MLP built from scratch can effectively learn discriminative features from raw pixel inputs.
## Technologies Used
Python, NumPy, Matplotlib, Scikit-learn, Pillow

## Output
The model predicts whether an image contains a face or not and evaluates performance using accuracy and classification metrics.
