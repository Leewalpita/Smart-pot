# Smart-pot
# Plant Disease Detection System
Tomato Plant Disease Classification using CNN with Real-Time Image Capture
This repository contains a machine learning project that uses a Convolutional Neural Network (CNN) to classify different types of diseases in tomato plants from images. Additionally, the project now includes real-time image capture functionality using an IP webcam, allowing live images to be classified for plant disease detection.

# Project Overview
The goal of this project is to classify tomato plant diseases using a dataset of images, training a CNN to recognize different disease classes. The model takes as input images of tomato plant leaves and outputs predictions for 10 different disease classes. The project has been enhanced with real-time image capture from an IP webcam, which can automatically predict the disease class for live-captured images.

# Dataset
Dataset Path: The dataset used is located in Kaggle Tomato leaf disease detection
The dataset is split into train and val directories containing the training and validation images, respectively.
# Classes
The dataset contains images for the following tomato plant diseases:

Septoria Leaf Spot
Bacterial Spot
Early Blight
Late Blight
Leaf Mold
Mosaic Virus
Spider Mite
Target Spot
Yellow Leaf Curl Virus
Healthy (no disease)

# Model Architecture
The CNN model consists of the following layers:

Convolutional Layers: Several convolutional layers with ReLU activation and same padding.
MaxPooling Layers: For down-sampling feature maps.
Dense Layers: Two fully connected layers with 1024 and 512 units respectively.
Dropout Layers: Used to prevent overfitting.
Output Layer: A softmax layer for 10-class classification.

# Optimizer
Optimizer: Adam with a learning rate of 1e-5.
Loss Function: Sparse categorical cross-entropy.
Real-Time Image Capture Using IP Webcam
The project includes a feature that captures images from an IP webcam and uses the trained CNN model to predict the disease in real-time. Here’s how it works:

IP Webcam Setup: The webcam feed is accessed using a URL (e.g., http://192.168.55.39:8080/photo.jpg).
Image Processing: The image is fetched, decoded using OpenCV, and saved to a local directory for prediction.
Prediction: The captured image is processed, resized, and passed to the trained model for disease prediction.

# Model Training
Training Image Size: 128x128 pixels.
Batch Size: 20.
Epochs: 50 (with early stopping).
Callbacks: Early stopping is used to monitor validation loss and stop training after 7 epochs without improvement.
Real-Time Image Prediction
Once the image is captured from the IP webcam, it can be passed to the trained CNN model for prediction. Here’s the process:

Load the Pre-trained Model: The model is loaded from a saved file (my_model2_new.h5).
Image Preprocessing: The captured image is resized and normalized.
Prediction: The model predicts the disease class for the image

# Future Improvements
Enhance the IP webcam feature with real-time video streaming and continuous predictions.
Explore transfer learning with pre-trained models such as VGG16 or ResNet for improved accuracy.
Add data augmentation to improve model generalizatio

# graphs 
![Figure_2](https://github.com/user-attachments/assets/5874d351-3302-4b3e-95e8-3b07fe24ff90)
![Figure_1](https://github.com/user-attachments/assets/7587f2be-6513-49c9-8201-3a49a4df0e1d)


