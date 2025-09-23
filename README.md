# Sugarcane-Leaf-Disease-Classification-Using-Deep-Learning
This project utilizes convolutional neural networks with transfer learning to classify sugarcane leaf diseases from images. It includes preprocessing, augmentation, and fine-tuning to improve multi-class disease detection accuracy, providing tools to help farmers with practical AI-based plant health diagnosis.
# Sugarcane Leaf Disease Classification Using Deep Learning

## Overview

Sugarcane diseases reduce crop yield and farmer income significantly. This project provides an AI-based solution to automatically detect diseases from sugarcane leaf images using deep learning models (ResNet50/ResNet152V2). It facilitates timely interventions for better crop protection in regions like Baramati, Maharashtra.

## Features

- Dataset preprocessing, cleaning, and augmentation
- Transfer learning with pretrained ResNet architectures
- Training, validation, and testing code with metrics
- Visualization of confusion matrix and classification reports
- Prediction interface for inference on new images
- Model saving and loading for future use

## Dataset

The dataset is sourced from Kaggle: [Sugarcane Leaf Disease Dataset](https://www.kaggle.com/datasets/nirmalsankalana/sugarcane-leaf-disease-dataset). It includes images labeled with six disease categories:
- Healthy
- Mosaic
- Red Rot
- Rust
- Yellow Leaf
- Bacterial Blight

## Setup Instructions

### Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- scikit-learn
- Kaggle API and credentials (for dataset download)

Install dependencies:


