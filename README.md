Project Overview

This project explores image classification techniques using Bag-of-Words (BOW) representations and lays the groundwork for future implementation of machine learning models. It provides a modular structure for feature extraction, BOW generation, and basic prediction approaches.

Project Structure

The project is organized into the following core files:

init.sh: This script downloads the dataset and sets up environment variables. It requires the user to have saved Kaggle credentials beforehand.
main.py: This is the central control script that utilizes the other modules for data loading, feature extraction (SIFT or HOG), BOW generation, and performing naive predictions using cosine similarity or K-Nearest Neighbors (KNN).
Prediction.py: This module implements the naive prediction methods:
Cosine Similarity: Calculates the cosine similarity between the input image's BOW and the BOWs of images in the dataset, returning the image with the highest similarity as the predicted class.
KNN: Employs the KNN algorithm to find the k most similar images to the input image based on their BOWs. The predicted class is determined by majority vote among the nearest neighbors' classes.
Get_BOWs.py: This module handles BOW generation and histogram calculations:
K-Means Clustering: Performs K-Means clustering to create the visual vocabulary (codebook) for generating BOWs.
Single Image Histogram: Computes the histogram of feature descriptors for a single image.
Mega Histogram: Creates a combined histogram representing the entire dataset, capturing the distribution of features across all classes.
FeatureExtraction.py: This module contains implementations for extracting feature descriptors:
SIFT (Scale-Invariant Feature Transform): Detects and describes keypoints in images that are invariant to scale and rotation.
HOG (Histogram of Oriented Gradients): Computes the distribution of local gradients over fixed-size blocks, capturing local object shape and appearance.
Future Work

The project lays a solid foundation for further exploration of machine learning techniques:

Implement an SVM (Support Vector Machine) classifier for each class based on the BOW representations. This can significantly improve prediction accuracy compared to the current naive methods.
Experiment with different feature descriptors and clustering algorithms to optimize performance for your specific dataset.
Explore more advanced machine learning models like convolutional neural networks (CNNs) for more robust and accurate image classification.
Running the Project

Set Up Environment: Ensure you have Python (version 3 recommended) and any necessary libraries installed (e.g., scikit-learn, OpenCV). Consider using a virtual environment for project isolation.
Download Dataset and Set Credentials: Run init.sh to download the dataset and set environment variables using your Kaggle credentials (follow instructions in the script).
Run the Code: Navigate to the project directory and execute python main.py. This will load the data, extract features, generate BOWs, and perform naive predictions. You can modify parameters in main.py to experiment with different settings.
Disclaimer:

This project is for educational and research purposes only. Adapt it responsibly for your specific use case.
The provided code might require adjustments depending on the downloaded dataset.
