## Image Classification with Bag-of-Words (BOW)

**Introduction**

This project explores image classification using BOW representations and lays the groundwork for machine learning exploration ([[https://en.wikipedia.org/wiki/Machine_learning](https://en.wikipedia.org/wiki/Machine_learning)]). It provides a modular structure for feature extraction, BOW generation, and basic prediction approaches.

**Project Structure**

* `init.sh`: Downloads dataset and sets environment variables (requires pre-saved Kaggle credentials).
* `main.py`: Central control script for data loading, feature extraction (SIFT/HOG), BOW generation, and naive predictions (cosine similarity/KNN).
* `Prediction.py`: Implements naive prediction methods:
    * Cosine similarity
    * K-Nearest Neighbors (KNN)
* `Get_BOWs.py`: Handles BOW generation and histograms:
    * K-Means clustering for visual vocabulary creation
    * Single image histogram
    * Mega histogram (entire dataset)
* `FeatureExtraction.py`: Implements feature descriptors:
    * SIFT (Scale-Invariant Feature Transform)
    * HOG (Histogram of Oriented Gradients)

**Future Work**

* Implement SVM classifiers for each class based on BOWs.
* Experiment with feature descriptors and clustering algorithms.
* Explore advanced models like convolutional neural networks (CNNs).

**Running the Project**

1. Ensure Python (v3 recommended) and libraries (scikit-learn, OpenCV) are installed.
2. Run `init.sh` to download data and set credentials.
3. Navigate to the project directory and run `python main.py`.

**Disclaimer**

Educational and research purposes only. Adapt responsibly. Code may require adjustments.
