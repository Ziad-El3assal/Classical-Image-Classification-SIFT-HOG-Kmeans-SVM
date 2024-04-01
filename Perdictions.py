from FeatureExtraction import Get_Discriptor
import cv2
import numpy as np
from Get_BOWs import ClassHistoGram
from sklearn.neighbors import KNeighborsClassifier
def cosineSimilarity(vec_a,vec_b):
    return np.dot(vec_a,vec_b)/(np.linalg.norm(vec_a)*np.linalg.norm(vec_b))
def NaivePredicions(image,ClassDHistograms,clf,Type='SIFT'):
    """
    Here The problem is handled as it is an Unsupervised Learning problem
    And the similarity between the histograms is used to make the prediction.
    This function takes an image and computes the HOG descriptor.

    Args:
        image (numpy.ndarray): Grayscale image as a NumPy array.
        ClassDiscriptor (dict): Dictionary of class descriptors
        Type (str): Type of descriptor to use ('SIFT' or 'HOG')
        
    Returns:
        str: Predicted class label.
    """
    # Get the descriptor function
    get_descriptor = Get_Discriptor(Type)
    Discriptors=get_descriptor(image)
    # Compute the histogram of the predicted classes
    hist = ClassHistoGram(Discriptors,clf)
    # Find the class with the highest similarity
    max_similarity = 0
    predicted_class = None
    for label, class_hist in ClassDHistograms.items():
        for i in class_hist:
            #print(len(i))
            if len(i) != len(hist):
                continue
            
            similarity = cosineSimilarity(i,hist)
            if similarity > max_similarity:
                max_similarity = similarity
                predicted_class = label
    return predicted_class

def getNearseteKImages(Bow,CLH,label,numK=5):
    """
    This Function takes a Bag of Words and a dictionary of class histograms and returns the nearest K images.
    Args:
        Bow (numpy.ndarray): Bag of Words as a NumPy array.
        classHistogram (dict): Dictionary of class histograms.
    Returns:
        list: List of K nearest images.
    """
    knn=KNeighborsClassifier(n_neighbors=numK)
    knn.fit(CLH[label])
    nearest=knn.kneighbors(Bow)
    
