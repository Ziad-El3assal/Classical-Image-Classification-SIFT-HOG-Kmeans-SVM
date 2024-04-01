
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
def cast_vector(row):
    return np.array(list(map(lambda x: x.astype('double'), row)))
def train_svm(descriptors, k=10):
  """
  This function trains an SVM classifier on the provided descriptors and labels.

  Args:
      descriptors (list): List of feature descriptors (each descriptor as a NumPy array).
      labels (list): List of labels corresponding to each descriptor.

  Returns:
      sklearn.svm.SVC: The trained SVM model.
  """
 
  print("Training SVM...")
  print("Descriptors shape: ", descriptors.shape)   
  clf = KMeans(n_clusters=k, max_iter=1000, random_state=True, n_init=50)
  clf.fit(descriptors)
  return clf


def ClassHistoGram(Discriptors,clf,plot=False):
    """
    This function takes a list of descriptors and a trained SVM model, and returns the histogram of predicted classes.
    
    Args:
        descriptors (list): List of feature descriptors (each descriptor as a NumPy array).
        clf (sklearn.svm.SVC): Trained SVM model.
    
    Returns:
        numpy.ndarray: Histogram of predicted classes.
    """
    Discriptors = np.array(Discriptors)
    # print("Predicting...")
    # print("Descriptors shape: ", Discriptors.shape)
    try:
        predictions = clf.predict(Discriptors)
        # print(predictions)
        hist = np.histogram(predictions, bins=range(clf.n_clusters+1))[0]
        if plot:
            plt.bar(range(clf.n_clusters), hist)
            plt.show()
        return hist
    except:
        return []

def getClassesHistogram(CDiscriptor,clf,):
    """
    This Function gets the classes Disriptors and returns the histogram of the classes
    From This point we can make a predicition using the similarity of the histogram
    Args:
        Classes (dict): Dictionary of classes with descriptors for each class.
        clf (sklearn.svm.SVC): Trained SVM model.
    """
    Histograms={}
    for i in CDiscriptor.keys():
        Histograms[i]=[]
        for j in CDiscriptor[i]:
            x=ClassHistoGram(j,clf)
            if len(x)==0 or x is None:
                continue
            Histograms[i].append(x)
    return Histograms
