import cv2
import os 
import numpy as np
import cv2


def get_sift_descriptors(image_path):
  """
  This function reads an image, detects SIFT keypoints and descriptors,
  and returns them.

  Args:
      image_path (str): Path to the image file.

  Returns:
      tuple: (keypoints, descriptors)
          - keypoints: List of cv2.KeyPoint objects representing detected keypoints.
          - descriptors: NumPy array of shape (num_keypoints, 128) containing SIFT descriptors.

  Raises:
      IOError: If image reading fails.
  """
  # Read the image in grayscale
  try:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
      raise IOError("Error reading image: {}".format(image_path))
  except IOError as e:
    print(e)
    return None, None

  # Create a SIFT object
  sift = cv2.SIFT_create()

  # Detect keypoints and compute descriptors
  keypoints, descriptors = sift.detectAndCompute(img, None)

  return descriptors


def get_hog_descriptor(image):
  """
  This function takes an image and computes the HOG descriptor.

  Args:
      image (numpy.ndarray): Grayscale image as a NumPy array.

  Returns:
      numpy.ndarray: HOG descriptor as a NumPy array.
  """
  # Convert to grayscale if needed (assuming color input)
  if len(image.shape) > 2:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Define HOG parameters (experiment for optimal values)
  win_size = (16, 16)
  block_size = (8, 8)
  block_stride = (4, 4)
  num_bins = 9

  # Create HOG descriptor object
  hog = cv2.HOGDescriptor(win_size, block_size, block_stride, num_bins)

  # Compute HOG descriptor
  hog_descriptor = hog.compute(image)

  return np.array(hog_descriptor)


def Get_Discriptor(type='SIFT'):
    if type == 'SIFT':
        return get_sift_descriptors
    elif type == 'HOG':
        return get_hog_descriptor
    else:
        print('Invalid Discriptor Type')
        return None