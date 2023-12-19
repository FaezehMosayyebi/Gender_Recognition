import numpy as np
import dlib

class LandmakDetector(object):
  def __init__(self, landmark_number) -> None:

    self.landmark_number = landmark_number

    # Landmark detection model
    if self.landmark_number == 5:
      landmarks ="shape_predictor_5_face_landmarks.dat"
    elif self.landmark_number == 68:
      landmarks="shape_predictor_68_face_landmarks.dat"

    self.landmark_predictor = dlib.shape_predictor(landmarks)

  def shape_to_np(landmarks,n, dtype='int'):
    """
      Convert landmarks to an array of points
    """
    coords = np.zeros((n, 2), dtype=dtype)
    for i in range (0,n):
      coords[i][0] = landmarks.part(i).x
      coords[i][1] = landmarks.part(i).y
    return coords
  
  def detect(self, image, rect):

    landmarks = self.landmark_predictor(image, rect)
    coords = self.shape_to_np(landmarks, self.landmark_number)

    return coords
