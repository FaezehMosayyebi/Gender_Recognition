# author: Faezeh Mosayyebi
# face landmark detector using dlib library 

import numpy as np
import dlib
import cv2
import os
from face_processes.facedetection.face_detection import FaceDetection
from utils import *


class LandmakDetector(object):
  def __init__(self, landmark_number:int) -> None:

    self.landmark_number = landmark_number
    self.face_detector = FaceDetection()

    # Landmark detection model
    if self.landmark_number == 5:
      landmarks ="shape_predictor_5_face_landmarks.dat"
    elif self.landmark_number == 68:
      landmarks="shape_predictor_68_face_landmarks.dat"

    self.landmark_predictor = dlib.shape_predictor(landmarks)

  def shape_to_np(self, landmarks, num:int, dtype='int'):
    """
      Convert landmarks to an array of points
    """
    coords = np.zeros((num, 2), dtype=dtype)
    for i in range (0,num):
      coords[i][0] = landmarks.part(i).x
      coords[i][1] = landmarks.part(i).y
    return coords
  
  def detect(self, image, rect):

    """
    Detects all landmarks of a deteted face.
    params:
      input:
        image
        rect of detected faces in the image
      output:
        coords of the landmarks of a face.
    """

    landmarks = self.landmark_predictor(image, rect)
    coords = self.shape_to_np(landmarks, self.landmark_number)

    return coords
  
  def flow_from_directory(self, src_dir:str, dest_dir:str, prefix:str) -> None:

    """
    Reads all the images of a directory and detects landmarks of all faces in them.
    params:
      src_dir: The source directory to read images.
      src_dir: The destination directory to save images.
      prefix: The prefix to save images with.
    """

    if path_valiadtor(src_dir):
      directory_maker(dest_dir)
      for file in os.listdir(src_dir): 
          if file.endswith(".jpg"): 

              img = cv2.imread(os.path.join(src_dir, file))

              if img is not None:
                  rects =self.face_detector.detect(img)
                  for i in range(len(rects)):
                    coords = self.detect(img, rects[i])
                    for (x, y) in coords:
                      cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)

                  if prefix is not None:
                      file = prefix + file
                  cv2.imwrite(os.path.join(dest_dir, file), img)
